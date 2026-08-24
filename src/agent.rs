use crate::client::Client;
use crate::config::Config;
use crate::dsl::{Cmd, Lexer};
use crate::ledger::{est_tokens, Kind, Ledger};
use crate::render::Renderer;
use crate::tools::{prune, run_shell, Workspace};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_TOOL_TIMEOUT_MS: u64 = 120_000;
const MAX_EMPTY_TURNS: u32 = 3;

/// Live events for a UI. Headless runs pass Ctl::default() and pay nothing.
pub enum Ev {
    Turn(u32),
    /// Raw model stream chunk (top-level agent only).
    Delta(String),
    Action(u8, String),
    Result(u8, String),
    /// The agent speaking to the user mid-run (S verb).
    Say(String),
    /// End-of-run stats line.
    Report(String),
    Done(String),
    /// Real billed context size (prompt_tokens) of the latest request.
    Ctx(u64),
    /// A named subagent's latest activity line (for a UI's agents pane).
    Sub(String, String),
    /// The named subagent finished.
    SubDone(String),
    /// Current compact plan view (for a UI's plan pane).
    Plan(String),
}

#[derive(Clone, Default)]
pub struct Ctl {
    pub sink: Option<std::sync::mpsc::Sender<Ev>>,
    pub stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Messages the user sends mid-run; drained into the ledger at the top of
    /// each top-level turn, so you can steer the leader while it works.
    pub inbox: Option<std::sync::Arc<std::sync::Mutex<Vec<String>>>>,
    /// Set on a subagent's Ctl: its activity is re-labeled Ev::Sub(name, …)
    /// so a UI can route it to an agents pane instead of the main chat.
    pub tag: Option<String>,
}

impl Ctl {
    fn emit(&self, ev: Ev) {
        let Some(s) = &self.sink else { return };
        let ev = match (&self.tag, ev) {
            (Some(t), Ev::Action(d, a)) if d > 0 => Ev::Sub(t.clone(), a),
            (Some(t), Ev::Result(d, r)) if d > 0 => Ev::Sub(t.clone(), r),
            // A subagent's turn counter must not fight the leader's.
            (Some(_), Ev::Turn(_)) => return,
            (_, ev) => ev,
        };
        let _ = s.send(ev);
    }
    fn stopped(&self) -> bool {
        self.stop
            .as_ref()
            .is_some_and(|s| s.load(std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Debug, Default)]
pub struct Report {
    pub final_msg: String,
    pub turns: u32,
    pub wall_ms: u128,
    pub model_ms: u128,
    pub ttft_ms_sum: u128,
    pub tool_ms: u128,
    pub render_us: u128,
    pub sent_tokens: usize,
    pub out_chars: usize,
    pub commands: usize,
    /// Exact provider-reported usage (0 when the provider doesn't send it).
    pub tok_in: u64,
    pub tok_cached: u64,
    pub tok_out: u64,
    /// History seals (model compactions) performed during the run.
    pub seals: u32,
}

/// A continuable session: the ledger, workspace, and renderer survive across
/// tasks so a TUI conversation has memory. Headless runs make one and drop it.
pub struct Session {
    pub ledger: Ledger,
    pub ws: Workspace,
    pub renderer: Renderer,
    root: PathBuf,
    turn_base: u32,
}

impl Session {
    pub fn new(cfg: &Config, root: PathBuf, depth: u8) -> Session {
        let tee = (depth == 0).then(|| root.join(".haste").join("ledger.jsonl"));
        let mut ledger = Ledger::new(tee.as_deref());
        if depth == 0 && cfg.context.bootstrap {
            ledger.push(
                Kind::Pin,
                0,
                crate::bootstrap::workspace_state(&root, &cfg.exec.shell, &cfg.context.instruction_files),
                None,
            );
        }
        Session {
            ledger,
            ws: Workspace::new(root.clone()),
            renderer: Renderer::new(),
            root,
            turn_base: 0,
        }
    }
}

pub fn run(cfg: Arc<Config>, root: PathBuf, task: &str, profile: Option<&str>, depth: u8, ctl: Ctl) -> Report {
    let mut session = Session::new(&cfg, root, depth);
    run_session(cfg, &mut session, task, profile, depth, ctl)
}

pub fn run_session(
    cfg: Arc<Config>,
    session: &mut Session,
    task: &str,
    profile: Option<&str>,
    depth: u8,
    ctl: Ctl,
) -> Report {
    let t_start = Instant::now();
    let prof = profile.and_then(|p| cfg.profile.get(p));
    let (max_turns, budget) = match prof {
        Some(p) => (p.max_turns, p.budget_tokens),
        None => (cfg.context.max_turns, cfg.context.budget_tokens),
    };
    let mut ctx_cfg = cfg.context.clone();
    ctx_cfg.budget_tokens = budget;

    let root = session.root.clone();
    let ledger = &mut session.ledger;
    let ws = &mut session.ws;
    let renderer = &mut session.renderer;
    ledger.push(Kind::Task, session.turn_base, task.to_string(), None);
    let client = Client::new(cfg.model.clone(), cfg.api_key());
    let system = build_system(&cfg, prof.map(|p| p.system.as_str()), prof.map(|p| p.tools.as_str()));
    let allowed: Option<Vec<char>> = prof.map(|p| p.tools.chars().collect());

    let mut rep = Report::default();
    let mut empty_turns = 0u32;
    let mut degens = 0u32;
    let mut refusals = 0u32;
    let mut consec_errs = 0u32;
    // Real billed context: prompt_tokens from the provider's last usage
    // report. Budget decisions prefer this over any estimate.
    let mut last_prompt: u64 = 0;
    // Legend-delta cursor: files at index < this were already announced (or
    // are baked into a seal); the per-turn tail only lists what's new.
    let mut legend_base = 0usize;
    // "Now scanning..." endings: how often we have refused to stop on
    // trailing-ellipsis narration. Capped so a model that really does end
    // every sentence with dots can still finish.
    let mut ellipsis_refusals = 0u32;
    // Images queued by V: attached to exactly the next request, then dropped —
    // the model sees them once and can V again if it needs another look.
    let mut images: Vec<(String, String)> = Vec::new();
    let base = session.turn_base;
    // Background subagents: spawned here, never blocking a turn. Finished ones
    // are harvested at the top of each turn; D waits for stragglers.
    // (profile, task, handle) — task kept for duplicate-spawn refusal.
    let mut pending: Vec<(String, String, std::thread::JoinHandle<Report>)> = Vec::new();
    // Background auto-verify: (ok, ready-to-inject note). Spawned after an
    // editing turn, harvested at a turn top, force-joined by D.
    let mut verify_bg: Option<std::thread::JoinHandle<(bool, String)>> = None;
    // Loop breaker: per exact command, how many times in a row it produced the
    // exact same result. 3 → warn; 5 → refuse to execute it again.
    let mut repeats: std::collections::HashMap<u64, (u32, u64)> = std::collections::HashMap::new();
    // Plan state machine: last-seen step statuses, for detecting done-transitions.
    let plan_path = root.join(&cfg.plan.file);
    let mut plan_seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    let mut i = 0u32;
    loop {
        i += 1;
        // max_turns = 0 means unlimited: the loop breaker, empty-turn abort,
        // and Esc are the real guards against runaways.
        if max_turns > 0 && i > max_turns {
            break;
        }
        // Ledger turn numbers keep counting across tasks in a continued
        // session so the renderer's age-based folding stays correct.
        let turn = base + i;
        if ctl.stopped() {
            rep.final_msg = "(stopped)".into();
            break;
        }
        rep.turns = i;
        ctl.emit(Ev::Turn(i));

        // Mid-run steering: user messages land as fresh task lines.
        if depth == 0 {
            if let Some(inbox) = &ctl.inbox {
                let msgs: Vec<String> = std::mem::take(&mut *inbox.lock().unwrap());
                for m in msgs {
                    ledger.push(Kind::Task, turn, m, None);
                }
            }
        }

        // Harvest any subagents that finished while we were working — their
        // briefs enter the context now, without ever having blocked a turn.
        let mut k = 0;
        while k < pending.len() {
            if pending[k].2.is_finished() {
                let (name, _, h) = pending.remove(k);
                let sub = h.join().unwrap_or_default();
                harvest(&name, sub, ledger, &ctl, depth, turn, &mut rep);
            } else {
                k += 1;
            }
        }

        // Harvest a finished background verify: its PASS/FAIL note enters the
        // context like any tool result, without ever having blocked a turn.
        if verify_bg.as_ref().is_some_and(|h| h.is_finished()) {
            if let Some(h) = verify_bg.take() {
                if let Ok((_, text)) = h.join() {
                    note(ledger, &ctl, depth, turn, text);
                }
            }
        }

        // Plan tick: verify fresh "done"s, get the always-visible view. Runs
        // BEFORE compaction so a completed step can trigger a phase seal.
        let dones_before = plan_seen.values().filter(|s| s.as_str() == "done").count();
        let (plan_view, _) = if cfg.plan.enforce {
            plan_tick(&plan_path, &mut plan_seen, &cfg.exec.shell, &root, ledger, &ctl, depth, turn)
        } else {
            (String::new(), Vec::new())
        };
        let phase_done = plan_seen.values().filter(|s| s.as_str() == "done").count() > dones_before;
        if depth == 0 {
            ctl.emit(Ev::Plan(plan_view.clone()));
        }

        // Model compaction: ask the model to summarize its own history. The
        // prompt is the SAME document the provider's KV cache already holds
        // plus one appended instruction — prefill is ~free, we pay only the
        // summary's decode. The ledger stays lossless; the summary is a
        // render-layer seal. Falls back to structural folding.
        // Triggers: over budget, OR a plan step just completed and the doc is
        // past the phase floor — that phase's raw history is dead weight, and
        // the model summarizes best right at the boundary. Long tasks then
        // hold a near-constant context instead of sawtoothing to the budget.
        let phase_floor = if ctx_cfg.compact_phase_tokens > 0 {
            ctx_cfg.compact_phase_tokens
        } else {
            ctx_cfg.budget_tokens / 3
        };
        let real_ctx = (last_prompt > 0).then_some(last_prompt as usize);
        let seal_now = ctx_cfg.compact == "model"
            && (renderer.seal_due(ledger, &ctx_cfg, ctx_cfg.budget_tokens, real_ctx)
                || (phase_done && renderer.seal_due(ledger, &ctx_cfg, phase_floor, real_ctx)));
        if seal_now {
            let doc = renderer.render(ledger, &ctx_cfg, turn);
            let cuser = format!(
                "{doc}\n## COMPRESS\nCompact your working memory: write a terse briefing (max 40 lines) \
                 preserving the current task and its exact state, files touched (#ids and paths), key \
                 findings and decisions, remaining steps, and pitfalls already hit. Output only the briefing.\n"
            );
            let mut summary = String::new();
            if let Ok(s) = client.stream(&system, &cuser, &[], &mut |d| summary.push_str(d)) {
                rep.model_ms += s.total_ms;
                rep.tok_in += s.prompt_tokens;
                rep.tok_cached += s.cached_tokens;
                rep.tok_out += s.completion_tokens;
                let summary = clean_final(summary.trim());
                if !summary.is_empty() {
                    // Bake the FULL file legend into the seal — it rides the
                    // cached prefix from here on, so the per-turn tail only
                    // needs to announce newly interned files.
                    let sealed = format!("{summary}\n{}", ws.legend()).trim().to_string();
                    renderer.seal_summary(ledger, ctx_cfg.compact_keep_last, sealed);
                    legend_base = ws.file_count();
                    rep.seals += 1;
                    let note = "(history compacted via prompt-cached model summary)";
                    ctl.emit(Ev::Result(depth, note.into()));
                }
            }
        }

        let t_r = Instant::now();
        let doc = renderer.render(ledger, &ctx_cfg, turn);
        // File legend and plan view go at the BOTTOM: they change as work
        // progresses, and anything that changes near the top of the document
        // would shift every byte after it and invalidate the prefix cache.
        // With model sealing, the legend is delta-only: each file announced
        // once, the full table lives inside the seals.
        let delta_legend = ctx_cfg.mode == "append" && ctx_cfg.compact == "model";
        let legend = if delta_legend { ws.legend_from(legend_base) } else { ws.legend() };
        let legend_snapshot = ws.file_count();
        let user = format!("{}\n{}{}## NOW\nNext commands:\n", doc, legend, plan_view);
        rep.render_us += t_r.elapsed().as_micros();
        rep.sent_tokens += est_tokens(&system) + est_tokens(&user);

        let turn_images = std::mem::take(&mut images);
        let mut lexer = Lexer::new();
        // S and D wait for stream end (solo-S detection and rescue_done need
        // the whole turn); every other command executes the moment the lexer
        // completes it, WHILE the model is still generating — tool time hides
        // inside generation time instead of following it.
        let mut tail: Vec<Cmd> = Vec::new();
        let mut chunk: Vec<Cmd> = Vec::new();
        let mut tc = TurnCtx {
            cfg: &cfg,
            root: &root,
            ledger: &mut *ledger,
            ws: &mut *ws,
            client: &client,
            ctl: &ctl,
            ctx_cfg: &ctx_cfg,
            allowed: &allowed,
            task,
            depth,
            turn,
            images: &mut images,
            pending: &mut pending,
            repeats: &mut repeats,
            refusals: &mut refusals,
            done: None,
            edited: false,
            executed: 0,
            tool_us: 0,
            says: Vec::new(),
        };
        let stream_res = client.stream(&system, &user, &turn_images, &mut |delta| {
            if tc.depth == 0 {
                tc.ctl.emit(Ev::Delta(delta.to_string()));
            }
            lexer.feed(delta, &mut chunk);
            for cmd in chunk.drain(..) {
                if matches!(cmd, Cmd::Say { .. } | Cmd::Done { .. }) {
                    tail.push(cmd);
                } else {
                    tc.dispatch(cmd);
                }
            }
        });
        let (length_capped, degenerated) = match stream_res {
            Ok(s) => {
                consec_errs = 0;
                if delta_legend {
                    legend_base = legend_snapshot;
                }
                if s.prompt_tokens > 0 {
                    last_prompt = s.prompt_tokens;
                    // Fold the real usage into the estimator (text-only
                    // requests — image tokens would skew the ratio).
                    if turn_images.is_empty() {
                        crate::ledger::calibrate(system.len() + user.len(), s.prompt_tokens);
                    }
                    if depth == 0 {
                        ctl.emit(Ev::Ctx(s.prompt_tokens));
                    }
                }
                rep.model_ms += s.total_ms;
                rep.ttft_ms_sum += s.ttft_ms;
                rep.out_chars += s.out_chars;
                rep.tok_in += s.prompt_tokens;
                rep.tok_cached += s.cached_tokens;
                rep.tok_out += s.completion_tokens;
                let fr = s.finish_reason.as_deref();
                (fr == Some("length"), fr == Some("degenerate"))
            }
            Err(e) => {
                consec_errs += 1;
                tc.note(format!("model error: {e}"));
                if consec_errs >= 6 {
                    rep.final_msg = format!(
                        "(aborted: the model endpoint failed {consec_errs} times in a row — is the server at {} up?)",
                        cfg.model.base_url
                    );
                    break;
                }
                // Endpoint hiccups must not burn turns: back off (0.5s
                // doubling to a 4s cap) and retry the SAME turn.
                std::thread::sleep(std::time::Duration::from_millis(500u64 << (consec_errs - 1).min(3)));
                i -= 1;
                continue;
            }
        };
        lexer.finish(&mut tail);
        rescue_done(&mut tail, &cfg);

        if degenerated {
            degens += 1;
            // Every note is unique text: stacking IDENTICAL lines was itself a
            // repetition attractor that made the next collapse more likely.
            let note_text = match degens {
                1 => "(output collapsed into repetition and was cut — retry)".to_string(),
                2 => "(collapsed again — reply with ONE short command only, e.g. `X dir`)".to_string(),
                n => format!("(repetition collapse #{n} — emit a single tiny command, nothing else)"),
            };
            tc.note(note_text);
            // Commands parsed before the collapse already executed mid-stream;
            // a deferred D is dropped — its message may be full of the spam tail.
            tail.retain(|c| !matches!(c, Cmd::Done { .. }));
            // A stuck model produces the same collapse forever — stop burning
            // turns and tell the user instead of retrying unboundedly.
            if degens >= 6 {
                rep.final_msg =
                    "(aborted: output collapsed 6 times in a row — the model is stuck on this prompt; rephrase or try again)".into();
                break;
            }
            // A spam-only turn retries without counting toward the empty-turn abort.
            if tc.executed == 0 && tail.is_empty() {
                continue;
            }
        } else {
            degens = 0;
        }

        if tc.executed == 0 && tail.is_empty() {
            empty_turns += 1;
            if empty_turns >= MAX_EMPTY_TURNS {
                rep.final_msg = "(aborted: model produced no commands)".into();
                break;
            }
            tc.ledger.push(
                Kind::Result,
                turn,
                "no commands parsed — plain prose is discarded. Use S <text> to talk to the user, or D to finish".into(),
                None,
            );
            continue;
        }
        empty_turns = 0;

        // Prose mixed into a command turn vanishes silently — the classic
        // symptom is "here are the results:" followed by a list the user
        // never sees. Tell the model exactly what it lost.
        if lexer.dropped > 0 {
            tc.note(format!(
                "({} plain-prose line{} DISCARDED — anything meant for the user must be inside one S line or the D message)",
                lexer.dropped,
                if lexer.dropped == 1 { " was" } else { "s were" }
            ));
        }

        // A turn that is ONLY talk (S with no work and no subagents pending)
        // means the model is waiting on the user: hand the mic back instead of
        // looping into ever-rephrased clarification questions.
        // (dropped >= 3 means a whole prose block was lost — the model was
        // mid-report, not waiting on the user: let it retry with the note.)
        let mut say_final = false;
        let mut nudge_continue = false;
        if tc.executed == 0
            && tc.pending.is_empty()
            && !tail.is_empty()
            && tail.iter().all(|c| matches!(c, Cmd::Say { .. }))
            && lexer.dropped < 3
        {
            // "Found 11 dirs. Now scanning each..." is announcing the NEXT
            // step, not waiting on the user — a trailing ellipsis must not
            // hand the mic back and end the run mid-task.
            let trailing = tail
                .iter()
                .rev()
                .find_map(|c| match c {
                    Cmd::Say { text } => Some(text.trim_end()),
                    _ => None,
                })
                .is_some_and(|t| t.ends_with("...") || t.ends_with('…'));
            if trailing && ellipsis_refusals < 3 {
                ellipsis_refusals += 1;
                nudge_continue = true;
            } else {
                say_final = true;
                tc.executed += tail.len();
                let text = tail
                    .drain(..)
                    .map(|c| match c {
                        Cmd::Say { text } => text,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                tc.done = Some(text);
            }
        }
        for cmd in tail.drain(..) {
            tc.dispatch(cmd);
        }
        if nudge_continue {
            tc.note("(you sound mid-task — do the next step NOW with commands; S alone ends nothing but stops the work, D is only for finished results)".into());
        }
        rep.commands += tc.executed;
        rep.tool_ms += tc.tool_us / 1000;
        let TurnCtx { done, edited, says, .. } = tc;
        // Auto-verify: after any editing turn, the configured check runs in a
        // BACKGROUND thread — the model keeps working while tests run; the
        // result is harvested at a later turn top. A D joins the run first
        // (below), so a failing verify still refuses the same-turn D. The
        // model's explicit "run the tests" turn stays deleted.
        if edited {
            if let Some(vcmd) = &cfg.verify.cmd {
                // A fresh edit makes any in-flight verify stale: the handle is
                // replaced below, detaching the old thread; its result is discarded.
                let vcmd = vcmd.clone();
                let root2 = root.clone();
                let shell = cfg.exec.shell.clone();
                let spec = cfg.verify.prune.clone();
                let tmo = cfg.verify.timeout_ms;
                verify_bg = Some(std::thread::spawn(move || {
                    let out = run_shell(&vcmd, &root2, tmo, &shell);
                    let ok = out.starts_with("ok");
                    let text = format!("(auto-verify {}: `{vcmd}`)\n{}", if ok { "PASS" } else { "FAIL" }, prune(&spec, &out));
                    (ok, text)
                }));
            }
        }

        // A model that ignores refusals and re-sends the same command forever
        // (deterministic samplers have no variance to escape with) must not
        // burn unlimited turns.
        if refusals >= 12 {
            rep.final_msg =
                "(aborted: the model kept repeating a refused command — it is stuck; rephrase the task or use a different model)".into();
            break;
        }

        if length_capped {
            note(ledger, &ctl, depth, turn, "(your output hit the max_tokens limit mid-message — any cut-off file was written PARTIALLY. Read it, then CONTINUE it with I <id> <last-line> or E; do NOT rewrite it from scratch)".into());
        }

        if let Some(msg) = done {
            // Every D gate below is skipped for solo-S mic-backs: those are
            // conversation, not completion.
            if !say_final {
                // A D that merely repeats this turn's S, or trails off in an
                // ellipsis, is narration — the model is mid-work and reached
                // for the wrong verb ("S checking X…" + "D checking X…").
                let d = clean_final(&msg).to_ascii_lowercase();
                let echoes = |s: &String| {
                    let s = s.trim().to_ascii_lowercase();
                    s == d || s.starts_with(&d) || d.starts_with(&s)
                };
                let trailing = (d.trim_end().ends_with("...") || d.trim_end().ends_with('…')) && ellipsis_refusals < 3;
                if trailing {
                    ellipsis_refusals += 1;
                }
                if !d.is_empty() && (says.iter().any(echoes) || trailing) {
                    note(
                        ledger,
                        &ctl,
                        depth,
                        turn,
                        "(D refused — that message narrates work in progress, and you already told the user with S. Keep working; D only with the RESULTS)".into(),
                    );
                    continue;
                }
                // A pending background verify gets the first word: join it
                // now, so a failing check still refuses the same D that rode
                // in with the edits.
                if let Some(h) = verify_bg.take() {
                    if !h.is_finished() {
                        ctl.emit(Ev::Result(depth, "(waiting on auto-verify before finishing…)".into()));
                    }
                    if let Ok((ok, text)) = h.join() {
                        note(ledger, &ctl, depth, turn, text);
                        if !ok {
                            note(ledger, &ctl, depth, turn, "(D refused — auto-verify FAILED after your edits; fix it first)".into());
                            continue;
                        }
                    }
                }
                // The plan state machine gets the last word: verify statuses
                // changed THIS turn, then refuse D while steps are open.
                if cfg.plan.enforce {
                    let (_, open) =
                        plan_tick(&plan_path, &mut plan_seen, &cfg.exec.shell, &root, ledger, &ctl, depth, turn);
                    if !open.is_empty() {
                        note(
                            ledger,
                            &ctl,
                            depth,
                            turn,
                            format!(
                                "(D refused — the plan has open steps: {}. Finish and mark them done, or edit {} to descope with status \"skip\". To narrate progress use S, not D)",
                                open.join(", "),
                                cfg.plan.file
                            ),
                        );
                        continue;
                    }
                }
            }
            if !pending.is_empty() {
                // D while subagents are still out: wait for them, feed their
                // briefs to the model, and let it finish with full knowledge.
                ctl.emit(Ev::Result(
                    depth,
                    format!("(waiting on {} subagent{} before finishing…)", pending.len(), if pending.len() == 1 { "" } else { "s" }),
                ));
                for (name, _task, h) in pending.drain(..) {
                    let sub = h.join().unwrap_or_default();
                    harvest(&name, sub, ledger, &ctl, depth, turn, &mut rep);
                }
                note(ledger, &ctl, depth, turn, "(all subagents have returned — incorporate their briefs above, then finish with D)".into());
                continue;
            }
            let msg = clean_final(&msg);
            let msg = if msg.is_empty() { "done.".to_string() } else { msg };
            ledger.push(Kind::Final, turn, msg.clone(), None);
            rep.final_msg = msg;
            break;
        }
    }
    // Abort paths can leave subagents running: collect them so their work
    // still lands in the ledger and no thread outlives the session silently.
    for (name, _task, h) in pending.drain(..) {
        let sub = h.join().unwrap_or_default();
        harvest(&name, sub, ledger, &ctl, depth, base + rep.turns, &mut rep);
    }
    if rep.final_msg.is_empty() {
        rep.final_msg = "(max turns reached)".into();
    }
    session.turn_base = base + rep.turns;
    rep.wall_ms = t_start.elapsed().as_millis();
    if depth == 0 {
        let tokens = if rep.tok_in > 0 && rep.tok_cached > 0 {
            format!(
                "in {} ({}% cached) · out {}",
                fmt_k(rep.tok_in),
                rep.tok_cached * 100 / rep.tok_in.max(1),
                fmt_k(rep.tok_out)
            )
        } else if rep.tok_in > 0 {
            format!("in {} · out {}", fmt_k(rep.tok_in), fmt_k(rep.tok_out))
        } else {
            format!("~{}t sent", rep.sent_tokens)
        };
        let seals = if rep.seals > 0 { format!(" · {} seal{}", rep.seals, if rep.seals == 1 { "" } else { "s" }) } else { String::new() };
        ctl.emit(Ev::Report(format!(
            "{} turn{} · {} cmd{} · {:.1}s · {tokens}{seals}",
            rep.turns,
            if rep.turns == 1 { "" } else { "s" },
            rep.commands,
            if rep.commands == 1 { "" } else { "s" },
            rep.wall_ms as f64 / 1000.0,
        )));
        ctl.emit(Ev::Done(rep.final_msg.clone()));
    }
    rep
}

fn fmt_k(n: u64) -> String {
    if n >= 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{n}t")
    }
}

fn first_lines(s: &str, n: usize) -> String {
    s.lines().take(n).collect::<Vec<_>>().join(" | ")
}

/// Land a finished subagent's brief in the ledger and UI.
fn harvest(name: &str, sub: Report, ledger: &mut Ledger, ctl: &Ctl, depth: u8, turn: u32, rep: &mut Report) {
    ctl.emit(Ev::SubDone(name.to_string()));
    ctl.emit(Ev::Result(
        depth,
        crate::tools::clip(&format!("[{name}] ({} turns) {}", sub.turns, first_lines(&sub.final_msg, 2)), 400),
    ));
    ledger.push(
        Kind::Result,
        turn,
        format!("[{name}] ({} turns) {}", sub.turns, sub.final_msg),
        None,
    );
    rep.model_ms += sub.model_ms; // subagent time is still model time
}

/// D swallows the rest of the stream, so a degenerating model can append
/// hallucinated prompt scaffolding after its real answer. Cut at the first
/// scaffold marker; nothing legitimate starts a line with these.
/// Prompt scaffolding AND hallucinated tool-call XML: models bleed their
/// prompt structure or tool-training syntax after a real answer — the final
/// is cut at the first line starting with any of these.
const SCAFFOLD_MARKERS: &[&str] = &[
    "## TASK", "## LOG", "## NOW", "files: #", "<|", "<invoke", "</invoke", "<function_", "</function", "[TOOL_CALLS]",
];

fn clean_final(msg: &str) -> String {
    let mut out = Vec::new();
    for l in msg.lines() {
        let t = l.trim_start();
        if SCAFFOLD_MARKERS.iter().any(|m| t.starts_with(m)) {
            break;
        }
        out.push(l);
    }
    out.join("\n").trim().to_string()
}

/// Every action string starts with its verb letter, so the verb needs no
/// second lookup table.
fn action_of(cmd: &Cmd) -> String {
    match cmd {
        Cmd::Read { target, range } => match range {
            Some((x, y)) => format!("R {target} {x}:{y}"),
            None => format!("R {target}"),
        },
        Cmd::Edit { target, a, b, body } => format!("E {target} {a}:{b} (+{} lines)", body.lines().count()),
        Cmd::Insert { target, after, body } => format!("I {target} {after} (+{} lines)", body.lines().count()),
        Cmd::New { path, body } => format!("N {path} (+{} lines)", body.lines().count()),
        Cmd::Grep { pat, target } => match target {
            Some(t) => format!("G \"{pat}\" {t}"),
            None => format!("G \"{pat}\""),
        },
        Cmd::Exec { line } => format!("X {line}"),
        Cmd::Custom { verb, args } => format!("{verb} {args}"),
        Cmd::Agent { profile, task } => format!("A {profile} {task}"),
        Cmd::Done { .. } => "D".into(),
        Cmd::View { target } => format!("V {target}"),
        Cmd::Say { text } => format!("S {}", crate::tools::clip(text, 60)),
        Cmd::Outline { target } => format!("O {target}"),
    }
}

/// One turn's dispatch state. Commands run through here both from inside the
/// stream callback (the moment the lexer completes them, mid-generation) and
/// from the deferred tail after the stream ends — identical semantics, so the
/// only difference is WHEN the work happens.
struct TurnCtx<'a> {
    cfg: &'a Arc<Config>,
    root: &'a PathBuf,
    ledger: &'a mut Ledger,
    ws: &'a mut Workspace,
    client: &'a Client,
    ctl: &'a Ctl,
    ctx_cfg: &'a crate::config::CtxCfg,
    allowed: &'a Option<Vec<char>>,
    task: &'a str,
    depth: u8,
    turn: u32,
    images: &'a mut Vec<(String, String)>,
    pending: &'a mut Vec<(String, String, std::thread::JoinHandle<Report>)>,
    repeats: &'a mut std::collections::HashMap<u64, (u32, u64)>,
    refusals: &'a mut u32,
    done: Option<String>,
    edited: bool,
    executed: usize,
    tool_us: u128,
    /// S texts from this turn, for catching a D that merely repeats one.
    says: Vec<String>,
}

impl TurnCtx<'_> {
    fn note(&mut self, text: String) {
        note(self.ledger, self.ctl, self.depth, self.turn, text);
    }

    fn dispatch(&mut self, cmd: Cmd) {
        self.executed += 1;
        let act = action_of(&cmd);
        let v = act.chars().next().unwrap_or(' ');
        if let Some(allow) = self.allowed {
            if !allow.contains(&v) && v != 'D' {
                self.ledger.push(Kind::Result, self.turn, format!("verb {v} not allowed in this profile"), None);
                return;
            }
        }
        // Game-mod style override: a config tool with `override = true` on a
        // single-line native verb (RGXOV) replaces the built-in — the raw
        // argument text routes to the mod's command as {args}.
        if crate::config::OVERRIDABLE_VERBS.contains(v)
            && self.cfg.tool.get(&v.to_string()).is_some_and(|t| t.override_native)
        {
            let args = act.split_once(' ').map(|x| x.1).unwrap_or("").trim().to_string();
            self.exec_tool(Cmd::Custom { verb: v, args });
            return;
        }
        match cmd {
            Cmd::Done { msg } => self.done = Some(msg),
            Cmd::Say { text } => {
                if self.depth == 0 {
                    self.ctl.emit(Ev::Say(text.clone()));
                }
                self.ledger.push(Kind::Result, self.turn, format!("(you told the user: {text})"), None);
                self.says.push(text);
            }
            Cmd::View { target } => {
                let act = format!("V {target}");
                self.ctl.emit(Ev::Action(self.depth, act.clone()));
                self.ledger.push(Kind::Action, self.turn, act, None);
                let result = match self.ws.load_image(&target) {
                    Ok((id, mime, b64, bytes)) => {
                        self.images.push((mime, b64));
                        format!("(image #{id} attached — you will SEE it in the next turn; {}KB)", bytes / 1024)
                    }
                    Err(e) => format!("err: {e}"),
                };
                self.note(result);
            }
            Cmd::Agent { profile, task } => {
                if self.depth >= 2 {
                    self.ledger.push(Kind::Result, self.turn, "subagent depth limit reached".into(), None);
                    return;
                }
                if !self.cfg.profile.contains_key(&profile) {
                    self.ledger.push(Kind::Result, self.turn, format!("no profile '{profile}'"), None);
                    return;
                }
                // The spawn-storm guard: an identical spawn while the first is
                // still running is a model mistake, not a wish for two of them.
                let norm = |s: &str| s.trim_matches('"').trim().to_ascii_lowercase();
                if self.pending.iter().any(|(p, t, _)| *p == profile && norm(t) == norm(&task)) {
                    let msg = format!("({profile} is ALREADY RUNNING on that task — its brief arrives in a later turn; do not spawn it again)");
                    self.note(msg);
                    return;
                }
                self.ledger.push(Kind::Action, self.turn, format!("A {profile} {task}"), None);
                self.ctl.emit(Ev::Action(self.depth, format!("A {profile} {task}")));
                // Immediate acknowledgment, so the next turn's context
                // proves the spawn happened and nothing needs repeating.
                let ack = format!("({profile} started in background — its [{profile}] brief will arrive in a later turn)");
                self.note(ack);
                let cfg2 = Arc::clone(self.cfg);
                let root2 = self.root.clone();
                let p2 = profile.clone();
                let t2 = task.clone();
                let mut ctl2 = self.ctl.clone();
                // Nested spawns keep the OUTERMOST name: the UI attributes a
                // grandchild's work to the agent the user can actually see.
                if ctl2.tag.is_none() {
                    ctl2.tag = Some(profile.clone());
                }
                let d = self.depth;
                self.pending.push((
                    profile,
                    task,
                    std::thread::spawn(move || run(cfg2, root2, &t2, Some(&p2), d + 1, ctl2)),
                ));
            }
            other => {
                let key = crate::ledger::fnv(&format!("{other:?}"));
                if self.repeats.get(&key).is_some_and(|(n, _)| *n >= 5) {
                    *self.refusals += 1;
                    // Every note is UNIQUE text with a rotating concrete
                    // suggestion: deterministic samplers (diffusion) can
                    // only escape a loop if their input actually changes.
                    let hint = match *self.refusals % 4 {
                        0 => "read the file again with R before editing",
                        1 => "rewrite the ENTIRE function in one E covering its full line range",
                        2 => "run the failing check with X and act on its exact message",
                        _ => "explain your plan to the user with S, then try a different command",
                    };
                    let msg = format!("(refusal #{}: that command keeps repeating with the same result — {hint})", *self.refusals);
                    self.note(msg);
                    return;
                }
                let is_edit = matches!(other, Cmd::Edit { .. } | Cmd::Insert { .. } | Cmd::New { .. });
                let t = Instant::now();
                self.exec_tool(other);
                self.tool_us += t.elapsed().as_micros();
                if is_edit && self.ledger.entries.last().is_some_and(|e| !e.text.starts_with("err")) {
                    self.edited = true;
                }
                let res_hash = self.ledger.entries.last().map(|e| e.hash).unwrap_or(0);
                let cnt = {
                    let e = self.repeats.entry(key).or_insert((0, 0));
                    if e.1 == res_hash {
                        e.0 += 1;
                    } else {
                        *e = (1, res_hash);
                    }
                    e.0
                };
                if cnt >= 3 {
                    // Also unique per occurrence (see refusal note above).
                    let msg = format!(
                        "(warning {cnt}x: that command gives the identical result every time — running it again will not help; change approach)"
                    );
                    self.note(msg);
                }
            }
        }
    }

    /// Run one workspace/shell command and land its (possibly dedup'd) result.
    fn exec_tool(&mut self, cmd: Cmd) {
        // Announce BEFORE executing: a slow tool must show up in the status
        // bar while it runs, not after it finishes.
        let action = action_of(&cmd);
        self.ctl.emit(Ev::Action(self.depth, crate::tools::clip(&action, 200)));
        let flat = |r: Result<String, String>| r.unwrap_or_else(|e| format!("err: {e}"));
        let mut file: Option<u32> = None;
        let result = match cmd {
            Cmd::Read { target, range } => {
                let r = flat(self.ws.read(&target, range));
                file = r.strip_prefix('#').and_then(|s| s.split(' ').next()).and_then(|s| s.parse().ok());
                r
            }
            Cmd::Edit { target, a, b, body } => flat(self.ws.edit(&target, a, b, &body)),
            Cmd::Insert { target, after, body } => flat(self.ws.insert(&target, after, &body)),
            Cmd::New { path, body } => flat(self.ws.new_file(&path, &body)),
            Cmd::Grep { pat, target } => flat(self.ws.grep(&pat, target.as_deref())),
            Cmd::Outline { target } => flat(self.ws.outline(&target)),
            Cmd::Exec { line } => run_shell(&line, &self.ws.root, DEFAULT_TOOL_TIMEOUT_MS, &self.cfg.exec.shell),
            Cmd::Custom { verb, args } => match self.cfg.tool.get(&verb.to_string()) {
                Some(t) => {
                    let line = t.cmd.replace("{args}", &args);
                    let raw = crate::tools::run_shell_env(&line, &self.ws.root, t.timeout_ms.unwrap_or(DEFAULT_TOOL_TIMEOUT_MS), &self.cfg.exec.shell, &t.env);
                    let spec = t.prune.as_deref().unwrap_or("");
                    if spec.split('|').any(|s| s.trim() == "distill") {
                        distill(self.client, self.cfg, self.task, &prune(spec, &raw))
                    } else {
                        prune(spec, &raw)
                    }
                }
                None => format!("err: unknown verb {verb}"),
            },
            Cmd::Agent { .. } | Cmd::Done { .. } | Cmd::View { .. } | Cmd::Say { .. } => unreachable!("handled by dispatch"),
        };
        let capped = cap(result, self.ctx_cfg.result_cap_chars);
        // Append-time dedup: a result byte-identical to an earlier one becomes
        // a pointer — append mode only; working_set dedups at render time and
        // must keep the LATEST read as the full survivor, not the first.
        let stored = if self.ctx_cfg.mode == "append" && capped.len() > 160 {
            self.ledger.dup_of(&capped).unwrap_or(capped)
        } else {
            capped
        };
        self.ctl.emit(Ev::Result(self.depth, crate::tools::clip(&first_lines(&stored, 3), 400)));
        self.ledger.push(Kind::Action, self.turn, action, None);
        self.ledger.push(Kind::Result, self.turn, stored, file);
    }
}

/// Models habitually write "D let me check that\nW <url>" — but D swallows the
/// rest of the stream, so the W never runs and the run dies mid-thought. When
/// a D message contains lines that parse into valid commands, convert: first
/// line becomes an S (the talk), the commands execute, the run continues.
fn rescue_done(cmds: &mut Vec<Cmd>, cfg: &Config) {
    let Some(pos) = cmds.iter().position(|c| matches!(c, Cmd::Done { .. })) else { return };
    let Cmd::Done { msg } = &cmds[pos] else { return };
    let mut lines = msg.lines();
    let first = lines.next().unwrap_or("").trim().to_string();
    let rest = lines.collect::<Vec<_>>().join("\n");
    if rest.trim().is_empty() {
        return;
    }
    let mut lx = Lexer::new();
    let mut sub = Vec::new();
    lx.feed(&format!("{rest}\n"), &mut sub);
    lx.finish(&mut sub);
    let valid = |c: &Cmd| match c {
        Cmd::Done { .. } => false,
        Cmd::Agent { profile, .. } => cfg.profile.contains_key(profile),
        Cmd::Custom { verb, .. } => cfg.tool.contains_key(&verb.to_string()),
        _ => true,
    };
    if !sub.iter().any(valid) {
        return; // a normal multi-line final report — leave it alone
    }
    sub.retain(|c| valid(c));
    let mut replacement = Vec::new();
    if !first.is_empty() {
        replacement.push(Cmd::Say { text: first });
    }
    replacement.extend(sub);
    cmds.splice(pos..=pos, replacement);
}

/// Emit + record an advisory line (loop-breaker notes, spawn acks, errors…).
fn note(ledger: &mut Ledger, ctl: &Ctl, depth: u8, turn: u32, text: String) {
    ctl.emit(Ev::Result(depth, crate::tools::clip(&text, 400)));
    ledger.push(Kind::Result, turn, text, None);
}

/// Load the plan, auto-verify steps freshly marked done (revert on failure),
/// and return (compact view, open step ids). A broken plan file blocks D.
#[allow(clippy::too_many_arguments)]
fn plan_tick(
    plan_path: &std::path::Path,
    seen: &mut std::collections::HashMap<String, String>,
    shell: &str,
    root: &std::path::Path,
    ledger: &mut Ledger,
    ctl: &Ctl,
    depth: u8,
    turn: u32,
) -> (String, Vec<String>) {
    use crate::plan::Plan;
    let Some(res) = Plan::load(plan_path) else {
        return (String::new(), Vec::new());
    };
    let mut p = match res {
        Ok(p) => p,
        Err(e) => {
            return (
                format!("## PLAN: PARSE ERROR — fix the file: {}\n", crate::tools::clip(&e, 200)),
                vec!["(unparseable plan)".into()],
            )
        }
    };
    let mut dirty = false;
    for i in 0..p.steps.len() {
        let id = p.steps[i].id.clone();
        let newly_done = p.steps[i].status == "done"
            && seen.get(&id).map(String::as_str).unwrap_or("todo") != "done";
        if !newly_done {
            continue;
        }
        if let Some(v) = p.steps[i].verify.clone() {
            let out = run_shell(&v, root, DEFAULT_TOOL_TIMEOUT_MS, shell);
            if !out.starts_with("ok") {
                p.steps[i].status = "doing".into();
                dirty = true;
                note(
                    ledger,
                    ctl,
                    depth,
                    turn,
                    format!(
                        "(plan step '{id}' marked done but its verify FAILED — reverted to doing)\n{}",
                        prune("first_failure", &out)
                    ),
                );
            }
        }
    }
    if dirty {
        let _ = p.save(plan_path);
    }
    seen.clear();
    for s in &p.steps {
        seen.insert(s.id.clone(), s.status.clone());
    }
    (p.compact(), p.open_ids())
}

fn distill(client: &Client, cfg: &Config, task: &str, text: &str) -> String {
    if text.len() < 600 {
        return text.to_string();
    }
    let prompt = cfg.distill.prompt.replace("{task}", task).replace("{text}", text);
    match client.complete(&prompt, cfg.distill.max_tokens) {
        Ok(d) => format!("(distilled from {} chars)\n{}", text.len(), d.trim()),
        Err(_) => crate::tools::prune("head_tail:30,10", text),
    }
}

fn cap(s: String, max: usize) -> String {
    if s.len() > max {
        // clip is UTF-8-boundary safe; String::truncate would panic mid-char.
        format!("{}\n…(result capped)", crate::tools::clip(&s, max))
    } else {
        s
    }
}

fn build_system(cfg: &Config, profile_system: Option<&str>, allowed: Option<&str>) -> String {
    let allow = |v: char| allowed.is_none_or(|a| a.contains(v));
    let mut s = match &cfg.prompt.system {
        Some(id) => format!("{}\n", id.trim_end()),
        None => String::from(
            "You are haste, a fast coding agent. You act ONLY by emitting command lines. \
             No prose, no markdown, no explanations — command lines only.\n",
        ),
    };
    if let Some(ps) = profile_system {
        s.push_str(ps);
        s.push('\n');
    }
    s.push_str("Commands (one per line):\n");
    if allow('R') { s.push_str("R <id|path> [a:b]   read file, numbered lines\n"); }
    if allow('E') { s.push_str("E <id> <a>:<b>      replace lines a..b; content lines follow; end with a line that is only \".\"\n"); }
    if allow('I') { s.push_str("I <id> <a>          insert after line a (0=top); content follows, end \".\"\n"); }
    if allow('N') { s.push_str("N <path>            create file; content follows, end \".\"\n"); }
    if allow('G') { s.push_str("G <regex> [id|path] search files, results as #id:line:text\n"); }
    if allow('O') { s.push_str("O [id|path|dir]     outline: signatures with line numbers, bodies elided — orient cheaply BEFORE reading\n"); }
    if allow('X') { s.push_str("X <command>         run shell command in the repo root\n"); }
    if allow('V') { s.push_str("V <id|path>         view an image file (png/jpg/webp/gif) — you SEE it in the next turn\n"); }
    if allow('A') && !cfg.profile.is_empty() {
        let names: Vec<&str> = cfg.profile.keys().map(String::as_str).collect();
        s.push_str(&format!(
            "A <profile> <task>  start a BACKGROUND subagent; its [profile] brief arrives as a result in a later turn — keep working meanwhile. Profiles: {}\n",
            names.join(", ")
        ));
    }
    for (v, t) in &cfg.tool {
        if allow(v.chars().next().unwrap()) {
            s.push_str(&format!("{v} <args>            {}\n", t.desc));
        }
    }
    s.push_str("S <text>            say one line to the user WITHOUT finishing (progress, questions, findings)\n");
    s.push_str("D <message>         done — ENDS THE RUN; message is your final report (may span lines to end of message). Progress updates are S, never D\n");
    let plan_file = &cfg.plan.file;
    s.push_str(
        "Rules: files get ids (#0,#1..) listed in the files: header — refer to them by id (with or without #). \
         In E/I/N content, a line that must start with \".\" gets one extra \".\" prefix. \
         Batch independent commands in one message. Results arrive in the next message. \
         Edit results already show the updated lines — do not re-read a file after editing it. \
         Never emit long runs of one repeated character (dashes, =, !); keep separators under 40 chars. \
         Verify edits by running checks/tests. Be terse.\n\
         Example message (nothing but commands):\n\
         R config.py\n\
         G \"load_cfg\" src\n\
         X python tests.py\n",
    );
    if !cfg.prompt_extra.is_empty() {
        s.push_str(&cfg.prompt_extra);
    }
    if !cfg.prompt.extra.trim().is_empty() {
        s.push_str(cfg.prompt.extra.trim());
        s.push('\n');
    }
    if let Some(v) = &cfg.verify.cmd {
        s.push_str(&format!(
            "After any turn where you edit files, `{v}` runs AUTOMATICALLY and its result is shown — never run it yourself.\n"
        ));
    }
    s.push_str(&format!(
        "Plan: for multi-step work, create {plan_file} with N — \
         {{\"goal\":\"...\",\"steps\":[{{\"id\":\"..\",\"what\":\"..\",\"status\":\"todo\",\"needs\":[],\"verify\":\"shell cmd\"}}]}}. \
         It is a live state machine you keep current by editing it (status: todo/doing/done/skip). \
         Marking a step done runs its verify automatically and REVERTS the status if it fails. \
         D is refused while steps are open — finish them or descope with status \"skip\".\n"
    ));
    s
}

