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
}

pub fn run(cfg: Arc<Config>, root: PathBuf, task: &str, profile: Option<&str>, depth: u8) -> Report {
    let t_start = Instant::now();
    let prof = profile.and_then(|p| cfg.profile.get(p));
    let (max_turns, budget) = match prof {
        Some(p) => (p.max_turns, p.budget_tokens),
        None => (cfg.context.max_turns, cfg.context.budget_tokens),
    };
    let mut ctx_cfg = cfg.context.clone();
    ctx_cfg.budget_tokens = budget;

    let tee = (depth == 0).then(|| root.join(".haste").join("ledger.jsonl"));
    let mut ledger = Ledger::new(tee.as_deref());
    if depth == 0 && cfg.context.bootstrap {
        ledger.push(Kind::Pin, 0, crate::bootstrap::workspace_state(&root), None);
    }
    ledger.push(Kind::Task, 0, task.to_string(), None);
    let mut ws = Workspace::new(root.clone());
    let mut renderer = Renderer::new();
    let client = Client::new(cfg.model.clone(), cfg.api_key());
    let system = build_system(&cfg, prof.map(|p| p.system.as_str()), prof.map(|p| p.tools.as_str()));
    let allowed: Option<Vec<char>> = prof.map(|p| p.tools.chars().collect());

    let mut rep = Report::default();
    let mut empty_turns = 0u32;

    for turn in 1..=max_turns {
        rep.turns = turn;
        let t_r = Instant::now();
        let doc = renderer.render(&ledger, &ctx_cfg, turn);
        let user = format!("{}{}\n## NOW\nNext commands:\n", ws.legend(), doc);
        rep.render_us += t_r.elapsed().as_micros();
        rep.sent_tokens += est_tokens(&system) + est_tokens(&user);

        let mut lexer = Lexer::new();
        let mut cmds: Vec<Cmd> = Vec::new();
        let stream_res = client.stream(&system, &user, &mut |delta| {
            lexer.feed(delta, &mut cmds);
        });
        match stream_res {
            Ok(s) => {
                rep.model_ms += s.total_ms;
                rep.ttft_ms_sum += s.ttft_ms;
                rep.out_chars += s.out_chars;
            }
            Err(e) => {
                ledger.push(Kind::Result, turn, format!("model error: {e}"), None);
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
        }
        lexer.finish(&mut cmds);

        if cmds.is_empty() {
            empty_turns += 1;
            if empty_turns >= MAX_EMPTY_TURNS {
                rep.final_msg = "(aborted: model produced no commands)".into();
                break;
            }
            ledger.push(Kind::Result, turn, "no commands parsed — emit DSL command lines only".into(), None);
            continue;
        }
        empty_turns = 0;
        rep.commands += cmds.len();

        let t_tools = Instant::now();
        let mut spawned: Vec<(String, std::thread::JoinHandle<Report>)> = Vec::new();
        let mut done: Option<String> = None;
        for cmd in cmds {
            if let Some(allow) = &allowed {
                let v = verb_of(&cmd);
                if !allow.contains(&v) && v != 'D' {
                    ledger.push(Kind::Result, turn, format!("verb {v} not allowed in this profile"), None);
                    continue;
                }
            }
            match cmd {
                Cmd::Done { msg } => done = Some(msg),
                Cmd::Agent { profile, task } => {
                    if depth >= 2 {
                        ledger.push(Kind::Result, turn, "subagent depth limit reached".into(), None);
                        continue;
                    }
                    if !cfg.profile.contains_key(&profile) {
                        ledger.push(Kind::Result, turn, format!("no profile '{profile}'"), None);
                        continue;
                    }
                    ledger.push(Kind::Action, turn, format!("A {profile} {task}"), None);
                    let cfg2 = Arc::clone(&cfg);
                    let root2 = root.clone();
                    let p2 = profile.clone();
                    let t2 = task.clone();
                    spawned.push((
                        profile,
                        std::thread::spawn(move || run(cfg2, root2, &t2, Some(&p2), depth + 1)),
                    ));
                }
                other => exec_one(other, &mut ws, &mut ledger, &cfg, &client, task, turn, &ctx_cfg),
            }
        }
        for (name, h) in spawned {
            let sub = h.join().unwrap_or_default();
            ledger.push(
                Kind::Result,
                turn,
                format!("[{name}] ({} turns) {}", sub.turns, sub.final_msg),
                None,
            );
            rep.model_ms += sub.model_ms; // subagent time is still model time
        }
        rep.tool_ms += t_tools.elapsed().as_millis();

        if let Some(msg) = done {
            ledger.push(Kind::Final, turn, msg.clone(), None);
            rep.final_msg = msg;
            break;
        }
    }
    if rep.final_msg.is_empty() {
        rep.final_msg = "(max turns reached)".into();
    }
    rep.wall_ms = t_start.elapsed().as_millis();
    rep
}

fn verb_of(cmd: &Cmd) -> char {
    match cmd {
        Cmd::Read { .. } => 'R',
        Cmd::Edit { .. } => 'E',
        Cmd::Insert { .. } => 'I',
        Cmd::New { .. } => 'N',
        Cmd::Grep { .. } => 'G',
        Cmd::Exec { .. } => 'X',
        Cmd::Agent { .. } => 'A',
        Cmd::Done { .. } => 'D',
        Cmd::Custom { verb, .. } => *verb,
    }
}

#[allow(clippy::too_many_arguments)]
fn exec_one(
    cmd: Cmd,
    ws: &mut Workspace,
    ledger: &mut Ledger,
    cfg: &Config,
    client: &Client,
    task: &str,
    turn: u32,
    ctx: &crate::config::CtxCfg,
) {
    let (action, result, file) = match cmd {
        Cmd::Read { target, range } => {
            let a = match range {
                Some((x, y)) => format!("R {target} {x}:{y}"),
                None => format!("R {target}"),
            };
            match ws.read(&target, range) {
                Ok(r) => {
                    let fid = r.strip_prefix('#').and_then(|s| s.split(' ').next()).and_then(|s| s.parse().ok());
                    (a, r, fid)
                }
                Err(e) => (a, format!("err: {e}"), None),
            }
        }
        Cmd::Edit { target, a, b, body } => {
            let act = format!("E {target} {a}:{b} (+{} lines)", body.lines().count());
            match ws.edit(&target, a, b, &body) {
                Ok(r) => (act, r, None),
                Err(e) => (act, format!("err: {e}"), None),
            }
        }
        Cmd::Insert { target, after, body } => {
            let act = format!("I {target} {after} (+{} lines)", body.lines().count());
            match ws.insert(&target, after, &body) {
                Ok(r) => (act, r, None),
                Err(e) => (act, format!("err: {e}"), None),
            }
        }
        Cmd::New { path, body } => {
            let act = format!("N {path} (+{} lines)", body.lines().count());
            match ws.new_file(&path, &body) {
                Ok(r) => (act, r, None),
                Err(e) => (act, format!("err: {e}"), None),
            }
        }
        Cmd::Grep { pat, target } => {
            let act = match &target {
                Some(t) => format!("G \"{pat}\" {t}"),
                None => format!("G \"{pat}\""),
            };
            match ws.grep(&pat, target.as_deref()) {
                Ok(r) => (act, r, None),
                Err(e) => (act, format!("err: {e}"), None),
            }
        }
        Cmd::Exec { line } => {
            let act = format!("X {line}");
            let r = run_shell(&line, &ws.root, DEFAULT_TOOL_TIMEOUT_MS);
            (act, r, None)
        }
        Cmd::Custom { verb, args } => {
            let key = verb.to_string();
            let act = format!("{verb} {args}");
            match cfg.tool.get(&key) {
                Some(t) => {
                    let line = t.cmd.replace("{args}", &args);
                    let raw = run_shell(&line, &ws.root, t.timeout_ms.unwrap_or(DEFAULT_TOOL_TIMEOUT_MS));
                    let spec = t.prune.as_deref().unwrap_or("");
                    let pruned = if spec.split('|').any(|s| s.trim() == "distill") {
                        distill(client, cfg, task, &prune(spec, &raw))
                    } else {
                        prune(spec, &raw)
                    };
                    (act, pruned, None)
                }
                None => (act, format!("err: unknown verb {verb}"), None),
            }
        }
        Cmd::Agent { .. } | Cmd::Done { .. } => unreachable!("handled by caller"),
    };
    ledger.push(Kind::Action, turn, action, None);
    let result = cap(result, ctx.result_cap_chars);
    ledger.push(Kind::Result, turn, result, file);
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

fn cap(mut s: String, max: usize) -> String {
    if s.len() > max {
        s.truncate(max);
        s.push_str("\n…(result capped)");
    }
    s
}

fn build_system(cfg: &Config, profile_system: Option<&str>, allowed: Option<&str>) -> String {
    let allow = |v: char| allowed.map_or(true, |a| a.contains(v));
    let mut s = String::from(
        "You are haste, a fast coding agent. You act ONLY by emitting command lines. \
         No prose, no markdown, no explanations — command lines only.\n",
    );
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
    if allow('X') { s.push_str("X <command>         run shell command in the repo root\n"); }
    if allow('A') && !cfg.profile.is_empty() {
        let names: Vec<&str> = cfg.profile.keys().map(String::as_str).collect();
        s.push_str(&format!("A <profile> <task>  delegate to a subagent; profiles: {}\n", names.join(", ")));
    }
    for (v, t) in &cfg.tool {
        if allow(v.chars().next().unwrap()) {
            s.push_str(&format!("{v} <args>            {}\n", t.desc));
        }
    }
    s.push_str("D <message>         done; message is your final report (may span lines to end of message)\n");
    s.push_str(
        "Rules: files get ids (#0,#1..) listed in the files: header — refer to them by id (with or without #). \
         In E/I/N content, a line that must start with \".\" gets one extra \".\" prefix. \
         Batch independent commands in one message. Results arrive in the next message. \
         Verify edits by running checks/tests. Be terse.\n\
         Example message (nothing but commands):\n\
         R config.py\n\
         G \"load_cfg\" src\n\
         X python tests.py\n",
    );
    s
}
