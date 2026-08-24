//! End-to-end loop test against an in-process mock model server, plus a
//! renderer benchmark. The mock replies instantly, so (wall - model) â‰ˆ pure
//! harness overhead â€” the number this project exists to minimize.

use haste::config::Config;
use haste::ledger::{Kind, Ledger};
use haste::render::Renderer;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

/// Minimal scripted OpenAI-compatible SSE server: each request pops the next
/// scripted assistant message and streams it back in small chunks.
fn mock_server(scripts: Vec<&'static str>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let mut i = 0usize;
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            // read request until end of headers, then the content-length body
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let (mut cl, mut hdr_end) = (0usize, 0usize);
            loop {
                let n = s.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if hdr_end == 0 {
                    if let Some(p) = find(&buf, b"\r\n\r\n") {
                        hdr_end = p + 4;
                        let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                        cl = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                    }
                }
                if hdr_end > 0 && buf.len() >= hdr_end + cl {
                    break;
                }
            }
            let msg = scripts.get(i).copied().unwrap_or("D out of script\n");
            i += 1;
            // "LENGTH:" prefix simulates a max_tokens-guillotined stream.
            let (msg, fr) = match msg.strip_prefix("LENGTH:") {
                Some(m) => (m, "length"),
                None => (msg, "stop"),
            };
            let mut resp = String::from(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            );
            // stream in 7-char chunks to exercise the streaming lexer
            let chars: Vec<char> = msg.chars().collect();
            for chunk in chars.chunks(7) {
                let piece: String = chunk.iter().collect();
                let ev = serde_json::json!({"choices":[{"delta":{"content":piece}}]});
                resp.push_str(&format!("data: {ev}\n\n"));
            }
            let fin = serde_json::json!({
                "choices":[{"delta":{}, "finish_reason": fr}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7,
                          "prompt_tokens_details": {"cached_tokens": 80}}
            });
            resp.push_str(&format!("data: {fin}\n\n"));
            resp.push_str("data: [DONE]\n\n");
            let _ = s.write_all(resp.as_bytes());
        }
    });
    port
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn temp_repo() -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("haste-e2e-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    std::fs::write(p.join("greet.txt"), "hello\nwrold\ngoodbye\n").unwrap();
    p
}

fn cfg_for(port: u16) -> Config {
    let toml = format!(
        r#"
[model]
base_url = "http://127.0.0.1:{port}/v1"
model = "mock"

[context]
mode = "working_set"
budget_tokens = 8000
max_turns = 10
"#
    );
    toml::from_str(&toml).unwrap()
}

#[test]
fn full_loop_fixes_typo_and_measures_overhead() {
    let port = mock_server(vec![
        "G \"wrold\"\nR 0\n",
        "E 0 2:2\nworld\n.\n",
        "X echo verified\nD fixed the typo in greet.txt\n",
    ]);
    let root = temp_repo();
    let cfg = Arc::new(cfg_for(port));
    let rep = haste::agent::run(cfg, root.clone(), "fix the typo in greet.txt", None, 0, haste::agent::Ctl::default());

    assert_eq!(rep.turns, 3, "final: {}", rep.final_msg);
    assert_eq!(rep.final_msg, "fixed the typo in greet.txt");
    let fixed = std::fs::read_to_string(root.join("greet.txt")).unwrap();
    assert_eq!(fixed, "hello\nworld\ngoodbye\n");
    assert_eq!(rep.commands, 5);

    // Ledger tee exists and is lossless (one line per entry).
    let tee = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(tee.lines().count() >= 10, "tee lines: {}", tee.lines().count());

    // The headline number: everything that is not model or tool time.
    let overhead = rep.wall_ms.saturating_sub(rep.model_ms).saturating_sub(rep.tool_ms);
    println!(
        "HARNESS OVERHEAD: {overhead}ms across {} turns ({:.2}ms/turn) | wall {}ms model {}ms tools {}ms render {:.2}ms",
        rep.turns,
        overhead as f64 / rep.turns as f64,
        rep.wall_ms,
        rep.model_ms,
        rep.tool_ms,
        rep.render_us as f64 / 1000.0
    );
    assert!(
        overhead < 250,
        "harness overhead {overhead}ms too high (loopback network included)"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn profile_restriction_blocks_edits() {
    let port = mock_server(vec![
        "E 0 1:1\nhacked\n.\nD tried\n",
    ]);
    let root = temp_repo();
    let toml = format!(
        r#"
[model]
base_url = "http://127.0.0.1:{port}/v1"
model = "mock"

[profile.reader]
system = "You only read."
tools = "RG"
max_turns = 3
"#
    );
    let cfg: Config = toml::from_str(&toml).unwrap();
    let rep = haste::agent::run(Arc::new(cfg), root.clone(), "look around", Some("reader"), 1, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "tried");
    let untouched = std::fs::read_to_string(root.join("greet.txt")).unwrap();
    assert_eq!(untouched, "hello\nwrold\ngoodbye\n", "read-only profile must not edit");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn event_stream_feeds_a_ui() {
    let port = mock_server(vec![
        "R greet.txt\n",
        "D looked\n",
    ]);
    let root = temp_repo();
    let (tx, rx) = std::sync::mpsc::channel();
    let ctl = haste::agent::Ctl { sink: Some(tx), stop: None, inbox: None, tag: None };
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "look at greet.txt", None, 0, ctl);
    assert_eq!(rep.final_msg, "looked");
    let evs: Vec<haste::agent::Ev> = rx.try_iter().collect();
    use haste::agent::Ev;
    assert!(evs.iter().any(|e| matches!(e, Ev::Turn(1))), "no Turn event");
    assert!(evs.iter().any(|e| matches!(e, Ev::Delta(_))), "no Delta events");
    assert!(
        evs.iter().any(|e| matches!(e, Ev::Action(0, a) if a == "R greet.txt")),
        "no Action event"
    );
    assert!(evs.iter().any(|e| matches!(e, Ev::Result(0, _))), "no Result event");
    assert!(evs.iter().any(|e| matches!(e, Ev::Report(_))), "no Report event");
    assert!(
        matches!(evs.last(), Some(Ev::Done(m)) if m == "looked"),
        "Done must be last"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn model_compaction_seals_history_but_ledger_stays_lossless() {
    let port = mock_server(vec![
        // Enough commands that run 1 leaves >=8 ledger entries (hysteresis
        // arms) and a verbose finish that busts the tiny budget.
        "X echo a\nX echo b\nX echo c\n",
        "D one — this deliberately verbose completion note pads the ledger well past the ten-token budget so run two must compact first\n",
        // run 2, request 1 is the compaction call (over budget) — the mock's
        // reply becomes the summary, not a command stream.
        "COMPACT BRIEF: finished task one; greet.txt untouched.\n",
        "D two\n",
    ]);
    let root = temp_repo();
    let toml = format!(
        r#"
[model]
base_url = "http://127.0.0.1:{port}/v1"
model = "mock"

[context]
mode = "append"
budget_tokens = 10
bootstrap = false
compact = "model"
compact_keep_last = 2
"#
    );
    let cfg: Arc<Config> = Arc::new(toml::from_str(&toml).unwrap());
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let r1 = haste::agent::run_session(Arc::clone(&cfg), &mut session, "first task", None, 0, haste::agent::Ctl::default());
    assert!(r1.final_msg.starts_with("one"), "{}", r1.final_msg);
    let r2 = haste::agent::run_session(Arc::clone(&cfg), &mut session, "second task", None, 0, haste::agent::Ctl::default());
    assert_eq!(r2.final_msg, "two");
    // The rendered view carries the model summary…
    let mut cfg2 = cfg.context.clone();
    cfg2.budget_tokens = 100_000; // render without re-triggering structural fold
    let doc = session.renderer.render(&session.ledger, &cfg2, 99);
    assert!(doc.contains("[history compressed]") && doc.contains("COMPACT BRIEF"), "{doc}");
    // …but the lossless ledger never absorbed it.
    assert!(
        session.ledger.entries.iter().all(|e| !e.text.contains("COMPACT BRIEF")),
        "summary leaked into the ledger"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn length_cap_writes_partial_and_tells_model_to_continue() {
    let port = mock_server(vec![
        // Payload guillotined mid-file: no "." terminator, finish_reason=length.
        "LENGTH:N page.html\n<!DOCTYPE html>\n<body>",
        // Model follows the note and continues instead of rewriting.
        "I 0 2\n</body>\n.\nD finished the page\n",
    ]);
    let root = temp_repo();
    let cfg = Arc::new(cfg_for(port));
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "make page", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "finished the page");
    assert!(
        session.ledger.entries.iter().any(|e| e.text.contains("hit the max_tokens limit")),
        "continuation note missing"
    );
    let page = std::fs::read_to_string(root.join("page.html")).unwrap();
    assert_eq!(page, "<!DOCTYPE html>\n<body>\n</body>\n");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn say_speaks_without_ending_the_run() {
    let port = mock_server(vec!["S found the bug, fixing it now\nX echo work\n", "D fixed\n"]);
    let root = temp_repo();
    let (tx, rx) = std::sync::mpsc::channel();
    let ctl = haste::agent::Ctl { sink: Some(tx), stop: None, inbox: None, tag: None };
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "task", None, 0, ctl);
    assert_eq!(rep.final_msg, "fixed");
    assert_eq!(rep.turns, 2, "S must not end the run");
    let says: Vec<String> = rx
        .try_iter()
        .filter_map(|e| match e {
            haste::agent::Ev::Say(t) => Some(t),
            _ => None,
        })
        .collect();
    assert_eq!(says, vec!["found the bug, fixing it now".to_string()]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn done_with_trailing_commands_is_rescued() {
    // "D let me check\nX echo hi" — the classic misuse: talk then work in one
    // D. Must become S + executed command, run continuing.
    let port = mock_server(vec!["D let me check that\nX echo hi\n", "D really done\n"]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "task", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "really done");
    assert_eq!(rep.turns, 2);
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("you told the user: let me check that"), "say part missing");
    assert!(ledger.contains("X echo hi"), "command part not executed");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn outline_verb_maps_a_file() {
    let port = mock_server(vec!["O util.py\nD outlined\n"]);
    let root = temp_repo();
    std::fs::write(root.join("util.py"), "import os\n\nclass Tool:\n    def run(self):\n        return 1\n").unwrap();
    let mut session = {
        let cfg = Arc::new(cfg_for(port));
        let mut s = haste::agent::Session::new(&cfg, root.clone(), 0);
        haste::agent::run_session(Arc::clone(&cfg), &mut s, "map it", None, 0, haste::agent::Ctl::default());
        s
    };
    let texts: Vec<&str> = session.ledger.entries.iter().map(|e| e.text.as_str()).collect();
    let out = texts.iter().find(|t| t.starts_with("outline #")).expect("no outline result");
    assert!(out.contains("3: class Tool:") && out.contains("4:     def run(self):"), "{out}");
    assert!(!out.contains("return 1"), "body leaked: {out}");
    session.ledger.entries.clear();
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn auto_verify_runs_after_edits_and_gates_done() {
    // Failing verify: the same-turn D is refused; a later D (no edits) passes.
    let port = mock_server(vec![
        "E 0 2:2\nworld\n.\nD fixed it\n",
        "D giving my report anyway\n",
    ]);
    let root = temp_repo();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n\
         [context]\nbootstrap = false\n[verify]\ncmd = \"exit 5\"\n"
    );
    let cfg: Arc<Config> = Arc::new(toml::from_str(&toml).unwrap());
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    // Pre-intern greet.txt as #0 for the scripted edit.
    session.ws.read("greet.txt", None).unwrap();
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "fix", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "giving my report anyway");
    assert_eq!(rep.turns, 2);
    let texts: Vec<&str> = session.ledger.entries.iter().map(|e| e.text.as_str()).collect();
    assert!(texts.iter().any(|t| t.starts_with("(auto-verify FAIL")), "no auto-verify: {texts:?}");
    assert!(texts.iter().any(|t| t.contains("D refused — auto-verify")), "no gate: {texts:?}");

    // Passing verify: edit + D completes in one turn.
    let port2 = mock_server(vec!["E 0 2:2\nworld\n.\nD done\n"]);
    let root2 = temp_repo();
    let toml2 = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port2}/v1\"\nmodel = \"mock\"\n\
         [context]\nbootstrap = false\n[verify]\ncmd = \"echo tests pass\"\n"
    );
    let cfg2: Arc<Config> = Arc::new(toml::from_str(&toml2).unwrap());
    let mut s2 = haste::agent::Session::new(&cfg2, root2.clone(), 0);
    s2.ws.read("greet.txt", None).unwrap();
    let rep2 = haste::agent::run_session(Arc::clone(&cfg2), &mut s2, "fix", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep2.final_msg, "done");
    assert_eq!(rep2.turns, 1, "pass case must finish in one turn");
    assert!(s2.ledger.entries.iter().any(|e| e.text.starts_with("(auto-verify PASS")));
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(root2);
}

#[test]
fn plan_state_machine_enforces_and_verifies() {
    let port = mock_server(vec![
        // t1: write a plan with one verifiable step, then try to D early.
        "N plan.json\n{\"goal\":\"demo\",\"steps\":[{\"id\":\"fix\",\"what\":\"fix greet\",\"status\":\"doing\",\"verify\":\"echo ok\"}]}\n.\nD all done\n",
        // t2 (D was refused): mark the step done, then D for real.
        "E 0 1:1\n{\"goal\":\"demo\",\"steps\":[{\"id\":\"fix\",\"what\":\"fix greet\",\"status\":\"done\",\"verify\":\"echo ok\"}]}\n.\nD actually done\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "do the plan", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "actually done");
    assert_eq!(rep.turns, 2);
    let ledger_texts: Vec<String> = std::fs::read_to_string(root.join(".haste/ledger.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().unwrap().to_string())
        .collect();
    assert!(
        ledger_texts.iter().any(|t| t.contains("D refused") && t.contains("fix")),
        "refusal missing: {ledger_texts:?}"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn lying_about_done_gets_reverted_by_verify() {
    let port = mock_server(vec![
        // One step whose verify fails — model claims done and tries to D.
        "N plan.json\n{\"goal\":\"demo\",\"steps\":[{\"id\":\"broken\",\"status\":\"done\",\"verify\":\"exit 3\"}]}\n.\nD shipped it\n",
        // After the revert + refusal, model descopes honestly (N overwrites —
        // the revert-save pretty-printed the file, so line edits are stale).
        "N plan.json\n{\"goal\":\"demo\",\"steps\":[{\"id\":\"broken\",\"status\":\"skip\",\"verify\":\"exit 3\"}]}\n.\nD descoped\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "ship", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "descoped");
    let plan = std::fs::read_to_string(root.join("plan.json")).unwrap();
    assert!(plan.contains("skip"), "{plan}");
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("verify FAILED"), "revert note missing");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn solo_say_hands_the_mic_back() {
    // A clarification question with no work must END the run, not loop.
    let port = mock_server(vec!["S what would you like me to do?\n"]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "wasu", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "what would you like me to do?");
    assert_eq!(rep.turns, 1, "solo S must not loop");
    // Exact usage flows through from the provider.
    assert_eq!((rep.tok_in, rep.tok_cached, rep.tok_out), (100, 80, 7));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn view_attaches_image_for_next_turn() {
    let port = mock_server(vec!["V pic.png\n", "D saw it\n"]);
    let root = temp_repo();
    std::fs::write(root.join("pic.png"), [0x89u8, 0x50, 0x4E, 0x47, 1, 2, 3, 4]).unwrap();
    let cfg = Arc::new(cfg_for(port));
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "look at pic", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "saw it");
    assert!(
        session.ledger.entries.iter().any(|e| e.text.contains("attached") && e.text.contains("SEE")),
        "attach note missing"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn duplicate_subagent_spawns_are_refused() {
    let sub_turn = if cfg!(windows) { "X powershell -Command Start-Sleep -Milliseconds 900\nD sub done\n" } else { "X sleep 1\nD sub done\n" };
    let parent_t1 = if cfg!(windows) {
        "A researcher dig into the repo\nX powershell -Command Start-Sleep -Milliseconds 400\n"
    } else {
        "A researcher dig into the repo\nX sleep 0.4\n"
    };
    let port = mock_server(vec![
        // t1: spawn + slow tool (guarantees the researcher requests next).
        parent_t1,
        sub_turn, // researcher's single concurrent turn
        // t2 (sub still sleeping): the SAME spawn again — must be refused.
        "A researcher dig into the repo\n",
        // t3: premature D forces the wait-join; t4 finishes for real.
        "D finished\n",
        "D finished\n",
    ]);
    let root = temp_repo();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n\
         [context]\nbootstrap = false\n\
         [profile.researcher]\nsystem = \"research\"\ntools = \"RGX\"\n"
    );
    let cfg: Arc<Config> = Arc::new(toml::from_str(&toml).unwrap());
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "go", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "finished");
    let texts: Vec<&str> = session.ledger.entries.iter().map(|e| e.text.as_str()).collect();
    assert!(texts.iter().any(|t| t.contains("started in background")), "spawn ack missing: {texts:?}");
    assert!(texts.iter().any(|t| t.contains("ALREADY RUNNING")), "duplicate refusal missing: {texts:?}");
    // Exactly one researcher actually ran.
    let spawns = texts.iter().filter(|t| t.starts_with("A researcher")).count();
    assert_eq!(spawns, 1, "duplicate was spawned anyway");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn degenerate_spam_is_cut_and_retried() {
    let spam: &'static str = Box::leak(format!("R greet.txt\n{}", "!".repeat(600)).into_boxed_str());
    let port = mock_server(vec![spam, "D recovered\n"]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "task", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "recovered");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn ascii_art_survives_the_degeneration_guard() {
    // 120 dashes = a markdown table rule / box border. Must NOT trip the guard.
    let art: &'static str = Box::leak(format!("D here is the diagram\n{}\nstate machine\n{}\n", "-".repeat(120), "-".repeat(120)).into_boxed_str());
    let port = mock_server(vec![art]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "draw it", None, 0, haste::agent::Ctl::default());
    assert!(rep.final_msg.contains("state machine"), "art was cut: {}", rep.final_msg);
    assert_eq!(rep.turns, 1);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn phrase_loops_are_cut_like_char_spam() {
    // Sentence-level repetition: no char/pair run, but the same phrase forever.
    let spam: &'static str = Box::leak("the model is stuck here. ".repeat(40).into_boxed_str());
    let looped: &'static str = Box::leak(format!("D {spam}\n").into_boxed_str());
    let port = mock_server(vec![looped, "D recovered\n"]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "task", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "recovered", "phrase loop was not cut");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn endless_refusal_loop_aborts() {
    // A deterministic model re-sending one refused command forever.
    let same: &'static str = "G \"wrold\" greet.txt\n";
    let mut scripts = vec![same; 30];
    scripts.push("D never reached\n");
    let port = mock_server(scripts);
    let root = temp_repo();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n[context]\nmax_turns = 0\nbootstrap = false\n"
    );
    let cfg: Config = toml::from_str(&toml).unwrap();
    let rep = haste::agent::run(Arc::new(cfg), root.clone(), "loop", None, 0, haste::agent::Ctl::default());
    assert!(rep.final_msg.contains("kept repeating"), "{}", rep.final_msg);
    assert!(rep.turns < 25, "ran {} turns", rep.turns);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn endless_collapse_aborts_with_explanation() {
    let spam: &'static str = Box::leak("!".repeat(600).into_boxed_str());
    let port = mock_server(vec![spam, spam, spam, spam, spam, spam, spam, spam]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "task", None, 0, haste::agent::Ctl::default());
    assert!(rep.final_msg.contains("collapsed 6 times"), "{}", rep.final_msg);
    assert!(rep.turns <= 7);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn midrun_inbox_message_lands_in_ledger() {
    let port = mock_server(vec!["X echo one\n", "D ok\n"]);
    let root = temp_repo();
    let cfg = Arc::new(cfg_for(port));
    let inbox = std::sync::Arc::new(std::sync::Mutex::new(vec!["also check the docs folder".to_string()]));
    let ctl = haste::agent::Ctl { sink: None, stop: None, inbox: Some(inbox), tag: None };
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "main task", None, 0, ctl);
    assert_eq!(rep.final_msg, "ok");
    assert!(
        session.ledger.entries.iter().any(|e| e.text == "also check the docs folder"),
        "steering message missing from ledger"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn subagents_stream_in_and_done_waits_for_stragglers() {
    let sleep = if cfg!(windows) { "X powershell -Command Start-Sleep -Milliseconds 900\n" } else { "X sleep 1\n" };
    let port = mock_server(vec![
        // parent turn 1: spawn researcher, then D immediately (premature).
        "A researcher find the answer\nD too early\n",
        // researcher turn 1 (runs concurrently, consumes next script): slow tool then finish.
        Box::leak(format!("{sleep}D sub-answer: 42\n").into_boxed_str()),
        // parent turn 2, after forced wait + briefs landed: real finish.
        "D final with sub knowledge\n",
    ]);
    let root = temp_repo();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n\
         [context]\nbootstrap = false\n\
         [profile.researcher]\nsystem = \"research\"\ntools = \"RGX\"\n"
    );
    let cfg: Arc<Config> = Arc::new(toml::from_str(&toml).unwrap());
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "big question", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "final with sub knowledge");
    let texts: Vec<&str> = session.ledger.entries.iter().map(|e| e.text.as_str()).collect();
    let brief = texts.iter().position(|t| t.contains("[researcher]") && t.contains("sub-answer: 42"));
    let final_pos = texts.iter().position(|t| t.contains("final with sub knowledge"));
    assert!(brief.is_some(), "brief missing: {texts:?}");
    assert!(
        texts.iter().any(|t| t.contains("incorporate their briefs")),
        "wait note missing"
    );
    assert!(brief.unwrap() < final_pos.unwrap(), "brief must land before the final answer");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn zero_max_turns_means_unlimited() {
    let port = mock_server(vec!["X echo a\n", "X echo b\n", "X echo c\n", "D unbounded\n"]);
    let root = temp_repo();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n[context]\nmax_turns = 0\nbootstrap = false\n"
    );
    let cfg: Config = toml::from_str(&toml).unwrap();
    let rep = haste::agent::run(Arc::new(cfg), root.clone(), "go", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "unbounded");
    assert_eq!(rep.turns, 4);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn empty_done_is_not_max_turns() {
    let port = mock_server(vec!["D\n"]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "cool", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "done.");
    assert_eq!(rep.turns, 1);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn final_message_truncates_hallucinated_scaffolding() {
    let port = mock_server(vec![
        "D found it in E:\\Repos\\haste\nsecond real line\n## TASK\ndäm\n## LOG\n> X junk\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "find repo", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "found it in E:\\Repos\\haste\nsecond real line");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn loop_breaker_warns_then_refuses() {
    // Model stuck repeating the identical command every turn.
    let same = "G \"wrold\" greet.txt\n";
    let port = mock_server(vec![same, same, same, same, same, same, same, "D gave up\n"]);
    let root = temp_repo();
    let cfg = Arc::new(cfg_for(port));
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "loop", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "gave up");
    let texts: Vec<&str> = session.ledger.entries.iter().map(|e| e.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("identical result every time")),
        "warning missing: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("refusal #1")),
        "refusal missing: {texts:?}"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn continued_session_shares_ledger() {
    let port = mock_server(vec!["D one\n", "D two\n"]);
    let root = temp_repo();
    let cfg = Arc::new(cfg_for(port));
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let r1 = haste::agent::run_session(Arc::clone(&cfg), &mut session, "first", None, 0, haste::agent::Ctl::default());
    let r2 = haste::agent::run_session(Arc::clone(&cfg), &mut session, "second", None, 0, haste::agent::Ctl::default());
    assert_eq!(r1.final_msg, "one");
    assert_eq!(r2.final_msg, "two");
    let tasks = session.ledger.entries.iter().filter(|e| matches!(e.kind, Kind::Task)).count();
    let finals = session.ledger.entries.iter().filter(|e| matches!(e.kind, Kind::Final)).count();
    assert_eq!((tasks, finals), (2, 2), "one ledger must hold both tasks");
    // Turn numbering continues across tasks so age-based folding stays sane.
    assert_eq!(session.ledger.entries.last().unwrap().turn, 2);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn renderer_stress_huge_ledger() {
    // 10k entries, ~5MB of raw ledger text â€” a long session far past any budget.
    let mut ledger = Ledger::new(None);
    ledger.push(Kind::Task, 0, "stress task".into(), None);
    for turn in 1..=2500u32 {
        ledger.push(Kind::Action, turn, format!("X step {turn}"), None);
        ledger.push(
            Kind::Result,
            turn,
            format!("output for step {turn}\n{}", format!("detail {turn} line\n").repeat(30)),
            Some(turn % 120),
        );
        ledger.push(Kind::Action, turn, format!("R {}", turn % 120), None);
        ledger.push(
            Kind::Result,
            turn,
            format!("#{} file.rs 1:40 of 40\n{}", turn % 120, "1:code line\n".repeat(40)),
            Some(turn % 120),
        );
    }
    let raw: usize = ledger.entries.iter().map(|e| e.text.len()).sum();
    let cfg = haste::config::CtxCfg::default();

    let mut r = Renderer::new();
    let doc = r.render(&ledger, &cfg, 2501);
    let t = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters {
        let _ = r.render(&ledger, &cfg, 2501);
    }
    let ws_ms = t.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let mut acfg = cfg.clone();
    acfg.mode = "append".into();
    let mut ra = Renderer::new();
    let adoc = ra.render(&ledger, &acfg, 2501);
    let t2 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = ra.render(&ledger, &acfg, 2501);
    }
    let ap_ms = t2.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    println!(
        "STRESS: {} entries, {:.1}MB raw -> working_set {} chars in {ws_ms:.2}ms | append {} chars in {ap_ms:.2}ms",
        ledger.entries.len(),
        raw as f64 / 1e6,
        doc.len(),
        adoc.len()
    );
    assert!(ws_ms < 50.0, "working_set render too slow: {ws_ms}ms");
    assert!(ap_ms < 50.0, "append render too slow: {ap_ms}ms");
}

#[test]
fn renderer_bench_1000_entries() {
    let mut ledger = Ledger::new(None);
    ledger.push(Kind::Task, 0, "benchmark task".into(), None);
    for turn in 1..=250u32 {
        ledger.push(Kind::Action, turn, format!("X step {turn}"), None);
        ledger.push(
            Kind::Result,
            turn,
            format!("output for step {turn}\n{}", "detail line\n".repeat(20)),
            Some(turn % 40),
        );
    }
    let cfg = haste::config::CtxCfg::default();
    let mut r = Renderer::new();
    // warmup + timed
    let doc = r.render(&ledger, &cfg, 251);
    let t = std::time::Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ = r.render(&ledger, &cfg, 251);
    }
    let per = t.elapsed().as_micros() as f64 / iters as f64;
    println!(
        "RENDER BENCH: {} entries -> {} chars in {per:.0}us/render",
        ledger.entries.len(),
        doc.len()
    );
    assert!(per < 5000.0, "render too slow: {per}us");
    // budget respected (approximately: est_tokens is chars/4)
    assert!(doc.len() / 4 < cfg.budget_tokens + 2000, "doc {} chars", doc.len());
}


#[test]
fn narration_d_is_refused_and_scaffold_is_stripped() {
    // Turn 1: the model narrates with S and then reaches for D with the same
    // text — that must NOT end the run. Turn 2: a real D, with hallucinated
    // tool-XML after the answer, which must be stripped from the final.
    let port = mock_server(vec![
        "S checking sibling projects for patterns\nD checking sibling projects for patterns\n",
        "D swedish only, no i18n layer\n</invoke>\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "answer", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "swedish only, no i18n layer");
    assert_eq!(rep.turns, 2, "narration D must not end turn 1");
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("narrates work in progress"), "no refusal note");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn trailing_ellipsis_narration_does_not_end_the_run() {
    // Solo-S "Now scanning..." announces the NEXT step — the run must
    // continue, not hand the mic back mid-task.
    let port = mock_server(vec![
        "S Found 11 directories. Now scanning each for project indicators...\n",
        "X echo scanned\nD 11 projects found, 3 are Rust\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "survey", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "11 projects found, 3 are Rust");
    assert_eq!(rep.turns, 2, "ellipsis solo-S must not end turn 1");
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("mid-task"), "no continue nudge in ledger");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn completed_plan_step_triggers_phase_seal_under_budget() {
    // Budget 100 is never hit, but the phase floor (compact_phase_tokens=10)
    // is — so finishing step one must compact right at the boundary.
    let port = mock_server(vec![
        // Turn 1: plan with one verified step + padding work (arms hysteresis).
        "N plan.json\n{\"goal\":\"demo\",\"steps\":[{\"id\":\"s1\",\"what\":\"pad\",\"status\":\"todo\",\"verify\":\"echo ok\"}]}\n.\nX echo a\nX echo b\nX echo c\nX echo d\nX echo e\n",
        // Turn 2: mark the step done — next turn top is the phase boundary.
        "E 0 1:1\n{\"goal\":\"demo\",\"steps\":[{\"id\":\"s1\",\"what\":\"pad\",\"status\":\"done\",\"verify\":\"echo ok\"}]}\n.\n",
        // Turn 3, request 1: the phase-seal compaction call.
        "PHASE BRIEF: step s1 finished; nothing pending.\n",
        // Turn 3, request 2: the real turn.
        "D phase sealed\n",
    ]);
    let root = temp_repo();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n\
         [context]\nmode = \"append\"\nbudget_tokens = 100000\ncompact_phase_tokens = 10\nbootstrap = false\ncompact = \"model\"\ncompact_keep_last = 2\n"
    );
    let cfg: Arc<Config> = Arc::new(toml::from_str(&toml).unwrap());
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "do the phase", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "phase sealed");
    assert_eq!(rep.seals, 1, "phase boundary must seal exactly once");
    let mut wide = cfg.context.clone();
    wide.budget_tokens = 100_000;
    let doc = session.renderer.render(&session.ledger, &wide, 99);
    assert!(doc.contains("PHASE BRIEF"), "summary missing from render:\n{doc}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn append_mode_dedups_identical_results_at_write_time() {
    // Reading the same unchanged file twice: the second result must be a
    // pointer, not a re-send — and the ledger (not just the render) holds it,
    // so the frozen append-mode prefix never carried the duplicate at all.
    let port = mock_server(vec![
        "R big.txt\n",
        "R big.txt\n",
        "D compared\n",
    ]);
    let root = temp_repo();
    let body: String = (1..=30).map(|i| format!("line number {i} with some padding text\n")).collect();
    std::fs::write(root.join("big.txt"), body).unwrap();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n\
         [context]\nmode = \"append\"\nbootstrap = false\n"
    );
    let cfg: Arc<Config> = Arc::new(toml::from_str(&toml).unwrap());
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "compare", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "compared");
    let results: Vec<String> = session
        .ledger
        .entries
        .iter()
        .map(|e| e.text.chars().take(60).collect())
        .collect();
    let dups = session.ledger.entries.iter().filter(|e| e.text.starts_with("(= identical")).count();
    assert_eq!(dups, 1, "second read must be a pointer: {results:?}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn override_tool_replaces_native_verb() {
    // A [tool.G] with override = true takes the G verb over from the builtin.
    let port = mock_server(vec!["G anything at all\n", "D overridden\n"]);
    let root = temp_repo();
    let toml = format!(
        "[model]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"mock\"\n\
         [context]\nbootstrap = false\n\
         [tool.G]\ndesc = \"custom search\"\ncmd = \"echo MODGREP {{args}}\"\noverride = true\n"
    );
    let cfg: Arc<Config> = Arc::new(toml::from_str(&toml).unwrap());
    let mut session = haste::agent::Session::new(&cfg, root.clone(), 0);
    let rep = haste::agent::run_session(Arc::clone(&cfg), &mut session, "search", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "overridden");
    assert!(
        session.ledger.entries.iter().any(|e| e.text.contains("MODGREP")),
        "override tool did not run"
    );
    // Without the flag, a native-verb tool is refused at load.
    let bad = "[model]\nbase_url = \"x\"\nmodel = \"m\"\n[tool.G]\ndesc = \"d\"\ncmd = \"c\"\n";
    let parsed: Config = toml::from_str(bad).unwrap();
    let _ = parsed; // parse is fine; Config::load's validation rejects it
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn pure_prose_turn_is_rescued_as_say() {
    // The model writes its whole report in prose with no verbs — that must
    // reach the user and end the run, not abort as "no commands".
    let port = mock_server(vec![
        "The rounding bug is fixed in tax.py.\nAll three tests pass now.\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "report", None, 0, haste::agent::Ctl::default());
    assert!(rep.final_msg.contains("rounding bug is fixed"), "{}", rep.final_msg);
    assert!(rep.final_msg.contains("tests pass"), "{}", rep.final_msg);
    assert_eq!(rep.turns, 1, "prose rescue must not loop");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn plan_with_trailing_commas_still_parses() {
    let p = std::env::temp_dir().join(format!("haste-plan-{}.json", std::process::id()));
    std::fs::write(&p, "{\"goal\":\"g\",\"steps\":[{\"id\":\"a\",\"status\":\"todo\",},],}").unwrap();
    let plan = haste::plan::Plan::load(&p).unwrap().expect("trailing commas must be forgiven");
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].id, "a");
    let _ = std::fs::remove_file(p);
}

#[test]
fn blocked_step_cannot_be_marked_done() {
    let port = mock_server(vec![
        // s2 depends on s1 (still todo) — the model lies s2 done anyway.
        "N plan.json\n{\"goal\":\"gate\",\"steps\":[{\"id\":\"s1\",\"what\":\"first\",\"status\":\"todo\"},{\"id\":\"s2\",\"what\":\"second\",\"status\":\"done\",\"needs\":[\"s1\"]}]}\n.\n",
        // After the revert note: do it in order.
        "N plan.json\n{\"goal\":\"gate\",\"steps\":[{\"id\":\"s1\",\"what\":\"first\",\"status\":\"done\"},{\"id\":\"s2\",\"what\":\"second\",\"status\":\"done\",\"needs\":[\"s1\"]}]}\n.\nD in order\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "gated", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "in order");
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("BLOCKED by open step"), "no gate note in ledger");
    let plan = std::fs::read_to_string(root.join("plan.json")).unwrap();
    assert!(plan.contains("\"done\""), "{plan}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn step_kickoff_protocol_fires_on_doing_transition() {
    let port = mock_server(vec![
        "N plan.json\n{\"goal\":\"kick\",\"steps\":[{\"id\":\"s1\",\"what\":\"the work\",\"status\":\"todo\"}]}\n.\n",
        "E 0 1:1\n{\"goal\":\"kick\",\"steps\":[{\"id\":\"s1\",\"what\":\"the work\",\"status\":\"doing\"}]}\n.\n",
        "E 0 1:1\n{\"goal\":\"kick\",\"steps\":[{\"id\":\"s1\",\"what\":\"the work\",\"status\":\"done\"}]}\n.\nD kicked\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "go", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "kicked");
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("step 's1' started — protocol"), "no kickoff note");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn talk_only_turn_with_open_plan_does_not_end_the_run() {
    let port = mock_server(vec![
        "N plan.json\n{\"goal\":\"keep going\",\"steps\":[{\"id\":\"s1\",\"what\":\"the fix\",\"status\":\"doing\"}]}\n.\n",
        // Solo-S narration mid-plan — must be nudged back to work, not mic-backed.
        "S still working through the rounding logic\n",
        "N plan.json\n{\"goal\":\"keep going\",\"steps\":[{\"id\":\"s1\",\"what\":\"the fix\",\"status\":\"done\"}]}\n.\nD finished\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "go", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "finished");
    assert_eq!(rep.turns, 3, "solo-S with open plan must not end turn 2");
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("mid-task"), "no continue nudge");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn d_with_discarded_prose_block_is_bounced_until_resent() {
    // The model narrates its report as prose (discarded) then D's with "see
    // summary above" — the D must bounce so the report actually reaches the user.
    let port = mock_server(vec![
        "S here is the lay of the land:\nfirst finding in prose\nsecond finding in prose\nthird finding in prose\nD done - see summary above\n",
        "D full report: alpha, beta, gamma\n",
    ]);
    let root = temp_repo();
    let rep = haste::agent::run(Arc::new(cfg_for(port)), root.clone(), "survey", None, 0, haste::agent::Ctl::default());
    assert_eq!(rep.final_msg, "full report: alpha, beta, gamma");
    assert_eq!(rep.turns, 2, "D with discarded prose must bounce");
    let ledger = std::fs::read_to_string(root.join(".haste/ledger.jsonl")).unwrap();
    assert!(ledger.contains("Resend the report INSIDE the D"), "no bounce note");
    let _ = std::fs::remove_dir_all(root);
}
