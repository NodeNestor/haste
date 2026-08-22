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
            let fin = serde_json::json!({"choices":[{"delta":{}, "finish_reason": fr}]});
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
    let p = std::env::temp_dir().join(format!("haste-e2e-{}-{:?}", std::process::id(), std::time::Instant::now()).replace([':', ' ', '.'], "-"));
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
    let ctl = haste::agent::Ctl { sink: Some(tx), stop: None };
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
        // Long final message so run 1's history alone busts the tiny budget.
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
        texts.iter().any(|t| t.contains("identical result 3 times")),
        "warning missing: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("refused: this exact command")),
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

