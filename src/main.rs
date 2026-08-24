use haste::{agent, config::Config};
use std::sync::Arc;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Answer CLI probes natively — shells and editors call these, and each one
    // must not become an LLM conversation.
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("haste {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "haste — micro agent harness for wafer-speed inference\n\n\
             usage: haste [-c config.toml] [-p profile] [-C root] [--tui] [task...]\n       haste update\n\n\
             With no task, opens the TUI in the current directory.\n\
             Config lookup: -c, ./haste.toml, ~/.haste.toml, built-in default."
        );
        return;
    }
    if args.first().map(String::as_str) == Some("update") {
        match haste::update::self_update() {
            Ok(msg) => println!("haste: {msg}"),
            Err(e) => {
                eprintln!("haste: update failed — {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    let mut cfg_path: Option<String> = None;
    let mut profile: Option<String> = None;
    let mut root = std::env::current_dir().expect("cwd");
    let mut want_tui = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                cfg_path = args.get(i + 1).cloned();
                args.drain(i..(i + 2).min(args.len()));
            }
            "--profile" | "-p" => {
                profile = args.get(i + 1).cloned();
                args.drain(i..(i + 2).min(args.len()));
            }
            "--root" | "-C" => {
                if let Some(p) = args.get(i + 1) {
                    root = p.into();
                }
                args.drain(i..(i + 2).min(args.len()));
            }
            "--tui" => {
                want_tui = true;
                args.remove(i);
            }
            _ => i += 1,
        }
    }
    let cfg = match Config::load(cfg_path.as_deref()) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("haste: {e}");
            std::process::exit(2);
        }
    };
    for n in &cfg.mod_notes {
        eprintln!("haste: {n}");
    }
    let task = args.join(" ");

    // No task = interactive. Explicit --tui always wins.
    if want_tui || task.trim().is_empty() {
        if let Err(e) = haste::tui::run(cfg, root) {
            eprintln!("haste tui: {e}");
            std::process::exit(1);
        }
        return;
    }

    let rep = agent::run(cfg, root, &task, profile.as_deref(), 0, agent::Ctl::default());

    println!("{}", rep.final_msg);
    eprintln!(
        "\n-- haste: {} turns, {} cmds, {:.1}s wall | model {:.1}s (ttft sum {:.1}s) | tools {:.1}s | render {:.1}ms | in {}t ({}t cached) out {}t (~{}t est, {}ch) --",
        rep.turns,
        rep.commands,
        rep.wall_ms as f64 / 1000.0,
        rep.model_ms as f64 / 1000.0,
        rep.ttft_ms_sum as f64 / 1000.0,
        rep.tool_ms as f64 / 1000.0,
        rep.render_us as f64 / 1000.0,
        rep.tok_in,
        rep.tok_cached,
        rep.tok_out,
        rep.sent_tokens,
        rep.out_chars,
    );
}
