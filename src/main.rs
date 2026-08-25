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
             usage: haste [-c config.toml] [-p profile] [-C root] [--tui] [task...]\n       haste init | update\n\n\
             With no task, opens the TUI in the current directory.\n\
             Config lookup: -c, ./haste.toml, ~/.haste.toml, built-in default."
        );
        return;
    }
    if args.first().map(String::as_str) == Some("init") {
        let path = Config::home_config_path();
        if path.exists() {
            println!("haste: {} already exists — edit it directly", path.display());
        } else {
            match std::fs::write(&path, haste::config::INIT_TOML) {
                Ok(()) => println!(
                    "haste: wrote {} — uncomment the block for your provider (Cerebras, Ollama, LM Studio, vLLM, OpenRouter), then just run `haste`",
                    path.display()
                ),
                Err(e) => {
                    eprintln!("haste: could not write {}: {e}", path.display());
                    std::process::exit(1);
                }
            }
        }
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
    let mut events = false;
    let mut model_choice: Option<String> = None;
    let mut reason_choice: Option<String> = None;
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
            "--events" => {
                events = true;
                args.remove(i);
            }
            "--model" | "-m" => {
                model_choice = args.get(i + 1).cloned();
                args.drain(i..(i + 2).min(args.len()));
            }
            "--reason" | "-r" => {
                reason_choice = args.get(i + 1).cloned();
                args.drain(i..(i + 2).min(args.len()));
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
    let cfg = match model_choice {
        Some(name) => {
            let mut c = (*cfg).clone();
            match c.models.get(&name).cloned() {
                Some(m) => {
                    c.model = m;
                    Arc::new(c)
                }
                None => {
                    let have: Vec<&str> = c.models.keys().map(String::as_str).collect();
                    eprintln!("haste: no [models.{name}] in config (available: {})", have.join(", "));
                    std::process::exit(2);
                }
            }
        }
        None => cfg,
    };
    let cfg = match reason_choice {
        Some(r) => {
            if !cfg.model.reasoning.contains_key(&r) {
                let have: Vec<&str> = cfg.model.reasoning.keys().map(String::as_str).collect();
                eprintln!("haste: no [model.reasoning.{r}] for this model (available: {})", have.join(", "));
                std::process::exit(2);
            }
            Config::effective(&cfg, &None, &Some(r))
        }
        None => cfg,
    };
    for n in &cfg.mod_notes {
        eprintln!("haste: {n}");
    }
    let task = args.join(" ");

    // No task = interactive. Explicit --tui always wins.
    if want_tui || task.trim().is_empty() {
        let initial = (!task.trim().is_empty()).then(|| task.clone());
        if let Err(e) = haste::tui::run(cfg, root, initial) {
            eprintln!("haste tui: {e}");
            std::process::exit(1);
        }
        return;
    }

    // --events: one JSON line per agent event on stdout — the interface a
    // fleet manager or any supervisor scripts against.
    let (ctl, ev_thread) = if events {
        let (tx, rx) = std::sync::mpsc::channel();
        let h = std::thread::spawn(move || {
            for ev in rx {
                if let Ok(line) = serde_json::to_string(&ev) {
                    println!("{line}");
                }
            }
        });
        (agent::Ctl { sink: Some(tx), ..Default::default() }, Some(h))
    } else {
        (agent::Ctl::default(), None)
    };
    let rep = agent::run(cfg, root, &task, profile.as_deref(), 0, ctl);
    if let Some(h) = ev_thread {
        let _ = h.join(); // sender dropped with the run's Ctl — drains cleanly
    }

    if !events {
        println!("{}", rep.final_msg);
    }
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
