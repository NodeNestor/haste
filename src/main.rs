use haste::{agent, config::Config};
use std::sync::Arc;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
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
        "\n-- haste: {} turns, {} cmds, {:.1}s wall | model {:.1}s (ttft sum {:.1}s) | tools {:.1}s | render {:.1}ms | sent ~{}t, out {}ch --",
        rep.turns,
        rep.commands,
        rep.wall_ms as f64 / 1000.0,
        rep.model_ms as f64 / 1000.0,
        rep.ttft_ms_sum as f64 / 1000.0,
        rep.tool_ms as f64 / 1000.0,
        rep.render_us as f64 / 1000.0,
        rep.sent_tokens,
        rep.out_chars,
    );
}
