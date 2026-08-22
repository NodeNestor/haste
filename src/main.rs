use haste::{agent, config::Config};
use std::sync::Arc;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg_path: Option<String> = None;
    let mut profile: Option<String> = None;
    let mut root = std::env::current_dir().expect("cwd");
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
            _ => i += 1,
        }
    }
    let task = args.join(" ");
    if task.trim().is_empty() {
        eprintln!("usage: haste [-c haste.toml] [-p profile] [-C root] <task...>");
        std::process::exit(2);
    }
    let cfg = match Config::load(cfg_path.as_deref()) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("haste: {e}");
            std::process::exit(2);
        }
    };

    let rep = agent::run(cfg, root, &task, profile.as_deref(), 0);

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
