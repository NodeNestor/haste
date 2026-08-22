//! Split-screen TUI: top = session log (actions/results/reports), bottom =
//! the live agent stream (raw model output as it generates, F2 toggles) plus
//! status bar and input line. Headless mode never touches this module.

use crate::agent::{self, Ctl, Ev};
use crate::config::Config;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::style::{Attribute, Print, SetAttribute};
use crossterm::{cursor, execute, queue, terminal};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

const LOG_CAP: usize = 4000;

struct App {
    log: Vec<String>,
    live: String,
    input: String,
    root: PathBuf,
    running: bool,
    stream_on: bool,
    scroll: usize,
    turn: u32,
    started: Option<Instant>,
    rx: Option<Receiver<Ev>>,
    stop: Option<Arc<AtomicBool>>,
    quit: bool,
}

impl App {
    fn push(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > LOG_CAP {
            self.log.drain(..self.log.len() - LOG_CAP);
        }
        self.scroll = 0;
    }
}

pub fn run(cfg: Arc<Config>, root: PathBuf) -> io::Result<()> {
    let mut out = io::BufWriter::new(io::stdout());
    terminal::enable_raw_mode()?;
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;
    let res = main_loop(&cfg, root, &mut out);
    execute!(out, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    res
}

fn main_loop(cfg: &Arc<Config>, root: PathBuf, out: &mut impl Write) -> io::Result<()> {
    let mut app = App {
        log: vec![
            format!("haste — {} @ {}", cfg.model.model, cfg.model.base_url),
            "type a task and press Enter · :cd <path> · F2 stream pane · Esc stop · :q quit".into(),
            String::new(),
        ],
        live: String::new(),
        input: String::new(),
        root,
        running: false,
        stream_on: true,
        scroll: 0,
        turn: 0,
        started: None,
        rx: None,
        stop: None,
        quit: false,
    };
    let mut dirty = true;
    let mut last_tick = Instant::now();

    while !app.quit {
        if drain_events(&mut app) {
            dirty = true;
        }
        if event::poll(Duration::from_millis(40))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != event::KeyEventKind::Release {
                    handle_key(&mut app, cfg, k);
                    dirty = true;
                }
            }
        }
        // tick the elapsed clock while running
        if app.running && last_tick.elapsed() > Duration::from_millis(250) {
            dirty = true;
        }
        if dirty {
            draw(&app, out)?;
            dirty = false;
            last_tick = Instant::now();
        }
    }
    Ok(())
}

fn drain_events(app: &mut App) -> bool {
    let mut evs = Vec::new();
    let mut disconnected = false;
    {
        let Some(rx) = &app.rx else { return false };
        loop {
            match rx.try_recv() {
                Ok(ev) => evs.push(ev),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
    }
    let changed = !evs.is_empty() || disconnected;
    for ev in evs {
        match ev {
            Ev::Turn(t) => {
                app.turn = t;
                app.live.clear();
            }
            Ev::Delta(d) => app.live.push_str(&d),
            Ev::Action(depth, a) => {
                let pad = "  ".repeat(depth as usize);
                app.push(format!("{pad}> {a}"));
            }
            Ev::Result(depth, r) => {
                let pad = "  ".repeat(depth as usize);
                app.push(format!("{pad}  {r}"));
            }
            Ev::Report(r) => app.push(format!("-- {r}")),
            Ev::Done(msg) => {
                for l in msg.lines() {
                    app.push(format!("DONE: {l}"));
                }
                app.push(String::new());
                app.running = false;
                app.live.clear();
                app.rx = None;
                app.stop = None;
                return true;
            }
        }
    }
    if disconnected {
        app.running = false;
        app.rx = None;
    }
    changed
}

fn handle_key(app: &mut App, cfg: &Arc<Config>, k: KeyEvent) {
    match (k.code, k.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            if app.running {
                request_stop(app);
            } else {
                app.quit = true;
            }
        }
        (KeyCode::Esc, _) => request_stop(app),
        (KeyCode::F(2), _) => app.stream_on = !app.stream_on,
        (KeyCode::PageUp, _) => app.scroll = (app.scroll + 5).min(app.log.len()),
        (KeyCode::PageDown, _) => app.scroll = app.scroll.saturating_sub(5),
        (KeyCode::Backspace, _) => {
            app.input.pop();
        }
        (KeyCode::Enter, _) => submit(app, cfg),
        (KeyCode::Char(c), m) if !m.contains(KeyModifiers::CONTROL) => app.input.push(c),
        _ => {}
    }
}

fn request_stop(app: &mut App) {
    if let Some(s) = &app.stop {
        s.store(true, Ordering::Relaxed);
        app.push("(stop requested — finishing current turn)".into());
    }
}

fn submit(app: &mut App, cfg: &Arc<Config>) {
    let line = app.input.trim().to_string();
    app.input.clear();
    if line.is_empty() {
        return;
    }
    if let Some(rest) = line.strip_prefix(":cd ") {
        if app.running {
            app.push("(busy — :cd after the run finishes)".into());
            return;
        }
        let p = PathBuf::from(rest.trim());
        match p.canonicalize() {
            Ok(abs) if abs.is_dir() => {
                app.push(format!("root -> {}", abs.display()));
                app.root = abs;
            }
            _ => app.push(format!("no such directory: {rest}")),
        }
        return;
    }
    if line == ":q" {
        app.quit = true;
        return;
    }
    if app.running {
        app.push("(a run is active — Esc to stop it first)".into());
        return;
    }
    app.push(format!("TASK: {line}"));
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    app.rx = Some(rx);
    app.stop = Some(stop.clone());
    app.running = true;
    app.turn = 0;
    app.started = Some(Instant::now());
    let cfg = Arc::clone(cfg);
    let root = app.root.clone();
    std::thread::spawn(move || {
        agent::run(cfg, root, &line, None, 0, Ctl { sink: Some(tx), stop: Some(stop) });
    });
}

fn wrap_into<'a>(dst: &mut Vec<String>, line: &'a str, w: usize) {
    if line.len() <= w {
        dst.push(line.to_string());
        return;
    }
    let mut rest = line;
    while !rest.is_empty() {
        let cut = rest
            .char_indices()
            .take_while(|(i, _)| *i < w)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(rest.len());
        dst.push(rest[..cut].to_string());
        rest = &rest[cut..];
    }
}

fn draw(app: &App, out: &mut impl Write) -> io::Result<()> {
    let (tw, th) = terminal::size()?;
    let (w, h) = (tw as usize, th as usize);
    if h < 6 || w < 20 {
        return Ok(());
    }
    queue!(out, terminal::Clear(terminal::ClearType::All), cursor::MoveTo(0, 0))?;

    // Bottom-up budget: input line, status line, then optional stream pane.
    let stream_h = if app.stream_on { (h - 2) / 2 } else { 0 };
    let log_h = h - 2 - stream_h - if stream_h > 0 { 1 } else { 0 };

    // Session log (top).
    let mut wrapped: Vec<String> = Vec::new();
    for l in app.log.iter().rev().take(log_h + app.scroll + 40) {
        let mut tmp = Vec::new();
        wrap_into(&mut tmp, l, w.saturating_sub(1));
        for t in tmp.into_iter().rev() {
            wrapped.push(t);
        }
        if wrapped.len() > log_h + app.scroll {
            break;
        }
    }
    let visible: Vec<&String> = wrapped.iter().skip(app.scroll).take(log_h).collect();
    for (row, line) in visible.iter().rev().enumerate() {
        let start = log_h.saturating_sub(visible.len());
        queue!(out, cursor::MoveTo(0, (start + row) as u16), Print(line))?;
    }

    // Stream pane (bottom half).
    if stream_h > 0 {
        let div_row = log_h as u16;
        let title = format!("── agent stream (F2 hides) {}", "─".repeat(w.saturating_sub(28)));
        queue!(out, cursor::MoveTo(0, div_row), SetAttribute(Attribute::Dim), Print(&title[..title.len().min(w)]), SetAttribute(Attribute::Reset))?;
        let mut lines: Vec<String> = Vec::new();
        for l in app.live.lines() {
            wrap_into(&mut lines, l, w.saturating_sub(1));
        }
        let take = lines.len().saturating_sub(stream_h - 1);
        for (i, l) in lines[take..].iter().enumerate() {
            queue!(out, cursor::MoveTo(0, div_row + 1 + i as u16), SetAttribute(Attribute::Dim), Print(l), SetAttribute(Attribute::Reset))?;
        }
    }

    // Status bar.
    let state = if app.running {
        let secs = app.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        format!("RUNNING turn {} · {:.1}s", app.turn, secs)
    } else {
        "idle".into()
    };
    let status = format!(
        " {} · {} · stream {} · Esc stop · :q quit ",
        app.root.display(),
        state,
        if app.stream_on { "ON" } else { "off" }
    );
    let mut bar = status;
    bar.truncate(w);
    queue!(out, cursor::MoveTo(0, (h - 2) as u16), SetAttribute(Attribute::Reverse), Print(format!("{bar:<w$}")), SetAttribute(Attribute::Reset))?;

    // Input line.
    let prompt = format!("> {}", app.input);
    let shown: String = if prompt.len() >= w { prompt[prompt.len() - w + 1..].to_string() } else { prompt };
    queue!(out, cursor::MoveTo(0, (h - 1) as u16), Print(&shown))?;
    out.flush()
}
