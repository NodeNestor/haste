//! Split-screen TUI, styled like a chat: your prompt echoed as "> ...", the
//! agent's answer as plain text, all machinery (actions, results, stats)
//! dimmed. One continuable Session per root, so follow-up prompts share the
//! ledger and workspace — conversation memory. Bottom pane = live model
//! stream (F2 toggles). Headless mode never touches this module.

use crate::agent::{self, Ctl, Ev, Session};
use crate::config::Config;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::{cursor, execute, queue, terminal};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const LOG_CAP: usize = 4000;

/// Line styles in the session log.
const PLAIN: u8 = 0; // agent's answer
const DIM: u8 = 1; // results, stats, system notes
const USER: u8 = 2; // the user's prompt
const ACT: u8 = 3; // tool actions
const ERR: u8 = 4; // errors, aborts, refusals

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

struct App {
    log: Vec<(u8, String)>,
    live: String,
    input: String,
    root: PathBuf,
    running: bool,
    stream_on: bool,
    scroll: usize,
    turn: u32,
    /// Real billed context size (prompt_tokens) of the latest request.
    ctx_tokens: u64,
    last_action: String,
    /// Sidebar: (name, latest activity line, finished) per live subagent.
    agents: Vec<(String, String, bool)>,
    /// Sidebar: current compact plan view.
    plan: String,
    spin: usize,
    started: Option<Instant>,
    rx: Option<Receiver<Ev>>,
    sess_rx: Option<Receiver<Session>>,
    session: Option<Session>,
    stop: Option<Arc<AtomicBool>>,
    inbox: Option<Arc<Mutex<Vec<String>>>>,
    quit: bool,
}

impl App {
    fn push(&mut self, kind: u8, line: String) {
        self.log.push((kind, crate::tools::clip(&line, 600)));
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
        log: {
            let mut l = vec![
                (DIM, format!("haste · {} · {}", cfg.model.model, root.display())),
                (DIM, "Enter task · :cd <path> · F2 stream · Esc stop · :q quit".into()),
            ];
            for n in &cfg.mod_notes {
                l.push((DIM, format!("  {n}")));
            }
            l.push((DIM, String::new()));
            l
        },
        live: String::new(),
        input: String::new(),
        root,
        running: false,
        stream_on: true,
        scroll: 0,
        turn: 0,
        ctx_tokens: 0,
        last_action: String::new(),
        agents: Vec::new(),
        plan: String::new(),
        spin: 0,
        started: None,
        rx: None,
        sess_rx: None,
        session: None,
        stop: None,
        inbox: None,
        quit: false,
    };
    let mut dirty = true;
    let mut last_tick = Instant::now();
    // One background release check per TUI session; at most one line arrives.
    let mut update_rx = Some(crate::update::spawn_check());

    while !app.quit {
        if let Some(rx) = &update_rx {
            if let Ok(msg) = rx.try_recv() {
                app.push(DIM, format!("  {msg}"));
                update_rx = None;
                dirty = true;
            }
        }
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
        if app.running && last_tick.elapsed() > Duration::from_millis(250) {
            dirty = true;
        }
        if dirty {
            draw(&mut app, out)?;
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
            Ev::Action(_, a) => {
                app.last_action = a.clone();
                app.push(ACT, format!("  · {a}"));
            }
            Ev::Result(_, r) => {
                let k = if r.starts_with("err") || r.starts_with("exit ") || r.starts_with("model error") || r.starts_with("(aborted") {
                    ERR
                } else {
                    DIM
                };
                app.push(k, format!("    {r}"));
            }
            Ev::Say(text) => app.push(PLAIN, text),
            Ev::Report(r) => app.push(DIM, format!("  {r}")),
            Ev::Ctx(t) => app.ctx_tokens = t,
            Ev::Sub(name, line) => {
                let line = crate::tools::clip(line.lines().next().unwrap_or(""), 80);
                match app.agents.iter_mut().find(|(n, _, _)| *n == name) {
                    Some(row) => {
                        row.1 = line;
                        row.2 = false;
                    }
                    None => app.agents.push((name, line, false)),
                }
            }
            Ev::SubDone(name) => {
                if let Some(row) = app.agents.iter_mut().find(|(n, _, _)| *n == name) {
                    row.2 = true;
                }
            }
            Ev::Plan(p) => app.plan = p,
            Ev::Done(msg) => {
                finish_run(app, msg);
                return true;
            }
        }
    }
    if disconnected {
        finish_run(app, "(worker died)".into());
    }
    changed
}

fn finish_run(app: &mut App, msg: String) {
    for l in msg.lines() {
        app.push(PLAIN, l.to_string());
    }
    app.push(PLAIN, String::new());
    app.running = false;
    app.live.clear();
    app.last_action.clear();
    app.rx = None;
    app.stop = None;
    app.inbox = None;
    // Take the continued session back from the worker thread.
    if let Some(rx) = app.sess_rx.take() {
        if let Ok(s) = rx.recv_timeout(Duration::from_secs(2)) {
            app.session = Some(s);
        }
    }
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
        app.push(DIM, "  (stopping after this turn)".into());
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
            app.push(DIM, "  (busy — :cd after the run finishes)".into());
            return;
        }
        let p = PathBuf::from(rest.trim());
        match p.canonicalize() {
            Ok(abs) if abs.is_dir() => {
                app.push(DIM, format!("  root → {} (new session)", abs.display()));
                app.root = abs;
                app.session = None;
            }
            _ => app.push(DIM, format!("  no such directory: {rest}")),
        }
        return;
    }
    if line == ":q" {
        app.quit = true;
        return;
    }
    if app.running {
        // Talk to the leader mid-run: the message lands in its context at the
        // top of its next turn.
        if let Some(inbox) = &app.inbox {
            inbox.lock().unwrap().push(line.clone());
            app.push(USER, format!("> {line}"));
            app.push(DIM, "  (delivered to the agent — lands next turn)".into());
        }
        return;
    }
    app.push(USER, format!("> {line}"));
    app.agents.clear();
    let (tx, rx) = std::sync::mpsc::channel();
    let (sess_tx, sess_rx) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let inbox = Arc::new(Mutex::new(Vec::<String>::new()));
    app.rx = Some(rx);
    app.sess_rx = Some(sess_rx);
    app.stop = Some(stop.clone());
    app.inbox = Some(inbox.clone());
    app.running = true;
    app.turn = 0;
    app.started = Some(Instant::now());
    let cfg2 = Arc::clone(cfg);
    let mut session = app
        .session
        .take()
        .unwrap_or_else(|| Session::new(cfg, app.root.clone(), 0));
    std::thread::spawn(move || {
        agent::run_session(
            cfg2,
            &mut session,
            &line,
            None,
            0,
            Ctl { sink: Some(tx), stop: Some(stop), inbox: Some(inbox), tag: None },
        );
        let _ = sess_tx.send(session);
    });
}

fn wrap_into(dst: &mut Vec<(u8, String)>, kind: u8, line: &str, w: usize) {
    if line.len() <= w {
        dst.push((kind, line.to_string()));
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
        dst.push((kind, rest[..cut].to_string()));
        rest = &rest[cut..];
    }
}

fn styled_print(out: &mut impl Write, kind: u8, text: &str) -> io::Result<()> {
    match kind {
        DIM => queue!(out, SetAttribute(Attribute::Dim), Print(text), SetAttribute(Attribute::Reset)),
        USER => queue!(out, SetForegroundColor(Color::Green), SetAttribute(Attribute::Bold), Print(text), SetAttribute(Attribute::Reset), ResetColor),
        ACT => queue!(out, SetForegroundColor(Color::DarkCyan), Print(text), ResetColor),
        ERR => queue!(out, SetForegroundColor(Color::Red), Print(text), ResetColor),
        _ => queue!(out, Print(text)),
    }
}

/// Right sidebar: live agents on top, the plan below. Only drawn while it has
/// content; the chat gets the full width back otherwise.
fn draw_sidebar(app: &App, out: &mut impl Write, x: u16, rows: usize, sw: usize) -> io::Result<()> {
    let mut row: u16 = 0;
    if !app.agents.is_empty() {
        side_line(out, x, row, DIM, "── agents ──", sw)?;
        row += 1;
        for (name, last, done) in &app.agents {
            if row as usize + 2 > rows {
                break;
            }
            let head = if *done {
                format!("✔ {name}")
            } else {
                format!("{} {name}", SPINNER[app.spin % SPINNER.len()])
            };
            side_line(out, x, row, if *done { DIM } else { ACT }, &head, sw)?;
            row += 1;
            side_line(out, x, row, DIM, &format!("  {last}"), sw)?;
            row += 1;
        }
        row += 1;
    }
    if !app.plan.is_empty() && (row as usize) < rows {
        side_line(out, x, row, DIM, "── plan ──", sw)?;
        row += 1;
        for l in app.plan.lines() {
            if row as usize >= rows {
                break;
            }
            let l = l.strip_prefix("## PLAN: ").unwrap_or(l);
            let kind = if l.starts_with("[x]") {
                USER // green: done
            } else if l.starts_with("[>]") {
                ACT // cyan: in progress
            } else {
                DIM
            };
            side_line(out, x, row, kind, l, sw)?;
            row += 1;
        }
    }
    Ok(())
}

fn side_line(out: &mut impl Write, x: u16, r: u16, kind: u8, text: &str, sw: usize) -> io::Result<()> {
    let clipped: String = text.chars().take(sw).collect();
    queue!(out, cursor::MoveTo(x, r))?;
    styled_print(out, kind, &clipped)
}

fn draw(app: &mut App, out: &mut impl Write) -> io::Result<()> {
    let (tw, th) = terminal::size()?;
    let (w, h) = (tw as usize, th as usize);
    if h < 6 || w < 20 {
        return Ok(());
    }
    if app.running {
        app.spin = app.spin.wrapping_add(1);
    }
    // Sidebar (agents + plan) claims the right edge only when it has content
    // and the terminal is wide enough for both panes to breathe.
    let side = if w >= 90 && (!app.agents.is_empty() || !app.plan.is_empty()) {
        34.min(w / 3)
    } else {
        0
    };
    let cw = w - side; // chat width (sidebar text starts after a divider)
    queue!(out, terminal::Clear(terminal::ClearType::All), cursor::MoveTo(0, 0))?;

    // The stream pane only takes rows while something is streaming, and only
    // as many as its content needs (capped at half the screen) — otherwise the
    // log hugs the input line at the bottom, chat-style.
    let live_lines: usize = app
        .live
        .lines()
        .map(|l| 1 + l.len() / w.saturating_sub(1).max(1))
        .sum();
    // Bottom rows: input, status bar, and two blank padding rows so the chat
    // floats a little above the bar instead of sitting on it.
    let usable = h - 4;
    let stream_h = if app.stream_on && app.running && live_lines > 0 {
        (live_lines + 1).min(usable / 2)
    } else {
        0
    };
    let log_h = usable - stream_h;

    // Session log (top), bottom-anchored, wrapped to the chat pane width.
    let mut wrapped: Vec<(u8, String)> = Vec::new();
    for (kind, l) in app.log.iter().rev().take(log_h + app.scroll + 40) {
        let mut tmp = Vec::new();
        wrap_into(&mut tmp, *kind, l, cw.saturating_sub(2));
        for t in tmp.into_iter().rev() {
            wrapped.push(t);
        }
        if wrapped.len() > log_h + app.scroll {
            break;
        }
    }
    let visible: Vec<&(u8, String)> = wrapped.iter().skip(app.scroll).take(log_h).collect();
    for (row, (kind, line)) in visible.iter().rev().enumerate() {
        let start = log_h.saturating_sub(visible.len());
        queue!(out, cursor::MoveTo(0, (start + row) as u16))?;
        styled_print(out, *kind, line)?;
    }

    // Sidebar: divider column + agents/plan panes over the log region.
    if side > 0 {
        for r in 0..log_h {
            queue!(out, cursor::MoveTo(cw as u16 - 1, r as u16))?;
            styled_print(out, DIM, "│")?;
        }
        draw_sidebar(app, out, cw as u16 + 1, log_h, side.saturating_sub(2))?;
    }

    // Live stream pane.
    if stream_h > 0 {
        let div_row = log_h as u16;
        // Build to exact column width — never byte-slice multi-byte dashes.
        let title = format!("── agent stream (F2 hides) {}", "─".repeat(w.saturating_sub(28).max(1)));
        let title: String = title.chars().take(w).collect();
        queue!(out, cursor::MoveTo(0, div_row))?;
        styled_print(out, DIM, &title)?;
        let mut lines: Vec<(u8, String)> = Vec::new();
        for l in app.live.lines() {
            wrap_into(&mut lines, DIM, l, w.saturating_sub(1));
        }
        let take = lines.len().saturating_sub(stream_h - 1);
        for (i, (_, l)) in lines[take..].iter().enumerate() {
            queue!(out, cursor::MoveTo(0, div_row + 1 + i as u16))?;
            styled_print(out, DIM, l)?;
        }
    }

    // Status bar.
    let state = if app.running {
        let secs = app.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        let act = if app.last_action.is_empty() {
            String::new()
        } else {
            format!(" · {}", app.last_action)
        };
        format!("{} turn {} · {:.0}s{act}", SPINNER[app.spin % SPINNER.len()], app.turn, secs)
    } else {
        "idle".into()
    };
    let ctx = if app.ctx_tokens > 0 {
        format!(" · ctx {:.1}k", app.ctx_tokens as f64 / 1000.0)
    } else {
        String::new()
    };
    let status = format!(
        " {} · {}{} · stream {} ",
        app.root.display(),
        state,
        ctx,
        if app.stream_on { "on" } else { "off" }
    );
    let bar: String = status.chars().take(w).collect();
    queue!(out, cursor::MoveTo(0, (h - 2) as u16), SetAttribute(Attribute::Reverse), Print(format!("{bar:<w$}")), SetAttribute(Attribute::Reset))?;

    // Input line.
    let prompt = format!("> {}", app.input);
    let nch = prompt.chars().count();
    let shown: String = if nch >= w {
        prompt.chars().skip(nch - w + 1).collect()
    } else {
        prompt
    };
    queue!(out, cursor::MoveTo(0, (h - 1) as u16), Print(&shown))?;
    out.flush()
}
