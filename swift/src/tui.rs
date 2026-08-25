//! The fleet TUI: an overview page (every agent's live state at a glance)
//! plus one page per agent (its event log, and an input line that drops
//! straight into that agent's inbox — mid-run it lands in the session as a
//! user message). Left/Right or Tab cycles pages; type + Enter to talk.

use crate::{drop_task, file_log, fmt_ev, stamp};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::{cursor, execute, queue, terminal};
use haste::agent::Ev;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

const PLAIN: u8 = 0;
const DIM: u8 = 1;
const USER: u8 = 2;
const ACT: u8 = 3;
const ERR: u8 = 4;
const LOG_CAP: usize = 2000;
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

struct View {
    running: bool,
    turn: u32,
    ctx: u64,
    last: String,
    plan: String,
    log: Vec<(u8, String)>,
}

impl View {
    fn new() -> View {
        View { running: false, turn: 0, ctx: 0, last: String::new(), plan: String::new(), log: Vec::new() }
    }
    fn push(&mut self, kind: u8, line: String) {
        for l in line.lines() {
            self.log.push((kind, l.to_string()));
        }
        if self.log.len() > LOG_CAP {
            self.log.drain(..self.log.len() - LOG_CAP);
        }
    }
}

pub fn run(
    rx: Receiver<(String, Ev)>,
    roots: BTreeMap<String, PathBuf>,
    _stop: Arc<AtomicBool>,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(io::stdout());
    terminal::enable_raw_mode()?;
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;
    let res = main_loop(rx, roots, &mut out);
    execute!(out, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    res
}

fn main_loop(
    rx: Receiver<(String, Ev)>,
    roots: BTreeMap<String, PathBuf>,
    out: &mut impl Write,
) -> io::Result<()> {
    let names: Vec<String> = roots.keys().cloned().collect();
    let mut views: BTreeMap<String, View> = names.iter().map(|n| (n.clone(), View::new())).collect();
    let mut files: BTreeMap<String, std::fs::File> = BTreeMap::new();
    // page 0 = overview; 1..=N = agents by index.
    let mut page = 0usize;
    let mut selected = 0usize;
    let mut input = String::new();
    let mut spin = 0usize;
    let mut dirty = true;
    let mut disconnected = false;

    loop {
        // Drain fleet events.
        loop {
            match rx.try_recv() {
                Ok((name, ev)) => {
                    dirty = true;
                    if let Some(line) = fmt_ev(&ev) {
                        file_log(&mut files, &roots, &name, &line);
                    }
                    let Some(v) = views.get_mut(&name) else { continue };
                    match ev {
                        Ev::Turn(t) => {
                            v.running = true;
                            v.turn = t;
                        }
                        Ev::Ctx(t) => v.ctx = t,
                        Ev::Action(_, a) => {
                            v.last = a.clone();
                            v.push(ACT, format!("  · {a}"));
                        }
                        Ev::Result(_, r) => {
                            let k = if r.starts_with("err") || r.starts_with("exit ") || r.starts_with("model error") || r.starts_with("(aborted") {
                                ERR
                            } else {
                                DIM
                            };
                            v.push(k, format!("    {r}"));
                        }
                        Ev::Say(s) => {
                            let k = if s.starts_with("task: ") || s.starts_with("source task: ") || s.starts_with("(mid-run message)") {
                                USER
                            } else {
                                PLAIN
                            };
                            v.push(k, s);
                        }
                        Ev::Sub(n, l) => v.push(DIM, format!("  [{n}] {}", l.lines().next().unwrap_or(""))),
                        Ev::SubDone(n) => v.push(DIM, format!("  [{n}] finished")),
                        Ev::Plan(p) => v.plan = p,
                        Ev::Report(r) => {
                            v.push(DIM, format!("  {r}"));
                            v.running = false;
                        }
                        Ev::Done(m) => {
                            v.push(PLAIN, format!("DONE: {m}"));
                            v.push(DIM, format!("── finished {} ──", stamp()));
                            v.running = false;
                        }
                        Ev::Delta(_) => {}
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected && views.values().all(|v| !v.running) {
            return Ok(());
        }

        // Keys.
        if event::poll(Duration::from_millis(40))? {
            match event::read()? {
                Event::Key(k) if k.kind != event::KeyEventKind::Release => {
                    dirty = true;
                    match (k.code, k.modifiers) {
                        (KeyCode::Char('c'), KeyModifiers::CONTROL) => return Ok(()),
                        (KeyCode::Right, _) | (KeyCode::Tab, _) => page = (page + 1) % (names.len() + 1),
                        (KeyCode::Left, _) | (KeyCode::BackTab, _) => page = (page + names.len()) % (names.len() + 1),
                        (KeyCode::Up, _) if page == 0 => selected = selected.saturating_sub(1),
                        (KeyCode::Down, _) if page == 0 => selected = (selected + 1).min(names.len().saturating_sub(1)),
                        (KeyCode::Enter, _) => {
                            if page == 0 {
                                page = selected + 1;
                            } else {
                                let line = input.trim().to_string();
                                input.clear();
                                if line == "/q" {
                                    return Ok(());
                                }
                                if !line.is_empty() {
                                    let name = &names[page - 1];
                                    drop_task(&roots[name], &line);
                                    if let Some(v) = views.get_mut(name) {
                                        v.push(USER, format!("> {line}"));
                                    }
                                }
                            }
                        }
                        (KeyCode::Backspace, _) => {
                            input.pop();
                        }
                        (KeyCode::Esc, _) => page = 0,
                        (KeyCode::Char(c), m) if !m.contains(KeyModifiers::CONTROL) => input.push(c),
                        _ => {}
                    }
                }
                Event::Resize(_, _) => dirty = true,
                _ => {}
            }
        }
        if views.values().any(|v| v.running) {
            dirty = true; // keep spinners moving
        }
        if dirty {
            spin = spin.wrapping_add(1);
            draw(&names, &views, page, selected, &input, spin, out)?;
            dirty = false;
        }
    }
}

fn draw(
    names: &[String],
    views: &BTreeMap<String, View>,
    page: usize,
    selected: usize,
    input: &str,
    spin: usize,
    out: &mut impl Write,
) -> io::Result<()> {
    let (tw, th) = terminal::size()?;
    let (w, h) = (tw as usize, th as usize);
    if h < 8 || w < 30 {
        return Ok(());
    }
    for r in 0..h as u16 {
        queue!(out, cursor::MoveTo(0, r), terminal::Clear(terminal::ClearType::UntilNewLine))?;
    }

    // Tab bar.
    let mut tabs = String::from(if page == 0 { "[fleet]" } else { " fleet " });
    for (i, n) in names.iter().enumerate() {
        let live = views[n].running;
        let mark = if live { "*" } else { "" };
        if page == i + 1 {
            tabs.push_str(&format!(" [{n}{mark}]"));
        } else {
            tabs.push_str(&format!("  {n}{mark} "));
        }
    }
    let bar: String = tabs.chars().take(w).collect();
    queue!(out, cursor::MoveTo(0, 0), SetAttribute(Attribute::Reverse), Print(format!("{bar:<w$}")), SetAttribute(Attribute::Reset))?;

    if page == 0 {
        draw_overview(names, views, selected, spin, w, out)?;
    } else {
        draw_agent(&names[page - 1], &views[&names[page - 1]], spin, w, h, out)?;
    }

    // Input box (agent pages) / hint (overview).
    let border = "─".repeat(w.saturating_sub(1));
    queue!(out, cursor::MoveTo(0, (h - 4) as u16))?;
    styled(out, DIM, &border)?;
    let prompt = if page == 0 {
        "↑↓ select · Enter open · Tab cycle · Ctrl+C quit".to_string()
    } else {
        format!("> {input}")
    };
    let nch = prompt.chars().count();
    let shown: String = if nch >= w { prompt.chars().skip(nch - w + 1).collect() } else { prompt };
    queue!(out, cursor::MoveTo(0, (h - 3) as u16))?;
    styled(out, if page == 0 { DIM } else { PLAIN }, &shown)?;
    queue!(out, cursor::MoveTo(0, (h - 2) as u16))?;
    styled(out, DIM, &border)?;
    out.flush()
}

fn draw_overview(
    names: &[String],
    views: &BTreeMap<String, View>,
    selected: usize,
    spin: usize,
    w: usize,
    out: &mut impl Write,
) -> io::Result<()> {
    for (i, n) in names.iter().enumerate() {
        let v = &views[n];
        let state = if v.running {
            format!("{} turn {:<3} ctx {:>6}", SPINNER[spin % SPINNER.len()], v.turn, fmt_k(v.ctx))
        } else {
            "· idle              ".to_string()
        };
        let marker = if i == selected { ">" } else { " " };
        let line = format!("{marker} {:<14} {state}  {}", n, v.last);
        let line: String = line.chars().take(w.saturating_sub(1)).collect();
        queue!(out, cursor::MoveTo(0, (2 + i * 2) as u16))?;
        styled(out, if v.running { ACT } else { DIM }, &line)?;
    }
    Ok(())
}

fn draw_agent(name: &str, v: &View, spin: usize, w: usize, h: usize, out: &mut impl Write) -> io::Result<()> {
    let _ = name;
    let head = if v.running {
        format!(" {} turn {} · ctx {}", SPINNER[spin % SPINNER.len()], v.turn, fmt_k(v.ctx))
    } else {
        format!(" idle · ctx {}", fmt_k(v.ctx))
    };
    queue!(out, cursor::MoveTo(0, 1))?;
    styled(out, DIM, &head.chars().take(w).collect::<String>())?;
    // Log region rows 2..h-4, bottom-anchored, soft-wrapped.
    let log_h = h.saturating_sub(6);
    let mut wrapped: Vec<(u8, String)> = Vec::new();
    for (kind, l) in v.log.iter().rev().take(log_h + 40) {
        let mut parts: Vec<(u8, String)> = Vec::new();
        let mut rest = l.as_str();
        if rest.is_empty() {
            parts.push((*kind, String::new()));
        }
        while !rest.is_empty() {
            let cut = rest
                .char_indices()
                .take_while(|(i, _)| *i < w.saturating_sub(2))
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(rest.len());
            parts.push((*kind, rest[..cut].to_string()));
            rest = &rest[cut..];
        }
        for p in parts.into_iter().rev() {
            wrapped.push(p);
        }
        if wrapped.len() > log_h {
            break;
        }
    }
    let visible: Vec<&(u8, String)> = wrapped.iter().take(log_h).collect();
    for (row, (kind, line)) in visible.iter().rev().enumerate() {
        let start = 2 + log_h.saturating_sub(visible.len());
        queue!(out, cursor::MoveTo(0, (start + row) as u16))?;
        styled(out, *kind, line)?;
    }
    Ok(())
}

fn fmt_k(n: u64) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{n}t")
    }
}

fn styled(out: &mut impl Write, kind: u8, text: &str) -> io::Result<()> {
    match kind {
        DIM => queue!(out, SetAttribute(Attribute::Dim), Print(text), SetAttribute(Attribute::Reset)),
        USER => queue!(out, SetForegroundColor(Color::Green), SetAttribute(Attribute::Bold), Print(text), SetAttribute(Attribute::Reset), ResetColor),
        ACT => queue!(out, SetForegroundColor(Color::DarkCyan), Print(text), ResetColor),
        ERR => queue!(out, SetForegroundColor(Color::Red), Print(text), ResetColor),
        _ => queue!(out, Print(text)),
    }
}
