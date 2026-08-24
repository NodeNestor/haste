//! swift — hyper-light fleet manager for haste agents.
//!
//! A fleet is a TOML file of named agents. Each agent has a workspace root
//! and an inbox directory (`<root>/.swift/inbox/`); any text file dropped
//! there is a task. Persistent agents keep ONE haste Session across tasks —
//! conversation memory, interned files, sealed history, near-constant
//! context forever. One-shot agents get a fresh session per task.
//!
//!   swift fleet.toml                 run the fleet (watches inboxes)
//!   swift send <agent> <task...>     drop a task (fleet.toml in cwd)
//!
//! Everything else — what an agent can do, which model, which mods — is the
//! agent's own haste.toml. swift only routes work and multiplexes events.

use haste::agent::{self, Ctl, Ev, Session};
use haste::config::Config;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

#[derive(Deserialize)]
struct Fleet {
    #[serde(default)]
    agent: BTreeMap<String, AgentCfg>,
}

#[derive(Deserialize, Clone)]
struct AgentCfg {
    /// Workspace root; the inbox lives at <root>/.swift/inbox/.
    root: String,
    /// Keep one Session across tasks (memory) instead of fresh-per-task.
    #[serde(default)]
    persistent: bool,
    /// haste profile to run under (defaults to the leader defaults).
    #[serde(default)]
    profile: Option<String>,
    /// Explicit haste config path (default: the root's own lookup chain).
    #[serde(default)]
    config: Option<String>,
    /// Task feeder: a shell command polled every `interval_s`; every output
    /// line NEVER SEEN BEFORE becomes one task. `gh issue list --label
    /// coder --json number,title -q '.[] | "fix issue #\(.number): \(.title)"'`
    /// turns labeled issues into work.
    #[serde(default)]
    source: Option<String>,
    #[serde(default = "d_interval")]
    interval_s: u64,
    /// One-shot agents: tasks that may run at once (persistent is always 1 —
    /// one session, one timeline). >1 on a shared root suits read/research
    /// work; concurrent EDITORS want separate roots or worktrees.
    #[serde(default = "d_parallel")]
    parallel: usize,
}
fn d_interval() -> u64 {
    60
}
fn d_parallel() -> usize {
    1
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("send") => {
            let (agent, task) = (args.get(1), args[2..].join(" "));
            let Some(agent) = agent else { usage() };
            if task.trim().is_empty() {
                usage();
            }
            let fleet = load_fleet(Path::new("fleet.toml"));
            let Some(a) = fleet.agent.get(agent) else {
                eprintln!("swift: no agent '{agent}' in fleet.toml");
                std::process::exit(2);
            };
            let path = drop_task(Path::new(&a.root), &task);
            println!("swift: queued for {agent}: {}", path.display());
        }
        Some(p) if p.ends_with(".toml") => run_fleet(Path::new(p)),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage: swift <fleet.toml> | swift send <agent> <task...>");
    std::process::exit(2);
}

fn load_fleet(path: &Path) -> Fleet {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("swift: {}: {e}", path.display());
        std::process::exit(2);
    });
    let fleet: Fleet = toml::from_str(&text).unwrap_or_else(|e| {
        eprintln!("swift: {}: {e}", path.display());
        std::process::exit(2);
    });
    if fleet.agent.is_empty() {
        eprintln!("swift: {} declares no [agent.*]", path.display());
        std::process::exit(2);
    }
    fleet
}

fn inbox_dir(root: &Path) -> PathBuf {
    root.join(".swift").join("inbox")
}

/// Write one task file, timestamp-named so the queue stays ordered.
fn drop_task(root: &Path, task: &str) -> PathBuf {
    let dir = inbox_dir(root);
    let _ = std::fs::create_dir_all(&dir);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("{stamp}-{}.task", std::process::id()));
    let _ = std::fs::write(&path, task);
    path
}

/// Oldest task file in the inbox, consumed (read + deleted) atomically enough
/// for a single manager: rename first so a half-written file is never lost.
fn next_task(inbox: &Path) -> Option<String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(inbox)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "task"))
        .collect();
    files.sort();
    let path = files.into_iter().next()?;
    let claimed = path.with_extension("task.claimed");
    std::fs::rename(&path, &claimed).ok()?;
    let text = std::fs::read_to_string(&claimed).ok()?;
    let _ = std::fs::remove_file(&claimed);
    let t = text.trim().to_string();
    (!t.is_empty()).then_some(t)
}

fn run_fleet(path: &Path) {
    let fleet = load_fleet(path);
    let stop = Arc::new(AtomicBool::new(false));
    let (log_tx, log_rx) = std::sync::mpsc::channel::<(String, Ev)>();
    let mut handles = Vec::new();
    let mut roots: BTreeMap<String, PathBuf> = BTreeMap::new();
    for (name, acfg) in fleet.agent {
        println!(
            "swift: {} {} @ {}{}",
            if acfg.persistent { "persistent" } else { "one-shot " },
            name,
            acfg.root,
            acfg.source.as_deref().map(|s| format!(" ← `{s}`")).unwrap_or_default()
        );
        let _ = std::fs::create_dir_all(inbox_dir(Path::new(&acfg.root)));
        roots.insert(name.clone(), PathBuf::from(&acfg.root));
        handles.push(spawn_agent(name, acfg, log_tx.clone(), Arc::clone(&stop)));
    }
    drop(log_tx);
    // The manager's whole UI: one multiplexed event log — mirrored into each
    // agent's own <root>/.swift/log so history survives the console.
    let mut logs: BTreeMap<String, std::fs::File> = BTreeMap::new();
    for (name, ev) in log_rx {
        if let Some(line) = fmt_ev(&ev) {
            println!("[{name}] {line}");
            if let Some(f) = logs.get_mut(&name) {
                use std::io::Write;
                let _ = writeln!(f, "{} {line}", stamp());
            } else if let Some(root) = roots.get(&name) {
                if let Ok(f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(root.join(".swift").join("log"))
                {
                    logs.insert(name.clone(), f);
                    if let Some(f) = logs.get_mut(&name) {
                        use std::io::Write;
                        let _ = writeln!(f, "{} {line}", stamp());
                    }
                }
            }
        }
    }
    for h in handles {
        let _ = h.join();
    }
}

/// HH:MM:SS (UTC) — enough to read a log, no chrono dependency.
fn stamp() -> String {
    let s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

/// One thread per agent: watch the inbox, run tasks. Persistent agents reuse
/// one Session (the fleet's memory); one-shot agents start clean every time.
fn spawn_agent(
    name: String,
    acfg: AgentCfg,
    log: Sender<(String, Ev)>,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let root = PathBuf::from(&acfg.root);
        let inbox = inbox_dir(&root);
        let cfg = match load_agent_config(&acfg, &root) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                let _ = log.send((name.clone(), Ev::Say(format!("config error: {e}"))));
                return;
            }
        };
        let mut session: Option<Session> = None;
        let mut running: Vec<std::thread::JoinHandle<()>> = Vec::new();
        let mut last_poll: Option<std::time::Instant> = None;
        while !stop.load(Ordering::Relaxed) {
            // Feeder: poll the source command; every never-seen line = a task.
            if let Some(src) = &acfg.source {
                let due = last_poll.is_none_or(|t| t.elapsed().as_secs() >= acfg.interval_s);
                if due {
                    last_poll = Some(std::time::Instant::now());
                    for line in new_source_lines(src, &root, &cfg.exec.shell) {
                        let _ = log.send((name.clone(), Ev::Say(format!("source task: {line}"))));
                        drop_task(&root, &line);
                    }
                }
            }
            running.retain(|h| !h.is_finished());
            if !acfg.persistent && running.len() >= acfg.parallel.max(1) {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            let Some(task) = next_task(&inbox) else {
                std::thread::sleep(Duration::from_millis(500));
                continue;
            };
            let _ = log.send((name.clone(), Ev::Say(format!("task: {task}"))));
            // Forward this run's events into the shared log, tagged by name.
            let (tx, rx) = std::sync::mpsc::channel();
            let fwd_log = log.clone();
            let fwd_name = name.clone();
            let fwd = std::thread::spawn(move || {
                for ev in rx {
                    let _ = fwd_log.send((fwd_name.clone(), ev));
                }
            });
            let ctl = Ctl { sink: Some(tx), ..Default::default() };
            if acfg.persistent {
                // One session, one timeline: tasks run inline, sequentially.
                let s = session.get_or_insert_with(|| Session::new(&cfg, root.clone(), 0));
                agent::run_session(Arc::clone(&cfg), s, &task, acfg.profile.as_deref(), 0, ctl);
                let _ = fwd.join();
            } else {
                // One-shot: fresh session per task, up to `parallel` at once.
                let cfg2 = Arc::clone(&cfg);
                let root2 = root.clone();
                let profile = acfg.profile.clone();
                running.push(std::thread::spawn(move || {
                    agent::run(cfg2, root2, &task, profile.as_deref(), 0, ctl);
                    let _ = fwd.join();
                }));
            }
        }
        for h in running {
            let _ = h.join();
        }
    })
}

/// Run the source command; return only lines never seen before (dedup ledger
/// at <root>/.swift/seen, one fnv hash per line — an issue queued once stays
/// queued once, forever).
fn new_source_lines(cmd: &str, root: &Path, shell: &str) -> Vec<String> {
    let out = haste::tools::run_shell(cmd, root, 30_000, shell);
    let Some(body) = out.strip_prefix("ok") else {
        return Vec::new(); // source command failed — silent this round
    };
    let seen_path = root.join(".swift").join("seen");
    let mut seen: std::collections::HashSet<u64> = std::fs::read_to_string(&seen_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let mut fresh = Vec::new();
    let mut appended = String::new();
    for line in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let h = haste::ledger::fnv(line);
        if seen.insert(h) {
            appended.push_str(&format!("{h}\n"));
            fresh.push(line.to_string());
        }
    }
    if !appended.is_empty() {
        let _ = std::fs::create_dir_all(seen_path.parent().unwrap_or(root));
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&seen_path) {
            let _ = f.write_all(appended.as_bytes());
        }
    }
    fresh
}

fn load_agent_config(acfg: &AgentCfg, root: &Path) -> Result<Config, String> {
    match &acfg.config {
        Some(p) => Config::load(Some(p)),
        // No explicit config: prefer <root>/haste.toml, else the normal chain.
        None => {
            let local = root.join("haste.toml");
            if local.is_file() {
                Config::load(local.to_str())
            } else {
                Config::load(None)
            }
        }
    }
}

fn fmt_ev(ev: &Ev) -> Option<String> {
    Some(match ev {
        Ev::Turn(t) => format!("turn {t}"),
        Ev::Action(_, a) => format!("· {a}"),
        Ev::Result(_, r) => format!("  {}", r.lines().next().unwrap_or("")),
        Ev::Say(s) => format!("S {s}"),
        Ev::Sub(n, l) => format!("[{n}] {}", l.lines().next().unwrap_or("")),
        Ev::SubDone(n) => format!("[{n}] done"),
        Ev::Report(r) => r.clone(),
        Ev::Done(m) => format!("DONE: {}", m.lines().next().unwrap_or("")),
        Ev::Delta(_) | Ev::Ctx(_) | Ev::Plan(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inbox_queue_is_ordered_and_consumed() {
        let root = std::env::temp_dir().join(format!("swift-inbox-{}", std::process::id()));
        let inbox = inbox_dir(&root);
        std::fs::create_dir_all(&inbox).unwrap();
        std::fs::write(inbox.join("2-b.task"), "second").unwrap();
        std::fs::write(inbox.join("1-a.task"), "first").unwrap();
        assert_eq!(next_task(&inbox).as_deref(), Some("first"));
        assert_eq!(next_task(&inbox).as_deref(), Some("second"));
        assert_eq!(next_task(&inbox), None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn drop_task_lands_in_inbox() {
        let root = std::env::temp_dir().join(format!("swift-drop-{}", std::process::id()));
        drop_task(&root, "hello fleet");
        assert_eq!(next_task(&inbox_dir(&root)).as_deref(), Some("hello fleet"));
        let _ = std::fs::remove_dir_all(root);
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;

    #[test]
    fn source_lines_dedup_across_polls() {
        let root = std::env::temp_dir().join(format!("swift-src-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let shell = if cfg!(windows) { "powershell" } else { "sh" };
        let cmd = "echo issue-1; echo issue-2";
        let first = new_source_lines(cmd, &root, shell);
        assert_eq!(first, vec!["issue-1", "issue-2"]);
        let second = new_source_lines(cmd, &root, shell);
        assert!(second.is_empty(), "already-seen lines must not requeue: {second:?}");
        let third = new_source_lines("echo issue-1; echo issue-3", &root, shell);
        assert_eq!(third, vec!["issue-3"]);
        let _ = std::fs::remove_dir_all(root);
    }
}
