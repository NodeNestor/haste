use regex::Regex;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

const GREP_MAX_HITS: usize = 50;
pub const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".haste", "__pycache__", "dist", ".venv"];

/// The workspace: root dir + the file intern table. Paths are token sinks, so
/// after first contact every file is a small integer (#3) in both directions.
pub struct Workspace {
    pub root: PathBuf,
    files: Vec<PathBuf>,
}

impl Workspace {
    pub fn new(root: PathBuf) -> Workspace {
        Workspace { root, files: Vec::new() }
    }

    pub fn intern(&mut self, rel: &Path) -> u32 {
        if let Some(i) = self.files.iter().position(|p| p == rel) {
            return i as u32;
        }
        self.files.push(rel.to_path_buf());
        (self.files.len() - 1) as u32
    }

    /// Target is either an intern id ("3" or "#3") or a path (interned on first use).
    pub fn resolve(&mut self, target: &str) -> Result<(u32, PathBuf), String> {
        let target = target.strip_prefix('#').unwrap_or(target);
        if let Ok(id) = target.parse::<u32>() {
            let rel = self
                .files
                .get(id as usize)
                .cloned()
                .ok_or(format!("no file #{id}"))?;
            return Ok((id, self.root.join(rel)));
        }
        let rel = PathBuf::from(target.replace('\\', "/"));
        let abs = self.root.join(&rel);
        if !abs.is_file() {
            return Err(format!("not a file: {target}"));
        }
        Ok((self.intern(&rel), abs))
    }

    pub fn read(&mut self, target: &str, range: Option<(usize, usize)>) -> Result<String, String> {
        let (id, abs) = self.resolve(target)?;
        let text = std::fs::read_to_string(&abs).map_err(|e| format!("read {target}: {e}"))?;
        let lines: Vec<&str> = text.lines().collect();
        let (a, b) = match range {
            Some((a, b)) => (a.max(1), b.min(lines.len())),
            None => (1, lines.len()),
        };
        if a > lines.len() {
            return Err(format!("#{id} has only {} lines", lines.len()));
        }
        let mut out = format!("#{id} {} {}:{} of {}\n", self.files[id as usize].display(), a, b, lines.len());
        for (i, l) in lines.iter().enumerate().take(b).skip(a - 1) {
            out.push_str(&format!("{}:{}\n", i + 1, l));
        }
        out.pop();
        Ok(out)
    }

    /// Replace lines a..=b (1-indexed, inclusive) with body. Returns the fresh
    /// numbering of the touched region so the model never works from stale lines.
    pub fn edit(&mut self, target: &str, a: usize, b: usize, body: &str) -> Result<String, String> {
        let (id, abs) = self.resolve(target)?;
        let text = std::fs::read_to_string(&abs).map_err(|e| e.to_string())?;
        let mut lines: Vec<String> = text.lines().map(String::from).collect();
        if a < 1 || b < a || b > lines.len() {
            return Err(format!("bad range {a}:{b} (#{id} has {} lines)", lines.len()));
        }
        let new: Vec<String> = if body.is_empty() { Vec::new() } else { body.lines().map(String::from).collect() };
        let nnew = new.len();
        lines.splice(a - 1..b, new);
        write_lines(&abs, &lines, text.ends_with('\n'))?;
        Ok(region_report(id, &lines, a, nnew))
    }

    /// Insert body after line `after` (0 = top of file).
    pub fn insert(&mut self, target: &str, after: usize, body: &str) -> Result<String, String> {
        let (id, abs) = self.resolve(target)?;
        let text = std::fs::read_to_string(&abs).map_err(|e| e.to_string())?;
        let mut lines: Vec<String> = text.lines().map(String::from).collect();
        if after > lines.len() {
            return Err(format!("insert point {after} past end ({} lines)", lines.len()));
        }
        let new: Vec<String> = body.lines().map(String::from).collect();
        let nnew = new.len();
        lines.splice(after..after, new);
        write_lines(&abs, &lines, text.ends_with('\n'))?;
        Ok(region_report(id, &lines, after + 1, nnew))
    }

    pub fn new_file(&mut self, path: &str, body: &str) -> Result<String, String> {
        let rel = PathBuf::from(path.replace('\\', "/"));
        let abs = self.root.join(&rel);
        if let Some(dir) = abs.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(&abs, format!("{body}\n")).map_err(|e| e.to_string())?;
        let id = self.intern(&rel);
        Ok(format!("#{id} {} created ({} lines)", rel.display(), body.lines().count()))
    }

    pub fn grep(&mut self, pat: &str, target: Option<&str>) -> Result<String, String> {
        let re = Regex::new(pat).map_err(|e| format!("bad regex: {e}"))?;
        let base = match target {
            Some(t) => {
                if let Ok((_, abs)) = self.resolve(t) {
                    abs
                } else {
                    let p = self.root.join(t.replace('\\', "/"));
                    if p.exists() { p } else { return Err(format!("no such target: {t}")); }
                }
            }
            None => self.root.clone(),
        };
        let mut hits = Vec::new();
        let mut stack = vec![base];
        while let Some(p) = stack.pop() {
            if hits.len() >= GREP_MAX_HITS {
                break;
            }
            if p.is_dir() {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if SKIP_DIRS.contains(&name) {
                    continue;
                }
                if let Ok(rd) = std::fs::read_dir(&p) {
                    for e in rd.flatten() {
                        stack.push(e.path());
                    }
                }
            } else if let Ok(text) = std::fs::read_to_string(&p) {
                let mut id: Option<u32> = None;
                for (i, line) in text.lines().enumerate() {
                    if re.is_match(line) {
                        let id = *id.get_or_insert_with(|| {
                            let rel = p.strip_prefix(&self.root).unwrap_or(&p).to_path_buf();
                            self.intern(&rel)
                        });
                        hits.push(format!("#{id}:{}:{}", i + 1, line.trim()));
                        if hits.len() >= GREP_MAX_HITS {
                            break;
                        }
                    }
                }
            }
        }
        if hits.is_empty() {
            return Ok(format!("no hits for /{pat}/"));
        }
        let mut out = self.legend_delta();
        out.push_str(&hits.join("\n"));
        Ok(out)
    }

    /// Map of intern ids -> paths, for the prompt header.
    pub fn legend(&self) -> String {
        if self.files.is_empty() {
            return String::new();
        }
        let mut s = String::from("files: ");
        for (i, p) in self.files.iter().enumerate() {
            s.push_str(&format!("#{i}={} ", p.display()));
        }
        s.push('\n');
        s
    }

    fn legend_delta(&self) -> String {
        String::new() // ids appear inline as #id:line:text; full map lives in the header
    }
}

fn write_lines(abs: &Path, lines: &[String], trailing_nl: bool) -> Result<(), String> {
    let mut text = lines.join("\n");
    if trailing_nl {
        text.push('\n');
    }
    std::fs::write(abs, text).map_err(|e| e.to_string())
}

/// "ok #2 now 120 lines" + the touched region freshly numbered (±2 context).
fn region_report(id: u32, lines: &[String], start: usize, nnew: usize) -> String {
    let total = lines.len();
    let lo = start.saturating_sub(3).max(1);
    let hi = (start + nnew + 1).min(total);
    let mut out = format!("ok #{id} now {total} lines");
    for i in lo..=hi {
        if i >= 1 && i <= total {
            out.push_str(&format!("\n{}:{}", i, lines[i - 1]));
        }
    }
    out
}

/// Run a shell line, kill on timeout, merge stdout+stderr, report exit code.
pub fn run_shell(line: &str, cwd: &Path, timeout_ms: u64) -> String {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", line]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", line]);
        c
    };
    cmd.current_dir(cwd).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return format!("spawn failed: {e}"),
    };
    let child = Arc::new(Mutex::new(child));
    let watchdog = {
        let child = Arc::clone(&child);
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let h = std::thread::spawn(move || {
            if rx.recv_timeout(std::time::Duration::from_millis(timeout_ms)).is_err() {
                let _ = child.lock().unwrap().kill();
            }
        });
        (h, tx)
    };
    let (mut out_pipe, mut err_pipe) = {
        let mut c = child.lock().unwrap();
        (c.stdout.take().unwrap(), c.stderr.take().unwrap())
    };
    let err_h = std::thread::spawn(move || {
        let mut s = String::new();
        use std::io::Read;
        let _ = err_pipe.read_to_string(&mut s);
        s
    });
    let mut out = String::new();
    {
        use std::io::Read;
        let _ = out_pipe.read_to_string(&mut out);
    }
    let err = err_h.join().unwrap_or_default();
    let status = child.lock().unwrap().wait();
    let _ = watchdog.1.send(());
    let _ = watchdog.0.join();
    let code = status.ok().and_then(|s| s.code()).unwrap_or(-1);
    let mut text = out;
    if !err.is_empty() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&err);
    }
    let text = text.trim_end();
    if code == 0 {
        if text.is_empty() { "ok".into() } else { format!("ok\n{text}") }
    } else {
        format!("exit {code}\n{text}")
    }
}

/// Structural pruners. Spec: name[:args], chained with '|'.
pub fn prune(spec: &str, text: &str) -> String {
    let mut cur = text.to_string();
    for stage in spec.split('|') {
        let stage = stage.trim();
        let (name, args) = stage.split_once(':').unwrap_or((stage, ""));
        cur = match name {
            "none" | "" => cur,
            "head_tail" => {
                let (a, b) = args.split_once(',').unwrap_or(("20", "10"));
                head_tail(&cur, a.trim().parse().unwrap_or(20), b.trim().parse().unwrap_or(10))
            }
            "first_failure" => first_failure(&cur),
            "errors_only" => keep_matching(&cur, r"(?i)\b(error|panic|failed|exception|traceback)\b"),
            "keep" => keep_matching(&cur, args),
            "drop" => drop_matching(&cur, args),
            _ => cur, // unknown stage (incl. "distill", handled upstream): pass through
        };
    }
    cur
}

fn head_tail(text: &str, head: usize, tail: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= head + tail + 1 {
        return text.to_string();
    }
    let mut out: Vec<&str> = lines[..head].to_vec();
    let elided = format!("…({} lines elided)…", lines.len() - head - tail);
    let mut s = out.join("\n");
    s.push('\n');
    s.push_str(&elided);
    out = lines[lines.len() - tail..].to_vec();
    s.push('\n');
    s.push_str(&out.join("\n"));
    s
}

/// First line matching a failure signature, with a window of context after it.
fn first_failure(text: &str) -> String {
    let re = Regex::new(r"(?i)\b(FAILED|error(\[|:)|panicked|assert|Traceback|Exception)\b").unwrap();
    let lines: Vec<&str> = text.lines().collect();
    match lines.iter().position(|l| re.is_match(l)) {
        Some(i) => {
            let hi = (i + 25).min(lines.len());
            let mut s = lines[i..hi].join("\n");
            if hi < lines.len() {
                s.push_str(&format!("\n…({} more lines)", lines.len() - hi));
            }
            s
        }
        None => head_tail(text, 5, 5),
    }
}

fn keep_matching(text: &str, pat: &str) -> String {
    match Regex::new(pat) {
        Ok(re) => {
            let kept: Vec<&str> = text.lines().filter(|l| re.is_match(l)).collect();
            if kept.is_empty() { "(no matching lines)".into() } else { kept.join("\n") }
        }
        Err(_) => text.to_string(),
    }
}

fn drop_matching(text: &str, pat: &str) -> String {
    match Regex::new(pat) {
        Ok(re) => text.lines().filter(|l| !re.is_match(l)).collect::<Vec<_>>().join("\n"),
        Err(_) => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> (Workspace, tempdir::TempDir) {
        let td = tempdir::TempDir::new();
        std::fs::write(td.path().join("a.txt"), "one\ntwo\nthree\nfour\nfive\n").unwrap();
        (Workspace::new(td.path().to_path_buf()), td)
    }

    // minimal tempdir without a dependency
    mod tempdir {
        pub struct TempDir(std::path::PathBuf);
        impl TempDir {
            pub fn new() -> TempDir {
                let p = std::env::temp_dir().join(format!("haste-test-{}-{:x}", std::process::id(), crate::ledger::fnv(&format!("{:?}", std::time::Instant::now()))));
                std::fs::create_dir_all(&p).unwrap();
                TempDir(p)
            }
            pub fn path(&self) -> &std::path::Path { &self.0 }
        }
        impl Drop for TempDir {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }
    }

    #[test]
    fn read_edit_renumber() {
        let (mut w, _td) = ws();
        let r = w.read("a.txt", Some((2, 4))).unwrap();
        assert!(r.starts_with("#0 a.txt 2:4 of 5"));
        assert!(r.contains("2:two"));
        let rep = w.edit("0", 2, 3, "TWO\nTWO-B\nTWO-C").unwrap();
        assert!(rep.starts_with("ok #0 now 6 lines"), "{rep}");
        let r2 = w.read("0", None).unwrap();
        assert!(r2.contains("2:TWO") && r2.contains("4:TWO-C") && r2.contains("5:four"));
    }

    #[test]
    fn insert_and_delete() {
        let (mut w, _td) = ws();
        w.read("a.txt", None).unwrap();
        w.insert("0", 0, "ZERO").unwrap();
        let r = w.read("0", Some((1, 2))).unwrap();
        assert!(r.contains("1:ZERO") && r.contains("2:one"));
        // empty body edit = delete lines
        w.edit("0", 1, 1, "").unwrap();
        let r = w.read("0", Some((1, 1))).unwrap();
        assert!(r.contains("1:one"));
    }

    #[test]
    fn resolve_accepts_hash_ids() {
        let (mut w, _td) = ws();
        w.read("a.txt", None).unwrap();
        assert!(w.read("#0", Some((1, 1))).unwrap().contains("1:one"));
        assert!(w.edit("#0", 1, 1, "ONE").is_ok());
    }

    #[test]
    fn grep_interns() {
        let (mut w, _td) = ws();
        let hits = w.grep("thr", None).unwrap();
        assert!(hits.contains("#0:3:three"), "{hits}");
        assert!(w.legend().contains("#0=a.txt"));
    }

    #[test]
    fn pruners() {
        let noisy: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        let ht = prune("head_tail:3,2", &noisy);
        assert!(ht.contains("(95 lines elided)"));
        let failing = "running 30 tests\nok ok ok\nerror[E0308]: mismatched types\n --> src/x.rs:9\nnote: blah\n";
        let ff = prune("first_failure", failing);
        assert!(ff.starts_with("error[E0308]"));
        let chain = prune("drop:^note|first_failure", failing);
        assert!(!chain.contains("note:"));
    }

    #[test]
    fn shell_timeout_and_exit() {
        let cwd = std::env::temp_dir();
        let r = run_shell(if cfg!(windows) { "exit 3" } else { "exit 3" }, &cwd, 5000);
        assert!(r.starts_with("exit 3"), "{r}");
        let ok = run_shell("echo hi", &cwd, 5000);
        assert!(ok.contains("hi"));
    }
}
