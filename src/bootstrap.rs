//! What the agent knows before it is asked anything (ported from nestor).
//!
//! Orientation an agent would otherwise burn round trips discovering — or
//! worse, guess at: what kind of project this is, how to build and test it,
//! git state, and the shape of the tree. Pinned at ledger head, so in append
//! mode it seals into the cached prefix: paid for once.

use std::path::Path;
use std::process::Command;

const MAX_CHARS: usize = 3000;
const TREE_DEPTH: usize = 3;

pub fn workspace_state(root: &Path) -> String {
    let mut out = String::from("--- workspace ---\n");
    if cfg!(windows) {
        out.push_str("os: Windows — X runs cmd.exe (dir, type, findstr; no ls/cat/grep)\n");
    } else {
        out.push_str("os: unix — X runs sh\n");
    }
    if let Some(kind) = project_kind(root) {
        out.push_str(&format!("project: {kind}\n"));
    }
    // Only shell out to git when there is actually a repo here: process spawns
    // are the one cost in this module, and temp/scratch dirs shouldn't pay it.
    if root.join(".git").exists() {
        if let Some(g) = git_state(root) {
            out.push_str(&g);
        }
    }
    out.push_str("layout (paths below are RELATIVE to the workspace root):\n");
    let mut budget = MAX_CHARS.saturating_sub(out.len());
    tree(root, root, 0, &mut budget, &mut out);
    out
}

fn project_kind(root: &Path) -> Option<String> {
    let has = |f: &str| root.join(f).exists();
    let kind = if has("Cargo.toml") {
        "Rust (cargo build / cargo test)"
    } else if has("package.json") {
        "Node (npm install / npm test)"
    } else if has("pyproject.toml") || has("requirements.txt") {
        "Python (pip install -r requirements.txt / pytest)"
    } else if has("go.mod") {
        "Go (go build ./... / go test ./...)"
    } else if has("tests.py") {
        "Python (python tests.py)"
    } else {
        return None;
    };
    Some(kind.into())
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).current_dir(root).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn git_state(root: &Path) -> Option<String> {
    let branch = git(root, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    let mut out = format!("git: {branch}");
    match git(root, &["status", "--porcelain"]) {
        None => out.push_str(", clean\n"),
        Some(status) => {
            let names: Vec<&str> = status.lines().take(12).collect();
            out.push_str(&format!(", {} uncommitted: {}\n", status.lines().count(), names.join(" ")));
        }
    }
    if let Some(log) = git(root, &["log", "-3", "--format=%h %s"]) {
        out.push_str("recent: ");
        out.push_str(&log.replace('\n', " | "));
        out.push('\n');
    }
    Some(out)
}

/// Compact map: one line per directory. Small dirs list actual filenames
/// (names are what the model acts on — counts make it go exploring anyway);
/// big dirs fall back to per-extension counts.
fn tree(root: &Path, dir: &Path, depth: usize, budget: &mut usize, out: &mut String) {
    if *budget == 0 || depth > TREE_DEPTH {
        return;
    }
    let mut names = Vec::new();
    let mut subdirs = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if p.is_dir() {
            if !crate::tools::SKIP_DIRS.contains(&name.as_str()) && !name.starts_with('.') {
                subdirs.push(p);
            }
        } else {
            names.push(name);
        }
    }
    let rel = dir.strip_prefix(root).unwrap_or(dir);
    let label = if depth == 0 { "(root)".to_string() } else { format!("{}/", rel.display().to_string().replace('\\', "/")) };
    if !names.is_empty() || depth == 0 {
        names.sort();
        let listing = if names.len() <= 10 {
            names.join(" ")
        } else {
            let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
            for n in &names {
                let ext = n.rsplit_once('.').map(|(_, x)| format!(".{x}")).unwrap_or_else(|| n.clone());
                *counts.entry(ext).or_insert(0) += 1;
            }
            counts.iter().map(|(k, v)| format!("{v} {k}")).collect::<Vec<_>>().join(", ")
        };
        let line = format!("{}{} [{}]\n", "  ".repeat(depth.min(4) + 1), label, listing);
        if line.len() <= *budget {
            *budget -= line.len();
            out.push_str(&line);
        } else {
            *budget = 0;
            return;
        }
    }
    subdirs.sort();
    for sd in subdirs {
        tree(root, &sd, depth + 1, budget, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orients_in_own_repo() {
        let s = workspace_state(Path::new("."));
        assert!(s.contains("project: Rust"), "{s}");
        // src/ has >10 files, so the map falls back to extension counts.
        assert!(s.contains("src/ [") && s.contains(".rs"), "{s}");
        // A small dir still lists real filenames.
        assert!(s.contains("Cargo.toml"), "{s}");
        assert!(s.len() <= MAX_CHARS + 200, "bootstrap too big: {}", s.len());
    }

    #[test]
    fn no_git_spawn_outside_repos() {
        let p = std::env::temp_dir().join(format!("haste-boot-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        std::fs::write(p.join("x.py"), "pass\n").unwrap();
        let t = std::time::Instant::now();
        let s = workspace_state(&p);
        assert!(t.elapsed().as_millis() < 100, "bootstrap slow: {:?}", t.elapsed());
        assert!(!s.contains("git:"), "{s}");
        let _ = std::fs::remove_dir_all(p);
    }
}
