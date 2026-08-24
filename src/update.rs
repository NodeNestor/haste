//! Self-update against GitHub releases. Deliberately light: the TUI spawns
//! one background CHECK (a version string comparison, never a download) and
//! prints one line if something newer exists; `haste update` does the actual
//! swap, using the running-exe rename trick so it works on Windows too.
//! Everything fails silent/soft — no network must never break the tool.

use std::sync::mpsc::{channel, Receiver};

const REPO: &str = "NodeNestor/haste";
const CUR: &str = env!("CARGO_PKG_VERSION");

fn api_latest() -> Result<serde_json::Value, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(6))
        .build();
    let resp = agent
        .get(&format!("https://api.github.com/repos/{REPO}/releases/latest"))
        .set("User-Agent", "haste-updater")
        .call()
        .map_err(|e| format!("release lookup: {e}"))?;
    serde_json::from_reader(resp.into_reader()).map_err(|e| format!("release json: {e}"))
}

/// "v0.2.1" -> (0,2,1); tolerant of missing parts.
fn triple(v: &str) -> (u32, u32, u32) {
    let mut it = v.trim_start_matches('v').split('.').map(|p| {
        p.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0)
    });
    (it.next().unwrap_or(0), it.next().unwrap_or(0), it.next().unwrap_or(0))
}

fn asset_name() -> &'static str {
    if cfg!(windows) {
        "haste-windows-x86_64.exe"
    } else if cfg!(target_os = "macos") {
        "haste-macos-aarch64"
    } else {
        "haste-linux-x86_64"
    }
}

/// Background version check for the TUI: sends at most ONE line, only when a
/// newer release exists. Every failure mode is silence.
pub fn spawn_check() -> Receiver<String> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let Ok(v) = api_latest() else { return };
        let tag = v["tag_name"].as_str().unwrap_or("");
        if !tag.is_empty() && triple(tag) > triple(CUR) {
            let _ = tx.send(format!("update: {tag} available (you have v{CUR}) — run `haste update`"));
        }
    });
    rx
}

/// Replace the running executable with the latest release build.
pub fn self_update() -> Result<String, String> {
    let v = api_latest()?;
    let tag = v["tag_name"].as_str().unwrap_or("").to_string();
    if tag.is_empty() {
        return Err("no releases published yet".into());
    }
    if triple(&tag) <= triple(CUR) {
        return Ok(format!("already up to date (v{CUR})"));
    }
    let want = asset_name();
    let url = v["assets"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|a| a["name"].as_str() == Some(want))
        .and_then(|a| a["browser_download_url"].as_str())
        .ok_or_else(|| format!("release {tag} has no asset '{want}'"))?
        .to_string();
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .redirects(8)
        .build();
    let resp = agent.get(&url).set("User-Agent", "haste-updater").call().map_err(|e| format!("download: {e}"))?;
    let mut bytes = Vec::new();
    use std::io::Read;
    resp.into_reader()
        .take(100 * 1024 * 1024)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("download read: {e}"))?;
    if bytes.len() < 100_000 {
        return Err(format!("downloaded asset suspiciously small ({} bytes) — aborting", bytes.len()));
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let fresh = exe.with_extension("new");
    let old = exe.with_extension("old");
    std::fs::write(&fresh, &bytes).map_err(|e| format!("write: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&fresh, std::fs::Permissions::from_mode(0o755));
    }
    // The rename trick: a RUNNING exe can be renamed (even on Windows), just
    // not overwritten — so slide it aside and move the fresh one in.
    let _ = std::fs::remove_file(&old); // leftover from a previous update
    std::fs::rename(&exe, &old).map_err(|e| format!("rename current: {e}"))?;
    if let Err(e) = std::fs::rename(&fresh, &exe) {
        let _ = std::fs::rename(&old, &exe); // roll back
        return Err(format!("install: {e}"));
    }
    let _ = std::fs::remove_file(&old); // fails while running on Windows; next update cleans it
    Ok(format!("updated v{CUR} -> {tag} — restart haste"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_triples_compare() {
        assert!(triple("v0.2.0") > triple("0.1.9"));
        assert!(triple("1.0.0") > triple("v0.99.99"));
        assert_eq!(triple("v1.2.3-rc1"), (1, 2, 3));
        assert!(triple(env!("CARGO_PKG_VERSION")) >= (0, 1, 0));
    }
}
