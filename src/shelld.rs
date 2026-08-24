//! Persistent PowerShell daemon: powershell.exe takes 200-600ms to start on
//! Windows, and every X / config-tool call paid it. One shell process per
//! workspace survives across calls — commands go in as base64 on stdin,
//! output comes back terminated by a sentinel line carrying the exit code.
//! Anything suspicious (timeout, EOF, write failure) kills the daemon and the
//! caller falls back to the one-shot spawn — never wedged, only slower.
//! Concurrent callers (subagents) each check a daemon OUT of the pool, so
//! parallel tool calls never serialize on one shell.

use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

const SENTINEL: &str = "<<HASTE:DONE";
/// Daemons kept per workspace; a burst of subagents beyond this pays one-shot.
const POOL_MAX: usize = 4;

/// The in-shell driver loop. The command runs as `& {scriptblock} 2>&1` —
/// NOT Invoke-Expression: in PS 5.1 `iex $cmd 2>&1` does NOT merge errors
/// raised inside the expression (they bypass to the host's stderr, which is
/// nulled here), silently eating every cmdlet error. The child scope also
/// mirrors one-shot semantics: cd/$env: never persist between calls.
/// Exit code: $LASTEXITCODE for natives (cargo, python, git); any error
/// record -> 1, matching `powershell -Command`'s behavior closely enough.
const DRIVER: &str = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; \
    while($true){ $l=[Console]::In.ReadLine(); if($null -eq $l){break}; \
    $global:LASTEXITCODE=0; $c=0; $Error.Clear(); \
    try{ $b=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($l)); \
    & ([ScriptBlock]::Create($b)) 2>&1 | Out-String -Stream | Write-Output; \
    if($LASTEXITCODE){$c=$LASTEXITCODE} elseif($Error.Count){$c=1} } \
    catch { $_ | Out-String | Write-Output; $c=1 }; \
    Write-Output ('<<HASTE:DONE '+$c+'>>') }";

struct Daemon {
    key: String,
    child: Arc<Mutex<Child>>,
    stdin: ChildStdin,
    out: std::io::BufReader<ChildStdout>,
}

enum Fail {
    /// Process died on its own (stale daemon, crash): worth one retry.
    Dead,
    /// Our watchdog killed it: the command itself timed out — do NOT retry.
    Timeout(String),
}

fn pool() -> &'static Mutex<Vec<Daemon>> {
    static P: OnceLock<Mutex<Vec<Daemon>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(Vec::new()))
}

fn spawn(key: &str, cwd: &Path) -> Option<Daemon> {
    let mut child = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", DRIVER])
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdin = child.stdin.take()?;
    let out = std::io::BufReader::new(child.stdout.take()?);
    Some(Daemon { key: key.into(), child: Arc::new(Mutex::new(child)), stdin, out })
}

/// Run one command through a pooled daemon. None = powershell unavailable or
/// the daemon path broke — the caller falls back to a one-shot spawn.
pub fn run(line: &str, cwd: &Path, timeout_ms: u64) -> Option<(String, i32)> {
    let key = cwd.to_string_lossy().to_ascii_lowercase();
    let pooled = {
        let mut p = pool().lock().unwrap();
        p.iter().position(|d| d.key == key).map(|i| p.remove(i))
    };
    // A pooled daemon can have died while idle: one retry on a fresh process.
    // A FRESH daemon dying means powershell itself is broken here — give up.
    let mut retries = u32::from(pooled.is_some());
    let mut d = pooled.or_else(|| spawn(&key, cwd))?;
    loop {
        match exec(&mut d, line, timeout_ms) {
            Ok(res) => {
                let mut p = pool().lock().unwrap();
                if p.iter().filter(|x| x.key == key).count() < POOL_MAX {
                    p.push(d);
                } else {
                    let _ = d.child.lock().unwrap().kill();
                }
                return Some(res);
            }
            Err(Fail::Timeout(partial)) => {
                let _ = d.child.lock().unwrap().kill();
                let sep = if partial.is_empty() { "" } else { "\n" };
                return Some((format!("{partial}{sep}(killed after {timeout_ms}ms)"), -1));
            }
            Err(Fail::Dead) => {
                let _ = d.child.lock().unwrap().kill();
                if retries == 0 {
                    return None;
                }
                retries -= 1;
                d = spawn(&key, cwd)?;
            }
        }
    }
}

fn exec(d: &mut Daemon, line: &str, timeout_ms: u64) -> Result<(String, i32), Fail> {
    use std::io::{BufRead, Write};
    if d.stdin
        .write_all(b64(line.as_bytes()).as_bytes())
        .and_then(|_| d.stdin.write_all(b"\n"))
        .and_then(|_| d.stdin.flush())
        .is_err()
    {
        return Err(Fail::Dead);
    }
    let (wd, disarm, fired) = crate::tools::watchdog(Arc::clone(&d.child), timeout_ms);
    let mut out = String::new();
    let mut result: Option<i32> = None;
    let mut buf = String::new();
    loop {
        buf.clear();
        match d.out.read_line(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let t = buf.trim_end();
                if let Some(rest) = t.strip_prefix(SENTINEL) {
                    result = Some(rest.trim().trim_end_matches(">>").trim().parse().unwrap_or(-1));
                    break;
                }
                out.push_str(t);
                out.push('\n');
            }
        }
    }
    let _ = disarm.send(());
    let _ = wd.join();
    match result {
        Some(code) => Ok((out, code)),
        None if fired.load(Ordering::SeqCst) => Err(Fail::Timeout(out)),
        None => Err(Fail::Dead),
    }
}

/// Standard base64 (no dependency for 15 lines).
fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64_matches_reference() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn daemon_reuses_process_and_reports_exit_codes() {
        if !cfg!(windows) {
            return;
        }
        let cwd = std::env::temp_dir();
        let Some((out, code)) = run("Write-Output hello", &cwd, 20_000) else {
            return; // no powershell — the one-shot path covers this machine
        };
        assert_eq!(code, 0, "{out}");
        assert!(out.contains("hello"), "{out}");
        // Native exit codes propagate.
        let (_, code2) = run("cmd /c exit 3", &cwd, 20_000).unwrap();
        assert_eq!(code2, 3);
        // Reuse: a warm call must be far under powershell's own startup time.
        let t = std::time::Instant::now();
        let (out3, _) = run("Write-Output again", &cwd, 20_000).unwrap();
        assert!(out3.contains("again"), "{out3}");
        assert!(t.elapsed().as_millis() < 400, "no reuse: {:?}", t.elapsed());
        // Timeout kills the daemon and reports, instead of hanging.
        let t2 = std::time::Instant::now();
        let (out4, code4) = run("Start-Sleep -Seconds 30", &cwd, 600).unwrap();
        assert_eq!(code4, -1, "{out4}");
        assert!(t2.elapsed().as_secs() < 10, "timeout not enforced: {:?}", t2.elapsed());
    }
}
