use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;

#[derive(Clone, Copy, PartialEq, Serialize, Debug)]
pub enum Kind {
    Task,
    /// Pinned context (workspace orientation): rendered verbatim, never folded,
    /// never dropped by any budget pass.
    Pin,
    Action,
    Result,
    Final,
}

#[derive(Clone, Serialize)]
pub struct Entry {
    pub kind: Kind,
    pub turn: u32,
    pub text: String,
    pub hash: u64,
    /// For read results: interned file id, so the renderer can supersede stale reads.
    pub file: Option<u32>,
}

/// Append-only and lossless: entries are never edited. All compression is a
/// rendering decision (render.rs) recomputed against the intact ledger.
pub struct Ledger {
    pub entries: Vec<Entry>,
    tee: Option<File>,
    /// First occurrence per result hash, for append-time dedup pointers.
    seen: std::collections::HashMap<u64, usize>,
}

impl Ledger {
    pub fn new(tee_path: Option<&std::path::Path>) -> Ledger {
        let tee = tee_path.and_then(|p| {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            OpenOptions::new().create(true).append(true).open(p).ok()
        });
        Ledger {
            entries: Vec::new(),
            tee,
            seen: std::collections::HashMap::new(),
        }
    }

    pub fn push(&mut self, kind: Kind, turn: u32, text: String, file: Option<u32>) {
        let e = Entry {
            kind,
            turn,
            hash: fnv(&text),
            text,
            file,
        };
        if let Some(f) = &mut self.tee {
            if let Ok(line) = serde_json::to_string(&e) {
                let _ = writeln!(f, "{line}");
            }
        }
        if e.kind == Kind::Result {
            self.seen.entry(e.hash).or_insert(self.entries.len());
        }
        self.entries.push(e);
    }

    /// Append-time dedup: a tool result byte-identical to an earlier one is
    /// worth a pointer, not a re-send — same trick as render-time dedup, but
    /// applied at WRITE time so append mode (where rendered bytes are frozen
    /// for the prefix cache) gets it too. Returns the pointer text, naming
    /// the action whose result it duplicates so the model can find it above.
    pub fn dup_of(&self, text: &str) -> Option<String> {
        let &i = self.seen.get(&fnv(text))?;
        if self.entries[i].text != text {
            return None; // hash collision: keep the real body
        }
        let what = if i > 0 && self.entries[i - 1].kind == Kind::Action {
            format!(" of `{}`", crate::tools::clip(&self.entries[i - 1].text, 60))
        } else {
            String::new()
        };
        Some(format!("(= identical to the earlier result{what} above — content unchanged)"))
    }
}

pub fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Chars-per-token, in thousandths, continuously calibrated from the
/// provider's REAL usage reports (see calibrate). 4.0 until the first report.
static CPT_MILLI: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4000);

/// Token estimate used where a pre-send number is unavoidable (per-entry
/// budget math). Real API counts are always preferred where they exist —
/// this only fills the gap, and calibrate() keeps it honest.
pub fn est_tokens(s: &str) -> usize {
    s.len() * 1000 / CPT_MILLI.load(std::sync::atomic::Ordering::Relaxed) + 1
}

/// Fold one real (chars sent, prompt_tokens billed) measurement into the
/// estimator — an EMA, clamped to sane text ratios so a weird request
/// (huge image payload, provider quirk) cannot wreck it.
pub fn calibrate(chars: usize, tokens: u64) {
    if chars == 0 || tokens == 0 {
        return;
    }
    let measured = (chars * 1000 / tokens as usize).clamp(2000, 6000);
    let old = CPT_MILLI.load(std::sync::atomic::Ordering::Relaxed);
    CPT_MILLI.store((old * 7 + measured) / 8, std::sync::atomic::Ordering::Relaxed);
}
