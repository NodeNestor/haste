use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;

#[derive(Clone, Copy, PartialEq, Serialize, Debug)]
pub enum Kind {
    Task,
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
        self.entries.push(e);
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

/// Cheap token estimate used for all budget decisions.
pub fn est_tokens(s: &str) -> usize {
    s.len() / 4 + 1
}
