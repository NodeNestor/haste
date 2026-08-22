use crate::config::CtxCfg;
use crate::ledger::{est_tokens, Kind, Ledger};
use std::collections::HashMap;

/// Turns the lossless ledger into the prompt document for this turn.
///
/// working_set mode: everything below is recomputed from scratch every turn —
/// stale reads superseded, duplicates pointered, old results folded, budget
/// enforced by dropping oldest result bodies. Maximum compression, no prefix
/// stability (for providers without a prefix cache).
///
/// append mode: entries render verbatim; once the estimate exceeds budget,
/// reseal() folds everything older than the seal point exactly once and the
/// fold is then frozen (stored as an override) so the byte prefix stays
/// stable for provider-side prefix caches.
pub struct Renderer {
    sealed_upto: usize,
    overrides: HashMap<usize, String>,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            sealed_upto: 0,
            overrides: HashMap::new(),
        }
    }

    pub fn render(&mut self, ledger: &Ledger, cfg: &CtxCfg, turn: u32) -> String {
        if cfg.mode == "append" {
            self.render_append(ledger, cfg)
        } else {
            render_working_set(ledger, cfg, turn)
        }
    }

    fn render_append(&mut self, ledger: &Ledger, cfg: &CtxCfg) -> String {
        let mut total: usize = ledger
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| est_tokens(self.overrides.get(&i).map(String::as_str).unwrap_or(&e.text)))
            .sum();
        if total > cfg.budget_tokens {
            // Reseal: fold results older than the tail (last 25% of budget), once.
            let keep_from = ledger.entries.len().saturating_sub(8).max(self.sealed_upto);
            for (i, e) in ledger.entries.iter().enumerate() {
                if i >= keep_from || i < self.sealed_upto {
                    continue;
                }
                if e.kind == Kind::Result && !self.overrides.contains_key(&i) {
                    let folded = fold(&e.text);
                    total = total - est_tokens(&e.text) + est_tokens(&folded);
                    self.overrides.insert(i, folded);
                }
            }
            self.sealed_upto = keep_from;
            let _ = total;
        }
        let mut out = String::new();
        for (i, e) in ledger.entries.iter().enumerate() {
            let text = self.overrides.get(&i).map(String::as_str).unwrap_or(&e.text);
            push_entry(&mut out, e.kind, text);
        }
        out
    }
}

fn render_working_set(ledger: &Ledger, cfg: &CtxCfg, turn: u32) -> String {
    let n = ledger.entries.len();
    // Latest read per file wins; earlier reads of the same file are superseded.
    let mut last_read: HashMap<u32, usize> = HashMap::new();
    // First occurrence per content hash; later identical results are pointered.
    let mut first_hash: HashMap<u64, usize> = HashMap::new();
    for (i, e) in ledger.entries.iter().enumerate() {
        if e.kind == Kind::Result {
            if let Some(fid) = e.file {
                last_read.insert(fid, i);
            }
            first_hash.entry(e.hash).or_insert(i);
        }
    }

    // Pass 1: decide each entry's rendered form.
    let mut texts: Vec<Option<String>> = Vec::with_capacity(n);
    for (i, e) in ledger.entries.iter().enumerate() {
        let t = match e.kind {
            Kind::Result => {
                if let Some(fid) = e.file {
                    if last_read[&fid] != i {
                        Some(format!("(superseded by newer read of #{fid})"))
                    } else {
                        None
                    }
                } else if first_hash[&e.hash] != i {
                    Some("(= identical to an earlier result)".into())
                } else if e.turn + cfg.fold_after_turns < turn {
                    Some(fold(&e.text))
                } else {
                    None
                }
            }
            _ => None,
        };
        texts.push(t);
    }

    // Pass 2: budget — drop oldest full result bodies until under budget.
    let mut total: usize = ledger
        .entries
        .iter()
        .zip(&texts)
        .map(|(e, t)| est_tokens(t.as_deref().unwrap_or(&e.text)))
        .sum();
    if total > cfg.budget_tokens {
        for (i, e) in ledger.entries.iter().enumerate() {
            if total <= cfg.budget_tokens {
                break;
            }
            if e.kind == Kind::Result && texts[i].is_none() {
                let dropped = fold(&e.text);
                total = total - est_tokens(&e.text) + est_tokens(&dropped);
                texts[i] = Some(dropped);
            }
        }
    }

    let mut out = String::new();
    for (i, e) in ledger.entries.iter().enumerate() {
        push_entry(&mut out, e.kind, texts[i].as_deref().unwrap_or(&e.text));
    }
    out
}

fn push_entry(out: &mut String, kind: Kind, text: &str) {
    match kind {
        Kind::Task => {
            out.push_str("## TASK\n");
            out.push_str(text);
            out.push_str("\n\n## LOG\n");
        }
        Kind::Action => {
            out.push_str("> ");
            out.push_str(text);
            out.push('\n');
        }
        Kind::Result => {
            out.push_str(text);
            out.push_str("\n");
        }
        Kind::Pin => {
            out.push_str(text);
            out.push('\n');
        }
        Kind::Final => {
            out.push_str("DONE: ");
            out.push_str(text);
            out.push('\n');
        }
    }
}

/// Collapse a result to its first line plus a size note.
fn fold(text: &str) -> String {
    let first = text.lines().next().unwrap_or("");
    let lines = text.lines().count();
    if lines <= 1 {
        first.to_string()
    } else {
        format!("{first} …({} lines folded)", lines - 1)
    }
}
