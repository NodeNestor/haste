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
/// Compaction re-arms only after this many new entries since the last seal —
/// otherwise a session whose RECENT entries alone bust the budget would
/// compact every turn without ever getting smaller.
const MIN_ENTRIES_BETWEEN_SEALS: usize = 8;

pub struct Renderer {
    sealed_upto: usize,
    overrides: HashMap<usize, String>,
    last_seal_len: usize,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            sealed_upto: 0,
            overrides: HashMap::new(),
            last_seal_len: 0,
        }
    }

    /// Would a seal at `threshold` tokens pay off right now? (Append mode,
    /// with hysteresis — see MIN_ENTRIES_BETWEEN_SEALS.) `real` is the
    /// provider-reported prompt size of the last request — the ACTUAL billed
    /// context — and always wins over the estimate when available.
    pub fn seal_due(&self, ledger: &Ledger, cfg: &CtxCfg, threshold: usize, real: Option<usize>) -> bool {
        if cfg.mode != "append" {
            return false;
        }
        if ledger.entries.len() < self.last_seal_len + MIN_ENTRIES_BETWEEN_SEALS {
            return false;
        }
        let total = real.unwrap_or_else(|| {
            ledger
                .entries
                .iter()
                .enumerate()
                .map(|(i, e)| est_tokens(self.overrides.get(&i).map(String::as_str).unwrap_or(&e.text)))
                .sum()
        });
        total > threshold
    }

    /// Seal history behind a model-written summary. Render-layer only — the
    /// ledger is never touched (compression is a rendering decision). Pins,
    /// the most recent Task entry, and the last `keep_last` entries survive.
    pub fn seal_summary(&mut self, ledger: &Ledger, keep_last: usize, summary: String) {
        let n = ledger.entries.len();
        let keep_from = n.saturating_sub(keep_last).max(self.sealed_upto);
        let last_task = ledger
            .entries
            .iter()
            .rposition(|e| e.kind == Kind::Task)
            .unwrap_or(usize::MAX);
        // The new briefing was written while SEEING earlier summaries, so it
        // subsumes them: sweep from 0, replacing prior summary blocks instead
        // of letting them accumulate.
        let mut placed = false;
        for i in 0..keep_from {
            match ledger.entries[i].kind {
                Kind::Pin => continue,
                Kind::Task if i == last_task => continue,
                _ => {
                    if placed {
                        self.overrides.insert(i, String::new());
                    } else {
                        self.overrides.insert(i, format!("[history compressed]\n{summary}"));
                        placed = true;
                    }
                }
            }
        }
        self.sealed_upto = keep_from;
        self.last_seal_len = n;
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
            // Structural reseal (model compaction failed or is off): fold old
            // results, keeping the same tail the model seal would keep.
            let keep_from = ledger.entries.len().saturating_sub(cfg.compact_keep_last).max(self.sealed_upto);
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
            if text.is_empty() && self.overrides.contains_key(&i) {
                continue; // entry sealed behind the summary
            }
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
            out.push('\n');
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
