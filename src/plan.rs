//! The plan file: a JSON state machine the model writes and rewrites with its
//! ordinary N/E verbs, and the harness enforces.
//!
//! Lineage: build-test-loop (stop-hook that refuses to quit until stages
//! pass) -> nestor flow.rs/plan.rs (until-conditions, dependency graph) ->
//! this: the minimal fusion. The file IS the interface:
//!
//!   { "goal": "auth", "steps": [
//!       {"id": "schema", "what": "user table", "status": "todo",
//!        "needs": [], "verify": "python tests.py -k schema"} ] }
//!
//! Enforcement lives in the agent loop: a step freshly marked "done" gets its
//! verify command run — fail reverts it to "doing" with the failure injected;
//! D is refused while steps are open (edit the file to descope: status "skip").

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize, Serialize, Clone, Default)]
pub struct Plan {
    #[serde(default)]
    pub goal: String,
    #[serde(default)]
    pub steps: Vec<Step>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Step {
    pub id: String,
    #[serde(default)]
    pub what: String,
    #[serde(default = "d_todo")]
    pub status: String, // todo | doing | done | skip
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub needs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify: Option<String>,
}
fn d_todo() -> String {
    "todo".into()
}

/// Strip trailing commas before `}` / `]` — the single most common JSON
/// mistake weak models make; refusing the whole plan over one is cruel.
/// String-aware, so a comma inside a "what" text is never touched.
fn forgive_commas(t: &str) -> String {
    let mut out = String::with_capacity(t.len());
    let (mut in_str, mut esc) = (false, false);
    for c in t.chars() {
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                out.push(c);
            }
            '}' | ']' => {
                while out.ends_with(|x: char| x.is_whitespace()) {
                    out.pop();
                }
                if out.ends_with(',') {
                    out.pop();
                }
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

impl Plan {
    /// None = no plan file; Some(Err) = exists but broken (which blocks D
    /// until fixed — a corrupt state machine must not be quietly ignored).
    pub fn load(path: &Path) -> Option<Result<Plan, String>> {
        if !path.is_file() {
            return None;
        }
        Some(
            std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|t| serde_json::from_str(&forgive_commas(&t)).map_err(|e| e.to_string())),
        )
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| e.to_string())
    }

    pub fn open_ids(&self) -> Vec<String> {
        self.steps
            .iter()
            .filter(|s| s.status == "todo" || s.status == "doing")
            .map(|s| s.id.clone())
            .collect()
    }

    /// The always-visible one-glance view rendered into every prompt.
    pub fn compact(&self) -> String {
        let mut out = format!("## PLAN: {}\n", self.goal);
        for s in &self.steps {
            let mark = match s.status.as_str() {
                "done" => "x",
                "doing" => ">",
                "skip" => "-",
                _ => " ",
            };
            out.push_str(&format!("[{mark}] {}", s.id));
            if !s.what.is_empty() {
                out.push_str(&format!(" — {}", s.what));
            }
            let blocked: Vec<&str> = s
                .needs
                .iter()
                .filter(|n| {
                    self.steps
                        .iter()
                        .any(|o| o.id == **n && o.status != "done" && o.status != "skip")
                })
                .map(|n| n.as_str())
                .collect();
            if !blocked.is_empty() {
                out.push_str(&format!(" (blocked by: {})", blocked.join(",")));
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_compact_and_open() {
        let p: Plan = serde_json::from_str(
            r#"{"goal":"auth","steps":[
                {"id":"schema","status":"done","what":"tables"},
                {"id":"api","status":"doing","needs":["schema"],"verify":"echo ok"},
                {"id":"ui","needs":["api"]}
            ]}"#,
        )
        .unwrap();
        assert_eq!(p.open_ids(), vec!["api", "ui"]);
        let c = p.compact();
        assert!(c.contains("[x] schema"), "{c}");
        assert!(c.contains("[>] api"), "{c}");
        assert!(c.contains("[ ] ui") && c.contains("blocked by: api"), "{c}");
    }
}
