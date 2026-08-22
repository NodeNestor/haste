//! O verb: signature outlines — the codemap idea (nestor-lean) as a haste mod.
//!
//! Regex-lite per-language matchers pull declaration lines with their real
//! line numbers, bodies elided. ~100 tokens to orient in a file instead of a
//! full R; line numbers feed straight into range reads and E edits.

use crate::tools::clip;

/// (line_number, signature) pairs for one file's text.
pub fn outline(text: &str, ext: &str) -> Option<Vec<(usize, String)>> {
    let is_sig: fn(&str) -> bool = match ext {
        "rs" => |t| {
            let t2 = t.trim_start();
            let t2 = t2.strip_prefix("pub ").unwrap_or(t2);
            let t2 = t2.strip_prefix("async ").unwrap_or(t2);
            ["fn ", "struct ", "enum ", "trait ", "impl ", "mod ", "const ", "static ", "type ", "macro_rules!"]
                .iter()
                .any(|k| t2.starts_with(k))
        },
        "py" => |t| {
            let s = t.trim_start();
            s.starts_with("def ") || s.starts_with("async def ") || s.starts_with("class ")
                || (t == t.trim_start() && (s.starts_with("import ") || s.starts_with("from ")))
        },
        "js" | "ts" | "tsx" | "jsx" | "mjs" => |t| {
            let s = t.trim_start();
            let s2 = s.strip_prefix("export ").unwrap_or(s);
            let s2 = s2.strip_prefix("default ").unwrap_or(s2);
            let s2 = s2.strip_prefix("async ").unwrap_or(s2);
            ["function ", "class ", "interface ", "type ", "enum ", "const ", "import "]
                .iter()
                .any(|k| s2.starts_with(k))
                && (s != s2 || !s.starts_with("const ") || s.contains("=>") || s.contains("require("))
        },
        "go" => |t| {
            ["func ", "type ", "const ", "var ", "package ", "import"]
                .iter()
                .any(|k| t.starts_with(k))
        },
        "cs" | "java" => |t| {
            let s = t.trim_start();
            ["public ", "private ", "protected ", "internal ", "class ", "interface ", "enum ", "namespace ", "using ", "record "]
                .iter()
                .any(|k| s.starts_with(k))
                && !s.contains(" = new ")
        },
        _ => return None,
    };
    Some(
        text.lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty() && is_sig(l))
            .map(|(i, l)| (i + 1, clip(l.trim_end(), 120)))
            .collect(),
    )
}

pub fn is_code_ext(ext: &str) -> bool {
    matches!(ext, "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "mjs" | "go" | "cs" | "java")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_and_python_outlines() {
        let rs = "use std::io;\n\npub struct Foo {\n    x: u32,\n}\n\nimpl Foo {\n    pub fn new() -> Foo {\n        Foo { x: 1 }\n    }\n}\n";
        let o = outline(rs, "rs").unwrap();
        let sigs: Vec<&str> = o.iter().map(|(_, s)| s.as_str()).collect();
        assert!(sigs.contains(&"pub struct Foo {"));
        assert!(sigs.iter().any(|s| s.contains("pub fn new")));
        assert!(!sigs.iter().any(|s| s.contains("Foo { x: 1 }")), "body leaked: {sigs:?}");
        assert_eq!(o.iter().find(|(_, s)| s.contains("fn new")).unwrap().0, 8);

        let py = "import os\n\nclass Cart:\n    def add(self, x):\n        self.x = x\n\ndef main():\n    pass\n";
        let o = outline(py, "py").unwrap();
        let sigs: Vec<&str> = o.iter().map(|(_, s)| s.as_str()).collect();
        assert!(sigs.contains(&"class Cart:") && sigs.iter().any(|s| s.contains("def add")));
        assert!(!sigs.iter().any(|s| s.contains("self.x = x")));
    }
}
