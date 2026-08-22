//! Folder mods: drop a directory into the mods dir and haste grows new verbs.
//!
//!   ~/.haste/mods/
//!     mcp/
//!       mod.toml      <- manifest: tools, prompt injection, env
//!       bridge.py     <- the mod's own code, any language
//!
//! mod.toml:
//!   name = "mcp"
//!   prompt = "extra system-prompt lines the mod needs"
//!   [env]                      # exported to every tool call of this mod
//!   MCP_SERVERS = "..."
//!   [tool.M]                   # same schema as [tool.*] in haste.toml
//!   desc = "call an MCP tool"
//!   cmd  = "python {mod}/bridge.py {args}"   # {mod} = the mod's folder
//!
//! The harness stays dumb: a mod's tool is a process invocation. Mods that
//! need persistent state run their own daemon and talk to it — their
//! business, not the harness's. Native features (plan, vision, outline) are
//! compiled-in "native mods" toggled by config, not folders.

use crate::config::{Config, ToolCfg, NATIVE_VERBS};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    name: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    tool: BTreeMap<String, ToolCfg>,
}

/// Scan the mods dir and merge every mod's tools + prompt into the config.
/// Returns human-readable notes (loads and skips) for the UI/stderr.
pub fn apply(cfg: &mut Config) -> Vec<String> {
    let mut notes = Vec::new();
    let dir = expand_home(&cfg.mods_dir);
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return notes; // no mods dir = no mods, silently
    };
    let mut mod_dirs: Vec<_> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    mod_dirs.sort();
    for md in mod_dirs {
        let manifest_path = md.join("mod.toml");
        if !manifest_path.is_file() {
            continue;
        }
        let name = md.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let m: Manifest = match std::fs::read_to_string(&manifest_path)
            .map_err(|e| e.to_string())
            .and_then(|t| toml::from_str(&t).map_err(|e| e.to_string()))
        {
            Ok(m) => m,
            Err(e) => {
                notes.push(format!("mod {name}: SKIPPED — {}", crate::tools::clip(&e, 120)));
                continue;
            }
        };
        let label = if m.name.is_empty() { name.clone() } else { m.name.clone() };
        let mod_path = md.to_string_lossy().replace('\\', "/");
        let mut verbs = Vec::new();
        for (verb, mut tool) in m.tool {
            let ok = verb.len() == 1
                && verb.chars().next().unwrap().is_ascii_uppercase()
                && !NATIVE_VERBS.contains(&verb);
            if !ok {
                notes.push(format!("mod {label}: tool '{verb}' invalid (single uppercase letter outside {NATIVE_VERBS})"));
                continue;
            }
            if cfg.tool.contains_key(&verb) {
                notes.push(format!("mod {label}: tool '{verb}' already taken — skipped"));
                continue;
            }
            tool.cmd = tool.cmd.replace("{mod}", &mod_path);
            for (k, v) in &m.env {
                tool.env.insert(k.clone(), v.clone());
            }
            verbs.push(verb.clone());
            cfg.tool.insert(verb, tool);
        }
        if !m.prompt.trim().is_empty() {
            cfg.prompt_extra.push_str(m.prompt.trim());
            cfg.prompt_extra.push('\n');
        }
        notes.push(format!("mod {label}: loaded ({})", if verbs.is_empty() { "prompt only".into() } else { verbs.join(",") }));
    }
    notes
}

pub fn expand_home(path: &str) -> std::path::PathBuf {
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
        return std::path::Path::new(&home).join(rest);
    }
    std::path::PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_tools_prompt_and_env_with_mod_substitution() {
        let base = std::env::temp_dir().join(format!("haste-mods-{}", std::process::id()));
        let md = base.join("demo");
        std::fs::create_dir_all(&md).unwrap();
        std::fs::write(
            md.join("mod.toml"),
            "name = \"demo\"\nprompt = \"Demo mod: use K wisely.\"\n[env]\nDEMO_KEY = \"v1\"\n[tool.K]\ndesc = \"demo tool\"\ncmd = \"python {mod}/run.py {args}\"\n",
        )
        .unwrap();
        let mut cfg: Config = toml::from_str(crate::config::DEFAULT_TOML).unwrap();
        cfg.mods_dir = base.to_string_lossy().to_string();
        let notes = apply(&mut cfg);
        assert!(notes.iter().any(|n| n.contains("demo: loaded (K)")), "{notes:?}");
        let t = &cfg.tool["K"];
        assert!(t.cmd.contains("/demo/run.py") || t.cmd.contains("demo/run.py"), "{}", t.cmd);
        assert!(!t.cmd.contains("{mod}"));
        assert_eq!(t.env.get("DEMO_KEY").map(String::as_str), Some("v1"));
        assert!(cfg.prompt_extra.contains("use K wisely"));
        // Native collisions refused.
        std::fs::write(md.join("mod.toml"), "[tool.R]\ndesc=\"bad\"\ncmd=\"x\"\n").unwrap();
        let mut cfg2: Config = toml::from_str(crate::config::DEFAULT_TOML).unwrap();
        cfg2.mods_dir = base.to_string_lossy().to_string();
        let notes2 = apply(&mut cfg2);
        assert!(notes2.iter().any(|n| n.contains("invalid")), "{notes2:?}");
        let _ = std::fs::remove_dir_all(base);
    }
}
