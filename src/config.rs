use serde::Deserialize;
use std::collections::BTreeMap;

/// Verbs implemented natively in the binary. Config tools must not shadow these.
pub const NATIVE_VERBS: &str = "REIGNXADVSO";

#[derive(Deserialize, Clone)]
pub struct Config {
    pub model: ModelCfg,
    #[serde(default)]
    pub context: CtxCfg,
    #[serde(default)]
    pub tool: BTreeMap<String, ToolCfg>,
    #[serde(default)]
    pub profile: BTreeMap<String, ProfileCfg>,
    #[serde(default)]
    pub distill: DistillCfg,
    #[serde(default)]
    pub exec: ExecCfg,
    #[serde(default)]
    pub plan: PlanCfg,
    /// Folder mods live here; each subdir with a mod.toml adds verbs/prompt.
    #[serde(default = "d_mods_dir")]
    pub mods_dir: String,
    /// Accumulated prompt injections from loaded mods (not user-set).
    #[serde(skip)]
    pub prompt_extra: String,
    /// Loader notes (loaded/skipped mods) for the UI.
    #[serde(skip)]
    pub mod_notes: Vec<String>,
}
fn d_mods_dir() -> String {
    "~/.haste/mods".into()
}

/// The plan-file state machine (see plan.rs).
#[derive(Deserialize, Clone)]
pub struct PlanCfg {
    #[serde(default = "d_plan_file")]
    pub file: String,
    /// Refuse D while plan steps are open, auto-verify steps marked done.
    #[serde(default = "d_true")]
    pub enforce: bool,
}
fn d_plan_file() -> String {
    "plan.json".into()
}
impl Default for PlanCfg {
    fn default() -> Self {
        Self { file: d_plan_file(), enforce: true }
    }
}

/// How X and config tools run their command lines.
#[derive(Deserialize, Clone)]
pub struct ExecCfg {
    /// "powershell" | "cmd" | "sh". PowerShell is the Windows default: the
    /// model can use cmdlets directly with zero nested-quoting (cmd.exe
    /// mangles inner quotes, which sends weak models into retry loops).
    #[serde(default = "d_shell")]
    pub shell: String,
}
fn d_shell() -> String {
    if cfg!(windows) { "powershell".into() } else { "sh".into() }
}
impl Default for ExecCfg {
    fn default() -> Self {
        Self { shell: d_shell() }
    }
}

#[derive(Deserialize, Clone)]
pub struct ModelCfg {
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "d_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: f32,
    /// Provider has a token-prefix cache (llama.cpp, vLLM). Selects append mode defaults.
    #[serde(default)]
    pub prefix_cache: bool,
    /// Extra fields merged verbatim into every request body — provider quirks
    /// (e.g. vLLM/Qwen: chat_template_kwargs.enable_thinking = false) live in
    /// config, never in code.
    #[serde(default)]
    pub extra_body: Option<toml::value::Table>,
}
fn d_max_tokens() -> u32 {
    2048
}

#[derive(Deserialize, Clone)]
pub struct CtxCfg {
    /// "working_set": re-render aggressively every turn (no provider cache).
    /// "append": prefix-stable; folding only ever touches the tail at reseal.
    #[serde(default = "d_mode")]
    pub mode: String,
    #[serde(default = "d_budget")]
    pub budget_tokens: usize,
    /// Tool results older than this many turns collapse to their first line.
    #[serde(default = "d_fold")]
    pub fold_after_turns: u32,
    #[serde(default = "d_turns")]
    pub max_turns: u32,
    /// Cap on any single tool result, applied before pruners (chars).
    #[serde(default = "d_result_cap")]
    pub result_cap_chars: usize,
    /// Pin a workspace orientation block (project kind, git state, dir map)
    /// at the head of the session. Subagents never get one.
    #[serde(default = "d_true")]
    pub bootstrap: bool,
    /// Over-budget handling in append mode: "model" summarizes history with one
    /// prompt-cached model call (the context is already in the provider's KV
    /// cache, so the prefill is ~free); "structural" folds lines mechanically.
    /// Model compaction falls back to structural if the call fails.
    #[serde(default = "d_compact")]
    pub compact: String,
    #[serde(default = "d_compact_keep")]
    pub compact_keep_last: usize,
}
fn d_compact() -> String {
    "model".into()
}
fn d_compact_keep() -> usize {
    10
}
fn d_true() -> bool {
    true
}
fn d_mode() -> String {
    "working_set".into()
}
fn d_budget() -> usize {
    16_000
}
fn d_fold() -> u32 {
    6
}
fn d_turns() -> u32 {
    60
}
fn d_result_cap() -> usize {
    12_000
}
impl Default for CtxCfg {
    fn default() -> Self {
        Self {
            mode: d_mode(),
            budget_tokens: d_budget(),
            fold_after_turns: d_fold(),
            max_turns: d_turns(),
            result_cap_chars: d_result_cap(),
            bootstrap: true,
            compact: d_compact(),
            compact_keep_last: d_compact_keep(),
        }
    }
}

/// A config-declared tool: one DSL verb -> one shell command template.
#[derive(Deserialize, Clone)]
pub struct ToolCfg {
    pub desc: String,
    pub cmd: String,
    /// head_tail:A,B | first_failure | errors_only | drop:REGEX | keep:REGEX | distill | none
    /// Chain with '|', e.g. "drop:^warning\\b|head_tail:30,10"
    #[serde(default)]
    pub prune: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Env vars set for this tool's process (mods use this for their config).
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Deserialize, Clone)]
pub struct ProfileCfg {
    pub system: String,
    /// Verb letters this profile may use, e.g. "RGXW".
    #[serde(default = "d_ptools")]
    pub tools: String,
    #[serde(default = "d_pturns")]
    pub max_turns: u32,
    #[serde(default = "d_pbudget")]
    pub budget_tokens: usize,
}
fn d_ptools() -> String {
    "RGXD".into()
}
fn d_pturns() -> u32 {
    20
}
fn d_pbudget() -> usize {
    8_000
}

#[derive(Deserialize, Clone)]
pub struct DistillCfg {
    #[serde(default = "d_dprompt")]
    pub prompt: String,
    #[serde(default = "d_dmax")]
    pub max_tokens: u32,
}
fn d_dprompt() -> String {
    "Extract only the facts relevant to the task below. Terse bullet lines, no prose. \
     TASK: {task}\n---\n{text}"
        .into()
}
fn d_dmax() -> u32 {
    400
}
impl Default for DistillCfg {
    fn default() -> Self {
        Self {
            prompt: d_dprompt(),
            max_tokens: d_dmax(),
        }
    }
}

pub const DEFAULT_TOML: &str = r#"
[model]
base_url = "http://127.0.0.1:8000/v1"
model = "default"
"#;

impl Config {
    pub fn load(path: Option<&str>) -> Result<Config, String> {
        // Lookup chain: explicit -c, ./haste.toml, ~/.haste.toml, embedded default —
        // so `haste` works from any directory once ~/.haste.toml exists.
        let text = match path {
            Some(p) => std::fs::read_to_string(p).map_err(|e| format!("config {p}: {e}"))?,
            None => std::fs::read_to_string("haste.toml")
                .or_else(|_| {
                    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"));
                    home.map_err(|_| ()).and_then(|h| {
                        std::fs::read_to_string(std::path::Path::new(&h).join(".haste.toml"))
                            .map_err(|_| ())
                    })
                })
                .unwrap_or_else(|_| DEFAULT_TOML.to_string()),
        };
        let mut cfg: Config = toml::from_str(&text).map_err(|e| format!("config parse: {e}"))?;
        cfg.mod_notes = crate::mods::apply(&mut cfg);
        for verb in cfg.tool.keys() {
            let ok = verb.len() == 1
                && verb.chars().next().unwrap().is_ascii_uppercase()
                && !NATIVE_VERBS.contains(verb.as_str());
            if !ok {
                return Err(format!(
                    "tool verb '{verb}' must be a single uppercase letter outside {NATIVE_VERBS}"
                ));
            }
        }
        Ok(cfg)
    }

    pub fn api_key(&self) -> Option<String> {
        self.model
            .api_key_env
            .as_deref()
            .and_then(|v| std::env::var(v).ok())
    }
}
