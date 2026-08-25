use serde::Deserialize;
use std::collections::BTreeMap;

/// Verbs implemented natively in the binary. Config tools must not shadow
/// these — except the single-line ones below, with an explicit override flag.
pub const NATIVE_VERBS: &str = "REIGNXADVSO";
/// Natives a tool may replace with `override = true` (game-mod style). The
/// payload verbs (E/I/N) and protocol verbs (S/D/A) stay native.
pub const OVERRIDABLE_VERBS: &str = "RGXOV";

#[derive(Deserialize, Clone)]
pub struct Config {
    pub model: ModelCfg,
    /// Named alternate models: switch with /model <name> (TUI) or -m <name>
    /// (CLI). Same shape as [model]; [model] stays the default.
    #[serde(default)]
    pub models: BTreeMap<String, ModelCfg>,
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
    #[serde(default)]
    pub verify: VerifyCfg,
    #[serde(default)]
    pub prompt: PromptCfg,
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

/// System-prompt shaping. The command reference, rules, and plan protocol are
/// always generated (they must match what the binary parses); these hooks
/// replace the identity line and append free text after the rules.
#[derive(Deserialize, Clone, Default)]
pub struct PromptCfg {
    /// Replaces the built-in "You are haste…" identity line entirely.
    #[serde(default)]
    pub system: Option<String>,
    /// Appended verbatim after the rules (same slot mods inject into).
    #[serde(default)]
    pub extra: String,
}

/// Auto-verify: run this command automatically after any turn that edited
/// files, injecting the result — deletes the model's explicit "run tests"
/// turn (the most common turn in every trajectory). A failing verify also
/// refuses a same-turn D.
#[derive(Deserialize, Clone, Default)]
pub struct VerifyCfg {
    #[serde(default)]
    pub cmd: Option<String>,
    #[serde(default = "d_verify_timeout")]
    pub timeout_ms: u64,
    /// Pruner chain for the verify output (default: first failure only).
    #[serde(default = "d_verify_prune")]
    pub prune: String,
}
fn d_verify_timeout() -> u64 {
    180_000
}
fn d_verify_prune() -> String {
    "first_failure".into()
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
    /// Omitted = don't send the field (provider's own default). Explicit 0.0
    /// = greedy. Negative also means "don't send" (providers that reject it).
    #[serde(default = "d_temperature")]
    pub temperature: f32,
    /// Provider has a token-prefix cache (llama.cpp, vLLM). Selects append mode defaults.
    #[serde(default)]
    pub prefix_cache: bool,
    /// Extra fields merged verbatim into every request body — provider quirks
    /// (e.g. vLLM/Qwen: chat_template_kwargs.enable_thinking = false) live in
    /// config, never in code.
    #[serde(default)]
    pub extra_body: Option<toml::value::Table>,
    /// Named effort presets (off/low/high/dynamic — your names): each is a
    /// body fragment merged over extra_body when selected with /effort or
    /// --effort. The MAPPING is per provider, so it lives here, not in code.
    #[serde(default)]
    pub effort: BTreeMap<String, toml::value::Table>,
}
fn d_temperature() -> f32 {
    -1.0
}
fn d_max_tokens() -> u32 {
    8192 // big files must fit in one payload; cuts trigger rewrite loops
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
    /// Phase-boundary seal threshold: when a plan step completes and the doc
    /// is over this many tokens, compact NOW instead of waiting for budget —
    /// a long task's steady-state context stays near this floor.
    /// 0 = auto (budget_tokens / 3).
    #[serde(default)]
    pub compact_phase_tokens: usize,
    /// Project instruction files (CLAUDE.md / AGENTS.md style) pinned into the
    /// bootstrap block when they exist at the workspace root, in this order.
    #[serde(default = "d_instruction_files")]
    pub instruction_files: Vec<String>,
}
fn d_instruction_files() -> Vec<String> {
    vec!["HASTE.md".into(), "AGENTS.md".into(), "CLAUDE.md".into()]
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
            compact_phase_tokens: 0,
            instruction_files: d_instruction_files(),
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
    /// Replace the native implementation of this verb (RGXOV only): the
    /// command line routes to this tool instead of the built-in.
    #[serde(default, rename = "override")]
    pub override_native: bool,
    /// Polling tool: identical repeated results are its NORMAL behavior
    /// ("no new mail"), so the loop breaker leaves it alone. The tool's own
    /// command controls pacing (sleep/long-poll inside it).
    #[serde(default)]
    pub poll: bool,
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
    0 // unlimited, like the leader — the loop guards are the real rails
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

/// Starter config written by `haste init` — every popular way to connect,
/// one uncomment away.
pub const INIT_TOML: &str = r#"# haste config — point [model] at any OpenAI-compatible endpoint and go.
# Full reference of every knob: https://github.com/NodeNestor/haste/blob/master/haste.toml

[model]
# --- pick ONE block, uncomment, done ---

# Cerebras (wafer-speed; get a key at cloud.cerebras.ai):
# base_url = "https://api.cerebras.ai/v1"
# model = "zai-glm-4.7"
# api_key_env = "CEREBRAS_API_KEY"        # name of the env var holding your key

# Ollama (local):
# base_url = "http://127.0.0.1:11434/v1"
# model = "qwen3.5:9b"

# LM Studio (local):
# base_url = "http://127.0.0.1:1234/v1"
# model = "local-model"

# vLLM / llama.cpp server (local):
base_url = "http://127.0.0.1:8000/v1"
model = "default"

# OpenRouter:
# base_url = "https://openrouter.ai/api/v1"
# model = "qwen/qwen3.5-coder"
# api_key_env = "OPENROUTER_API_KEY"

max_tokens = 32768
temperature = 0.7
prefix_cache = true      # provider caches prompt prefixes (vLLM/llama.cpp/Cerebras: yes)

[context]
mode = "append"          # prefix-cache friendly; use "working_set" if your provider has no cache
budget_tokens = 40000

[verify]
# cmd = "cargo test"     # auto-runs after every editing turn; failing verify blocks D
"#;

impl Config {
    /// User-facing path for `haste init` and the first-run hint.
    pub fn home_config_path() -> std::path::PathBuf {
        let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
        std::path::Path::new(&home).join(".haste.toml")
    }

    pub fn load(path: Option<&str>) -> Result<Config, String> {
        // Lookup chain: explicit -c, ./haste.toml, ~/.haste.toml, embedded default —
        // so `haste` works from any directory once ~/.haste.toml exists.
        let mut found = true;
        let text = match path {
            Some(p) => std::fs::read_to_string(p).map_err(|e| format!("config {p}: {e}"))?,
            None => std::fs::read_to_string("haste.toml")
                .or_else(|_| std::fs::read_to_string(Self::home_config_path()).map_err(|_| ()))
                .unwrap_or_else(|_| {
                    found = false;
                    DEFAULT_TOML.to_string()
                }),
        };
        let mut cfg: Config = toml::from_str(&text).map_err(|e| format!("config parse: {e}"))?;
        cfg.mod_notes = crate::mods::apply(&mut cfg);
        if !found {
            cfg.mod_notes.insert(
                0,
                format!(
                    "no config found — using built-in default ({}). Run `haste init` to create {}",
                    cfg.model.base_url,
                    Self::home_config_path().display()
                ),
            );
        }
        for (verb, t) in &cfg.tool {
            let c = verb.chars().next().unwrap_or(' ');
            let native_ok = !NATIVE_VERBS.contains(c) || (t.override_native && OVERRIDABLE_VERBS.contains(c));
            if verb.len() != 1 || !c.is_ascii_uppercase() || !native_ok {
                return Err(format!(
                    "tool verb '{verb}' must be a single uppercase letter outside {NATIVE_VERBS} \
                     (or one of {OVERRIDABLE_VERBS} with override = true)"
                ));
            }
        }
        Ok(cfg)
    }

    /// The config a run actually uses: [models.<name>] swapped in, then the
    /// chosen effort preset merged over extra_body. Both optional.
    pub fn effective(cfg: &std::sync::Arc<Config>, model: &Option<String>, reason: &Option<String>) -> std::sync::Arc<Config> {
        if model.is_none() && reason.is_none() {
            return std::sync::Arc::clone(cfg);
        }
        let mut c = (**cfg).clone();
        if let Some(name) = model {
            if let Some(m) = c.models.get(name).cloned() {
                c.model = m;
            }
        }
        if let Some(r) = reason {
            if let Some(frag) = c.model.effort.get(r).cloned() {
                let eb = c.model.extra_body.get_or_insert_with(Default::default);
                for (k, v) in frag {
                    eb.insert(k, v);
                }
            }
        }
        std::sync::Arc::new(c)
    }

    pub fn api_key(&self) -> Option<String> {
        self.model
            .api_key_env
            .as_deref()
            .and_then(|v| std::env::var(v).ok())
    }
}
