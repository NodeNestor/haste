use crate::config::ModelCfg;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

pub struct Client {
    cfg: ModelCfg,
    key: Option<String>,
    agent: ureq::Agent,
}

pub struct StreamStats {
    pub ttft_ms: u128,
    pub total_ms: u128,
    pub out_chars: usize,
    /// "length" means the output was guillotined by max_tokens mid-message.
    pub finish_reason: Option<String>,
    /// Exact usage from the provider (0 when not reported).
    pub prompt_tokens: u64,
    /// None when the provider's usage block has no prompt_tokens_details —
    /// vLLM often omits it even with prefix caching active, and a printed 0
    /// reads as "cache broken" when the truth is "not reported".
    pub cached_tokens: Option<u64>,
    pub completion_tokens: u64,
}

/// Model ids served by one endpoint (GET /models). Short timeout: the TUI
/// calls this interactively. Failures are an empty list, never an error.
pub fn served_ids(ep: &ModelCfg) -> Vec<String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(3))
        .build();
    let url = format!("{}/models", ep.base_url.trim_end_matches('/'));
    let mut req = agent.get(&url);
    if let Some(k) = ep.api_key_env.as_deref().and_then(|v| std::env::var(v).ok()) {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }
    let Ok(resp) = req.call() else { return Vec::new() };
    let Ok(v) = resp.into_json::<Value>() else { return Vec::new() };
    v["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["id"].as_str().map(String::from))
        .collect()
}

/// Resolve a /model target: a declared [models.<name>] wins; otherwise every
/// KNOWN endpoint ([model] + all alternates) is asked what it serves, and a
/// matching id inherits that endpoint's settings with the id swapped in.
/// Declare an endpoint once, use every model on it by name.
pub fn resolve_model(cfg: &crate::config::Config, name: &str) -> Option<ModelCfg> {
    if let Some(m) = cfg.models.get(name) {
        return Some(m.clone());
    }
    let mut seen = std::collections::HashSet::new();
    for ep in std::iter::once(&cfg.model).chain(cfg.models.values()) {
        if !seen.insert(ep.base_url.clone()) {
            continue;
        }
        if served_ids(ep).iter().any(|i| i == name) {
            let mut m = ep.clone();
            m.model = name.to_string();
            return Some(m);
        }
    }
    None
}

impl Client {
    pub fn new(cfg: ModelCfg, key: Option<String>) -> Client {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(300))
            .build();
        Client { cfg, key, agent }
    }

    fn merge_extra(&self, body: &mut Value) {
        if let (Some(extra), Some(obj)) = (&self.cfg.extra_body, body.as_object_mut()) {
            for (k, v) in extra {
                if let Ok(jv) = serde_json::to_value(v) {
                    obj.insert(k.clone(), jv);
                }
            }
        }
    }

    fn post(&self, body: &Value) -> Result<ureq::Response, String> {
        let url = format!("{}/chat/completions", self.cfg.base_url.trim_end_matches('/'));
        let mut req = self.agent.post(&url).set("Content-Type", "application/json");
        if let Some(k) = &self.key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        req.send_string(&body.to_string()).map_err(|e| match e {
            ureq::Error::Status(code, resp) => {
                let text = resp.into_string().unwrap_or_default();
                format!("HTTP {code}: {}", &text[..text.len().min(300)])
            }
            other => format!("request failed: {other}"),
        })
    }

    /// Stream a completion; `on_delta` fires per content chunk so the caller
    /// can lex and execute commands while generation is still in flight.
    /// `images`: (mime, base64) pairs appended after the text — the text stays
    /// first so the provider's prefix cache keeps matching image-free turns.
    pub fn stream(
        &self,
        system: &str,
        user: &str,
        images: &[(String, String)],
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<StreamStats, String> {
        self.stream_with(system, user, images, None, on_delta)
    }

    /// `overlay`: a body fragment merged over extra_body for THIS request only
    /// — escalation thinking ([model.think]) rides here.
    pub fn stream_with(
        &self,
        system: &str,
        user: &str,
        images: &[(String, String)],
        overlay: Option<&toml::value::Table>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<StreamStats, String> {
        let user_content: Value = if images.is_empty() {
            Value::String(user.to_string())
        } else {
            let mut parts = vec![json!({"type": "text", "text": user})];
            for (mime, b64) in images {
                parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:{mime};base64,{b64}")}
                }));
            }
            Value::Array(parts)
        };
        let mut body = json!({
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        });
        // temperature < 0 means "don't send it" — diffusion models (vLLM
        // DiffusionGemma, Mercury) reject the parameter outright.
        if self.cfg.temperature >= 0.0 {
            body["temperature"] = json!(self.cfg.temperature);
        }
        self.merge_extra(&mut body);
        if let (Some(extra), Some(obj)) = (overlay, body.as_object_mut()) {
            for (k, v) in extra {
                if let Ok(jv) = serde_json::to_value(v) {
                    obj.insert(k.clone(), jv);
                }
            }
        }
        let t0 = Instant::now();
        let resp = self.post(&body)?;
        let mut ttft: Option<u128> = None;
        let mut out_chars = 0usize;
        let mut finish_reason: Option<String> = None;
        // Degeneration guard: quantized models sometimes collapse into "!!!!"
        // or alternating-pair spam. Detect the runaway in-stream, drop the
        // connection, and report it — a retry costs less than a ruined turn.
        let (mut last, mut run) = ('\0', 0u32);
        let (mut prev2, mut pair_run) = ('\0', 0u32);
        // Phrase-loop detection: a rolling tail of recent output, scanned for
        // three identical consecutive blocks (sentence-level repetition that
        // char/pair runs cannot see). The tail must hold 3 periods at the
        // MAX_PERIOD cap, or long-line loops become invisible.
        let mut tail: Vec<u8> = Vec::with_capacity(3 * Self::MAX_PERIOD + 64);
        let mut since_scan = 0usize;
        let mut degenerate = false;
        let mut usage = (0u64, None::<u64>, 0u64);
        let reader = BufReader::new(resp.into_reader());
        'outer: for line in reader.lines() {
            let line = line.map_err(|e| format!("stream read: {e}"))?;
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data.trim() == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
                finish_reason = Some(fr.to_string());
            }
            if v["usage"].is_object() {
                usage = (
                    v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                    v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64(),
                    v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
                );
            }
            if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                if !delta.is_empty() {
                    ttft.get_or_insert_with(|| t0.elapsed().as_millis());
                    out_chars += delta.len();
                    for c in delta.chars() {
                        if c == last { run += 1 } else { run = 1 }
                        if c == prev2 { pair_run += 1 } else { pair_run = 1 }
                        prev2 = last;
                        last = c;
                        // Thresholds must clear legitimate ASCII art: markdown
                        // tables and box diagrams run 80-150 identical chars,
                        // while true repetition collapse runs to max_tokens.
                        if run >= 240 || pair_run >= 480 {
                            degenerate = true;
                            finish_reason = Some("degenerate".into());
                            break 'outer;
                        }
                    }
                    tail.extend_from_slice(delta.as_bytes());
                    if tail.len() > 3 * Self::MAX_PERIOD {
                        tail.drain(..tail.len() - 3 * Self::MAX_PERIOD);
                    }
                    since_scan += delta.len();
                    if since_scan >= 48 {
                        since_scan = 0;
                        if Self::phrase_loop(&tail) {
                            degenerate = true;
                            finish_reason = Some("degenerate".into());
                            break 'outer;
                        }
                    }
                    on_delta(delta);
                }
            }
        }
        let _ = degenerate;
        Ok(StreamStats {
            ttft_ms: ttft.unwrap_or_else(|| t0.elapsed().as_millis()),
            total_ms: t0.elapsed().as_millis(),
            out_chars,
            finish_reason,
            prompt_tokens: usage.0,
            cached_tokens: usage.1,
            completion_tokens: usage.2,
        })
    }

    /// Longest repeated block the loop guard can see. Sized for a full
    /// command line: models loop on ~200-byte one-liner pipelines verbatim,
    /// which mid-stream exec turns into repeated real work.
    const MAX_PERIOD: usize = 640;

    /// Sentence-level repetition: the tail ends with three IDENTICAL
    /// consecutive blocks of 20-MAX_PERIOD bytes. Blocks of a single repeated
    /// character are the run-guard's job (and legit as ASCII art), so a
    /// phrase must have some variety (>=5 distinct bytes) to count.
    fn phrase_loop(tail: &[u8]) -> bool {
        let n = tail.len();
        for p in 20..=Self::MAX_PERIOD {
            if n < 3 * p {
                break;
            }
            let a = &tail[n - p..];
            let b = &tail[n - 2 * p..n - p];
            let c = &tail[n - 3 * p..n - 2 * p];
            if a == b && b == c {
                let mut seen = [false; 256];
                let distinct = a.iter().filter(|&&x| !std::mem::replace(&mut seen[x as usize], true)).count();
                if distinct >= 5 {
                    return true;
                }
            }
        }
        false
    }

    /// Non-streaming call, used by the distiller.
    pub fn complete(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        let mut body = json!({
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        });
        if self.cfg.temperature >= 0.0 {
            body["temperature"] = json!(0.0);
        }
        self.merge_extra(&mut body);
        let resp = self.post(&body)?;
        let v: Value = resp.into_json().map_err(|e| format!("bad json: {e}"))?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "no content in response".into())
    }
}
