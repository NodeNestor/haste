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
            "temperature": self.cfg.temperature,
            "stream": true,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        });
        self.merge_extra(&mut body);
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
        let mut degenerate = false;
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
        })
    }

    /// Non-streaming call, used by the distiller.
    pub fn complete(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        let mut body = json!({
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        });
        self.merge_extra(&mut body);
        let resp = self.post(&body)?;
        let v: Value = resp.into_json().map_err(|e| format!("bad json: {e}"))?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "no content in response".into())
    }
}
