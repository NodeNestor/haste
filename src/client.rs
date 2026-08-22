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
}

impl Client {
    pub fn new(cfg: ModelCfg, key: Option<String>) -> Client {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(300))
            .build();
        Client { cfg, key, agent }
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
    pub fn stream(
        &self,
        system: &str,
        user: &str,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<StreamStats, String> {
        let body = json!({
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "stream": true,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        let t0 = Instant::now();
        let resp = self.post(&body)?;
        let mut ttft: Option<u128> = None;
        let mut out_chars = 0usize;
        let reader = BufReader::new(resp.into_reader());
        for line in reader.lines() {
            let line = line.map_err(|e| format!("stream read: {e}"))?;
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data.trim() == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                if !delta.is_empty() {
                    ttft.get_or_insert_with(|| t0.elapsed().as_millis());
                    out_chars += delta.len();
                    on_delta(delta);
                }
            }
        }
        Ok(StreamStats {
            ttft_ms: ttft.unwrap_or_else(|| t0.elapsed().as_millis()),
            total_ms: t0.elapsed().as_millis(),
            out_chars,
        })
    }

    /// Non-streaming call, used by the distiller.
    pub fn complete(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        let body = json!({
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        });
        let resp = self.post(&body)?;
        let v: Value = resp.into_json().map_err(|e| format!("bad json: {e}"))?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "no content in response".into())
    }
}
