//! Qwen chat-template tokenization for the FLUX.2 text encoder.

use std::path::Path;

use anyhow::{anyhow, Context, Result};

pub struct QwenTokenizer {
    inner: tokenizers::Tokenizer,
    pad_id: u32,
}

impl QwenTokenizer {
    /// Load `tokenizer.json` (and `tokenizer_config.json` for the pad token) from `dir`.
    pub fn load(dir: &Path) -> Result<QwenTokenizer> {
        let inner = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading {}: {e}", dir.join("tokenizer.json").display()))?;
        let pad = std::fs::read(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| match &v["pad_token"] {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "<|endoftext|>".to_string());
        let pad_id = inner.token_to_id(&pad).context("pad token not in vocabulary")?;
        Ok(QwenTokenizer { inner, pad_id })
    }

    /// The single-turn chat prompt FLUX.2 [klein] feeds to Qwen3 (thinking disabled).
    pub fn chat_prompt(prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    /// Token ids padded (right) / truncated to `max_len`, plus the number of real tokens.
    pub fn encode_padded(&self, text: &str, max_len: usize) -> Result<(Vec<i32>, usize)> {
        let enc = self.inner.encode(text, true).map_err(|e| anyhow!("tokenizing: {e}"))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|v| *v as i32).collect();
        ids.truncate(max_len);
        let valid = ids.len();
        ids.resize(max_len, self.pad_id as i32);
        Ok((ids, valid))
    }
}
