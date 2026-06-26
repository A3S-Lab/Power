//! In-enclave mechanistic-interpretability tap.
//!
//! a3s-power owns the forward pass, so it has the residual stream natively. This module taps it at
//! one layer, encodes it with a Sparse Autoencoder, and emits only the sparse `(feature_id,
//! activation)` pairs — **the prompt/completion plaintext never leaves the enclave**. Downstream,
//! a3s-sentry's `SaeJudge` scores those features against a labeled dictionary, so the safety verdict
//! is white-box (judges the model's internal concepts), confidential (no text), and explainable.
//!
//! The SAE weights are an attestable artifact: sealed/measured alongside the model, so a client can
//! cryptographically prove *which* interpretability model scored the output.

use serde::Serialize;

/// A Sparse Autoencoder encoder: `f = TopK(ReLU(W_enc · h + b_enc))`. Decomposes a residual-stream
/// activation `h` (length `n_embd`) into a sparse set of interpretable features (length `n_features`).
/// Only the encoder is needed for monitoring (we read features, we don't reconstruct).
pub struct SaeEncoder {
    /// Row-major `[n_features × n_embd]` — feature `f`'s weights are `w_enc[f*n_embd .. (f+1)*n_embd]`.
    w_enc: Vec<f32>,
    b_enc: Vec<f32>,
    n_embd: usize,
    n_features: usize,
    /// Keep at most this many top features (sparsity of the emitted vector). 0 = keep all nonzero.
    top_k: usize,
    /// JumpReLU threshold — a pre-activation must exceed this to fire (0.0 = plain ReLU).
    threshold: f32,
}

impl SaeEncoder {
    /// Build from raw encoder weights. `w_enc` must be `n_features * n_embd` long.
    pub fn new(
        w_enc: Vec<f32>,
        b_enc: Vec<f32>,
        n_embd: usize,
        top_k: usize,
    ) -> anyhow::Result<Self> {
        let n_features = b_enc.len();
        anyhow::ensure!(n_embd > 0 && n_features > 0, "empty SAE dimensions");
        anyhow::ensure!(
            w_enc.len() == n_features * n_embd,
            "w_enc len {} != n_features {} * n_embd {}",
            w_enc.len(),
            n_features,
            n_embd
        );
        Ok(Self {
            w_enc,
            b_enc,
            n_embd,
            n_features,
            top_k,
            threshold: 0.0,
        })
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Encode one residual-stream activation into sparse `(feature_id, activation)` pairs, sorted by
    /// activation descending and truncated to `top_k`. Returns `[]` if `hidden` is the wrong length.
    pub fn encode(&self, hidden: &[f32]) -> Vec<(u32, f32)> {
        if hidden.len() != self.n_embd {
            return Vec::new();
        }
        let mut feats: Vec<(u32, f32)> = Vec::new();
        for f in 0..self.n_features {
            let row = &self.w_enc[f * self.n_embd..(f + 1) * self.n_embd];
            let mut pre = self.b_enc[f];
            for (w, h) in row.iter().zip(hidden.iter()) {
                pre += w * h;
            }
            // JumpReLU / ReLU gate.
            if pre > self.threshold && pre > 0.0 {
                feats.push((f as u32, pre));
            }
        }
        feats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if self.top_k > 0 && feats.len() > self.top_k {
            feats.truncate(self.top_k);
        }
        feats
    }
}

/// Taps the residual stream at a configured layer, encodes it, and produces the confidential
/// [`Event::LlmActivations`](https://docs.rs/a3s-sentry) NDJSON line a3s-sentry consumes. The caller
/// (the forward pass) hands it the post-layer `hidden`; the tap returns the line to emit on the
/// NDJSON sink. Only feature ids/activations are serialized — never the text.
pub struct InterpTap {
    encoder: SaeEncoder,
    /// Which transformer layer's residual to tap (mid-late, e.g. ~0.6–0.75 × depth).
    layer: u32,
}

#[derive(Serialize)]
struct Identity<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    agent: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session: Option<&'a str>,
}

#[derive(Serialize)]
struct LlmActivations<'a> {
    pid: u32,
    layer: u32,
    features: &'a [(u32, f32)],
}

#[derive(Serialize)]
struct EventLine<'a> {
    identity: Identity<'a>,
    event: EventBody<'a>,
}

#[derive(Serialize)]
enum EventBody<'a> {
    LlmActivations(LlmActivations<'a>),
}

impl InterpTap {
    pub fn new(encoder: SaeEncoder, layer: u32) -> Self {
        Self { encoder, layer }
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Call this for each layer of the forward pass. When `layer` matches the configured tap layer,
    /// encode `hidden` and return the confidential NDJSON event line to emit; otherwise `None`.
    pub fn observe(
        &self,
        pid: u32,
        layer: u32,
        hidden: &[f32],
        agent: Option<&str>,
        session: Option<&str>,
    ) -> Option<String> {
        if layer != self.layer {
            return None;
        }
        let features = self.encoder.encode(hidden);
        if features.is_empty() {
            return None;
        }
        let line = EventLine {
            identity: Identity { agent, session },
            event: EventBody::LlmActivations(LlmActivations {
                pid,
                layer,
                features: &features,
            }),
        };
        // Confidential by construction: this struct contains no prompt/completion text.
        serde_json::to_string(&line).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_sae() -> SaeEncoder {
        // 2 features over n_embd=4. feature0 reads dim0; feature1 reads dim2+dim3.
        let w_enc = vec![
            1.0, 0.0, 0.0, 0.0, // feature 0
            0.0, 0.0, 1.0, 1.0, // feature 1
        ];
        SaeEncoder::new(w_enc, vec![0.0, 0.0], 4, 8).unwrap()
    }

    #[test]
    fn encodes_top_features() {
        let sae = tiny_sae();
        let feats = sae.encode(&[2.0, 5.0, 0.5, 0.5]); // f0=2.0, f1=0.5+0.5=1.0
        assert_eq!(feats, vec![(0, 2.0), (1, 1.0)]);
    }

    #[test]
    fn relu_drops_negative_features() {
        let sae = tiny_sae();
        let feats = sae.encode(&[-3.0, 0.0, 0.0, 0.0]); // f0=-3 (gated off), f1=0 (off)
        assert!(feats.is_empty());
    }

    #[test]
    fn wrong_length_is_empty() {
        assert!(tiny_sae().encode(&[1.0, 2.0]).is_empty());
    }

    #[test]
    fn tap_emits_confidential_line_only_at_its_layer() {
        let tap = InterpTap::new(tiny_sae(), 18);
        assert!(tap
            .observe(7, 17, &[2.0, 0.0, 0.0, 0.0], Some("deep-finance"), None)
            .is_none());
        let line = tap
            .observe(7, 18, &[2.0, 0.0, 1.0, 1.0], Some("deep-finance"), None)
            .expect("emits at tap layer");
        // Shape matches a3s-sentry's Event::LlmActivations; carries NO prompt/completion text.
        assert!(line.contains("\"LlmActivations\""));
        assert!(line.contains("\"pid\":7"));
        assert!(line.contains("\"layer\":18"));
        assert!(line.contains("\"agent\":\"deep-finance\""));
        assert!(line.contains("[0,2.0]") || line.contains("[0,2]"));
        // sanity: no text fields leaked
        assert!(!line.to_lowercase().contains("prompt"));
        assert!(!line.to_lowercase().contains("completion"));
    }
}
