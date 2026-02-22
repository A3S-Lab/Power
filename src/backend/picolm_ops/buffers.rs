//! Pre-allocated working buffers for the transformer forward pass.
//!
//! Allocating `vec![0.0f32; n]` on every token × every layer is the single
//! largest source of overhead in the hot path (~312 allocations/token for a
//! 24-layer model). This struct allocates everything once at load time and
//! reuses the same memory across all tokens.
//!
//! # Buffer sizing
//!
//! All buffers are sized for the worst-case dimension at construction time:
//! - `n_embd`  — hidden state, normed, proj, down
//! - `n_heads * head_dim` — Q (= n_embd for MHA)
//! - `n_kv_heads * head_dim` — K, V
//! - `n_ff`    — gate, up
//! - `vocab_size` — logits
//! - `max_seq_len` — attention scores (grows with context)
//!
//! # TEE Security
//!
//! All buffers are zeroized on drop to prevent sensitive inference data
//! (prompts, activations, logits) from persisting in TEE memory.

use zeroize::Zeroize;

pub struct ForwardBuffers {
    /// Pre-attention / pre-FFN RMSNorm output  [n_embd]
    pub normed: Vec<f32>,
    /// Query projection output  [n_heads * head_dim]
    pub q: Vec<f32>,
    /// Key projection output    [n_kv_heads * head_dim]
    pub k: Vec<f32>,
    /// Value projection output  [n_kv_heads * head_dim]
    pub v: Vec<f32>,
    /// Attention output (pre-proj)  [n_heads * head_dim]
    pub attn_out: Vec<f32>,
    /// Attention scores for one head  [max_seq_len]
    pub scores: Vec<f32>,
    /// Output projection result  [n_embd]
    pub proj: Vec<f32>,
    /// FFN gate projection  [n_ff]
    pub gate: Vec<f32>,
    /// FFN up projection   [n_ff]
    pub up: Vec<f32>,
    /// FFN down projection result  [n_embd]
    pub down: Vec<f32>,
    /// Final normed hidden state before logit projection  [n_embd]
    pub normed_final: Vec<f32>,
    /// Logit vector  [vocab_size]
    pub logits: Vec<f32>,
    /// Temporary buffer for KV cache f16→f32 decode  [head_dim]
    pub kv_tmp: Vec<f32>,
}

impl ForwardBuffers {
    pub fn new(
        n_embd: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_ff: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Self {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        Self {
            normed: vec![0.0f32; n_embd],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; q_dim],
            scores: vec![0.0f32; max_seq_len],
            proj: vec![0.0f32; n_embd],
            gate: vec![0.0f32; n_ff],
            up: vec![0.0f32; n_ff],
            down: vec![0.0f32; n_embd],
            normed_final: vec![0.0f32; n_embd],
            logits: vec![0.0f32; vocab_size],
            kv_tmp: vec![0.0f32; head_dim],
        }
    }
}

/// Zeroize all buffers on drop to prevent sensitive inference data
/// (prompts, intermediate activations, logits) from persisting in memory.
/// Critical for TEE environments where memory pages may be inspected.
impl Drop for ForwardBuffers {
    fn drop(&mut self) {
        self.normed.zeroize();
        self.q.zeroize();
        self.k.zeroize();
        self.v.zeroize();
        self.attn_out.zeroize();
        self.scores.zeroize();
        self.proj.zeroize();
        self.gate.zeroize();
        self.up.zeroize();
        self.down.zeroize();
        self.normed_final.zeroize();
        self.logits.zeroize();
        self.kv_tmp.zeroize();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffers_sized_correctly() {
        let b = ForwardBuffers::new(16, 4, 2, 4, 32, 100, 512);
        assert_eq!(b.normed.len(), 16);
        assert_eq!(b.q.len(), 16); // 4 heads * 4 head_dim
        assert_eq!(b.k.len(), 8); // 2 kv_heads * 4 head_dim
        assert_eq!(b.v.len(), 8);
        assert_eq!(b.attn_out.len(), 16);
        assert_eq!(b.scores.len(), 512);
        assert_eq!(b.proj.len(), 16);
        assert_eq!(b.gate.len(), 32);
        assert_eq!(b.up.len(), 32);
        assert_eq!(b.down.len(), 16);
        assert_eq!(b.normed_final.len(), 16);
        assert_eq!(b.logits.len(), 100);
    }

    #[test]
    fn test_buffers_zero_initialized() {
        let b = ForwardBuffers::new(8, 2, 2, 4, 16, 50, 64);
        assert!(b.normed.iter().all(|&v| v == 0.0));
        assert!(b.logits.iter().all(|&v| v == 0.0));
    }
}
