//! Multi-head / Grouped-Query Attention for single-token decode.
//!
//! During autoregressive generation, we compute attention for only the last
//! token. The KV cache holds all previous tokens' key/value states.
//!
//! Supports GQA (Grouped-Query Attention) where multiple query heads share
//! the same KV head (LLaMA 3, Mistral).

use super::kv_cache::LayerKvCache;
use super::matmul::{extract_row, matvec};
use super::norm::rms_norm_out;
use super::rope::apply_rope;
use crate::backend::gguf_stream::GgufFile;
use crate::error::Result;

/// Model hyperparameters needed by the forward pass.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_layers: u32,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_dim: usize,
    pub context_length: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

/// Single-token attention for one transformer layer.
///
/// Reads Q/K/V/O weight tensors from GGUF, applies RMSNorm, projects,
/// applies RoPE, updates KV cache, computes scaled dot-product attention,
/// and projects output back to hidden dimension.
///
/// `hidden` is the residual stream `[n_embd]`, modified in-place (residual add).
/// `pos` is the absolute position of the current token.
pub fn attention_layer(
    hidden: &mut [f32],
    gguf: &GgufFile,
    layer: u32,
    pos: usize,
    kv_cache: &mut LayerKvCache,
    cfg: &ModelConfig,
) -> Result<()> {
    let n_embd = cfg.n_embd;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // 1. RMSNorm (pre-attention)
    let norm_name = format!("blk.{layer}.attn_norm.weight");
    let norm_raw = gguf.tensor_bytes(&norm_name)?;
    let norm_type = gguf.tensor_type(&norm_name)?;
    let mut normed = vec![0.0f32; n_embd];
    rms_norm_out(hidden, norm_raw, norm_type, cfg.norm_eps, &mut normed);

    // 2. Q/K/V projections
    let q_name = format!("blk.{layer}.attn_q.weight");
    let k_name = format!("blk.{layer}.attn_k.weight");
    let v_name = format!("blk.{layer}.attn_v.weight");

    let q_raw = gguf.tensor_bytes(&q_name)?;
    let q_type = gguf.tensor_type(&q_name)?;
    let k_raw = gguf.tensor_bytes(&k_name)?;
    let k_type = gguf.tensor_type(&k_name)?;
    let v_raw = gguf.tensor_bytes(&v_name)?;
    let v_type = gguf.tensor_type(&v_name)?;

    let mut q = vec![0.0f32; n_heads * head_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    matvec(q_raw, q_type, n_heads * head_dim, n_embd, &normed, &mut q);
    matvec(k_raw, k_type, kv_dim, n_embd, &normed, &mut k);
    matvec(v_raw, v_type, kv_dim, n_embd, &normed, &mut v);

    // 2b. Add bias if present (Qwen, Phi)
    add_bias_if_present(gguf, &format!("blk.{layer}.attn_q.bias"), &mut q);
    add_bias_if_present(gguf, &format!("blk.{layer}.attn_k.bias"), &mut k);
    add_bias_if_present(gguf, &format!("blk.{layer}.attn_v.bias"), &mut v);

    // 3. Apply RoPE to Q and K
    apply_rope(&mut q, n_heads, head_dim, pos, cfg.rope_theta, cfg.rope_dim);
    apply_rope(
        &mut k,
        n_kv_heads,
        head_dim,
        pos,
        cfg.rope_theta,
        cfg.rope_dim,
    );

    // 4. Store K, V in cache
    kv_cache.store(&k, &v);

    // 5. Scaled dot-product attention per query head
    let seq_len = kv_cache.len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads_per_kv = n_heads / n_kv_heads;

    let mut attn_out = vec![0.0f32; n_heads * head_dim];
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..n_heads {
        let kv_head = h / heads_per_kv;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Compute attention scores
        for (p, score) in scores[..seq_len].iter_mut().enumerate() {
            let k_cached = kv_cache.k_at(p, kv_head);
            let dot: f32 = q_head.iter().zip(k_cached.iter()).map(|(a, b)| a * b).sum();
            *score = dot * scale;
        }

        // Softmax
        softmax(&mut scores[..seq_len]);

        // Weighted sum of V
        let out_offset = h * head_dim;
        attn_out[out_offset..out_offset + head_dim].fill(0.0);
        for (p, &s) in scores[..seq_len].iter().enumerate() {
            let v_cached = kv_cache.v_at(p, kv_head);
            for d in 0..head_dim {
                attn_out[out_offset + d] += s * v_cached[d];
            }
        }
    }

    // 6. Output projection + residual
    let o_name = format!("blk.{layer}.attn_output.weight");
    let o_raw = gguf.tensor_bytes(&o_name)?;
    let o_type = gguf.tensor_type(&o_name)?;

    let mut proj = vec![0.0f32; n_embd];
    matvec(
        o_raw,
        o_type,
        n_embd,
        n_heads * head_dim,
        &attn_out,
        &mut proj,
    );

    // Residual connection
    for i in 0..n_embd {
        hidden[i] += proj[i];
    }

    Ok(())
}

/// Add a bias vector (dequantized from GGUF) to `out` if the tensor exists.
/// Silently does nothing if the tensor is not found.
fn add_bias_if_present(gguf: &GgufFile, name: &str, out: &mut [f32]) {
    if let Ok(bias_raw) = gguf.tensor_bytes(name) {
        if let Ok(bias_type) = gguf.tensor_type(name) {
            let mut bias = vec![0.0f32; out.len()];
            extract_row(bias_raw, bias_type, out.len(), 0, &mut bias);
            for (o, b) in out.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }
    }
}

/// In-place softmax over a slice.
fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_single() {
        let mut x = [1.0f32];
        softmax(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_uniform() {
        let mut x = [0.0f32; 4];
        softmax(&mut x);
        for v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
        assert!(x[2] < x[3]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not overflow
        let mut x = [1000.0f32, 1001.0, 1002.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1]);
    }

    #[test]
    fn test_softmax_empty() {
        let mut x: [f32; 0] = [];
        softmax(&mut x); // should not panic
    }
}
