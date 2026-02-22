//! Multi-head / Grouped-Query Attention for single-token decode.
//!
//! During autoregressive generation, we compute attention for only the last
//! token. The KV cache holds all previous tokens' key/value states.
//!
//! Supports GQA (Grouped-Query Attention) where multiple query heads share
//! the same KV head (LLaMA 3, Mistral).

use super::buffers::ForwardBuffers;
use super::kv_cache::LayerKvCache;
use super::matmul::{extract_row, matvec};
use super::norm::rms_norm_out_f32;
use super::rope::RopeTable;
use super::tensor_cache::{
    TensorCache, SLOT_ATTN_K, SLOT_ATTN_K_BIAS, SLOT_ATTN_O, SLOT_ATTN_Q, SLOT_ATTN_Q_BIAS,
    SLOT_ATTN_V, SLOT_ATTN_V_BIAS,
};
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

/// Single-token attention for one transformer layer (optimized).
///
/// Uses pre-computed RoPE tables, pre-dequantized norm weights, a pre-built
/// tensor cache (no HashMap lookups), and pre-allocated working buffers
/// (no heap allocation in the hot path).
#[allow(clippy::too_many_arguments)]
pub fn attention_layer(
    hidden: &mut [f32],
    tc: &TensorCache,
    layer: u32,
    pos: usize,
    kv_cache: &mut LayerKvCache,
    cfg: &ModelConfig,
    rope_table: &RopeTable,
    attn_norm_w: &[f32],
    buf: &mut ForwardBuffers,
) -> Result<()> {
    let n_embd = cfg.n_embd;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // 1. RMSNorm (pre-attention) — uses pre-dequantized weights, writes into buf.normed
    rms_norm_out_f32(hidden, attn_norm_w, cfg.norm_eps, &mut buf.normed);

    // 2. Q/K/V projections — tensor bytes from cache (zero HashMap lookups)
    let q_e = tc.get(layer, SLOT_ATTN_Q);
    let k_e = tc.get(layer, SLOT_ATTN_K);
    let v_e = tc.get(layer, SLOT_ATTN_V);

    let q_raw = q_e.bytes().unwrap();
    let k_raw = k_e.bytes().unwrap();
    let v_raw = v_e.bytes().unwrap();

    let q_buf = &mut buf.q[..n_heads * head_dim];
    let k_buf = &mut buf.k[..kv_dim];
    let v_buf = &mut buf.v[..kv_dim];

    matvec(
        q_raw,
        q_e.ggml_type,
        n_heads * head_dim,
        n_embd,
        &buf.normed,
        q_buf,
    );
    matvec(k_raw, k_e.ggml_type, kv_dim, n_embd, &buf.normed, k_buf);
    matvec(v_raw, v_e.ggml_type, kv_dim, n_embd, &buf.normed, v_buf);

    // 2b. Add bias if present (Qwen, Phi) — cache lookup, no string formatting
    add_bias_if_cached(tc, layer, SLOT_ATTN_Q_BIAS, q_buf);
    add_bias_if_cached(tc, layer, SLOT_ATTN_K_BIAS, k_buf);
    add_bias_if_cached(tc, layer, SLOT_ATTN_V_BIAS, v_buf);

    // 3. Apply RoPE — uses pre-computed tables (no powf/sin/cos)
    rope_table.apply(q_buf, n_heads, head_dim, pos);
    rope_table.apply(k_buf, n_kv_heads, head_dim, pos);

    // 4. Store K, V in cache
    kv_cache.store(k_buf, v_buf);

    // 5. Scaled dot-product attention per query head
    let seq_len = kv_cache.len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads_per_kv = n_heads / n_kv_heads;

    let attn_out = &mut buf.attn_out[..n_heads * head_dim];
    attn_out.fill(0.0);
    let scores = &mut buf.scores[..seq_len];

    for h in 0..n_heads {
        let kv_head = h / heads_per_kv;
        let q_offset = h * head_dim;
        let q_head = &buf.q[q_offset..q_offset + head_dim];

        // Compute Q·K scores — fused f16→f32 dot product (no intermediate buffer).
        for (p, score) in scores.iter_mut().enumerate() {
            *score = kv_cache.k_dot(p, kv_head, q_head) * scale;
        }

        softmax(scores);

        // Weighted sum of V vectors — fused f16→f32 accumulate (no intermediate buffer).
        let out_offset = h * head_dim;
        for (p, &s) in scores.iter().enumerate() {
            kv_cache.v_accumulate(
                p,
                kv_head,
                &mut attn_out[out_offset..out_offset + head_dim],
                s,
            );
        }
    }

    // 6. Output projection + residual
    let o_e = tc.get(layer, SLOT_ATTN_O);
    let o_raw = o_e.bytes().unwrap();
    let proj = &mut buf.proj[..n_embd];
    matvec(
        o_raw,
        o_e.ggml_type,
        n_embd,
        n_heads * head_dim,
        attn_out,
        proj,
    );

    for i in 0..n_embd {
        hidden[i] += proj[i];
    }

    Ok(())
}

/// Add a bias vector from the tensor cache to `out` if the slot is present.
#[inline]
fn add_bias_if_cached(tc: &TensorCache, layer: u32, slot: usize, out: &mut [f32]) {
    let entry = tc.get(layer, slot);
    if entry.is_present() {
        let bias_raw = entry.bytes().unwrap();
        // Bias is always a 1-D F32 or F16 vector; use extract_row (row 0).
        let mut tmp = vec![0.0f32; out.len()];
        extract_row(bias_raw, entry.ggml_type, out.len(), 0, &mut tmp);
        for (o, b) in out.iter_mut().zip(tmp.iter()) {
            *o += b;
        }
    }
}

/// In-place softmax over a slice.
fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if x.len() >= 4 {
            softmax_neon(x);
            return;
        }
    }

    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// NEON-accelerated softmax: fused max-find, exp-sum, and normalize.
#[cfg(target_arch = "aarch64")]
fn softmax_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let chunks = n / 4;

    unsafe {
        // Pass 1: find max
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            max_v = vmaxq_f32(max_v, v);
        }
        let mut max_val = vmaxvq_f32(max_v);
        for i in (chunks * 4)..n {
            if x[i] > max_val {
                max_val = x[i];
            }
        }

        // Pass 2: exp(x - max) and sum
        let max_v = vdupq_n_f32(max_val);
        let mut sum_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(x.as_ptr().add(off));
            let shifted = vsubq_f32(v, max_v);
            // Scalar exp per lane (no NEON exp intrinsic)
            let mut arr = [0.0f32; 4];
            vst1q_f32(arr.as_mut_ptr(), shifted);
            arr[0] = arr[0].exp();
            arr[1] = arr[1].exp();
            arr[2] = arr[2].exp();
            arr[3] = arr[3].exp();
            let exp_v = vld1q_f32(arr.as_ptr());
            vst1q_f32(x.as_mut_ptr().add(off), exp_v);
            sum_v = vaddq_f32(sum_v, exp_v);
        }
        let mut sum = vaddvq_f32(sum_v);
        for i in (chunks * 4)..n {
            x[i] = (x[i] - max_val).exp();
            sum += x[i];
        }

        // Pass 3: normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            let inv_v = vdupq_n_f32(inv_sum);
            for i in 0..chunks {
                let off = i * 4;
                let v = vld1q_f32(x.as_ptr().add(off));
                vst1q_f32(x.as_mut_ptr().add(off), vmulq_f32(v, inv_v));
            }
            for i in (chunks * 4)..n {
                x[i] *= inv_sum;
            }
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
