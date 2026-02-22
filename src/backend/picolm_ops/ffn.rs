//! SwiGLU Feed-Forward Network.
//!
//! LLaMA-family models use SwiGLU (Swish-Gated Linear Unit):
//!   gate = SiLU(W_gate × x)
//!   up   = W_up × x
//!   out  = W_down × (gate ⊙ up)
//!
//! Gemma uses GeGLU (GELU instead of SiLU). Controlled by `FfnActivation`.

use super::buffers::ForwardBuffers;
use super::matmul::matvec;
use super::norm::rms_norm_out_f32;
use super::tensor_cache::{TensorCache, SLOT_FFN_DOWN, SLOT_FFN_GATE, SLOT_FFN_UP};
use crate::error::Result;

use super::attention::ModelConfig;

/// FFN activation function variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnActivation {
    /// SiLU (Swish): x * sigmoid(x) — used by LLaMA, Mistral, Phi
    Silu,
    /// GELU: x * Φ(x) — used by Gemma
    Gelu,
}

/// SwiGLU/GeGLU FFN for one transformer layer (optimized).
///
/// Uses pre-dequantized norm weights, a pre-built tensor cache (no HashMap
/// lookups), and pre-allocated working buffers (no heap allocation in hot path).
pub fn ffn_layer(
    hidden: &mut [f32],
    tc: &TensorCache,
    layer: u32,
    cfg: &ModelConfig,
    activation: FfnActivation,
    ffn_norm_w: &[f32],
    buf: &mut ForwardBuffers,
) -> Result<()> {
    let n_embd = cfg.n_embd;
    let n_ff = cfg.n_ff;

    // 1. RMSNorm (pre-FFN) — uses pre-dequantized weights, writes into buf.normed
    rms_norm_out_f32(hidden, ffn_norm_w, cfg.norm_eps, &mut buf.normed);

    // 2. Gate and Up projections — tensor bytes from cache (zero HashMap lookups)
    let gate_e = tc.get(layer, SLOT_FFN_GATE);
    let up_e = tc.get(layer, SLOT_FFN_UP);

    let gate_buf = &mut buf.gate[..n_ff];
    let up_buf = &mut buf.up[..n_ff];

    matvec(
        gate_e.bytes().unwrap(),
        gate_e.ggml_type,
        n_ff,
        n_embd,
        &buf.normed,
        gate_buf,
    );
    matvec(
        up_e.bytes().unwrap(),
        up_e.ggml_type,
        n_ff,
        n_embd,
        &buf.normed,
        up_buf,
    );

    // 3. Activation + element-wise gate (in-place on gate_buf)
    match activation {
        FfnActivation::Silu => {
            silu_mul(gate_buf, up_buf);
        }
        FfnActivation::Gelu => {
            for i in 0..n_ff {
                gate_buf[i] = gelu(gate_buf[i]) * up_buf[i];
            }
        }
    }

    // 4. Down projection
    let down_e = tc.get(layer, SLOT_FFN_DOWN);
    let down_buf = &mut buf.down[..n_embd];
    matvec(
        down_e.bytes().unwrap(),
        down_e.ggml_type,
        n_embd,
        n_ff,
        gate_buf,
        down_buf,
    );

    // 5. Residual connection
    add_residual(hidden, down_buf);

    Ok(())
}

/// Fused SiLU(gate) * up — NEON-accelerated on aarch64.
#[inline]
fn silu_mul(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());

    #[cfg(target_arch = "aarch64")]
    {
        if gate.len() >= 4 {
            silu_mul_neon(gate, up);
            return;
        }
    }

    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }
}

#[cfg(target_arch = "aarch64")]
fn silu_mul_neon(gate: &mut [f32], up: &[f32]) {
    use std::arch::aarch64::*;
    let n = gate.len();
    let chunks = n / 4;

    unsafe {
        let one_v = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let off = i * 4;
            let gv = vld1q_f32(gate.as_ptr().add(off));
            let uv = vld1q_f32(up.as_ptr().add(off));

            // SiLU(x) = x / (1 + exp(-x)) — scalar exp per lane
            let neg_g = vnegq_f32(gv);
            let mut arr = [0.0f32; 4];
            vst1q_f32(arr.as_mut_ptr(), neg_g);
            arr[0] = arr[0].exp();
            arr[1] = arr[1].exp();
            arr[2] = arr[2].exp();
            arr[3] = arr[3].exp();
            let exp_v = vld1q_f32(arr.as_ptr());

            // x / (1 + exp(-x)) * up
            let denom = vaddq_f32(one_v, exp_v);
            let silu_v = vdivq_f32(gv, denom);
            let result = vmulq_f32(silu_v, uv);
            vst1q_f32(gate.as_mut_ptr().add(off), result);
        }
        for i in (chunks * 4)..n {
            gate[i] = silu(gate[i]) * up[i];
        }
    }
}

/// Residual connection: hidden[i] += buf[i] (NEON-accelerated).
#[inline]
fn add_residual(hidden: &mut [f32], buf: &[f32]) {
    debug_assert_eq!(hidden.len(), buf.len());

    #[cfg(target_arch = "aarch64")]
    {
        if hidden.len() >= 4 {
            add_residual_neon(hidden, buf);
            return;
        }
    }

    for i in 0..hidden.len() {
        hidden[i] += buf[i];
    }
}

#[cfg(target_arch = "aarch64")]
fn add_residual_neon(hidden: &mut [f32], buf: &[f32]) {
    use std::arch::aarch64::*;
    let n = hidden.len();
    let chunks = n / 4;

    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let hv = vld1q_f32(hidden.as_ptr().add(off));
            let bv = vld1q_f32(buf.as_ptr().add(off));
            vst1q_f32(hidden.as_mut_ptr().add(off), vaddq_f32(hv, bv));
        }
        for i in (chunks * 4)..n {
            hidden[i] += buf[i];
        }
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline]
fn gelu(x: f32) -> f32 {
    const C: f32 = 0.7978845608; // sqrt(2/π)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        // silu(x) ≈ x for large positive x (sigmoid → 1)
        let v = silu(10.0);
        assert!((v - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_negative() {
        // silu(x) ≈ 0 for large negative x (sigmoid → 0)
        let v = silu(-10.0);
        assert!(v.abs() < 0.01);
    }

    #[test]
    fn test_gelu_zero() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        // gelu(x) ≈ x for large positive x
        let v = gelu(10.0);
        assert!((v - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_negative() {
        // gelu(x) ≈ 0 for large negative x
        let v = gelu(-10.0);
        assert!(v.abs() < 0.01);
    }

    #[test]
    fn test_silu_at_one() {
        // silu(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
        let v = silu(1.0);
        assert!((v - 0.7311).abs() < 0.001);
    }
}
