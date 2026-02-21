//! SwiGLU Feed-Forward Network.
//!
//! LLaMA-family models use SwiGLU (Swish-Gated Linear Unit):
//!   gate = SiLU(W_gate × x)
//!   up   = W_up × x
//!   out  = W_down × (gate ⊙ up)
//!
//! Gemma uses GeGLU (GELU instead of SiLU). Controlled by `FfnActivation`.

use super::matmul::matvec;
use super::norm::rms_norm_out;
use crate::backend::gguf_stream::GgufFile;
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

/// SwiGLU/GeGLU FFN for one transformer layer.
///
/// Reads gate/up/down weight tensors from GGUF, applies RMSNorm,
/// computes gated FFN, and adds residual.
///
/// `hidden` is the residual stream `[n_embd]`, modified in-place.
pub fn ffn_layer(
    hidden: &mut [f32],
    gguf: &GgufFile,
    layer: u32,
    cfg: &ModelConfig,
    activation: FfnActivation,
) -> Result<()> {
    let n_embd = cfg.n_embd;
    let n_ff = cfg.n_ff;

    // 1. RMSNorm (pre-FFN)
    let norm_name = format!("blk.{layer}.ffn_norm.weight");
    let norm_raw = gguf.tensor_bytes(&norm_name)?;
    let norm_type = gguf.tensor_type(&norm_name)?;
    let mut normed = vec![0.0f32; n_embd];
    rms_norm_out(hidden, norm_raw, norm_type, cfg.norm_eps, &mut normed);

    // 2. Gate and Up projections
    let gate_name = format!("blk.{layer}.ffn_gate.weight");
    let up_name = format!("blk.{layer}.ffn_up.weight");

    let gate_raw = gguf.tensor_bytes(&gate_name)?;
    let gate_type = gguf.tensor_type(&gate_name)?;
    let up_raw = gguf.tensor_bytes(&up_name)?;
    let up_type = gguf.tensor_type(&up_name)?;

    let mut gate = vec![0.0f32; n_ff];
    let mut up = vec![0.0f32; n_ff];

    matvec(gate_raw, gate_type, n_ff, n_embd, &normed, &mut gate);
    matvec(up_raw, up_type, n_ff, n_embd, &normed, &mut up);

    // 3. Activation + element-wise gate
    match activation {
        FfnActivation::Silu => {
            for i in 0..n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
        }
        FfnActivation::Gelu => {
            for i in 0..n_ff {
                gate[i] = gelu(gate[i]) * up[i];
            }
        }
    }

    // 4. Down projection
    let down_name = format!("blk.{layer}.ffn_down.weight");
    let down_raw = gguf.tensor_bytes(&down_name)?;
    let down_type = gguf.tensor_type(&down_name)?;

    let mut down = vec![0.0f32; n_embd];
    matvec(down_raw, down_type, n_embd, n_ff, &gate, &mut down);

    // 5. Residual connection
    for i in 0..n_embd {
        hidden[i] += down[i];
    }

    Ok(())
}

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline]
fn gelu(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
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
