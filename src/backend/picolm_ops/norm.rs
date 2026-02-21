//! RMSNorm — Root Mean Square Layer Normalization.
//!
//! LLaMA-family models use RMSNorm (not LayerNorm) before attention and FFN.
//! x[i] = x[i] / rms(x) * weight[i]
//! where rms(x) = sqrt(mean(x²) + eps)

use super::dequant::{block_bytes, block_size, dequantize_block};

/// In-place RMSNorm.
///
/// `weight_raw` is the raw GGUF bytes for the norm weight tensor (1D, `x.len()` elements).
pub fn rms_norm(x: &mut [f32], weight_raw: &[u8], weight_ggml_type: u32, eps: f32) {
    let n = x.len();

    // Compute 1/rms
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms_inv = 1.0 / (ss + eps).sqrt();

    // Dequantize norm weights and apply
    let bs = block_size(weight_ggml_type);
    let bb = block_bytes(weight_ggml_type);
    let mut buf = [0.0f32; 256];

    let blocks = if bs == 1 { n } else { n / bs };
    for blk in 0..blocks {
        let blk_data = &weight_raw[blk * bb..(blk + 1) * bb];
        let offset = blk * bs;
        dequantize_block(blk_data, weight_ggml_type, &mut buf[..bs]);
        let end = bs.min(n - offset);
        for j in 0..end {
            x[offset + j] = x[offset + j] * rms_inv * buf[j];
        }
    }
}

/// RMSNorm into a separate output buffer (does not modify `x`).
pub fn rms_norm_out(x: &[f32], weight_raw: &[u8], weight_ggml_type: u32, eps: f32, out: &mut [f32]) {
    out.copy_from_slice(x);
    rms_norm(out, weight_raw, weight_ggml_type, eps);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_unit_weight() {
        // weight = [1.0, 1.0, 1.0, 1.0], x = [2.0, 2.0, 2.0, 2.0]
        // rms = sqrt(mean(4,4,4,4) + eps) = sqrt(4 + 1e-5) ≈ 2.0
        // result ≈ [1.0, 1.0, 1.0, 1.0]
        let mut weight = Vec::new();
        for _ in 0..4 {
            weight.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let mut x = [2.0f32; 4];
        rms_norm(&mut x, &weight, 0, 1e-5);
        for v in &x {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_scaling() {
        // weight = [2.0, 2.0], x = [3.0, 3.0]
        // rms = sqrt(9 + eps) ≈ 3.0
        // result ≈ [3/3 * 2, 3/3 * 2] = [2.0, 2.0]
        let mut weight = Vec::new();
        for _ in 0..2 {
            weight.extend_from_slice(&2.0f32.to_le_bytes());
        }
        let mut x = [3.0f32; 2];
        rms_norm(&mut x, &weight, 0, 1e-5);
        for v in &x {
            assert!((v - 2.0).abs() < 0.01, "expected ~2.0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_out_preserves_input() {
        let mut weight = Vec::new();
        for _ in 0..2 {
            weight.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let x = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        rms_norm_out(&x, &weight, 0, 1e-5, &mut out);
        // x should be unchanged
        assert!((x[0] - 3.0).abs() < 1e-6);
        assert!((x[1] - 4.0).abs() < 1e-6);
        // out should be normalized
        assert!(out[0] != 0.0);
    }
}
