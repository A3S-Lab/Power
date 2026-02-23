//! RMSNorm — Root Mean Square Layer Normalization.
//!
//! LLaMA-family models use RMSNorm (not LayerNorm) before attention and FFN.
//! x[i] = x[i] / rms(x) * weight[i]
//! where rms(x) = sqrt(mean(x²) + eps)

use super::dequant::{block_bytes, block_size, dequantize_block};

/// In-place RMSNorm with raw GGUF weight bytes (dequantizes on the fly).
pub fn rms_norm(x: &mut [f32], weight_raw: &[u8], weight_ggml_type: u32, eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms_inv = 1.0 / (ss + eps).sqrt();

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
pub fn rms_norm_out(
    x: &[f32],
    weight_raw: &[u8],
    weight_ggml_type: u32,
    eps: f32,
    out: &mut [f32],
) {
    out.copy_from_slice(x);
    rms_norm(out, weight_raw, weight_ggml_type, eps);
}

/// RMSNorm with pre-dequantized f32 weight vector (no dequant overhead).
pub fn rms_norm_f32(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();

    #[cfg(target_arch = "aarch64")]
    {
        if n >= 4 {
            rms_norm_f32_neon(x, weight, eps);
            return;
        }
    }

    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms_inv = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * rms_inv * weight[i];
    }
}

/// NEON-accelerated RMSNorm: fused sum-of-squares + scale in SIMD.
#[cfg(target_arch = "aarch64")]
fn rms_norm_f32_neon(x: &mut [f32], weight: &[f32], eps: f32) {
    use std::arch::aarch64::*;
    let n = x.len();

    unsafe {
        // Pass 1: sum of squares with 4-wide NEON accumulation
        let mut sum_v = vdupq_n_f32(0.0);
        let chunks = n / 4;

        for i in 0..chunks {
            let xv = vld1q_f32(x.as_ptr().add(i * 4));
            sum_v = vfmaq_f32(sum_v, xv, xv);
        }
        let mut ss = vaddvq_f32(sum_v);
        for val in &x[(chunks * 4)..n] {
            ss += val * val;
        }
        let rms_inv = 1.0 / (ss / n as f32 + eps).sqrt();
        let rms_inv_v = vdupq_n_f32(rms_inv);

        // Pass 2: x[i] = x[i] * rms_inv * weight[i]
        for i in 0..chunks {
            let off = i * 4;
            let xv = vld1q_f32(x.as_ptr().add(off));
            let wv = vld1q_f32(weight.as_ptr().add(off));
            let scaled = vmulq_f32(vmulq_f32(xv, rms_inv_v), wv);
            vst1q_f32(x.as_mut_ptr().add(off), scaled);
        }
        for (xv, wv) in x[(chunks * 4)..n].iter_mut().zip(&weight[(chunks * 4)..n]) {
            *xv = *xv * rms_inv * wv;
        }
    }
}

/// RMSNorm into a separate output buffer with pre-dequantized weights.
pub fn rms_norm_out_f32(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    out.copy_from_slice(x);
    rms_norm_f32(out, weight, eps);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_unit_weight() {
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
        assert!((x[0] - 3.0).abs() < 1e-6);
        assert!((x[1] - 4.0).abs() < 1e-6);
        assert!(out[0] != 0.0);
    }

    #[test]
    fn test_rms_norm_f32_matches_raw() {
        let weight_f32 = [1.0f32, 1.0, 1.0, 1.0];
        let mut weight_raw = Vec::new();
        for &w in &weight_f32 {
            weight_raw.extend_from_slice(&w.to_le_bytes());
        }

        let mut x1 = [2.0f32; 4];
        let mut x2 = [2.0f32; 4];
        rms_norm(&mut x1, &weight_raw, 0, 1e-5);
        rms_norm_f32(&mut x2, &weight_f32, 1e-5);
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
