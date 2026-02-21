//! Rotary Position Embeddings (RoPE).
//!
//! Encodes position information by rotating pairs of dimensions in Q and K
//! vectors. Used by all LLaMA-family models.

/// Apply RoPE to a flat vector of shape `[n_heads × head_dim]`.
///
/// `pos` is the absolute token position in the sequence.
/// `theta_base` is the RoPE base frequency (LLaMA 2: 10000.0, LLaMA 3: 500000.0).
/// `rope_dim` is the number of dimensions to rotate per head (typically `head_dim`).
pub fn apply_rope(
    qk: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta_base: f32,
    rope_dim: usize,
) {
    let rd = rope_dim.min(head_dim);
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in (0..rd).step_by(2) {
            let freq = 1.0 / theta_base.powf(i as f32 / rd as f32);
            let angle = pos as f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();
            let x0 = qk[offset + i];
            let x1 = qk[offset + i + 1];
            qk[offset + i] = x0 * cos_val - x1 * sin_val;
            qk[offset + i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_pos_zero_is_identity() {
        // At pos=0, angle=0 for all freqs → cos=1, sin=0 → identity
        let mut qk = vec![1.0f32, 2.0, 3.0, 4.0]; // 1 head, head_dim=4
        let original = qk.clone();
        apply_rope(&mut qk, 1, 4, 0, 10000.0, 4);
        for (a, b) in qk.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "pos=0 should be identity");
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation — it preserves the L2 norm of each (x0, x1) pair
        let mut qk = vec![3.0f32, 4.0, 1.0, 2.0]; // 1 head, head_dim=4
        let norm_before: f32 = qk.iter().map(|v| v * v).sum::<f32>().sqrt();
        apply_rope(&mut qk, 1, 4, 42, 10000.0, 4);
        let norm_after: f32 = qk.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE should preserve norm: {norm_before} vs {norm_after}"
        );
    }

    #[test]
    fn test_rope_multi_head() {
        // 2 heads, head_dim=2
        let mut qk = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_rope(&mut qk, 2, 2, 1, 10000.0, 2);
        // Both heads should be rotated by the same angle
        let norm0 = (qk[0] * qk[0] + qk[1] * qk[1]).sqrt();
        let norm1 = (qk[2] * qk[2] + qk[3] * qk[3]).sqrt();
        assert!((norm0 - 1.0).abs() < 1e-5);
        assert!((norm1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_partial_dim() {
        // rope_dim=2 on head_dim=4 → only first 2 dims rotated
        let mut qk = vec![1.0f32, 2.0, 3.0, 4.0];
        apply_rope(&mut qk, 1, 4, 5, 10000.0, 2);
        // dims 2,3 should be unchanged
        assert!((qk[2] - 3.0).abs() < 1e-6);
        assert!((qk[3] - 4.0).abs() < 1e-6);
        // dims 0,1 should be rotated (different from original)
        assert!((qk[0] - 1.0).abs() > 1e-6 || (qk[1] - 2.0).abs() > 1e-6);
    }
}
