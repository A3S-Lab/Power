//! Rotary Position Embeddings (RoPE).
//!
//! Encodes position information by rotating pairs of dimensions in Q and K
//! vectors. Used by all LLaMA-family models.
//!
//! Supports both on-the-fly computation and pre-computed cos/sin tables
//! for eliminating transcendental functions from the hot path.

/// Pre-computed RoPE cos/sin lookup table.
/// Layout: `[max_seq_len × half_dim]` for both cos and sin.
pub struct RopeTable {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half_dim: usize,
}

impl RopeTable {
    /// Build RoPE tables for all positions up to `max_seq_len`.
    pub fn new(max_seq_len: usize, head_dim: usize, rope_dim: usize, theta_base: f32) -> Self {
        let half_dim = rope_dim.min(head_dim) / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            let row = pos * half_dim;
            for i in 0..half_dim {
                let freq = 1.0 / theta_base.powf((i * 2) as f32 / rope_dim as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                cos[row + i] = c;
                sin[row + i] = s;
            }
        }

        Self { cos, sin, half_dim }
    }

    /// Apply RoPE using pre-computed tables. Much faster than `apply_rope`
    /// since it avoids powf/sin/cos per element.
    pub fn apply(&self, qk: &mut [f32], n_heads: usize, head_dim: usize, pos: usize) {
        let row = pos * self.half_dim;
        let cos_pos = &self.cos[row..row + self.half_dim];
        let sin_pos = &self.sin[row..row + self.half_dim];

        for h in 0..n_heads {
            let offset = h * head_dim;
            for i in 0..self.half_dim {
                let x0 = qk[offset + i * 2];
                let x1 = qk[offset + i * 2 + 1];
                qk[offset + i * 2] = x0 * cos_pos[i] - x1 * sin_pos[i];
                qk[offset + i * 2 + 1] = x0 * sin_pos[i] + x1 * cos_pos[i];
            }
        }
    }
}

/// Apply RoPE on the fly (no table, computes sin/cos per call).
/// Kept for backward compatibility and tests.
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
        let mut qk = vec![1.0f32, 2.0, 3.0, 4.0];
        let original = qk.clone();
        apply_rope(&mut qk, 1, 4, 0, 10000.0, 4);
        for (a, b) in qk.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "pos=0 should be identity");
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let mut qk = vec![3.0f32, 4.0, 1.0, 2.0];
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
        let mut qk = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_rope(&mut qk, 2, 2, 1, 10000.0, 2);
        let norm0 = (qk[0] * qk[0] + qk[1] * qk[1]).sqrt();
        let norm1 = (qk[2] * qk[2] + qk[3] * qk[3]).sqrt();
        assert!((norm0 - 1.0).abs() < 1e-5);
        assert!((norm1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_partial_dim() {
        let mut qk = vec![1.0f32, 2.0, 3.0, 4.0];
        apply_rope(&mut qk, 1, 4, 5, 10000.0, 2);
        assert!((qk[2] - 3.0).abs() < 1e-6);
        assert!((qk[3] - 4.0).abs() < 1e-6);
        assert!((qk[0] - 1.0).abs() > 1e-6 || (qk[1] - 2.0).abs() > 1e-6);
    }

    #[test]
    fn test_rope_table_matches_on_the_fly() {
        let table = RopeTable::new(64, 4, 4, 10000.0);

        let mut qk1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut qk2 = qk1.clone();

        apply_rope(&mut qk1, 2, 4, 7, 10000.0, 4);
        table.apply(&mut qk2, 2, 4, 7);

        for (a, b) in qk1.iter().zip(qk2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "table vs on-the-fly mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rope_table_pos_zero_identity() {
        let table = RopeTable::new(16, 4, 4, 10000.0);
        let mut qk = vec![1.0f32, 2.0, 3.0, 4.0];
        let original = qk.clone();
        table.apply(&mut qk, 1, 4, 0);
        for (a, b) in qk.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
