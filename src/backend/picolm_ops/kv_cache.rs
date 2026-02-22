//! KV cache for layer-streaming inference.
//!
//! Stores K and V as f16 (half-precision) to halve memory vs f32.
//! Conversion: f32→f16 on store, f16→f32 on read.
//!
//! Memory layout is position-major for cache-friendly sequential access
//! during attention score computation.
//!
//! # TEE Security
//!
//! All KV data is zeroized on drop to prevent cached conversation context
//! from persisting in TEE memory after session cleanup.

use half::f16;
use half::slice::HalfFloatSliceExt;

/// Fused dot product: compute dot(q_f32, k_f16) without intermediate f32 buffer.
/// This is the hot path in attention — called `n_heads × seq_len` times per token.
#[inline]
pub fn dot_f16_f32(f16_slice: &[f16], f32_slice: &[f32]) -> f32 {
    debug_assert_eq!(f16_slice.len(), f32_slice.len());
    let n = f16_slice.len();

    // Process in chunks of 8 to amortize f16→f32 conversion overhead.
    let chunks8 = n / 8;
    let mut sum = 0.0f32;
    let mut tmp = [0.0f32; 8];

    for i in 0..chunks8 {
        let off = i * 8;
        f16_slice[off..off + 8].convert_to_f32_slice(&mut tmp);
        sum += tmp[0] * f32_slice[off]
            + tmp[1] * f32_slice[off + 1]
            + tmp[2] * f32_slice[off + 2]
            + tmp[3] * f32_slice[off + 3]
            + tmp[4] * f32_slice[off + 4]
            + tmp[5] * f32_slice[off + 5]
            + tmp[6] * f32_slice[off + 6]
            + tmp[7] * f32_slice[off + 7];
    }
    for i in (chunks8 * 8)..n {
        sum += f16_slice[i].to_f32() * f32_slice[i];
    }
    sum
}

/// Fused accumulate: out[i] += scale * v_f16[i] without intermediate f32 buffer.
/// This is the hot path in attention V weighted sum.
#[inline]
pub fn accumulate_scaled_f16(out: &mut [f32], f16_slice: &[f16], scale: f32) {
    debug_assert_eq!(out.len(), f16_slice.len());
    let n = out.len();

    let chunks8 = n / 8;
    let mut tmp = [0.0f32; 8];

    for i in 0..chunks8 {
        let off = i * 8;
        f16_slice[off..off + 8].convert_to_f32_slice(&mut tmp);
        out[off] += scale * tmp[0];
        out[off + 1] += scale * tmp[1];
        out[off + 2] += scale * tmp[2];
        out[off + 3] += scale * tmp[3];
        out[off + 4] += scale * tmp[4];
        out[off + 5] += scale * tmp[5];
        out[off + 6] += scale * tmp[6];
        out[off + 7] += scale * tmp[7];
    }
    for i in (chunks8 * 8)..n {
        out[i] += scale * f16_slice[i].to_f32();
    }
}

/// Per-layer KV cache for one transformer layer.
pub struct LayerKvCache {
    /// K cache: flat `[max_seq_len × kv_dim]` stored as f16
    k: Vec<f16>,
    /// V cache: same shape as K, stored as f16
    v: Vec<f16>,
    /// Number of positions currently stored
    len: usize,
    /// Maximum sequence length (bounded)
    max_seq_len: usize,
    /// n_kv_heads * head_dim
    kv_dim: usize,
    #[allow(dead_code)]
    n_kv_heads: usize,
    head_dim: usize,
}

impl LayerKvCache {
    /// Create a new empty KV cache for one layer.
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        Self {
            k: vec![f16::ZERO; max_seq_len * kv_dim],
            v: vec![f16::ZERO; max_seq_len * kv_dim],
            len: 0,
            max_seq_len,
            kv_dim,
            n_kv_heads,
            head_dim,
        }
    }

    /// Store K and V vectors for the next position (f32 input → f16 stored).
    /// `k_vec` and `v_vec` must have length `kv_dim` (n_kv_heads * head_dim).
    ///
    /// If the cache is full, shifts the window: drops the oldest position.
    pub fn store(&mut self, k_vec: &[f32], v_vec: &[f32]) {
        debug_assert_eq!(k_vec.len(), self.kv_dim);
        debug_assert_eq!(v_vec.len(), self.kv_dim);

        if self.len >= self.max_seq_len {
            // Sliding window: shift everything left by one position
            let total = (self.max_seq_len - 1) * self.kv_dim;
            self.k.copy_within(self.kv_dim..self.kv_dim + total, 0);
            self.v.copy_within(self.kv_dim..self.kv_dim + total, 0);
            self.len = self.max_seq_len - 1;
        }

        let offset = self.len * self.kv_dim;
        self.k[offset..offset + self.kv_dim].convert_from_f32_slice(k_vec);
        self.v[offset..offset + self.kv_dim].convert_from_f32_slice(v_vec);
        self.len += 1;
    }

    /// Get K vector for a specific cached position and KV head (f16→f32).
    /// Writes into `out` which must have length `head_dim`.
    /// Uses `half` crate's batch conversion (SIMD-optimized on supported platforms).
    #[inline]
    pub fn k_at_into(&self, pos: usize, kv_head: usize, out: &mut [f32]) {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        self.k[offset..offset + self.head_dim].convert_to_f32_slice(out);
    }

    /// Get V vector for a specific cached position and KV head (f16→f32).
    /// Writes into `out` which must have length `head_dim`.
    /// Uses `half` crate's batch conversion (SIMD-optimized on supported platforms).
    #[inline]
    pub fn v_at_into(&self, pos: usize, kv_head: usize, out: &mut [f32]) {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        self.v[offset..offset + self.head_dim].convert_to_f32_slice(out);
    }

    /// Fused dot product: dot(q_f32, k_f16) without intermediate f32 buffer.
    /// Eliminates the kv_tmp write in the attention Q·K inner loop.
    #[inline]
    pub fn k_dot(&self, pos: usize, kv_head: usize, q: &[f32]) -> f32 {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        dot_f16_f32(&self.k[offset..offset + self.head_dim], q)
    }

    /// Fused accumulate: out[i] += scale * v_f16[i] without intermediate f32 buffer.
    /// Eliminates the kv_tmp write in the attention V weighted-sum inner loop.
    #[inline]
    pub fn v_accumulate(&self, pos: usize, kv_head: usize, out: &mut [f32], scale: f32) {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        accumulate_scaled_f16(out, &self.v[offset..offset + self.head_dim], scale);
    }

    /// Number of positions currently cached.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Truncate the cache to `new_len` positions.
    /// Used by speculative decoding to roll back rejected draft tokens.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.len);
        self.len = new_len;
    }

    /// Clear all cached positions and zeroize the data.
    pub fn clear(&mut self) {
        self.zeroize_data();
        self.len = 0;
    }

    /// Memory usage in bytes (f16 = 2 bytes/element).
    pub fn memory_bytes(&self) -> usize {
        self.k.len() * 2 + self.v.len() * 2
    }

    /// Zeroize all KV data in-place (overwrites with f16::ZERO).
    fn zeroize_data(&mut self) {
        self.k.fill(f16::ZERO);
        self.v.fill(f16::ZERO);
    }
}

/// Zeroize all cached K/V data on drop to prevent conversation context
/// from persisting in TEE memory after session cleanup.
impl Drop for LayerKvCache {
    fn drop(&mut self) {
        self.zeroize_data();
    }
}

/// Full KV cache for all layers.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    /// Create KV cache for all layers.
    pub fn new(n_layers: u32, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| LayerKvCache::new(n_kv_heads, head_dim, max_seq_len))
            .collect();
        Self { layers }
    }

    /// Get mutable reference to a specific layer's cache.
    #[inline]
    pub fn layer_mut(&mut self, layer: u32) -> &mut LayerKvCache {
        &mut self.layers[layer as usize]
    }

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Clear all cached positions across all layers and zeroize the data.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.zeroize_data();
            layer.len = 0;
        }
    }

    /// Number of cached positions (same across all layers).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.len()).unwrap_or(0)
    }

    /// Truncate all layers to `new_len` positions.
    /// Used by speculative decoding to roll back rejected draft tokens.
    pub fn truncate(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            layer.truncate(new_len);
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut cache = LayerKvCache::new(2, 4, 16); // 2 kv_heads, head_dim=4, max_seq=16
        let k = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // kv_dim=8
        let v = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        cache.store(&k, &v);

        assert_eq!(cache.len(), 1);

        let mut out = vec![0.0f32; 4];
        cache.k_at_into(0, 0, &mut out);
        for (got, exp) in out.iter().zip(&[1.0f32, 2.0, 3.0, 4.0]) {
            assert!((got - exp).abs() < 0.01, "k head0: {got} vs {exp}");
        }
        cache.k_at_into(0, 1, &mut out);
        for (got, exp) in out.iter().zip(&[5.0f32, 6.0, 7.0, 8.0]) {
            assert!((got - exp).abs() < 0.01, "k head1: {got} vs {exp}");
        }
        cache.v_at_into(0, 0, &mut out);
        for (got, exp) in out.iter().zip(&[0.1f32, 0.2, 0.3, 0.4]) {
            assert!((got - exp).abs() < 0.01, "v head0: {got} vs {exp}");
        }
    }

    #[test]
    fn test_multiple_positions() {
        let mut cache = LayerKvCache::new(1, 2, 16);
        cache.store(&[1.0f32, 2.0], &[3.0, 4.0]);
        cache.store(&[5.0f32, 6.0], &[7.0, 8.0]);

        assert_eq!(cache.len(), 2);
        let mut out = vec![0.0f32; 2];
        cache.k_at_into(0, 0, &mut out);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 2.0).abs() < 0.01);
        cache.k_at_into(1, 0, &mut out);
        assert!((out[0] - 5.0).abs() < 0.01);
        cache.v_at_into(1, 0, &mut out);
        assert!((out[0] - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_sliding_window() {
        let mut cache = LayerKvCache::new(1, 2, 3); // max_seq=3
        cache.store(&[1.0f32, 1.0], &[1.0, 1.0]); // pos 0
        cache.store(&[2.0f32, 2.0], &[2.0, 2.0]); // pos 1
        cache.store(&[3.0f32, 3.0], &[3.0, 3.0]); // pos 2 (full)
        assert_eq!(cache.len(), 3);

        cache.store(&[4.0f32, 4.0], &[4.0, 4.0]); // triggers shift
        assert_eq!(cache.len(), 3);

        let mut out = vec![0.0f32; 2];
        // Oldest (1.0) should be gone, now [2.0, 3.0, 4.0]
        cache.k_at_into(0, 0, &mut out);
        assert!((out[0] - 2.0).abs() < 0.01);
        cache.k_at_into(1, 0, &mut out);
        assert!((out[0] - 3.0).abs() < 0.01);
        cache.k_at_into(2, 0, &mut out);
        assert!((out[0] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut cache = LayerKvCache::new(1, 2, 16);
        cache.store(&[1.0f32, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_all_layers() {
        let mut kv = KvCache::new(4, 2, 4, 16);
        // f16: 2 bytes/element; 4 layers × 2 (k+v) × max_seq × kv_dim × 2
        assert_eq!(kv.memory_bytes(), 4 * 2 * (16 * 2 * 4 * 2));
        assert_eq!(kv.seq_len(), 0);

        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        kv.layer_mut(0).store(&k, &v);
        kv.layer_mut(1).store(&k, &v);
        assert_eq!(kv.layers[0].len(), 1);

        kv.clear();
        assert_eq!(kv.layers[0].len(), 0);
        assert_eq!(kv.layers[1].len(), 0);
    }

    #[test]
    fn test_memory_bytes() {
        let cache = LayerKvCache::new(8, 128, 2048);
        // f16: 2 bytes/element; 2 × max_seq × kv_dim × 2
        let expected = 2 * 2048 * (8 * 128) * 2;
        assert_eq!(cache.memory_bytes(), expected);
    }

    #[test]
    fn test_f16_precision_acceptable() {
        // f16 has ~3 decimal digits of precision; values used in attention
        // are typically in [-10, 10] range where f16 error is < 0.01.
        let mut cache = LayerKvCache::new(1, 1, 4);
        cache.store(&[3.14159f32], &[2.71828f32]);
        let mut out = vec![0.0f32; 1];
        cache.k_at_into(0, 0, &mut out);
        assert!((out[0] - 3.14159).abs() < 0.01);
        cache.v_at_into(0, 0, &mut out);
        assert!((out[0] - 2.71828).abs() < 0.01);
    }

    #[test]
    fn test_k_dot_fused() {
        // Fused dot product: dot(q, k_f16) should match manual k_at_into + dot.
        let mut cache = LayerKvCache::new(1, 4, 16);
        let k = vec![1.0f32, 2.0, 3.0, 4.0];
        let v = vec![0.0f32; 4];
        cache.store(&k, &v);

        let q = vec![1.0f32, 1.0, 1.0, 1.0];
        let dot = cache.k_dot(0, 0, &q);
        // 1+2+3+4 = 10.0 (with f16 precision)
        assert!(
            (dot - 10.0).abs() < 0.05,
            "k_dot: expected ~10.0, got {dot}"
        );
    }

    #[test]
    fn test_v_accumulate_fused() {
        // Fused accumulate: out += scale * v_f16 should match manual v_at_into + scale.
        let mut cache = LayerKvCache::new(1, 4, 16);
        let k = vec![0.0f32; 4];
        let v = vec![2.0f32, 4.0, 6.0, 8.0];
        cache.store(&k, &v);

        let mut out = vec![0.0f32; 4];
        cache.v_accumulate(0, 0, &mut out, 0.5);
        // out = 0.5 * [2, 4, 6, 8] = [1, 2, 3, 4]
        assert!((out[0] - 1.0).abs() < 0.01, "v_acc[0]: {}", out[0]);
        assert!((out[1] - 2.0).abs() < 0.01, "v_acc[1]: {}", out[1]);
        assert!((out[2] - 3.0).abs() < 0.01, "v_acc[2]: {}", out[2]);
        assert!((out[3] - 4.0).abs() < 0.01, "v_acc[3]: {}", out[3]);
    }

    #[test]
    fn test_fused_dot_matches_two_step() {
        // Verify fused k_dot matches k_at_into + manual dot product.
        let mut cache = LayerKvCache::new(2, 8, 16);
        let k: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let v = vec![0.0f32; 16];
        cache.store(&k, &v);

        let q: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();

        // Two-step: k_at_into + manual dot
        let mut k_buf = vec![0.0f32; 8];
        cache.k_at_into(0, 0, &mut k_buf);
        let expected: f32 = k_buf.iter().zip(q.iter()).map(|(a, b)| a * b).sum();

        // Fused
        let fused = cache.k_dot(0, 0, &q);
        assert!(
            (fused - expected).abs() < 0.01,
            "fused={fused}, expected={expected}"
        );
    }
}
