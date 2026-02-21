//! KV cache for layer-streaming inference.
//!
//! Stores K and V as f16 (half-precision) to halve memory vs f32.
//! Conversion: f32→f16 on store, f16→f32 on read.
//!
//! Memory layout is position-major for cache-friendly sequential access
//! during attention score computation.

use half::f16;

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
        for (i, (&kf, &vf)) in k_vec.iter().zip(v_vec.iter()).enumerate() {
            self.k[offset + i] = f16::from_f32(kf);
            self.v[offset + i] = f16::from_f32(vf);
        }
        self.len += 1;
    }

    /// Get K vector for a specific cached position and KV head (f16→f32).
    /// Writes into `out` which must have length `head_dim`.
    #[inline]
    pub fn k_at_into(&self, pos: usize, kv_head: usize, out: &mut [f32]) {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        for (o, &h) in out.iter_mut().zip(&self.k[offset..offset + self.head_dim]) {
            *o = h.to_f32();
        }
    }

    /// Get V vector for a specific cached position and KV head (f16→f32).
    /// Writes into `out` which must have length `head_dim`.
    #[inline]
    pub fn v_at_into(&self, pos: usize, kv_head: usize, out: &mut [f32]) {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        for (o, &h) in out.iter_mut().zip(&self.v[offset..offset + self.head_dim]) {
            *o = h.to_f32();
        }
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

    /// Clear all cached positions.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Memory usage in bytes (f16 = 2 bytes/element).
    pub fn memory_bytes(&self) -> usize {
        self.k.len() * 2 + self.v.len() * 2
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

    /// Clear all cached positions across all layers.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Number of cached positions (same across all layers).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.len()).unwrap_or(0)
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
}
