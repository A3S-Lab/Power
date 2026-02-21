//! KV cache for layer-streaming inference.
//!
//! The KV cache is the only large persistent allocation during layer-streaming.
//! Weights are streamed from mmap and dropped; the KV cache persists across
//! all generation steps.
//!
//! Memory layout is position-major for cache-friendly sequential access
//! during attention score computation.

/// Per-layer KV cache for one transformer layer.
pub struct LayerKvCache {
    /// K cache: flat `[max_seq_len × kv_dim]` where `kv_dim = n_kv_heads * head_dim`
    k: Vec<f32>,
    /// V cache: same shape as K
    v: Vec<f32>,
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
            k: vec![0.0; max_seq_len * kv_dim],
            v: vec![0.0; max_seq_len * kv_dim],
            len: 0,
            max_seq_len,
            kv_dim,
            n_kv_heads,
            head_dim,
        }
    }

    /// Store K and V vectors for the next position.
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
        self.k[offset..offset + self.kv_dim].copy_from_slice(k_vec);
        self.v[offset..offset + self.kv_dim].copy_from_slice(v_vec);
        self.len += 1;
    }

    /// Get K vector for a specific cached position and KV head.
    /// Returns a slice of length `head_dim`.
    #[inline]
    pub fn k_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        &self.k[offset..offset + self.head_dim]
    }

    /// Get V vector for a specific cached position and KV head.
    #[inline]
    pub fn v_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let offset = pos * self.kv_dim + kv_head * self.head_dim;
        &self.v[offset..offset + self.head_dim]
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

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.k.len() * 4 + self.v.len() * 4
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
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // kv_dim=8
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        cache.store(&k, &v);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.k_at(0, 0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.k_at(0, 1), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.v_at(0, 0), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(cache.v_at(0, 1), &[0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn test_multiple_positions() {
        let mut cache = LayerKvCache::new(1, 2, 16);
        cache.store(&[1.0, 2.0], &[3.0, 4.0]);
        cache.store(&[5.0, 6.0], &[7.0, 8.0]);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.k_at(0, 0), &[1.0, 2.0]);
        assert_eq!(cache.k_at(1, 0), &[5.0, 6.0]);
        assert_eq!(cache.v_at(1, 0), &[7.0, 8.0]);
    }

    #[test]
    fn test_sliding_window() {
        let mut cache = LayerKvCache::new(1, 2, 3); // max_seq=3
        cache.store(&[1.0, 1.0], &[1.0, 1.0]); // pos 0
        cache.store(&[2.0, 2.0], &[2.0, 2.0]); // pos 1
        cache.store(&[3.0, 3.0], &[3.0, 3.0]); // pos 2 (full)
        assert_eq!(cache.len(), 3);

        cache.store(&[4.0, 4.0], &[4.0, 4.0]); // triggers shift
        assert_eq!(cache.len(), 3);
        // Oldest (1.0) should be gone, now [2.0, 3.0, 4.0]
        assert_eq!(cache.k_at(0, 0), &[2.0, 2.0]);
        assert_eq!(cache.k_at(1, 0), &[3.0, 3.0]);
        assert_eq!(cache.k_at(2, 0), &[4.0, 4.0]);
    }

    #[test]
    fn test_clear() {
        let mut cache = LayerKvCache::new(1, 2, 16);
        cache.store(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_all_layers() {
        let mut kv = KvCache::new(4, 2, 4, 16);
        assert_eq!(kv.memory_bytes(), 4 * 2 * (16 * 2 * 4 * 4)); // 4 layers × 2 (k+v) × max_seq × kv_dim × sizeof(f32)
        assert_eq!(kv.seq_len(), 0);

        let k = vec![1.0; 8];
        let v = vec![2.0; 8];
        kv.layer_mut(0).store(&k, &v);
        kv.layer_mut(1).store(&k, &v);
        // seq_len reports from first layer
        assert_eq!(kv.layers[0].len(), 1);

        kv.clear();
        assert_eq!(kv.layers[0].len(), 0);
        assert_eq!(kv.layers[1].len(), 0);
    }

    #[test]
    fn test_memory_bytes() {
        let cache = LayerKvCache::new(8, 128, 2048);
        // 2 × max_seq × kv_dim × 4 bytes
        let expected = 2 * 2048 * (8 * 128) * 4;
        assert_eq!(cache.memory_bytes(), expected);
    }
}
