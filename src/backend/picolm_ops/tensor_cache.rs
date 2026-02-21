//! Per-layer tensor pointer cache — eliminates HashMap lookups from the hot path.
//!
//! At load time, every per-layer tensor name is resolved once into a raw pointer,
//! byte length, and GGML type. During inference the hot path indexes directly into
//! a flat Vec with no string hashing or HashMap overhead.
//!
//! # Layout (per layer, 10 slots)
//!
//! ```text
//! slot 0  blk.L.attn_q.weight
//! slot 1  blk.L.attn_k.weight
//! slot 2  blk.L.attn_v.weight
//! slot 3  blk.L.attn_output.weight
//! slot 4  blk.L.attn_q.bias      (optional — ptr=null if absent)
//! slot 5  blk.L.attn_k.bias      (optional)
//! slot 6  blk.L.attn_v.bias      (optional)
//! slot 7  blk.L.ffn_gate.weight
//! slot 8  blk.L.ffn_up.weight
//! slot 9  blk.L.ffn_down.weight
//! ```

use crate::backend::gguf_stream::GgufFile;
use crate::error::{PowerError, Result};

/// Number of tensor slots per layer.
pub const SLOTS_PER_LAYER: usize = 10;

pub const SLOT_ATTN_Q: usize = 0;
pub const SLOT_ATTN_K: usize = 1;
pub const SLOT_ATTN_V: usize = 2;
pub const SLOT_ATTN_O: usize = 3;
pub const SLOT_ATTN_Q_BIAS: usize = 4;
pub const SLOT_ATTN_K_BIAS: usize = 5;
pub const SLOT_ATTN_V_BIAS: usize = 6;
pub const SLOT_FFN_GATE: usize = 7;
pub const SLOT_FFN_UP: usize = 8;
pub const SLOT_FFN_DOWN: usize = 9;

/// A resolved tensor entry: raw pointer into the mmap, byte length, GGML type.
///
/// `ptr` is null for optional tensors that are absent from the model file.
#[derive(Clone, Copy)]
pub struct TensorEntry {
    ptr: *const u8,
    pub len: usize,
    pub ggml_type: u32,
}

// Safety: ptr points into a GgufFile mmap which is Send+Sync.
unsafe impl Send for TensorEntry {}
unsafe impl Sync for TensorEntry {}

impl TensorEntry {
    const ABSENT: Self = Self {
        ptr: std::ptr::null(),
        len: 0,
        ggml_type: 0,
    };

    /// Returns the tensor bytes, or `None` if this slot is absent.
    #[inline]
    pub fn bytes(&self) -> Option<&[u8]> {
        if self.ptr.is_null() {
            None
        } else {
            // Safety: ptr and len were validated against the mmap at build time.
            Some(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
        }
    }

    #[inline]
    pub fn is_present(&self) -> bool {
        !self.ptr.is_null()
    }
}

/// Flat cache of all per-layer tensor pointers, resolved once at load time.
pub struct TensorCache {
    /// `entries[layer * SLOTS_PER_LAYER + slot]`
    entries: Vec<TensorEntry>,
    n_layers: usize,
}

impl TensorCache {
    /// Build the cache by resolving every per-layer tensor name against `gguf`.
    ///
    /// Required tensors (Q/K/V/O projections, FFN gate/up/down) return an error
    /// if absent. Optional tensors (biases) are stored as null entries.
    pub fn build(gguf: &GgufFile, n_layers: u32) -> Result<Self> {
        let n = n_layers as usize;
        let mut entries = vec![TensorEntry::ABSENT; n * SLOTS_PER_LAYER];

        for layer in 0..n_layers {
            let base = layer as usize * SLOTS_PER_LAYER;

            // Required tensors
            let required: &[(usize, &str)] = &[
                (SLOT_ATTN_Q, "attn_q.weight"),
                (SLOT_ATTN_K, "attn_k.weight"),
                (SLOT_ATTN_V, "attn_v.weight"),
                (SLOT_ATTN_O, "attn_output.weight"),
                (SLOT_FFN_GATE, "ffn_gate.weight"),
                (SLOT_FFN_UP, "ffn_up.weight"),
                (SLOT_FFN_DOWN, "ffn_down.weight"),
            ];
            for &(slot, suffix) in required {
                let name = format!("blk.{layer}.{suffix}");
                let bytes = gguf.tensor_bytes(&name).map_err(|_| {
                    PowerError::InferenceFailed(format!("tensor_cache: missing {name}"))
                })?;
                let ggml_type = gguf.tensor_type(&name).unwrap();
                entries[base + slot] = TensorEntry {
                    ptr: bytes.as_ptr(),
                    len: bytes.len(),
                    ggml_type,
                };
            }

            // Optional bias tensors
            let optional: &[(usize, &str)] = &[
                (SLOT_ATTN_Q_BIAS, "attn_q.bias"),
                (SLOT_ATTN_K_BIAS, "attn_k.bias"),
                (SLOT_ATTN_V_BIAS, "attn_v.bias"),
            ];
            for &(slot, suffix) in optional {
                let name = format!("blk.{layer}.{suffix}");
                if let Ok(bytes) = gguf.tensor_bytes(&name) {
                    let ggml_type = gguf.tensor_type(&name).unwrap_or(0);
                    entries[base + slot] = TensorEntry {
                        ptr: bytes.as_ptr(),
                        len: bytes.len(),
                        ggml_type,
                    };
                }
                // else: stays TensorEntry::ABSENT
            }
        }

        Ok(Self {
            entries,
            n_layers: n,
        })
    }

    /// Get a tensor entry for a given layer and slot index.
    #[inline]
    pub fn get(&self, layer: u32, slot: usize) -> &TensorEntry {
        debug_assert!((layer as usize) < self.n_layers);
        debug_assert!(slot < SLOTS_PER_LAYER);
        &self.entries[layer as usize * SLOTS_PER_LAYER + slot]
    }

    /// Release the physical pages for all tensors in a layer via
    /// `madvise(MADV_DONTNEED)`.
    ///
    /// Call this after attention + FFN for `layer` completes. The OS will
    /// immediately free the backing pages; they will be re-faulted from disk
    /// on the next forward pass. This keeps peak RSS at O(layer_size).
    ///
    /// The norm weights (`attn_norm`, `ffn_norm`) are released separately via
    /// `GgufFile::advise_dontneed` in the forward pass.
    pub fn release_layer(&self, gguf: &GgufFile, layer: u32) -> Result<()> {
        let suffixes = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q.bias",
            "attn_k.bias",
            "attn_v.bias",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        for suffix in &suffixes {
            let name = format!("blk.{layer}.{suffix}");
            gguf.advise_dontneed(&name)?;
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_entry_absent_returns_none() {
        assert!(TensorEntry::ABSENT.bytes().is_none());
        assert!(!TensorEntry::ABSENT.is_present());
    }

    #[test]
    fn test_tensor_entry_present_returns_slice() {
        let data = [1u8, 2, 3, 4];
        let entry = TensorEntry {
            ptr: data.as_ptr(),
            len: 4,
            ggml_type: 0,
        };
        assert!(entry.is_present());
        assert_eq!(entry.bytes().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_slots_per_layer_constant() {
        assert_eq!(SLOTS_PER_LAYER, 10);
    }
}
