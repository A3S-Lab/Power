//! GGUF layer-streaming reader for TEE memory-constrained inference.
//!
//! Inspired by picolm's design: instead of loading all model weights into RAM,
//! this module memory-maps the GGUF file and provides an iterator that yields
//! one transformer layer at a time. Peak RAM stays at O(layer_size) rather
//! than O(model_size), enabling 7B+ models inside a 512MB TEE EPC budget.
//!
//! Supported: GGUF v2/v3, LLaMA-architecture models.
//! Quantization: Q4_K_M, Q4_0, Q8_0, F16, F32 (others skipped gracefully).

#[cfg(feature = "picolm")]
use std::collections::HashMap;
#[cfg(feature = "picolm")]
use std::path::Path;

#[cfg(all(feature = "picolm", unix))]
extern crate libc;

#[cfg(feature = "picolm")]
use memmap2::Mmap;

#[cfg(feature = "picolm")]
use crate::error::{PowerError, Result};

// Page size for madvise alignment (4 KiB on all supported platforms).
#[cfg(feature = "picolm")]
const PAGE_SIZE: usize = 4096;
#[cfg(feature = "picolm")]
const MAX_METADATA_KV_COUNT: u64 = 1_000_000;
#[cfg(feature = "picolm")]
const MAX_TENSOR_COUNT: u64 = 1_000_000;
#[cfg(feature = "picolm")]
const MAX_TENSOR_DIMS: u32 = 16;
#[cfg(feature = "picolm")]
const MAX_METADATA_STRING_BYTES: usize = 1_048_576;
#[cfg(feature = "picolm")]
const MAX_METADATA_ARRAY_ITEMS: usize = 1_000_000;

// ── GGUF constants ────────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
#[cfg(feature = "picolm")]
const GGUF_VERSION_MIN: u32 = 2;
#[cfg(feature = "picolm")]
const GGUF_VERSION_MAX: u32 = 3;

// GGUF metadata value types
#[cfg(feature = "picolm")]
const GGUF_TYPE_UINT8: u32 = 0;
#[cfg(feature = "picolm")]
const GGUF_TYPE_INT8: u32 = 1;
#[cfg(feature = "picolm")]
const GGUF_TYPE_UINT16: u32 = 2;
#[cfg(feature = "picolm")]
const GGUF_TYPE_INT16: u32 = 3;
#[cfg(feature = "picolm")]
const GGUF_TYPE_UINT32: u32 = 4;
#[cfg(feature = "picolm")]
const GGUF_TYPE_INT32: u32 = 5;
#[cfg(feature = "picolm")]
const GGUF_TYPE_FLOAT32: u32 = 6;
#[cfg(feature = "picolm")]
const GGUF_TYPE_BOOL: u32 = 7;
#[cfg(feature = "picolm")]
const GGUF_TYPE_STRING: u32 = 8;
#[cfg(feature = "picolm")]
const GGUF_TYPE_ARRAY: u32 = 9;
#[cfg(feature = "picolm")]
const GGUF_TYPE_UINT64: u32 = 10;
#[cfg(feature = "picolm")]
const GGUF_TYPE_INT64: u32 = 11;
#[cfg(feature = "picolm")]
const GGUF_TYPE_FLOAT64: u32 = 12;

// ── Model metadata ────────────────────────────────────────────────────────────

/// Metadata extracted from the GGUF header.
#[cfg(feature = "picolm")]
#[derive(Debug, Clone)]
pub struct GgufMeta {
    pub arch: String,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub context_length: u32,
    pub vocab_size: u32,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    /// FFN intermediate size (feed_forward_length).
    pub n_ff: u32,
    /// RMSNorm epsilon (default 1e-5).
    pub norm_eps: f32,
    /// RoPE base frequency (default 10000.0).
    pub rope_theta: f32,
    /// RoPE dimension count (None = full head_dim).
    pub rope_dim: Option<u32>,
    /// Chat template (Jinja2) from GGUF metadata, if present.
    pub chat_template: Option<String>,
    /// Tokenizer vocabulary strings.
    pub vocab_tokens: Vec<String>,
    /// Tokenizer merge priority scores.
    pub vocab_scores: Vec<f32>,
    /// Tokenizer token types (1=normal, 6=byte, etc.).
    pub vocab_types: Vec<i32>,
    /// Byte offset where tensor data begins in the file.
    pub tensor_data_offset: u64,
    /// Tensor descriptors: name → (offset, shape, ggml_type).
    pub tensors: HashMap<String, TensorDesc>,
}

/// Descriptor for a single tensor in the GGUF file.
#[cfg(feature = "picolm")]
#[derive(Debug, Clone)]
pub struct TensorDesc {
    /// Byte offset from `tensor_data_offset`.
    pub offset: u64,
    pub shape: Vec<u64>,
    /// GGML quantization type (0=F32, 1=F16, 2=Q4_0, ...).
    pub ggml_type: u32,
    /// Total number of elements.
    pub n_elements: u64,
}

// ── GGUF parser ───────────────────────────────────────────────────────────────

/// A memory-mapped GGUF file with parsed metadata.
///
/// The file is mmap'd read-only; no weights are copied into RAM at construction.
/// Use `layer_weights()` to stream individual layers on demand.
#[cfg(feature = "picolm")]
pub struct GgufFile {
    // Note: Debug is implemented manually below because raw pointers are not Debug.
    _mmap: Mmap,
    /// Raw pointer into the mmap for zero-copy reads.
    data: *const u8,
    data_len: usize,
    pub meta: GgufMeta,
}

#[cfg(feature = "picolm")]
// Safety: Mmap is Send+Sync; data pointer is valid for the lifetime of _mmap.
unsafe impl Send for GgufFile {}
#[cfg(feature = "picolm")]
unsafe impl Sync for GgufFile {}

#[cfg(feature = "picolm")]
impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("data_len", &self.data_len)
            .field("meta", &self.meta)
            .finish()
    }
}

#[cfg(feature = "picolm")]
impl GgufFile {
    /// Open and memory-map a GGUF file. Parses the header; does not load weights.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(PowerError::Io)?;
        // Safety: the file is opened read-only and we never mutate the mapping.
        let mmap = unsafe { Mmap::map(&file).map_err(PowerError::Io)? };
        let data = mmap.as_ptr();
        let data_len = mmap.len();

        let meta = parse_gguf_header(data, data_len)?;

        Ok(Self {
            _mmap: mmap,
            data,
            data_len,
            meta,
        })
    }

    /// Return the raw bytes for a named tensor (zero-copy slice into the mmap).
    ///
    /// The caller is responsible for dequantizing the bytes according to
    /// `TensorDesc::ggml_type`.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let desc = self.meta.tensors.get(name).ok_or_else(|| {
            PowerError::InvalidFormat(format!("Tensor '{name}' not found in GGUF file"))
        })?;

        let (start, byte_size) =
            checked_tensor_byte_range(name, self.meta.tensor_data_offset, self.data_len, desc)?;

        // Safety: start..end is within the mmap bounds verified above.
        Ok(unsafe { std::slice::from_raw_parts(self.data.add(start), byte_size) })
    }

    /// Return the GGML quantization type for a named tensor.
    pub fn tensor_type(&self, name: &str) -> Result<u32> {
        let desc = self.meta.tensors.get(name).ok_or_else(|| {
            PowerError::InvalidFormat(format!("Tensor '{name}' not found in GGUF file"))
        })?;
        Ok(desc.ggml_type)
    }

    /// Returns the names of all weight tensors for a given layer index.
    ///
    /// LLaMA naming convention: `blk.{layer}.{weight_name}`
    pub fn layer_tensor_names(&self, layer: u32) -> Vec<String> {
        let prefix = format!("blk.{layer}.");
        self.meta
            .tensors
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect()
    }

    /// Advise the OS that the physical pages backing a named tensor are no
    /// longer needed (`MADV_DONTNEED`).
    ///
    /// The mmap mapping stays intact — the virtual address range is still
    /// valid. The OS will free the physical pages immediately and re-fault
    /// them from disk on next access. This keeps peak RSS at O(layer_size)
    /// rather than O(model_size) during layer-streaming inference.
    ///
    /// The advice is applied to the page-aligned region covering the tensor.
    /// On macOS `MADV_DONTNEED` is equivalent to `MADV_FREE` (advisory only),
    /// but still reduces RSS in practice. On Linux it is a hard guarantee.
    ///
    /// Silently succeeds if the tensor is not found (non-fatal).
    pub fn advise_dontneed(&self, name: &str) -> Result<()> {
        let desc = match self.meta.tensors.get(name) {
            Some(d) => d,
            None => return Ok(()), // tensor absent — nothing to release
        };

        let (start_byte, byte_size) =
            checked_tensor_byte_range(name, self.meta.tensor_data_offset, self.data_len, desc)?;

        // Align down to page boundary.
        let aligned_start = start_byte & !(PAGE_SIZE - 1);
        let end_byte = start_byte.checked_add(byte_size).ok_or_else(|| {
            PowerError::InvalidFormat(format!("Tensor '{name}' byte range overflows usize"))
        })?;
        let aligned_end = checked_align_up(end_byte, PAGE_SIZE).ok_or_else(|| {
            PowerError::InvalidFormat(format!(
                "Tensor '{name}' aligned byte range overflows usize"
            ))
        })?;
        let aligned_len = aligned_end.checked_sub(aligned_start).ok_or_else(|| {
            PowerError::InvalidFormat(format!("Tensor '{name}' aligned byte range is invalid"))
        })?;

        if aligned_len == 0 || aligned_end > self.data_len {
            return Ok(());
        }

        // Safety: ptr is within the mmap; alignment and length are verified above.
        #[cfg(unix)]
        unsafe {
            let ptr = self.data.add(aligned_start) as *mut libc::c_void;
            if libc::madvise(ptr, aligned_len, libc::MADV_DONTNEED) != 0 {
                return Err(PowerError::Io(std::io::Error::other(format!(
                    "madvise(MADV_DONTNEED) failed for tensor {name}: {}",
                    std::io::Error::last_os_error()
                ))));
            }
        }

        Ok(())
    }
}

// ── GGUF header parser ────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
fn parse_gguf_header(data: *const u8, len: usize) -> Result<GgufMeta> {
    let mut cursor = 0usize;

    // Magic
    let magic = read_u32(data, len, &mut cursor)?;
    if magic != GGUF_MAGIC {
        return Err(PowerError::InvalidFormat(format!(
            "Not a GGUF file (magic={magic:#010x}, expected {GGUF_MAGIC:#010x})"
        )));
    }

    // Version
    let version = read_u32(data, len, &mut cursor)?;
    if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
        return Err(PowerError::InvalidFormat(format!(
            "Unsupported GGUF version {version} (supported: {GGUF_VERSION_MIN}–{GGUF_VERSION_MAX})"
        )));
    }

    // Tensor count and metadata KV count
    let n_tensors = read_u64(data, len, &mut cursor)?;
    let n_kv = read_u64(data, len, &mut cursor)?;
    if n_kv > MAX_METADATA_KV_COUNT {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: metadata count too large: {n_kv} entries"
        )));
    }
    if n_tensors > MAX_TENSOR_COUNT {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: tensor count too large: {n_tensors} tensors"
        )));
    }

    // Parse metadata key-value pairs
    let mut kv: HashMap<String, MetaValue> = HashMap::new();
    for _ in 0..n_kv {
        let key = read_gguf_string(data, len, &mut cursor)?;
        let val = read_meta_value(data, len, &mut cursor)?;
        kv.insert(key, val);
    }

    // Extract architecture
    let arch = optional_string_metadata(&kv, "general.architecture", "llama")?;

    let arch_prefix = arch.as_str();

    let n_layers = optional_u32_metadata(&kv, &format!("{arch_prefix}.block_count"), 32)?;

    let n_embd = optional_u32_metadata(&kv, &format!("{arch_prefix}.embedding_length"), 4096)?;

    let n_heads = optional_u32_metadata(&kv, &format!("{arch_prefix}.attention.head_count"), 32)?;

    let n_kv_heads = optional_u32_metadata(
        &kv,
        &format!("{arch_prefix}.attention.head_count_kv"),
        n_heads,
    )?;

    let context_length =
        optional_u32_metadata(&kv, &format!("{arch_prefix}.context_length"), 4096)?;

    let vocab_tokens = required_vocab_tokens(&kv)?;
    let vocab_size = u32::try_from(vocab_tokens.len()).map_err(|_| {
        PowerError::InvalidFormat("GGUF: tokenizer.ggml.tokens exceeds u32 length".to_string())
    })?;

    let bos_token_id = optional_i32_metadata(&kv, "tokenizer.ggml.bos_token_id", 1)?;
    let eos_token_id = optional_i32_metadata(&kv, "tokenizer.ggml.eos_token_id", 2)?;

    // FFN intermediate size
    let n_ff =
        match optional_u32_metadata_value(&kv, &format!("{arch_prefix}.feed_forward_length"))? {
            Some(n_ff) => n_ff,
            None => n_embd.checked_mul(4).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "GGUF: embedding_length is too large to derive feed_forward_length".to_string(),
                )
            })?,
        };

    // RMSNorm epsilon
    let norm_eps = optional_f32_metadata(
        &kv,
        &format!("{arch_prefix}.attention.layer_norm_rms_epsilon"),
        1e-5,
    )?;

    // RoPE base frequency
    let rope_theta = optional_f32_metadata(&kv, &format!("{arch_prefix}.rope.freq_base"), 10000.0)?;

    // RoPE dimension count (None = full head_dim)
    let rope_dim =
        optional_u32_metadata_value(&kv, &format!("{arch_prefix}.rope.dimension_count"))?;

    // Optional tokenizer metadata. Missing arrays are tolerated; the tokenizer
    // pads merge scores and treats missing token types as normal tokens.
    let vocab_scores = match kv.get("tokenizer.ggml.scores") {
        Some(value) => value.as_f32_array().ok_or_else(|| {
            PowerError::InvalidFormat(
                "GGUF: tokenizer.ggml.scores must be an array of finite numeric values".to_string(),
            )
        })?,
        None => Vec::new(),
    };

    let vocab_types = match kv.get("tokenizer.ggml.token_type") {
        Some(value) => value.as_i32_array().ok_or_else(|| {
            PowerError::InvalidFormat(
                "GGUF: tokenizer.ggml.token_type must be an array of i32-compatible integers"
                    .to_string(),
            )
        })?,
        None => Vec::new(),
    };

    // Chat template (Jinja2 format) — used for prompt formatting
    let chat_template = optional_string_metadata_value(&kv, "tokenizer.chat_template")?;

    // Parse tensor descriptors
    // Alignment: tensor data starts at next multiple of alignment after descriptors
    let alignment = optional_u32_metadata_value(&kv, "general.alignment")?
        .map(usize::try_from)
        .transpose()
        .map_err(|_| {
            PowerError::InvalidFormat("GGUF: general.alignment exceeds usize".to_string())
        })?
        .unwrap_or(32);
    if alignment != 0 && !alignment.is_power_of_two() {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: general.alignment must be zero or a power of two, got {alignment}"
        )));
    }

    let mut tensors = HashMap::new();

    for _ in 0..n_tensors {
        let name = read_gguf_string(data, len, &mut cursor)?;
        let n_dims = read_u32(data, len, &mut cursor)?;
        if n_dims > MAX_TENSOR_DIMS {
            return Err(PowerError::InvalidFormat(format!(
                "GGUF: tensor '{name}' has too many dimensions: {n_dims}"
            )));
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = read_u64(data, len, &mut cursor)?;
            shape.push(dim);
        }
        let n_elements = checked_tensor_element_count(&name, &shape)?;
        let ggml_type = read_u32(data, len, &mut cursor)?;
        checked_tensor_block_alignment(&name, ggml_type, &shape, n_elements)?;
        let offset = read_u64(data, len, &mut cursor)?;
        let byte_size = checked_ggml_type_size(ggml_type, n_elements)?;
        let byte_size_u64 = u64::try_from(byte_size).map_err(|_| {
            PowerError::InvalidFormat(format!("GGUF: tensor '{name}' byte size exceeds u64"))
        })?;
        offset.checked_add(byte_size_u64).ok_or_else(|| {
            PowerError::InvalidFormat(format!("GGUF: tensor '{name}' byte range overflows u64"))
        })?;

        // Use the offset from the file (relative to tensor_data_offset)
        tensors.insert(
            name,
            TensorDesc {
                offset,
                shape,
                ggml_type,
                n_elements,
            },
        );
    }

    // Tensor data starts at next aligned offset after the header
    let tensor_data_offset = checked_align_up(cursor, alignment).ok_or_else(|| {
        PowerError::InvalidFormat("GGUF: tensor data offset overflows usize".to_string())
    })? as u64;

    Ok(GgufMeta {
        arch,
        n_layers,
        n_embd,
        n_heads,
        n_kv_heads,
        context_length,
        vocab_size,
        bos_token_id,
        eos_token_id,
        n_ff,
        norm_eps,
        rope_theta,
        rope_dim,
        chat_template,
        vocab_tokens,
        vocab_scores,
        vocab_types,
        tensor_data_offset,
        tensors,
    })
}

#[cfg(feature = "picolm")]
fn required_vocab_tokens(kv: &HashMap<String, MetaValue>) -> Result<Vec<String>> {
    let value = kv.get("tokenizer.ggml.tokens").ok_or_else(|| {
        PowerError::InvalidFormat("GGUF: missing tokenizer.ggml.tokens".to_string())
    })?;

    let items = match value {
        MetaValue::Array(items) => items,
        _ => {
            return Err(PowerError::InvalidFormat(
                "GGUF: tokenizer.ggml.tokens must be an array of strings".to_string(),
            ));
        }
    };

    if items.is_empty() {
        return Err(PowerError::InvalidFormat(
            "GGUF: tokenizer.ggml.tokens is empty".to_string(),
        ));
    }

    let mut tokens = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item {
            MetaValue::Str(token) => tokens.push(token.clone()),
            _ => {
                return Err(PowerError::InvalidFormat(format!(
                    "GGUF: tokenizer.ggml.tokens[{index}] is not a string"
                )));
            }
        }
    }

    Ok(tokens)
}

#[cfg(feature = "picolm")]
fn optional_i32_metadata(kv: &HashMap<String, MetaValue>, key: &str, default: i32) -> Result<i32> {
    match kv.get(key) {
        Some(value) => value.as_i32().ok_or_else(|| {
            PowerError::InvalidFormat(format!("GGUF: {key} must be an i32-compatible integer"))
        }),
        None => Ok(default),
    }
}

#[cfg(feature = "picolm")]
fn optional_string_metadata(
    kv: &HashMap<String, MetaValue>,
    key: &str,
    default: &str,
) -> Result<String> {
    Ok(optional_string_metadata_value(kv, key)?.unwrap_or_else(|| default.to_string()))
}

#[cfg(feature = "picolm")]
fn optional_string_metadata_value(
    kv: &HashMap<String, MetaValue>,
    key: &str,
) -> Result<Option<String>> {
    match kv.get(key) {
        Some(value) => value
            .as_str()
            .map(|value| Some(value.to_string()))
            .ok_or_else(|| PowerError::InvalidFormat(format!("GGUF: {key} must be a string"))),
        None => Ok(None),
    }
}

#[cfg(feature = "picolm")]
fn optional_u32_metadata(kv: &HashMap<String, MetaValue>, key: &str, default: u32) -> Result<u32> {
    Ok(optional_u32_metadata_value(kv, key)?.unwrap_or(default))
}

#[cfg(feature = "picolm")]
fn optional_u32_metadata_value(kv: &HashMap<String, MetaValue>, key: &str) -> Result<Option<u32>> {
    match kv.get(key) {
        Some(value) => value.as_u32().map(Some).ok_or_else(|| {
            PowerError::InvalidFormat(format!("GGUF: {key} must be a u32-compatible integer"))
        }),
        None => Ok(None),
    }
}

#[cfg(feature = "picolm")]
fn optional_f32_metadata(kv: &HashMap<String, MetaValue>, key: &str, default: f32) -> Result<f32> {
    match kv.get(key) {
        Some(value) => {
            let value = value.as_f32().ok_or_else(|| {
                PowerError::InvalidFormat(format!("GGUF: {key} must be an f32-compatible number"))
            })?;
            if !value.is_finite() {
                return Err(PowerError::InvalidFormat(format!(
                    "GGUF: {key} must be finite"
                )));
            }
            Ok(value)
        }
        None => Ok(default),
    }
}

#[cfg(feature = "picolm")]
fn checked_tensor_element_count(tensor_name: &str, shape: &[u64]) -> Result<u64> {
    let mut n_elements = 1u64;
    for dim in shape {
        n_elements = n_elements.checked_mul(*dim).ok_or_else(|| {
            PowerError::InvalidFormat(format!(
                "GGUF: tensor '{tensor_name}' element count overflows u64"
            ))
        })?;
    }
    Ok(n_elements)
}

#[cfg(feature = "picolm")]
fn checked_ggml_type_size(ggml_type: u32, n_elements: u64) -> Result<usize> {
    let (block_size, bytes_per_block) = ggml_type_size_factor(ggml_type);
    if block_size > 1 && !n_elements.is_multiple_of(block_size) {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: tensor element count {n_elements} is not a multiple of block size {block_size} for type {ggml_type}"
        )));
    }
    let blocks = n_elements / block_size;
    let bytes = blocks.checked_mul(bytes_per_block).ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "GGUF: tensor byte size overflows u64 for type {ggml_type} with {n_elements} elements"
        ))
    })?;

    usize::try_from(bytes).map_err(|_| {
        PowerError::InvalidFormat(format!(
            "GGUF: tensor byte size exceeds usize for type {ggml_type} with {n_elements} elements"
        ))
    })
}

#[cfg(feature = "picolm")]
fn checked_tensor_block_alignment(
    tensor_name: &str,
    ggml_type: u32,
    shape: &[u64],
    n_elements: u64,
) -> Result<()> {
    let (block_size, _) = ggml_type_size_factor(ggml_type);
    if block_size == 1 {
        return Ok(());
    }

    let first_dimension = shape.first().copied().ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "GGUF: quantized tensor '{tensor_name}' has no dimensions for block size {block_size}"
        ))
    })?;
    if !first_dimension.is_multiple_of(block_size) {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: tensor '{tensor_name}' first dimension {first_dimension} is not a multiple of block size {block_size} for type {ggml_type}"
        )));
    }
    if !n_elements.is_multiple_of(block_size) {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: tensor '{tensor_name}' element count {n_elements} is not a multiple of block size {block_size} for type {ggml_type}"
        )));
    }

    Ok(())
}

#[cfg(feature = "picolm")]
fn checked_tensor_byte_range(
    tensor_name: &str,
    tensor_data_offset: u64,
    data_len: usize,
    desc: &TensorDesc,
) -> Result<(usize, usize)> {
    let byte_size = checked_ggml_type_size(desc.ggml_type, desc.n_elements)?;
    let start_u64 = tensor_data_offset.checked_add(desc.offset).ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "GGUF: tensor '{tensor_name}' start offset overflows u64"
        ))
    })?;
    let start = usize::try_from(start_u64).map_err(|_| {
        PowerError::InvalidFormat(format!(
            "GGUF: tensor '{tensor_name}' start offset exceeds usize"
        ))
    })?;
    let end = start.checked_add(byte_size).ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "GGUF: tensor '{tensor_name}' byte range overflows usize"
        ))
    })?;

    if end > data_len {
        return Err(PowerError::InvalidFormat(format!(
            "Tensor '{tensor_name}' extends beyond file bounds ({end} > {data_len})"
        )));
    }

    Ok((start, byte_size))
}

#[cfg(any(feature = "picolm", test))]
fn checked_align_up(offset: usize, alignment: usize) -> Option<usize> {
    if alignment == 0 {
        return Some(offset);
    }
    if !alignment.is_power_of_two() {
        return None;
    }
    offset
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

// ── Metadata value types ──────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
#[derive(Debug)]
#[allow(dead_code)] // variants are constructed during GGUF parsing but values not individually read
enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[cfg(feature = "picolm")]
impl MetaValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::U8(v) => Some((*v).into()),
            MetaValue::U16(v) => Some((*v).into()),
            MetaValue::U32(v) => Some(*v),
            MetaValue::U64(v) => u32::try_from(*v).ok(),
            MetaValue::I8(v) => u32::try_from(*v).ok(),
            MetaValue::I16(v) => u32::try_from(*v).ok(),
            MetaValue::I32(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            MetaValue::U32(v) => Some(*v as f32),
            MetaValue::I32(v) => Some(*v as f32),
            _ => None,
        }
    }

    fn as_i32(&self) -> Option<i32> {
        match self {
            MetaValue::I8(v) => Some((*v).into()),
            MetaValue::U8(v) => Some((*v).into()),
            MetaValue::I16(v) => Some((*v).into()),
            MetaValue::U16(v) => Some((*v).into()),
            MetaValue::I32(v) => Some(*v),
            MetaValue::U32(v) => i32::try_from(*v).ok(),
            MetaValue::I64(v) => i32::try_from(*v).ok(),
            MetaValue::U64(v) => i32::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    let value = item.as_f32()?;
                    if !value.is_finite() {
                        return None;
                    }
                    out.push(value);
                }
                Some(out)
            }
            _ => None,
        }
    }

    fn as_i32_array(&self) -> Option<Vec<i32>> {
        match self {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    out.push(item.as_i32()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

// ── Binary readers ────────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
fn checked_read_start(
    len: usize,
    cursor: &mut usize,
    byte_count: usize,
    context: &str,
) -> Result<usize> {
    let start = *cursor;
    let end = start.checked_add(byte_count).ok_or_else(|| {
        PowerError::InvalidFormat(format!("GGUF: {context} length overflows file bounds"))
    })?;
    if end > len {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: unexpected end of file ({context})"
        )));
    }
    *cursor = end;
    Ok(start)
}

#[cfg(feature = "picolm")]
fn read_u8(data: *const u8, len: usize, cursor: &mut usize) -> Result<u8> {
    let start = checked_read_start(len, cursor, 1, "u8")?;
    let v = unsafe { *data.add(start) };
    Ok(v)
}

#[cfg(feature = "picolm")]
fn read_u16(data: *const u8, len: usize, cursor: &mut usize) -> Result<u16> {
    let start = checked_read_start(len, cursor, 2, "u16")?;
    let mut buf = [0u8; 2];
    unsafe { std::ptr::copy_nonoverlapping(data.add(start), buf.as_mut_ptr(), 2) };
    Ok(u16::from_le_bytes(buf))
}

#[cfg(feature = "picolm")]
fn read_u32(data: *const u8, len: usize, cursor: &mut usize) -> Result<u32> {
    let start = checked_read_start(len, cursor, 4, "u32")?;
    let mut buf = [0u8; 4];
    unsafe { std::ptr::copy_nonoverlapping(data.add(start), buf.as_mut_ptr(), 4) };
    Ok(u32::from_le_bytes(buf))
}

#[cfg(feature = "picolm")]
fn read_u64(data: *const u8, len: usize, cursor: &mut usize) -> Result<u64> {
    let start = checked_read_start(len, cursor, 8, "u64")?;
    let mut buf = [0u8; 8];
    unsafe { std::ptr::copy_nonoverlapping(data.add(start), buf.as_mut_ptr(), 8) };
    Ok(u64::from_le_bytes(buf))
}

#[cfg(feature = "picolm")]
fn read_i8(data: *const u8, len: usize, cursor: &mut usize) -> Result<i8> {
    Ok(read_u8(data, len, cursor)? as i8)
}

#[cfg(feature = "picolm")]
fn read_i16(data: *const u8, len: usize, cursor: &mut usize) -> Result<i16> {
    Ok(read_u16(data, len, cursor)? as i16)
}

#[cfg(feature = "picolm")]
fn read_i32(data: *const u8, len: usize, cursor: &mut usize) -> Result<i32> {
    Ok(read_u32(data, len, cursor)? as i32)
}

#[cfg(feature = "picolm")]
fn read_i64(data: *const u8, len: usize, cursor: &mut usize) -> Result<i64> {
    Ok(read_u64(data, len, cursor)? as i64)
}

#[cfg(feature = "picolm")]
fn read_f32(data: *const u8, len: usize, cursor: &mut usize) -> Result<f32> {
    Ok(f32::from_le_bytes(
        read_u32(data, len, cursor)?.to_le_bytes(),
    ))
}

#[cfg(feature = "picolm")]
fn read_f64(data: *const u8, len: usize, cursor: &mut usize) -> Result<f64> {
    Ok(f64::from_le_bytes(
        read_u64(data, len, cursor)?.to_le_bytes(),
    ))
}

#[cfg(feature = "picolm")]
fn read_gguf_string(data: *const u8, len: usize, cursor: &mut usize) -> Result<String> {
    let str_len_u64 = read_u64(data, len, cursor)?;
    let str_len = usize::try_from(str_len_u64)
        .map_err(|_| PowerError::InvalidFormat("GGUF: string length exceeds usize".to_string()))?;
    if str_len > MAX_METADATA_STRING_BYTES {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: string too long: {str_len} bytes"
        )));
    }
    let start = checked_read_start(len, cursor, str_len, "string")?;
    let bytes = unsafe { std::slice::from_raw_parts(data.add(start), str_len) };
    String::from_utf8(bytes.to_vec())
        .map_err(|_| PowerError::InvalidFormat("GGUF: non-UTF8 string in metadata".to_string()))
}

#[cfg(feature = "picolm")]
fn read_meta_value(data: *const u8, len: usize, cursor: &mut usize) -> Result<MetaValue> {
    let type_id = read_u32(data, len, cursor)?;
    match type_id {
        GGUF_TYPE_UINT8 => Ok(MetaValue::U8(read_u8(data, len, cursor)?)),
        GGUF_TYPE_INT8 => Ok(MetaValue::I8(read_i8(data, len, cursor)?)),
        GGUF_TYPE_UINT16 => Ok(MetaValue::U16(read_u16(data, len, cursor)?)),
        GGUF_TYPE_INT16 => Ok(MetaValue::I16(read_i16(data, len, cursor)?)),
        GGUF_TYPE_UINT32 => Ok(MetaValue::U32(read_u32(data, len, cursor)?)),
        GGUF_TYPE_INT32 => Ok(MetaValue::I32(read_i32(data, len, cursor)?)),
        GGUF_TYPE_FLOAT32 => Ok(MetaValue::F32(read_f32(data, len, cursor)?)),
        GGUF_TYPE_BOOL => Ok(MetaValue::Bool(read_u8(data, len, cursor)? != 0)),
        GGUF_TYPE_STRING => Ok(MetaValue::Str(read_gguf_string(data, len, cursor)?)),
        GGUF_TYPE_UINT64 => Ok(MetaValue::U64(read_u64(data, len, cursor)?)),
        GGUF_TYPE_INT64 => Ok(MetaValue::I64(read_i64(data, len, cursor)?)),
        GGUF_TYPE_FLOAT64 => Ok(MetaValue::F64(read_f64(data, len, cursor)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(data, len, cursor)?;
            let count_u64 = read_u64(data, len, cursor)?;
            let count = usize::try_from(count_u64).map_err(|_| {
                PowerError::InvalidFormat("GGUF: metadata array length exceeds usize".to_string())
            })?;
            if count > MAX_METADATA_ARRAY_ITEMS {
                return Err(PowerError::InvalidFormat(format!(
                    "GGUF: metadata array too large: {count} items"
                )));
            }
            let mut items = Vec::with_capacity(count.min(65536));
            for _ in 0..count {
                // Push a dummy type_id then read the value
                let saved = *cursor;
                // We need to re-read with the elem_type, so we temporarily
                // write it back — instead, use a helper that takes type_id directly.
                let _ = saved;
                let item = read_meta_value_typed(data, len, cursor, elem_type)?;
                items.push(item);
            }
            Ok(MetaValue::Array(items))
        }
        other => Err(PowerError::InvalidFormat(format!(
            "GGUF: unknown metadata value type {other}"
        ))),
    }
}

/// Read a metadata value given an already-known type_id (used for array elements).
#[cfg(feature = "picolm")]
fn read_meta_value_typed(
    data: *const u8,
    len: usize,
    cursor: &mut usize,
    type_id: u32,
) -> Result<MetaValue> {
    match type_id {
        GGUF_TYPE_UINT8 => Ok(MetaValue::U8(read_u8(data, len, cursor)?)),
        GGUF_TYPE_INT8 => Ok(MetaValue::I8(read_i8(data, len, cursor)?)),
        GGUF_TYPE_UINT16 => Ok(MetaValue::U16(read_u16(data, len, cursor)?)),
        GGUF_TYPE_INT16 => Ok(MetaValue::I16(read_i16(data, len, cursor)?)),
        GGUF_TYPE_UINT32 => Ok(MetaValue::U32(read_u32(data, len, cursor)?)),
        GGUF_TYPE_INT32 => Ok(MetaValue::I32(read_i32(data, len, cursor)?)),
        GGUF_TYPE_FLOAT32 => Ok(MetaValue::F32(read_f32(data, len, cursor)?)),
        GGUF_TYPE_BOOL => Ok(MetaValue::Bool(read_u8(data, len, cursor)? != 0)),
        GGUF_TYPE_STRING => Ok(MetaValue::Str(read_gguf_string(data, len, cursor)?)),
        GGUF_TYPE_UINT64 => Ok(MetaValue::U64(read_u64(data, len, cursor)?)),
        GGUF_TYPE_INT64 => Ok(MetaValue::I64(read_i64(data, len, cursor)?)),
        GGUF_TYPE_FLOAT64 => Ok(MetaValue::F64(read_f64(data, len, cursor)?)),
        other => Err(PowerError::InvalidFormat(format!(
            "GGUF: unsupported array element type {other}"
        ))),
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Returns the byte size of a tensor given its GGML type and element count.
///
/// GGML quantization block sizes:
/// - F32 (0): 4 bytes/element
/// - F16 (1): 2 bytes/element
/// - Q4_0 (2): 18 bytes per 32 elements (2 + 16)
/// - Q4_1 (3): 20 bytes per 32 elements (4 + 16)
/// - Q5_0 (6): 22 bytes per 32 elements (2 + 4 + 16)
/// - Q5_1 (7): 24 bytes per 32 elements (4 + 4 + 16)
/// - Q8_0 (8): 34 bytes per 32 elements (2 + 32)
/// - Q2_K (10): 84 bytes per 256 elements
/// - Q3_K (11): 110 bytes per 256 elements
/// - Q4_K (12): 144 bytes per 256 elements
/// - Q5_K (13): 176 bytes per 256 elements
/// - Q6_K (14): 210 bytes per 256 elements
pub fn ggml_type_size(ggml_type: u32, n_elements: u64) -> usize {
    let (block_size, bytes_per_block) = ggml_type_size_factor(ggml_type);
    let blocks = n_elements / block_size;
    blocks
        .checked_mul(bytes_per_block)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .unwrap_or(usize::MAX)
}

fn ggml_type_size_factor(ggml_type: u32) -> (u64, u64) {
    match ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        _ => (1, 4),      // fallback: F32
    }
}

#[cfg(test)]
fn align_up(offset: usize, alignment: usize) -> usize {
    checked_align_up(offset, alignment).unwrap_or(usize::MAX)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_size_f32() {
        assert_eq!(ggml_type_size(0, 1024), 4096);
    }

    #[test]
    fn test_ggml_type_size_f16() {
        assert_eq!(ggml_type_size(1, 1024), 2048);
    }

    #[test]
    fn test_ggml_type_size_q4_0() {
        // 32 elements → 18 bytes
        assert_eq!(ggml_type_size(2, 32), 18);
        assert_eq!(ggml_type_size(2, 4096), 4096 / 32 * 18);
    }

    #[test]
    fn test_ggml_type_size_q8_0() {
        // 32 elements → 34 bytes
        assert_eq!(ggml_type_size(8, 32), 34);
    }

    #[test]
    fn test_ggml_type_size_q4_k() {
        // 256 elements → 144 bytes
        assert_eq!(ggml_type_size(12, 256), 144);
    }

    #[test]
    fn test_ggml_type_size_saturates_on_overflow() {
        assert_eq!(ggml_type_size(0, u64::MAX), usize::MAX);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 32), 128);
    }

    #[test]
    fn test_align_up_zero_alignment() {
        assert_eq!(align_up(42, 0), 42);
    }

    #[test]
    fn test_align_up_saturates_on_overflow() {
        assert_eq!(align_up(usize::MAX, 32), usize::MAX);
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_read_u32_rejects_cursor_overflow() {
        let bytes = [0u8; 8];
        let mut cursor = usize::MAX;

        let err = read_u32(bytes.as_ptr(), bytes.len(), &mut cursor).unwrap_err();

        assert!(err.to_string().contains("u32 length overflows"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_read_gguf_string_rejects_length_overflow() {
        let bytes = u64::MAX.to_le_bytes();
        let mut cursor = 0;

        let err = read_gguf_string(bytes.as_ptr(), bytes.len(), &mut cursor).unwrap_err();

        assert!(err.to_string().contains("string"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_read_meta_array_rejects_huge_truncated_array_without_large_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        bytes.extend_from_slice(&GGUF_TYPE_UINT8.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut cursor = 0;

        let err = read_meta_value(bytes.as_ptr(), bytes.len(), &mut cursor).unwrap_err();

        assert!(
            err.to_string()
                .contains("metadata array length exceeds usize")
                || err.to_string().contains("metadata array too large")
                || err.to_string().contains("unexpected end of file (u8)")
        );
    }

    #[cfg(feature = "picolm")]
    fn write_gguf_string(buf: &mut Vec<u8>, value: &str) {
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
    }

    #[cfg(feature = "picolm")]
    fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        write_gguf_string(buf, value);
    }

    #[cfg(feature = "picolm")]
    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    #[cfg(feature = "picolm")]
    fn write_kv_i32(buf: &mut Vec<u8>, key: &str, value: i32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    #[cfg(feature = "picolm")]
    fn write_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    #[cfg(feature = "picolm")]
    fn write_kv_f64(buf: &mut Vec<u8>, key: &str, value: f64) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_FLOAT64.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    #[cfg(feature = "picolm")]
    fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            write_gguf_string(buf, value);
        }
    }

    #[cfg(feature = "picolm")]
    fn write_kv_u32_array(buf: &mut Vec<u8>, key: &str, values: &[u32]) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    #[cfg(feature = "picolm")]
    fn write_tensor_desc(
        buf: &mut Vec<u8>,
        name: &str,
        shape: &[u64],
        ggml_type: u32,
        offset: u64,
    ) {
        write_gguf_string(buf, name);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&ggml_type.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    #[cfg(feature = "picolm")]
    fn base_meta() -> (Vec<u8>, u64) {
        let mut meta = Vec::new();
        let mut n_kv = 0;

        write_kv_string(&mut meta, "general.architecture", "llama");
        n_kv += 1;
        write_kv_string_array(&mut meta, "tokenizer.ggml.tokens", &["<s>", "</s>"]);
        n_kv += 1;

        (meta, n_kv)
    }

    #[cfg(feature = "picolm")]
    fn parse_test_header(
        meta: &[u8],
        n_kv: u64,
        tensors: &[u8],
        n_tensors: u64,
    ) -> Result<GgufMeta> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&GGUF_VERSION_MAX.to_le_bytes());
        bytes.extend_from_slice(&n_tensors.to_le_bytes());
        bytes.extend_from_slice(&n_kv.to_le_bytes());
        bytes.extend_from_slice(meta);
        bytes.extend_from_slice(tensors);

        parse_gguf_header(bytes.as_ptr(), bytes.len())
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_huge_metadata_count() {
        let err = parse_test_header(&[], MAX_METADATA_KV_COUNT + 1, &[], 0).unwrap_err();

        assert!(err.to_string().contains("metadata count too large"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_huge_tensor_count() {
        let err = parse_test_header(&[], 0, &[], MAX_TENSOR_COUNT + 1).unwrap_err();

        assert!(err.to_string().contains("tensor count too large"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_oversized_metadata_string() {
        let mut meta = Vec::new();
        meta.extend_from_slice(&((MAX_METADATA_STRING_BYTES as u64) + 1).to_le_bytes());

        let err = parse_test_header(&meta, 1, &[], 0).unwrap_err();

        assert!(err.to_string().contains("string too long"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_oversized_metadata_array() {
        let mut meta = Vec::new();
        write_gguf_string(&mut meta, "test.array");
        meta.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        meta.extend_from_slice(&GGUF_TYPE_UINT8.to_le_bytes());
        meta.extend_from_slice(&((MAX_METADATA_ARRAY_ITEMS as u64) + 1).to_le_bytes());

        let err = parse_test_header(&meta, 1, &[], 0).unwrap_err();

        assert!(err.to_string().contains("metadata array too large"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_too_many_tensor_dimensions() {
        let (meta, n_kv) = base_meta();
        let shape = vec![1u64; (MAX_TENSOR_DIMS + 1) as usize];
        let mut tensors = Vec::new();
        write_tensor_desc(&mut tensors, "overshaped.weight", &shape, 0, 0);

        let err = parse_test_header(&meta, n_kv, &tensors, 1).unwrap_err();

        assert!(err.to_string().contains("too many dimensions"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_meta_value_as_u32_rejects_out_of_range_values() {
        assert_eq!(MetaValue::U64((u32::MAX as u64) + 1).as_u32(), None);
        assert_eq!(MetaValue::I32(-1).as_u32(), None);
        assert_eq!(MetaValue::U16(7).as_u32(), Some(7));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_meta_value_as_i32_array_rejects_wrapping_u32() {
        let value = MetaValue::Array(vec![MetaValue::U32((i32::MAX as u32) + 1)]);

        assert_eq!(value.as_i32_array(), None);
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_out_of_range_token_type_array() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_u32_array(
            &mut meta,
            "tokenizer.ggml.token_type",
            &[(i32::MAX as u32) + 1],
        );
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("tokenizer.ggml.token_type"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_out_of_range_bos_token_id() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_u32(
            &mut meta,
            "tokenizer.ggml.bos_token_id",
            (i32::MAX as u32) + 1,
        );
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("tokenizer.ggml.bos_token_id"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_out_of_range_eos_token_id() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_u32(
            &mut meta,
            "tokenizer.ggml.eos_token_id",
            (i32::MAX as u32) + 1,
        );
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("tokenizer.ggml.eos_token_id"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_non_finite_norm_epsilon() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_f32(
            &mut meta,
            "llama.attention.layer_norm_rms_epsilon",
            f32::NAN,
        );
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err
            .to_string()
            .contains("llama.attention.layer_norm_rms_epsilon"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_rope_freq_base_exceeding_f32() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_f64(&mut meta, "llama.rope.freq_base", f64::MAX);
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("llama.rope.freq_base"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_negative_embedding_length() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_i32(&mut meta, "llama.embedding_length", -1);
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("llama.embedding_length"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_invalid_alignment_type() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_i32(&mut meta, "general.alignment", -1);
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("general.alignment"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_invalid_architecture_type() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_u32(&mut meta, "general.architecture", 42);
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("general.architecture"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_invalid_chat_template_type() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_u32(&mut meta, "tokenizer.chat_template", 42);
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("tokenizer.chat_template"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_feed_forward_default_overflow() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_u32(&mut meta, "llama.embedding_length", u32::MAX);
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("embedding_length is too large"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_tensor_element_count_overflow() {
        let (meta, n_kv) = base_meta();
        let mut tensors = Vec::new();
        write_tensor_desc(&mut tensors, "overflow.weight", &[u64::MAX, 2], 0, 0);

        let err = parse_test_header(&meta, n_kv, &tensors, 1).unwrap_err();

        assert!(err.to_string().contains("element count overflows"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_tensor_byte_size_overflow() {
        let (meta, n_kv) = base_meta();
        let mut tensors = Vec::new();
        write_tensor_desc(&mut tensors, "huge.weight", &[u64::MAX], 0, 0);

        let err = parse_test_header(&meta, n_kv, &tensors, 1).unwrap_err();

        assert!(err.to_string().contains("tensor byte size overflows"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_quantized_tensor_unaligned_first_dimension() {
        let (meta, n_kv) = base_meta();
        let mut tensors = Vec::new();
        write_tensor_desc(&mut tensors, "unaligned.weight", &[31], 2, 0);

        let err = parse_test_header(&meta, n_kv, &tensors, 1).unwrap_err();

        assert!(err.to_string().contains("block size"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_tensor_offset_overflow() {
        let (meta, n_kv) = base_meta();
        let mut tensors = Vec::new();
        write_tensor_desc(&mut tensors, "offset.weight", &[1], 0, u64::MAX);

        let err = parse_test_header(&meta, n_kv, &tensors, 1).unwrap_err();

        assert!(err.to_string().contains("byte range overflows u64"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_parse_rejects_non_power_of_two_alignment() {
        let (mut meta, mut n_kv) = base_meta();
        write_kv_u32(&mut meta, "general.alignment", 3);
        n_kv += 1;

        let err = parse_test_header(&meta, n_kv, &[], 0).unwrap_err();

        assert!(err.to_string().contains("general.alignment"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_checked_tensor_byte_range_rejects_out_of_bounds_tensor() {
        let desc = TensorDesc {
            offset: 8,
            shape: vec![1],
            ggml_type: 0,
            n_elements: 1,
        };

        let err = checked_tensor_byte_range("tiny.weight", 0, 10, &desc).unwrap_err();

        assert!(err.to_string().contains("extends beyond file bounds"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_required_vocab_tokens_accepts_string_array() {
        let mut kv = HashMap::new();
        kv.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array(vec![
                MetaValue::Str("<s>".to_string()),
                MetaValue::Str("</s>".to_string()),
            ]),
        );

        let tokens = required_vocab_tokens(&kv).unwrap();

        assert_eq!(tokens, vec!["<s>".to_string(), "</s>".to_string()]);
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_required_vocab_tokens_rejects_missing_tokens() {
        let kv = HashMap::new();

        let err = required_vocab_tokens(&kv).unwrap_err();

        assert!(err.to_string().contains("missing tokenizer.ggml.tokens"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_required_vocab_tokens_rejects_empty_tokens() {
        let mut kv = HashMap::new();
        kv.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array(vec![]),
        );

        let err = required_vocab_tokens(&kv).unwrap_err();

        assert!(err.to_string().contains("tokens is empty"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_required_vocab_tokens_rejects_non_string_token() {
        let mut kv = HashMap::new();
        kv.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array(vec![MetaValue::Str("<s>".to_string()), MetaValue::U32(42)]),
        );

        let err = required_vocab_tokens(&kv).unwrap_err();

        assert!(err.to_string().contains("tokens[1] is not a string"));
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_open_nonexistent_file_fails() {
        let result = GgufFile::open(std::path::Path::new("/nonexistent/model.gguf"));
        assert!(result.is_err());
    }

    #[cfg(feature = "picolm")]
    #[test]
    fn test_open_invalid_magic_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        // Write 32 bytes of zeros — wrong magic
        std::fs::write(&path, vec![0u8; 32]).unwrap();
        let result = GgufFile::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GGUF"));
    }
}
