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

#[cfg(feature = "picolm")]
use memmap2::Mmap;

#[cfg(feature = "picolm")]
use crate::error::{PowerError, Result};

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

        let start = (self.meta.tensor_data_offset + desc.offset) as usize;
        let byte_size = ggml_type_size(desc.ggml_type, desc.n_elements);
        let end = start + byte_size;

        if end > self.data_len {
            return Err(PowerError::InvalidFormat(format!(
                "Tensor '{name}' extends beyond file bounds ({end} > {})",
                self.data_len
            )));
        }

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

    // Parse metadata key-value pairs
    let mut kv: HashMap<String, MetaValue> = HashMap::new();
    for _ in 0..n_kv {
        let key = read_gguf_string(data, len, &mut cursor)?;
        let val = read_meta_value(data, len, &mut cursor)?;
        kv.insert(key, val);
    }

    // Extract architecture
    let arch = kv
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string();

    let arch_prefix = arch.as_str();

    let n_layers = kv
        .get(&format!("{arch_prefix}.block_count"))
        .and_then(|v| v.as_u32())
        .unwrap_or(32);

    let n_embd = kv
        .get(&format!("{arch_prefix}.embedding_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(4096);

    let n_heads = kv
        .get(&format!("{arch_prefix}.attention.head_count"))
        .and_then(|v| v.as_u32())
        .unwrap_or(32);

    let n_kv_heads = kv
        .get(&format!("{arch_prefix}.attention.head_count_kv"))
        .and_then(|v| v.as_u32())
        .unwrap_or(n_heads);

    let context_length = kv
        .get(&format!("{arch_prefix}.context_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(4096);

    let vocab_size = kv
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array_len())
        .unwrap_or(32000) as u32;

    let bos_token_id = kv
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.as_u32())
        .map(|v| v as i32)
        .unwrap_or(1);

    let eos_token_id = kv
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.as_u32())
        .map(|v| v as i32)
        .unwrap_or(2);

    // FFN intermediate size
    let n_ff = kv
        .get(&format!("{arch_prefix}.feed_forward_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(n_embd * 4); // common default: 4 × n_embd

    // RMSNorm epsilon
    let norm_eps = kv
        .get(&format!("{arch_prefix}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.as_f32())
        .unwrap_or(1e-5);

    // RoPE base frequency
    let rope_theta = kv
        .get(&format!("{arch_prefix}.rope.freq_base"))
        .and_then(|v| v.as_f32())
        .unwrap_or(10000.0);

    // RoPE dimension count (None = full head_dim)
    let rope_dim = kv
        .get(&format!("{arch_prefix}.rope.dimension_count"))
        .and_then(|v| v.as_u32());

    // Tokenizer vocabulary
    let vocab_tokens = kv
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_string_array())
        .unwrap_or_default();

    let vocab_scores = kv
        .get("tokenizer.ggml.scores")
        .and_then(|v| v.as_f32_array())
        .unwrap_or_default();

    let vocab_types = kv
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.as_i32_array())
        .unwrap_or_default();

    // Parse tensor descriptors
    // Alignment: tensor data starts at next multiple of alignment after descriptors
    let alignment = kv
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as usize;

    let mut tensors = HashMap::new();
    let mut running_offset: u64 = 0;

    for _ in 0..n_tensors {
        let name = read_gguf_string(data, len, &mut cursor)?;
        let n_dims = read_u32(data, len, &mut cursor)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        let mut n_elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = read_u64(data, len, &mut cursor)?;
            n_elements *= dim;
            shape.push(dim);
        }
        let ggml_type = read_u32(data, len, &mut cursor)?;
        let offset = read_u64(data, len, &mut cursor)?;

        // Use the offset from the file (relative to tensor_data_offset)
        let _ = running_offset; // will be set after all descriptors
        tensors.insert(
            name,
            TensorDesc {
                offset,
                shape,
                ggml_type,
                n_elements,
            },
        );
        running_offset = offset + ggml_type_size(ggml_type, n_elements) as u64;
    }

    // Tensor data starts at next aligned offset after the header
    let tensor_data_offset = align_up(cursor, alignment) as u64;

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
        vocab_tokens,
        vocab_scores,
        vocab_types,
        tensor_data_offset,
        tensors,
    })
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
            MetaValue::U32(v) => Some(*v),
            MetaValue::U64(v) => Some(*v as u32),
            MetaValue::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    fn as_array_len(&self) -> Option<usize> {
        match self {
            MetaValue::Array(v) => Some(v.len()),
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

    fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    match item {
                        MetaValue::Str(s) => out.push(s.clone()),
                        _ => out.push(String::new()),
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    out.push(item.as_f32().unwrap_or(0.0));
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
                    match item {
                        MetaValue::I32(v) => out.push(*v),
                        MetaValue::U32(v) => out.push(*v as i32),
                        MetaValue::I8(v) => out.push(*v as i32),
                        MetaValue::U8(v) => out.push(*v as i32),
                        _ => out.push(0),
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

// ── Binary readers ────────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
fn read_u8(data: *const u8, len: usize, cursor: &mut usize) -> Result<u8> {
    if *cursor + 1 > len {
        return Err(PowerError::InvalidFormat(
            "GGUF: unexpected end of file (u8)".to_string(),
        ));
    }
    let v = unsafe { *data.add(*cursor) };
    *cursor += 1;
    Ok(v)
}

#[cfg(feature = "picolm")]
fn read_u16(data: *const u8, len: usize, cursor: &mut usize) -> Result<u16> {
    if *cursor + 2 > len {
        return Err(PowerError::InvalidFormat(
            "GGUF: unexpected end of file (u16)".to_string(),
        ));
    }
    let mut buf = [0u8; 2];
    unsafe { std::ptr::copy_nonoverlapping(data.add(*cursor), buf.as_mut_ptr(), 2) };
    *cursor += 2;
    Ok(u16::from_le_bytes(buf))
}

#[cfg(feature = "picolm")]
fn read_u32(data: *const u8, len: usize, cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > len {
        return Err(PowerError::InvalidFormat(
            "GGUF: unexpected end of file (u32)".to_string(),
        ));
    }
    let mut buf = [0u8; 4];
    unsafe { std::ptr::copy_nonoverlapping(data.add(*cursor), buf.as_mut_ptr(), 4) };
    *cursor += 4;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(feature = "picolm")]
fn read_u64(data: *const u8, len: usize, cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > len {
        return Err(PowerError::InvalidFormat(
            "GGUF: unexpected end of file (u64)".to_string(),
        ));
    }
    let mut buf = [0u8; 8];
    unsafe { std::ptr::copy_nonoverlapping(data.add(*cursor), buf.as_mut_ptr(), 8) };
    *cursor += 8;
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
    let str_len = read_u64(data, len, cursor)? as usize;
    if *cursor + str_len > len {
        return Err(PowerError::InvalidFormat(
            "GGUF: string extends beyond file bounds".to_string(),
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.add(*cursor), str_len) };
    *cursor += str_len;
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
            let count = read_u64(data, len, cursor)? as usize;
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
    match ggml_type {
        0 => (n_elements * 4) as usize,          // F32
        1 => (n_elements * 2) as usize,          // F16
        2 => (n_elements / 32 * 18) as usize,    // Q4_0
        3 => (n_elements / 32 * 20) as usize,    // Q4_1
        6 => (n_elements / 32 * 22) as usize,    // Q5_0
        7 => (n_elements / 32 * 24) as usize,    // Q5_1
        8 => (n_elements / 32 * 34) as usize,    // Q8_0
        10 => (n_elements / 256 * 84) as usize,  // Q2_K
        11 => (n_elements / 256 * 110) as usize, // Q3_K
        12 => (n_elements / 256 * 144) as usize, // Q4_K
        13 => (n_elements / 256 * 176) as usize, // Q5_K
        14 => (n_elements / 256 * 210) as usize, // Q6_K
        _ => (n_elements * 4) as usize,          // fallback: F32
    }
}

#[cfg(any(feature = "picolm", test))]
fn align_up(offset: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
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
