// Lightweight GGUF metadata reader.
//
// Parses the GGUF binary header to extract key-value metadata and tensor
// descriptors without loading the full model weights into memory.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{PowerError, Result};

/// Magic number for GGUF files: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46475547; // "GGUF"

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Convert to a JSON-compatible value for API responses.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            GgufValue::Uint8(v) => serde_json::Value::Number((*v as u64).into()),
            GgufValue::Int8(v) => serde_json::Value::Number((*v as i64).into()),
            GgufValue::Uint16(v) => serde_json::Value::Number((*v as u64).into()),
            GgufValue::Int16(v) => serde_json::Value::Number((*v as i64).into()),
            GgufValue::Uint32(v) => serde_json::Value::Number((*v as u64).into()),
            GgufValue::Int32(v) => serde_json::Value::Number((*v as i64).into()),
            GgufValue::Float32(v) => serde_json::Number::from_f64(*v as f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            GgufValue::Bool(v) => serde_json::Value::Bool(*v),
            GgufValue::String(v) => serde_json::Value::String(v.clone()),
            GgufValue::Uint64(v) => serde_json::Value::Number((*v).into()),
            GgufValue::Int64(v) => serde_json::Value::Number((*v).into()),
            GgufValue::Float64(v) => serde_json::Number::from_f64(*v)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            GgufValue::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| v.to_json()).collect())
            }
        }
    }
}

/// Descriptor for a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
}

impl GgufTensor {
    /// Return the total number of elements in this tensor.
    pub fn element_count(&self) -> u64 {
        if self.dimensions.is_empty() {
            0
        } else {
            self.dimensions.iter().product()
        }
    }
}

/// Parsed GGUF file header with metadata and tensor descriptors.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensor>,
}

impl GgufMetadata {
    /// Convert all metadata to a JSON object for API responses.
    pub fn metadata_to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (key, value) in &self.metadata {
            map.insert(key.clone(), value.to_json());
        }
        serde_json::Value::Object(map)
    }

    /// Convert tensor descriptors to a JSON array for verbose show output.
    pub fn tensors_to_json(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.tensors
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "shape": t.dimensions,
                        "type": gguf_type_name(t.tensor_type),
                        "elements": t.element_count(),
                        "offset": t.offset,
                    })
                })
                .collect(),
        )
    }
}

/// Read GGUF metadata and tensor descriptors from a file path.
///
/// Only reads the header — does NOT load tensor data into memory.
pub fn read_metadata(path: &Path) -> Result<GgufMetadata> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        PowerError::Config(format!("Failed to open GGUF file {}: {e}", path.display()))
    })?;

    read_metadata_from_reader(&mut file)
}

/// Read GGUF metadata from any reader that supports Read + Seek.
fn read_metadata_from_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufMetadata> {
    // 1. Read and validate magic number
    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        return Err(PowerError::Config(format!(
            "Not a valid GGUF file (magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
        )));
    }

    // 2. Read version
    let version = read_u32(reader)?;
    if version < 2 || version > 3 {
        return Err(PowerError::Config(format!(
            "Unsupported GGUF version {version} (supported: 2, 3)"
        )));
    }

    // 3. Read counts
    let tensor_count = read_u64(reader)?;
    let metadata_kv_count = read_u64(reader)?;

    // 4. Read metadata key-value pairs
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(reader)?;
        let value = read_gguf_value(reader)?;
        metadata.insert(key, value);
    }

    // 5. Read tensor descriptors
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(reader)?;
        let n_dims = read_u32(reader)? as usize;
        let mut dimensions = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dimensions.push(read_u64(reader)?);
        }
        let tensor_type = read_u32(reader)?;
        let offset = read_u64(reader)?;

        tensors.push(GgufTensor {
            name,
            dimensions,
            tensor_type,
            offset,
        });
    }

    Ok(GgufMetadata {
        version,
        tensor_count,
        metadata,
        tensors,
    })
}

// ---------------------------------------------------------------------------
// Binary reading helpers (little-endian)
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF read error: {e}")))?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool<R: Read>(r: &mut R) -> Result<bool> {
    Ok(read_u8(r)? != 0)
}

/// Read a GGUF string: u64 length followed by UTF-8 bytes (no null terminator).
fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    // Sanity check: strings shouldn't be larger than 1MB in metadata
    if len > 1_048_576 {
        return Err(PowerError::Config(format!(
            "GGUF string too long: {len} bytes"
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| PowerError::Config(format!("GGUF string read error: {e}")))?;
    String::from_utf8(buf)
        .map_err(|e| PowerError::Config(format!("GGUF string is not valid UTF-8: {e}")))
}

/// Read a typed GGUF value (type tag + payload).
fn read_gguf_value<R: Read>(r: &mut R) -> Result<GgufValue> {
    let value_type = read_u32(r)?;
    read_gguf_value_of_type(r, value_type)
}

/// Read a GGUF value given its type tag.
fn read_gguf_value_of_type<R: Read>(r: &mut R, value_type: u32) -> Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::Uint8(read_u8(r)?)),
        1 => Ok(GgufValue::Int8(read_i8(r)?)),
        2 => Ok(GgufValue::Uint16(read_u16(r)?)),
        3 => Ok(GgufValue::Int16(read_i16(r)?)),
        4 => Ok(GgufValue::Uint32(read_u32(r)?)),
        5 => Ok(GgufValue::Int32(read_i32(r)?)),
        6 => Ok(GgufValue::Float32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool(read_bool(r)?)),
        8 => Ok(GgufValue::String(read_gguf_string(r)?)),
        9 => {
            // Array: element_type (u32) + count (u64) + elements
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            // Sanity check: arrays shouldn't have more than 1M elements in metadata
            if count > 1_000_000 {
                return Err(PowerError::Config(format!(
                    "GGUF array too large: {count} elements"
                )));
            }
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_gguf_value_of_type(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64(r)?)),
        11 => Ok(GgufValue::Int64(read_i64(r)?)),
        12 => Ok(GgufValue::Float64(read_f64(r)?)),
        _ => Err(PowerError::Config(format!(
            "Unknown GGUF value type: {value_type}"
        ))),
    }
}

/// Map GGUF tensor type ID to a human-readable name.
fn gguf_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "IQ2_XXS",
        16 => "IQ2_XS",
        17 => "IQ3_XXS",
        18 => "IQ1_S",
        19 => "IQ4_NL",
        20 => "IQ3_S",
        21 => "IQ2_S",
        22 => "IQ4_XS",
        23 => "I8",
        24 => "I16",
        25 => "I32",
        26 => "I64",
        27 => "F64",
        28 => "IQ1_M",
        29 => "BF16",
        _ => "unknown",
    }
}

/// Estimate the memory required to load a GGUF model (in bytes).
///
/// This is a rough estimate based on file size plus overhead for KV cache
/// and runtime buffers. The actual memory usage depends on context size,
/// batch size, and whether GPU offloading is used.
pub fn estimate_memory(path: &Path, ctx_size: u32) -> Result<MemoryEstimate> {
    let file_size = std::fs::metadata(path)
        .map_err(|e| {
            PowerError::Config(format!(
                "Failed to stat GGUF file {}: {e}",
                path.display()
            ))
        })?
        .len();

    let meta = read_metadata(path)?;

    // Extract key parameters from metadata
    let n_embd = meta
        .metadata
        .get("llama.embedding_length")
        .or_else(|| meta.metadata.get("phi3.embedding_length"))
        .or_else(|| meta.metadata.get("qwen2.embedding_length"))
        .and_then(|v| match v {
            GgufValue::Uint32(n) => Some(*n as u64),
            GgufValue::Uint64(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(4096);

    let n_layers = meta
        .metadata
        .get("llama.block_count")
        .or_else(|| meta.metadata.get("phi3.block_count"))
        .or_else(|| meta.metadata.get("qwen2.block_count"))
        .and_then(|v| match v {
            GgufValue::Uint32(n) => Some(*n as u64),
            GgufValue::Uint64(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(32);

    let n_head_kv = meta
        .metadata
        .get("llama.attention.head_count_kv")
        .or_else(|| meta.metadata.get("phi3.attention.head_count_kv"))
        .or_else(|| meta.metadata.get("qwen2.attention.head_count_kv"))
        .and_then(|v| match v {
            GgufValue::Uint32(n) => Some(*n as u64),
            GgufValue::Uint64(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(8);

    let n_head = meta
        .metadata
        .get("llama.attention.head_count")
        .or_else(|| meta.metadata.get("phi3.attention.head_count"))
        .or_else(|| meta.metadata.get("qwen2.attention.head_count"))
        .and_then(|v| match v {
            GgufValue::Uint32(n) => Some(*n as u64),
            GgufValue::Uint64(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(32);

    // KV cache size estimate: 2 (K+V) * n_layers * ctx_size * head_dim * n_head_kv * sizeof(f16)
    let head_dim = if n_head > 0 { n_embd / n_head } else { 128 };
    let kv_cache_bytes =
        2 * n_layers * (ctx_size as u64) * head_dim * n_head_kv * 2; // f16 = 2 bytes

    // Compute buffer overhead (~10% of model size for scratch buffers)
    let compute_overhead = file_size / 10;

    let total = file_size + kv_cache_bytes + compute_overhead;

    Ok(MemoryEstimate {
        model_size: file_size,
        kv_cache_size: kv_cache_bytes,
        compute_overhead,
        total,
        context_size: ctx_size,
    })
}

/// Estimated memory requirements for loading a model.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Size of the model weights on disk (bytes).
    pub model_size: u64,
    /// Estimated KV cache size for the given context length (bytes).
    pub kv_cache_size: u64,
    /// Estimated compute buffer overhead (bytes).
    pub compute_overhead: u64,
    /// Total estimated memory (bytes).
    pub total: u64,
    /// Context size used for the estimate.
    pub context_size: u32,
}

impl MemoryEstimate {
    /// Format total memory as a human-readable string.
    pub fn total_display(&self) -> String {
        format_bytes(self.total)
    }
}

/// Format bytes as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;
    const KB: u64 = 1_024;

    if bytes >= GB {
        format!("{:.1} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid GGUF v3 binary in memory for testing.
    fn build_test_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic: "GGUF" = 0x46475547
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version: 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV count: 2
        buf.extend_from_slice(&2u64.to_le_bytes());

        // KV 1: "general.architecture" = "llama" (string)
        let key1 = b"general.architecture";
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1);
        buf.extend_from_slice(&8u32.to_le_bytes()); // type = string
        let val1 = b"llama";
        buf.extend_from_slice(&(val1.len() as u64).to_le_bytes());
        buf.extend_from_slice(val1);

        // KV 2: "general.parameter_count" = 3200000000 (uint64)
        let key2 = b"general.parameter_count";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2);
        buf.extend_from_slice(&10u32.to_le_bytes()); // type = uint64
        buf.extend_from_slice(&3_200_000_000u64.to_le_bytes());

        // Tensor 1: "output.weight" with shape [4096, 32000], type F16, offset 0
        let tname = b"output.weight";
        buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        buf.extend_from_slice(tname);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        buf.extend_from_slice(&4096u64.to_le_bytes()); // dim[0]
        buf.extend_from_slice(&32000u64.to_le_bytes()); // dim[1]
        buf.extend_from_slice(&1u32.to_le_bytes()); // type = F16
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        buf
    }

    #[test]
    fn test_read_gguf_metadata() {
        let data = build_test_gguf();
        let mut cursor = Cursor::new(data);
        let meta = read_metadata_from_reader(&mut cursor).unwrap();

        assert_eq!(meta.version, 3);
        assert_eq!(meta.tensor_count, 1);
        assert_eq!(meta.metadata.len(), 2);

        // Check architecture key
        match meta.metadata.get("general.architecture") {
            Some(GgufValue::String(s)) => assert_eq!(s, "llama"),
            other => panic!("Expected String('llama'), got: {other:?}"),
        }

        // Check parameter count
        match meta.metadata.get("general.parameter_count") {
            Some(GgufValue::Uint64(n)) => assert_eq!(*n, 3_200_000_000),
            other => panic!("Expected Uint64(3200000000), got: {other:?}"),
        }
    }

    #[test]
    fn test_read_gguf_tensors() {
        let data = build_test_gguf();
        let mut cursor = Cursor::new(data);
        let meta = read_metadata_from_reader(&mut cursor).unwrap();

        assert_eq!(meta.tensors.len(), 1);
        let tensor = &meta.tensors[0];
        assert_eq!(tensor.name, "output.weight");
        assert_eq!(tensor.dimensions, vec![4096, 32000]);
        assert_eq!(tensor.tensor_type, 1); // F16
        assert_eq!(tensor.offset, 0);
        assert_eq!(tensor.element_count(), 4096 * 32000);
    }

    #[test]
    fn test_metadata_to_json() {
        let data = build_test_gguf();
        let mut cursor = Cursor::new(data);
        let meta = read_metadata_from_reader(&mut cursor).unwrap();

        let json = meta.metadata_to_json();
        assert_eq!(json["general.architecture"], "llama");
        assert_eq!(json["general.parameter_count"], 3_200_000_000u64);
    }

    #[test]
    fn test_tensors_to_json() {
        let data = build_test_gguf();
        let mut cursor = Cursor::new(data);
        let meta = read_metadata_from_reader(&mut cursor).unwrap();

        let json = meta.tensors_to_json();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["name"], "output.weight");
        assert_eq!(arr[0]["type"], "F16");
        assert_eq!(arr[0]["elements"], 4096 * 32000);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = build_test_gguf();
        data[0] = 0xFF; // corrupt magic
        let mut cursor = Cursor::new(data);
        let result = read_metadata_from_reader(&mut cursor);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not a valid GGUF"));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = build_test_gguf();
        // Version is at bytes 4..8, set to 99
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let mut cursor = Cursor::new(data);
        let result = read_metadata_from_reader(&mut cursor);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported GGUF"));
    }

    #[test]
    fn test_gguf_type_names() {
        assert_eq!(gguf_type_name(0), "F32");
        assert_eq!(gguf_type_name(1), "F16");
        assert_eq!(gguf_type_name(2), "Q4_0");
        assert_eq!(gguf_type_name(12), "Q4_K");
        assert_eq!(gguf_type_name(29), "BF16");
        assert_eq!(gguf_type_name(999), "unknown");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1_048_576), "1.0 MiB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
        assert_eq!(format_bytes(4_294_967_296), "4.0 GiB");
    }

    #[test]
    fn test_gguf_value_to_json() {
        assert_eq!(GgufValue::Uint32(42).to_json(), serde_json::json!(42));
        assert_eq!(
            GgufValue::String("hello".to_string()).to_json(),
            serde_json::json!("hello")
        );
        assert_eq!(GgufValue::Bool(true).to_json(), serde_json::json!(true));
        assert_eq!(
            GgufValue::Array(vec![GgufValue::Uint32(1), GgufValue::Uint32(2)]).to_json(),
            serde_json::json!([1, 2])
        );
    }

    #[test]
    fn test_tensor_element_count_empty() {
        let tensor = GgufTensor {
            name: "empty".to_string(),
            dimensions: vec![],
            tensor_type: 0,
            offset: 0,
        };
        assert_eq!(tensor.element_count(), 0);
    }

    #[test]
    fn test_tensor_element_count_scalar() {
        let tensor = GgufTensor {
            name: "scalar".to_string(),
            dimensions: vec![1],
            tensor_type: 0,
            offset: 0,
        };
        assert_eq!(tensor.element_count(), 1);
    }

    #[test]
    fn test_memory_estimate_display() {
        let est = MemoryEstimate {
            model_size: 4_000_000_000,
            kv_cache_size: 500_000_000,
            compute_overhead: 400_000_000,
            total: 4_900_000_000,
            context_size: 2048,
        };
        assert!(est.total_display().contains("GiB"));
    }
}
