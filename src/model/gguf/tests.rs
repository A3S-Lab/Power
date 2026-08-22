use super::*;
use std::io::Cursor;

/// Build a minimal valid GGUF v3 binary in memory for testing.
fn build_test_gguf() -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic: the literal ASCII bytes "GGUF".
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

fn push_header(buf: &mut Vec<u8>, tensor_count: u64, metadata_kv_count: u64) {
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_kv_count.to_le_bytes());
}

fn push_string(buf: &mut Vec<u8>, value: &[u8]) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value);
}

fn push_kv_u64(buf: &mut Vec<u8>, key: &[u8], value: u64) {
    push_string(buf, key);
    buf.extend_from_slice(&10u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

#[test]
fn test_read_gguf_metadata() {
    let data = build_test_gguf();
    assert_eq!(&data[..4], b"GGUF");
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
fn test_tensor_element_count_saturates_on_overflow() {
    let tensor = GgufTensor {
        name: "huge".to_string(),
        dimensions: vec![u64::MAX, 2],
        tensor_type: 0,
        offset: 0,
    };
    assert_eq!(tensor.element_count(), u64::MAX);
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

/// Build a GGUF with all value types for comprehensive testing.
fn build_test_gguf_all_types() -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic + Version 3
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count: 0
    buf.extend_from_slice(&0u64.to_le_bytes());
    // Metadata KV count: 10
    buf.extend_from_slice(&10u64.to_le_bytes());

    // KV: uint8
    let key = b"test.uint8";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&0u32.to_le_bytes()); // type = uint8
    buf.push(42u8);

    // KV: int8
    let key = b"test.int8";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&1u32.to_le_bytes()); // type = int8
    buf.push((-5i8) as u8);

    // KV: uint16
    let key = b"test.uint16";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&2u32.to_le_bytes()); // type = uint16
    buf.extend_from_slice(&1000u16.to_le_bytes());

    // KV: int16
    let key = b"test.int16";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&3u32.to_le_bytes()); // type = int16
    buf.extend_from_slice(&(-500i16).to_le_bytes());

    // KV: uint32
    let key = b"test.uint32";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&4u32.to_le_bytes()); // type = uint32
    buf.extend_from_slice(&100000u32.to_le_bytes());

    // KV: int32
    let key = b"test.int32";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&5u32.to_le_bytes()); // type = int32
    buf.extend_from_slice(&(-99999i32).to_le_bytes());

    // KV: float32
    let key = b"test.float32";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&6u32.to_le_bytes()); // type = float32
    buf.extend_from_slice(&std::f32::consts::PI.to_le_bytes());

    // KV: bool
    let key = b"test.bool";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&7u32.to_le_bytes()); // type = bool
    buf.push(1u8);

    // KV: int64
    let key = b"test.int64";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&11u32.to_le_bytes()); // type = int64
    buf.extend_from_slice(&(-123456789i64).to_le_bytes());

    // KV: float64
    let key = b"test.float64";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&12u32.to_le_bytes()); // type = float64
    buf.extend_from_slice(&std::f64::consts::E.to_le_bytes());

    buf
}

#[test]
fn test_read_all_value_types() {
    let data = build_test_gguf_all_types();
    let mut cursor = Cursor::new(data);
    let meta = read_metadata_from_reader(&mut cursor).unwrap();

    assert_eq!(meta.metadata.len(), 10);

    match meta.metadata.get("test.uint8") {
        Some(GgufValue::Uint8(v)) => assert_eq!(*v, 42),
        other => panic!("Expected Uint8(42), got: {other:?}"),
    }
    match meta.metadata.get("test.int8") {
        Some(GgufValue::Int8(v)) => assert_eq!(*v, -5),
        other => panic!("Expected Int8(-5), got: {other:?}"),
    }
    match meta.metadata.get("test.uint16") {
        Some(GgufValue::Uint16(v)) => assert_eq!(*v, 1000),
        other => panic!("Expected Uint16(1000), got: {other:?}"),
    }
    match meta.metadata.get("test.int16") {
        Some(GgufValue::Int16(v)) => assert_eq!(*v, -500),
        other => panic!("Expected Int16(-500), got: {other:?}"),
    }
    match meta.metadata.get("test.uint32") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 100000),
        other => panic!("Expected Uint32(100000), got: {other:?}"),
    }
    match meta.metadata.get("test.int32") {
        Some(GgufValue::Int32(v)) => assert_eq!(*v, -99999),
        other => panic!("Expected Int32(-99999), got: {other:?}"),
    }
    match meta.metadata.get("test.float32") {
        Some(GgufValue::Float32(v)) => assert!((v - std::f32::consts::PI).abs() < 0.001),
        other => panic!("Expected Float32(~3.14), got: {other:?}"),
    }
    match meta.metadata.get("test.bool") {
        Some(GgufValue::Bool(v)) => assert!(*v),
        other => panic!("Expected Bool(true), got: {other:?}"),
    }
    match meta.metadata.get("test.int64") {
        Some(GgufValue::Int64(v)) => assert_eq!(*v, -123456789),
        other => panic!("Expected Int64(-123456789), got: {other:?}"),
    }
    match meta.metadata.get("test.float64") {
        Some(GgufValue::Float64(v)) => assert!((v - std::f64::consts::E).abs() < 0.0001),
        other => panic!("Expected Float64(~2.718), got: {other:?}"),
    }
}

#[test]
fn test_gguf_value_to_json_all_types() {
    assert_eq!(GgufValue::Uint8(255).to_json(), serde_json::json!(255));
    assert_eq!(GgufValue::Int8(-1).to_json(), serde_json::json!(-1));
    assert_eq!(GgufValue::Uint16(65535).to_json(), serde_json::json!(65535));
    assert_eq!(
        GgufValue::Int16(-32768).to_json(),
        serde_json::json!(-32768)
    );
    assert_eq!(GgufValue::Int32(-1).to_json(), serde_json::json!(-1));
    assert_eq!(GgufValue::Float32(1.5).to_json(), serde_json::json!(1.5));
    assert_eq!(
        GgufValue::Uint64(u64::MAX).to_json(),
        serde_json::json!(u64::MAX)
    );
    assert_eq!(
        GgufValue::Int64(i64::MIN).to_json(),
        serde_json::json!(i64::MIN)
    );
    assert_eq!(GgufValue::Float64(2.5).to_json(), serde_json::json!(2.5));
    assert_eq!(GgufValue::Bool(false).to_json(), serde_json::json!(false));
}

#[test]
fn test_gguf_type_names_comprehensive() {
    assert_eq!(gguf_type_name(3), "Q4_1");
    assert_eq!(gguf_type_name(6), "Q5_0");
    assert_eq!(gguf_type_name(7), "Q5_1");
    assert_eq!(gguf_type_name(8), "Q8_0");
    assert_eq!(gguf_type_name(9), "Q8_1");
    assert_eq!(gguf_type_name(10), "Q2_K");
    assert_eq!(gguf_type_name(11), "Q3_K");
    assert_eq!(gguf_type_name(13), "Q5_K");
    assert_eq!(gguf_type_name(14), "Q6_K");
    assert_eq!(gguf_type_name(15), "IQ2_XXS");
    assert_eq!(gguf_type_name(16), "IQ2_XS");
    assert_eq!(gguf_type_name(17), "IQ3_XXS");
    assert_eq!(gguf_type_name(18), "IQ1_S");
    assert_eq!(gguf_type_name(19), "IQ4_NL");
    assert_eq!(gguf_type_name(20), "IQ3_S");
    assert_eq!(gguf_type_name(21), "IQ2_S");
    assert_eq!(gguf_type_name(22), "IQ4_XS");
    assert_eq!(gguf_type_name(23), "I8");
    assert_eq!(gguf_type_name(24), "I16");
    assert_eq!(gguf_type_name(25), "I32");
    assert_eq!(gguf_type_name(26), "I64");
    assert_eq!(gguf_type_name(27), "F64");
    assert_eq!(gguf_type_name(28), "IQ1_M");
}

#[test]
fn test_format_bytes_kib() {
    assert_eq!(format_bytes(1024), "1.0 KiB");
    assert_eq!(format_bytes(2048), "2.0 KiB");
    assert_eq!(format_bytes(0), "0 B");
}

#[test]
fn test_read_gguf_from_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    std::fs::write(&path, build_test_gguf()).unwrap();

    let meta = read_metadata(&path).unwrap();
    assert_eq!(meta.version, 3);
    assert_eq!(meta.tensor_count, 1);
}

#[test]
fn test_read_gguf_file_not_found() {
    let result = read_metadata(Path::new("/nonexistent/test.gguf"));
    assert!(result.is_err());
}

#[test]
fn test_read_gguf_truncated_file() {
    let data = build_test_gguf();
    // Truncate to just the magic number
    let truncated = &data[..4];
    let mut cursor = Cursor::new(truncated.to_vec());
    let result = read_metadata_from_reader(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_read_gguf_rejects_huge_metadata_count() {
    let mut buf = Vec::new();
    push_header(&mut buf, 0, MAX_METADATA_KV_COUNT + 1);

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(err.to_string().contains("metadata count too large"));
}

#[test]
fn test_read_gguf_rejects_huge_tensor_count() {
    let mut buf = Vec::new();
    push_header(&mut buf, MAX_TENSOR_COUNT + 1, 0);

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(err.to_string().contains("tensor count too large"));
}

#[test]
fn test_read_gguf_rejects_string_length_exceeding_usize() {
    let mut buf = Vec::new();
    push_header(&mut buf, 0, 1);
    buf.extend_from_slice(&u64::MAX.to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(
        err.to_string().contains("string length exceeds usize")
            || err.to_string().contains("GGUF string too long")
    );
}

#[test]
fn test_read_gguf_rejects_array_length_exceeding_usize() {
    let mut buf = Vec::new();
    push_header(&mut buf, 0, 1);
    push_string(&mut buf, b"test.array");
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(
        err.to_string().contains("array length exceeds usize")
            || err.to_string().contains("GGUF array too large")
    );
}

#[test]
fn test_read_gguf_rejects_too_many_tensor_dimensions() {
    let mut buf = Vec::new();
    push_header(&mut buf, 1, 0);
    push_string(&mut buf, b"overshaped.weight");
    buf.extend_from_slice(&(MAX_TENSOR_DIMS + 1).to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(err.to_string().contains("too many dimensions"));
}

#[test]
fn test_read_gguf_rejects_tensor_element_count_overflow() {
    let mut buf = Vec::new();
    push_header(&mut buf, 1, 0);
    push_string(&mut buf, b"overflow.weight");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(err.to_string().contains("element count overflows"));
}

#[test]
fn test_read_gguf_rejects_tensor_byte_size_overflow() {
    let mut buf = Vec::new();
    push_header(&mut buf, 1, 0);
    push_string(&mut buf, b"huge.weight");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32
    buf.extend_from_slice(&0u64.to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(err.to_string().contains("byte size overflows"));
}

#[test]
fn test_read_gguf_rejects_quantized_tensor_unaligned_first_dimension() {
    let mut buf = Vec::new();
    push_header(&mut buf, 1, 0);
    push_string(&mut buf, b"unaligned.weight");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&31u64.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0, 32-element blocks
    buf.extend_from_slice(&0u64.to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(err.to_string().contains("block size"));
}

#[test]
fn test_read_gguf_rejects_tensor_offset_range_overflow() {
    let mut buf = Vec::new();
    push_header(&mut buf, 1, 0);
    push_string(&mut buf, b"offset.weight");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32, 4 bytes
    buf.extend_from_slice(&u64::MAX.to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = read_metadata_from_reader(&mut cursor).unwrap_err();

    assert!(err.to_string().contains("byte range overflows"));
}

#[test]
fn test_unknown_value_type() {
    let mut buf = Vec::new();
    // Magic + Version 3
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&1u64.to_le_bytes()); // kv count

    // KV with unknown type 99
    let key = b"test.unknown";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&99u32.to_le_bytes()); // unknown type

    let mut cursor = Cursor::new(buf);
    let result = read_metadata_from_reader(&mut cursor);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unknown GGUF value type"));
}

/// Build a GGUF with an array value type.
fn build_test_gguf_with_array() -> Vec<u8> {
    let mut buf = Vec::new();

    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&1u64.to_le_bytes()); // kv count

    // KV: array of uint32
    let key = b"test.array";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&9u32.to_le_bytes()); // type = array
    buf.extend_from_slice(&4u32.to_le_bytes()); // element type = uint32
    buf.extend_from_slice(&3u64.to_le_bytes()); // count = 3
    buf.extend_from_slice(&10u32.to_le_bytes());
    buf.extend_from_slice(&20u32.to_le_bytes());
    buf.extend_from_slice(&30u32.to_le_bytes());

    buf
}

#[test]
fn test_read_gguf_array_value() {
    let data = build_test_gguf_with_array();
    let mut cursor = Cursor::new(data);
    let meta = read_metadata_from_reader(&mut cursor).unwrap();

    match meta.metadata.get("test.array") {
        Some(GgufValue::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            match &arr[0] {
                GgufValue::Uint32(v) => assert_eq!(*v, 10),
                other => panic!("Expected Uint32(10), got: {other:?}"),
            }
            match &arr[2] {
                GgufValue::Uint32(v) => assert_eq!(*v, 30),
                other => panic!("Expected Uint32(30), got: {other:?}"),
            }
        }
        other => panic!("Expected Array, got: {other:?}"),
    }
}

#[test]
fn test_estimate_memory_from_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");

    // Build a GGUF with embedding_length and block_count metadata
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&2u64.to_le_bytes()); // kv count

    // llama.embedding_length = 4096
    let key = b"llama.embedding_length";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&4u32.to_le_bytes()); // uint32
    buf.extend_from_slice(&4096u32.to_le_bytes());

    // llama.block_count = 32
    let key = b"llama.block_count";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&4u32.to_le_bytes()); // uint32
    buf.extend_from_slice(&32u32.to_le_bytes());

    std::fs::write(&path, &buf).unwrap();

    let est = estimate_memory(&path, 2048).unwrap();
    assert!(est.total > 0);
    assert_eq!(est.context_size, 2048);
    assert!(est.kv_cache_size > 0);
    assert!(est.model_size > 0);
    assert!(!est.total_display().is_empty());
}

#[test]
fn test_estimate_memory_file_not_found() {
    let result = estimate_memory(Path::new("/nonexistent/model.gguf"), 2048);
    assert!(result.is_err());
}

#[test]
fn test_estimate_memory_rejects_overflowing_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.gguf");

    let mut buf = Vec::new();
    push_header(&mut buf, 0, 4);
    push_kv_u64(&mut buf, b"llama.embedding_length", u64::MAX);
    push_kv_u64(&mut buf, b"llama.block_count", u64::MAX);
    push_kv_u64(&mut buf, b"llama.attention.head_count", 1);
    push_kv_u64(&mut buf, b"llama.attention.head_count_kv", u64::MAX);
    std::fs::write(&path, &buf).unwrap();

    let err = estimate_memory(&path, u32::MAX).unwrap_err();

    assert!(err.to_string().contains("memory estimate overflows"));
}

#[test]
fn test_gguf_version_2() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&0u64.to_le_bytes()); // kv count

    let mut cursor = Cursor::new(buf);
    let meta = read_metadata_from_reader(&mut cursor).unwrap();
    assert_eq!(meta.version, 2);
}

#[test]
fn test_tensor_3d_element_count() {
    let tensor = GgufTensor {
        name: "3d".to_string(),
        dimensions: vec![2, 3, 4],
        tensor_type: 0,
        offset: 0,
    };
    assert_eq!(tensor.element_count(), 24);
}

#[test]
fn test_memory_estimate_fields() {
    let est = MemoryEstimate {
        model_size: 1000,
        kv_cache_size: 200,
        compute_overhead: 100,
        total: 1300,
        context_size: 4096,
    };
    assert_eq!(est.model_size, 1000);
    assert_eq!(est.kv_cache_size, 200);
    assert_eq!(est.compute_overhead, 100);
    assert_eq!(est.total, 1300);
    assert_eq!(est.context_size, 4096);
}
