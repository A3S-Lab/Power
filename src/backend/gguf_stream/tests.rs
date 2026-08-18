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
fn test_ggml_type_size_bf16() {
    assert_eq!(ggml_type_size(30, 1024), 2048);
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
fn write_tensor_desc(buf: &mut Vec<u8>, name: &str, shape: &[u64], ggml_type: u32, offset: u64) {
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
fn parse_test_header(meta: &[u8], n_kv: u64, tensors: &[u8], n_tensors: u64) -> Result<GgufMeta> {
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

#[cfg(feature = "picolm")]
#[test]
fn test_open_rejects_truncated_tensor_payload() {
    let (meta, n_kv) = base_meta();
    let mut tensor = Vec::new();
    write_tensor_desc(&mut tensor, "truncated.weight", &[1], 0, 0);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&GGUF_VERSION_MAX.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&n_kv.to_le_bytes());
    bytes.extend_from_slice(&meta);
    bytes.extend_from_slice(&tensor);
    bytes.resize(align_up(bytes.len(), 32), 0);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.gguf");
    std::fs::write(&path, bytes).unwrap();

    let err = GgufFile::open(&path).unwrap_err();

    assert!(err.to_string().contains("extends beyond file bounds"));
}

#[cfg(feature = "picolm")]
#[test]
fn test_from_memory_decrypted_invalid_magic_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");
    std::fs::write(&path, vec![0u8; 32]).unwrap();
    let key = [0x42; 32];
    let enc_path = crate::tee::encrypted_model::encrypt_model_file(&path, &key).unwrap();
    let decrypted =
        crate::tee::encrypted_model::MemoryDecryptedModel::decrypt(&enc_path, &key).unwrap();

    let result = GgufFile::from_memory_decrypted(decrypted);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("GGUF"));
}

#[cfg(feature = "picolm")]
#[test]
fn test_from_layer_streaming_decrypted_invalid_magic_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");
    std::fs::write(&path, vec![0u8; 32]).unwrap();
    let key = [0x42; 32];
    let enc_path = crate::tee::encrypted_model::encrypt_model_file(&path, &key).unwrap();
    let decrypted =
        crate::tee::encrypted_model::LayerStreamingDecryptedModel::decrypt(&enc_path, &key)
            .unwrap();

    let result = GgufFile::from_layer_streaming_decrypted(decrypted);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("GGUF"));
}
