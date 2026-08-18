#![cfg(feature = "picolm")]

use std::collections::HashMap;
use std::path::PathBuf;

use a3s_power::backend::gguf_stream::GgufFile;
use a3s_power::backend::picolm_ops::qwen35::Qwen35TensorLayout;

const QWEN38_Q8_K_XL_BYTES: u64 = 31_457_991_680;

/// Opt-in artifact test for the reviewed Qwen3.8-27B UD-Q8_K_XL conversion.
///
/// The conversion name describes Unsloth's mixed-precision policy. Its GGUF
/// tensors are Q8_0, BF16, and F32 rather than one monolithic Q8_K tensor type.
#[test]
fn validates_real_qwen38_27b_ud_q8_k_xl_layout() {
    let Some(path) = std::env::var_os("A3S_POWER_QWEN38_Q8_TEST_MODEL").map(PathBuf::from) else {
        eprintln!(
            "A3S_POWER_QWEN38_Q8_TEST_MODEL is not set; skipping real Qwen3.8 Q8 artifact test"
        );
        return;
    };

    assert_eq!(
        std::fs::metadata(&path)
            .expect("real Qwen3.8 Q8 GGUF metadata must be readable")
            .len(),
        QWEN38_Q8_K_XL_BYTES
    );

    let model =
        GgufFile::open(&path).expect("real Qwen3.8 Q8 GGUF must parse with bounded tensors");
    assert_eq!(model.meta.architecture.name(), "qwen35");

    let layout = Qwen35TensorLayout::validate(&model.meta)
        .expect("real Qwen3.8 Q8 GGUF must match the typed qwen35 layout");
    assert_eq!(layout.trunk_layers, 64);
    assert_eq!(layout.recurrent_layers, 48);
    assert_eq!(layout.full_attention_layers, 16);
    assert_eq!(layout.mtp_layers, 1);

    let mut type_counts = HashMap::<u32, usize>::new();
    for tensor in model.meta.tensors.values() {
        *type_counts.entry(tensor.ggml_type).or_default() += 1;
    }
    assert_eq!(model.meta.tensors.len(), 866);
    assert_eq!(type_counts, HashMap::from([(0, 360), (8, 453), (30, 53)]));

    assert_eq!(
        model.tensor_type("blk.64.nextn.eh_proj.weight").unwrap(),
        30
    );
    assert_eq!(
        model
            .tensor_bytes("blk.64.nextn.eh_proj.weight")
            .unwrap()
            .len(),
        104_857_600
    );
}
