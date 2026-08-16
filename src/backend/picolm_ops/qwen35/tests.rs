use std::collections::HashMap;

use super::Qwen35TensorLayout;
use crate::backend::gguf_stream::{
    GgufArchitecture, GgufMeta, Qwen35Architecture, Qwen35ArchitectureArgs, Qwen35LayerKind,
    TensorDesc,
};

const N_EMBD: u64 = 1024;
const N_FF: u64 = 3584;
const VOCAB: u64 = 128;

fn architecture(total_layers: u32, nextn_predict_layers: u32) -> Qwen35Architecture {
    Qwen35Architecture::new(Qwen35ArchitectureArgs {
        total_layers,
        attention_key_length: 256,
        attention_value_length: 256,
        rope_sections: [11, 11, 10, 0],
        ssm_conv_kernel: 4,
        ssm_state_size: 128,
        ssm_group_count: 16,
        ssm_time_step_rank: 16,
        ssm_inner_size: 2048,
        nextn_predict_layers,
        recurrent_layers: None,
        full_attention_interval: Some(4),
    })
    .unwrap()
}

fn insert(tensors: &mut HashMap<String, TensorDesc>, name: impl Into<String>, shape: &[u64]) {
    tensors.insert(
        name.into(),
        TensorDesc {
            offset: 0,
            shape: shape.to_vec(),
            ggml_type: 0,
            n_elements: shape.iter().product(),
        },
    );
}

fn insert_common(tensors: &mut HashMap<String, TensorDesc>, layer: usize) {
    for (suffix, shape) in [
        ("attn_norm.weight", vec![N_EMBD]),
        ("post_attention_norm.weight", vec![N_EMBD]),
        ("ffn_gate.weight", vec![N_EMBD, N_FF]),
        ("ffn_up.weight", vec![N_EMBD, N_FF]),
        ("ffn_down.weight", vec![N_FF, N_EMBD]),
    ] {
        insert(tensors, format!("blk.{layer}.{suffix}"), &shape);
    }
}

fn insert_recurrent(tensors: &mut HashMap<String, TensorDesc>, layer: usize) {
    for (suffix, shape) in [
        ("attn_qkv.weight", vec![N_EMBD, 6144]),
        ("attn_gate.weight", vec![N_EMBD, 2048]),
        ("ssm_conv1d.weight", vec![4, 6144]),
        ("ssm_dt.bias", vec![16]),
        ("ssm_a", vec![16]),
        ("ssm_beta.weight", vec![N_EMBD, 16]),
        ("ssm_alpha.weight", vec![N_EMBD, 16]),
        ("ssm_norm.weight", vec![128]),
        ("ssm_out.weight", vec![2048, N_EMBD]),
    ] {
        insert(tensors, format!("blk.{layer}.{suffix}"), &shape);
    }
}

fn insert_full_attention(tensors: &mut HashMap<String, TensorDesc>, layer: usize) {
    for (suffix, shape) in [
        ("attn_q.weight", vec![N_EMBD, 4096]),
        ("attn_k.weight", vec![N_EMBD, 512]),
        ("attn_v.weight", vec![N_EMBD, 512]),
        ("attn_output.weight", vec![2048, N_EMBD]),
        ("attn_q_norm.weight", vec![256]),
        ("attn_k_norm.weight", vec![256]),
    ] {
        insert(tensors, format!("blk.{layer}.{suffix}"), &shape);
    }
}

fn insert_mtp(tensors: &mut HashMap<String, TensorDesc>, layer: usize) {
    insert(
        tensors,
        format!("blk.{layer}.nextn.eh_proj.weight"),
        &[2 * N_EMBD, N_EMBD],
    );
    insert(
        tensors,
        format!("blk.{layer}.nextn.enorm.weight"),
        &[N_EMBD],
    );
    insert(
        tensors,
        format!("blk.{layer}.nextn.hnorm.weight"),
        &[N_EMBD],
    );
}

fn metadata(total_layers: u32, nextn_predict_layers: u32) -> GgufMeta {
    let architecture = architecture(total_layers, nextn_predict_layers);
    let layer_kinds = architecture.layer_kinds().to_vec();
    let mut tensors = HashMap::new();
    insert(&mut tensors, "token_embd.weight", &[N_EMBD, VOCAB]);
    insert(&mut tensors, "output_norm.weight", &[N_EMBD]);

    for (layer, kind) in layer_kinds.into_iter().enumerate() {
        insert_common(&mut tensors, layer);
        match kind {
            Qwen35LayerKind::Recurrent => insert_recurrent(&mut tensors, layer),
            Qwen35LayerKind::FullAttention => insert_full_attention(&mut tensors, layer),
            Qwen35LayerKind::Mtp => {
                insert_full_attention(&mut tensors, layer);
                insert_mtp(&mut tensors, layer);
            }
        }
    }

    GgufMeta {
        architecture: GgufArchitecture::Qwen35(architecture),
        n_layers: total_layers,
        n_embd: N_EMBD as u32,
        n_heads: 8,
        n_kv_heads: 2,
        context_length: 262_144,
        vocab_size: VOCAB as u32,
        bos_token_id: 1,
        eos_token_id: 2,
        n_ff: N_FF as u32,
        norm_eps: 1e-6,
        rope_theta: 10_000_000.0,
        rope_dim: Some(64),
        chat_template: None,
        vocab_tokens: vec!["<s>".to_string(), "</s>".to_string()],
        vocab_scores: Vec::new(),
        vocab_types: Vec::new(),
        tensor_data_offset: 0,
        tensors,
    }
}

#[test]
fn validates_hybrid_qwen35_layout() {
    let meta = metadata(4, 0);

    let layout = Qwen35TensorLayout::validate(&meta).unwrap();

    assert_eq!(layout.trunk_layers, 4);
    assert_eq!(layout.recurrent_layers, 3);
    assert_eq!(layout.full_attention_layers, 1);
    assert_eq!(layout.mtp_layers, 0);
    assert!(layout.tied_output);
}

#[test]
fn validates_combined_mtp_layout() {
    let mut meta = metadata(5, 1);
    insert(
        &mut meta.tensors,
        "blk.4.nextn.shared_head_norm.weight",
        &[N_EMBD],
    );

    let layout = Qwen35TensorLayout::validate(&meta).unwrap();

    assert_eq!(layout.trunk_layers, 4);
    assert_eq!(layout.mtp_layers, 1);
}

#[test]
fn rejects_missing_architecture_specific_tensor() {
    let mut meta = metadata(4, 0);
    meta.tensors.remove("blk.0.ssm_alpha.weight");

    let err = Qwen35TensorLayout::validate(&meta).unwrap_err();

    assert!(err.to_string().contains("blk.0.ssm_alpha.weight"));
}

#[test]
fn rejects_wrong_full_attention_shape() {
    let mut meta = metadata(4, 0);
    meta.tensors.get_mut("blk.3.attn_q.weight").unwrap().shape = vec![N_EMBD, 2048];

    let err = Qwen35TensorLayout::validate(&meta).unwrap_err();

    assert!(err.to_string().contains("blk.3.attn_q.weight"));
    assert!(err.to_string().contains("4096"));
}

#[test]
fn validates_separate_output_head_when_present() {
    let mut meta = metadata(4, 0);
    insert(&mut meta.tensors, "output.weight", &[N_EMBD, VOCAB]);

    let layout = Qwen35TensorLayout::validate(&meta).unwrap();

    assert!(!layout.tied_output);
}
