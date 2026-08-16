use crate::backend::gguf_stream::{GgufMeta, Qwen35Architecture, Qwen35LayerKind, TensorDesc};
use crate::error::{PowerError, Result};

/// Validated tensor inventory for a dense Qwen3.5 GGUF artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TensorLayout {
    pub trunk_layers: usize,
    pub recurrent_layers: usize,
    pub full_attention_layers: usize,
    pub mtp_layers: usize,
    pub tied_output: bool,
}

impl Qwen35TensorLayout {
    pub fn validate(meta: &GgufMeta) -> Result<Self> {
        let architecture = meta.architecture.qwen35().ok_or_else(|| {
            PowerError::InvalidFormat(format!(
                "picolm Qwen3.5 layout requires architecture 'qwen35', got '{}'",
                meta.architecture.name()
            ))
        })?;
        validate_architecture_dimensions(meta, architecture)?;

        let n_embd = u64::from(meta.n_embd);
        let n_ff = u64::from(meta.n_ff);
        let vocab = u64::from(meta.vocab_size);

        require_shape(meta, "token_embd.weight", &[n_embd, vocab])?;
        require_shape(meta, "output_norm.weight", &[n_embd])?;
        let output_present = optional_shape(meta, "output.weight", &[n_embd, vocab])?;

        let mut recurrent_layers = 0;
        let mut full_attention_layers = 0;
        let mut mtp_layers = 0;

        for (layer, kind) in architecture.layer_kinds().iter().copied().enumerate() {
            validate_common_block(meta, layer, n_embd, n_ff)?;
            match kind {
                Qwen35LayerKind::Recurrent => {
                    recurrent_layers += 1;
                    validate_recurrent_block(meta, architecture, layer, n_embd)?;
                }
                Qwen35LayerKind::FullAttention => {
                    full_attention_layers += 1;
                    validate_full_attention_block(meta, architecture, layer, n_embd)?;
                }
                Qwen35LayerKind::Mtp => {
                    mtp_layers += 1;
                    validate_full_attention_block(meta, architecture, layer, n_embd)?;
                    validate_mtp_block(meta, layer, n_embd, vocab)?;
                }
            }
        }

        Ok(Self {
            trunk_layers: architecture.trunk_layer_count(),
            recurrent_layers,
            full_attention_layers,
            mtp_layers,
            tied_output: !output_present,
        })
    }
}

fn validate_architecture_dimensions(
    meta: &GgufMeta,
    architecture: &Qwen35Architecture,
) -> Result<()> {
    if architecture.total_layer_count() != meta.n_layers as usize {
        return Err(PowerError::InvalidFormat(format!(
            "qwen35 block plan contains {} layers but block_count is {}",
            architecture.total_layer_count(),
            meta.n_layers
        )));
    }

    let value_head = architecture.ssm_inner_size / architecture.ssm_time_step_rank;
    if value_head != architecture.ssm_state_size {
        return Err(PowerError::InvalidFormat(format!(
            "picolm qwen35 requires ssm.state_size ({}) to equal ssm.inner_size / ssm.time_step_rank ({value_head})",
            architecture.ssm_state_size
        )));
    }
    if architecture.attention_key_length != architecture.attention_value_length {
        return Err(PowerError::InvalidFormat(format!(
            "picolm qwen35 requires equal attention key/value lengths, got {}/{}",
            architecture.attention_key_length, architecture.attention_value_length
        )));
    }

    let rope_pairs = checked_sum(&architecture.rope_sections, "RoPE section pairs")?;
    let rope_dimension = checked_product(&[rope_pairs, 2], "RoPE section dimension")?;
    if meta.rope_dim != Some(rope_dimension) {
        return Err(PowerError::InvalidFormat(format!(
            "qwen35 rope.dimension_count {:?} does not match twice the dimension sections ({rope_dimension})",
            meta.rope_dim
        )));
    }
    if rope_dimension > architecture.attention_key_length {
        return Err(PowerError::InvalidFormat(format!(
            "qwen35 RoPE dimension ({rope_dimension}) exceeds attention key length ({})",
            architecture.attention_key_length
        )));
    }

    Ok(())
}

fn validate_common_block(meta: &GgufMeta, layer: usize, n_embd: u64, n_ff: u64) -> Result<()> {
    for (suffix, shape) in [
        ("attn_norm.weight", vec![n_embd]),
        ("post_attention_norm.weight", vec![n_embd]),
        ("ffn_gate.weight", vec![n_embd, n_ff]),
        ("ffn_up.weight", vec![n_embd, n_ff]),
        ("ffn_down.weight", vec![n_ff, n_embd]),
    ] {
        require_shape(meta, &block_name(layer, suffix), &shape)?;
    }
    Ok(())
}

fn validate_recurrent_block(
    meta: &GgufMeta,
    architecture: &Qwen35Architecture,
    layer: usize,
    n_embd: u64,
) -> Result<()> {
    let key_total = checked_product(
        &[architecture.ssm_state_size, architecture.ssm_group_count],
        "recurrent key dimension",
    )?;
    let value_total = architecture.ssm_inner_size;
    let qkv_total = checked_sum(
        &[key_total, key_total, value_total],
        "recurrent QKV dimension",
    )?;
    let value_head = architecture.ssm_inner_size / architecture.ssm_time_step_rank;

    for (suffix, shape) in [
        ("attn_qkv.weight", vec![n_embd, u64::from(qkv_total)]),
        ("attn_gate.weight", vec![n_embd, u64::from(value_total)]),
        (
            "ssm_conv1d.weight",
            vec![
                u64::from(architecture.ssm_conv_kernel),
                u64::from(qkv_total),
            ],
        ),
        (
            "ssm_dt.bias",
            vec![u64::from(architecture.ssm_time_step_rank)],
        ),
        ("ssm_a", vec![u64::from(architecture.ssm_time_step_rank)]),
        (
            "ssm_beta.weight",
            vec![n_embd, u64::from(architecture.ssm_time_step_rank)],
        ),
        (
            "ssm_alpha.weight",
            vec![n_embd, u64::from(architecture.ssm_time_step_rank)],
        ),
        ("ssm_norm.weight", vec![u64::from(value_head)]),
        ("ssm_out.weight", vec![u64::from(value_total), n_embd]),
    ] {
        require_shape(meta, &block_name(layer, suffix), &shape)?;
    }
    Ok(())
}

fn validate_full_attention_block(
    meta: &GgufMeta,
    architecture: &Qwen35Architecture,
    layer: usize,
    n_embd: u64,
) -> Result<()> {
    let q_dim = checked_product(
        &[architecture.attention_key_length, meta.n_heads],
        "full-attention query dimension",
    )?;
    let q_gate_dim = q_dim.checked_mul(2).ok_or_else(|| {
        PowerError::InvalidFormat(
            "qwen35 full-attention query/gate dimension overflows u32".to_string(),
        )
    })?;
    let k_dim = checked_product(
        &[architecture.attention_key_length, meta.n_kv_heads],
        "full-attention key dimension",
    )?;
    let v_dim = checked_product(
        &[architecture.attention_value_length, meta.n_kv_heads],
        "full-attention value dimension",
    )?;
    let output_dim = checked_product(
        &[architecture.attention_value_length, meta.n_heads],
        "full-attention output dimension",
    )?;

    for (suffix, shape) in [
        ("attn_q.weight", vec![n_embd, u64::from(q_gate_dim)]),
        ("attn_k.weight", vec![n_embd, u64::from(k_dim)]),
        ("attn_v.weight", vec![n_embd, u64::from(v_dim)]),
        ("attn_output.weight", vec![u64::from(output_dim), n_embd]),
        (
            "attn_q_norm.weight",
            vec![u64::from(architecture.attention_key_length)],
        ),
        (
            "attn_k_norm.weight",
            vec![u64::from(architecture.attention_key_length)],
        ),
    ] {
        require_shape(meta, &block_name(layer, suffix), &shape)?;
    }
    Ok(())
}

fn validate_mtp_block(meta: &GgufMeta, layer: usize, n_embd: u64, vocab: u64) -> Result<()> {
    let double_embd = n_embd.checked_mul(2).ok_or_else(|| {
        PowerError::InvalidFormat("qwen35 MTP embedding dimension overflows u64".to_string())
    })?;
    for (suffix, shape) in [
        ("nextn.eh_proj.weight", vec![double_embd, n_embd]),
        ("nextn.enorm.weight", vec![n_embd]),
        ("nextn.hnorm.weight", vec![n_embd]),
    ] {
        require_shape(meta, &block_name(layer, suffix), &shape)?;
    }
    optional_shape(
        meta,
        &block_name(layer, "nextn.embed_tokens.weight"),
        &[n_embd, vocab],
    )?;
    optional_shape(
        meta,
        &block_name(layer, "nextn.shared_head_head.weight"),
        &[n_embd, vocab],
    )?;
    optional_shape(
        meta,
        &block_name(layer, "nextn.shared_head_norm.weight"),
        &[n_embd],
    )?;
    Ok(())
}

fn block_name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

fn require_shape(meta: &GgufMeta, name: &str, expected: &[u64]) -> Result<()> {
    let tensor = meta.tensors.get(name).ok_or_else(|| {
        PowerError::InvalidFormat(format!("qwen35 GGUF is missing required tensor '{name}'"))
    })?;
    validate_shape(name, tensor, expected)
}

/// Returns true when the optional tensor is present.
fn optional_shape(meta: &GgufMeta, name: &str, expected: &[u64]) -> Result<bool> {
    match meta.tensors.get(name) {
        Some(tensor) => {
            validate_shape(name, tensor, expected)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

fn validate_shape(name: &str, tensor: &TensorDesc, expected: &[u64]) -> Result<()> {
    if tensor.shape != expected {
        return Err(PowerError::InvalidFormat(format!(
            "qwen35 tensor '{name}' has shape {:?}, expected {expected:?}",
            tensor.shape
        )));
    }
    Ok(())
}

fn checked_product(values: &[u32], label: &str) -> Result<u32> {
    values.iter().try_fold(1u32, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| PowerError::InvalidFormat(format!("qwen35 {label} overflows u32")))
    })
}

fn checked_sum(values: &[u32], label: &str) -> Result<u32> {
    values.iter().try_fold(0u32, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| PowerError::InvalidFormat(format!("qwen35 {label} overflows u32")))
    })
}
