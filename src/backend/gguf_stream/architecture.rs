//! Architecture-specific GGUF metadata used by the native picolm executor.
//!
//! Container parsing is intentionally separate from execution support. A GGUF
//! file can be structurally valid while describing a graph that picolm does not
//! implement yet; callers must match the typed architecture before building an
//! execution plan.

use std::collections::HashMap;

use super::MetaValue;
use crate::error::{PowerError, Result};

/// Architecture metadata extracted from `general.architecture`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufArchitecture {
    /// Qwen3.5 dense hybrid attention / Gated DeltaNet.
    Qwen35(Qwen35Architecture),
    /// An architecture without a dedicated typed metadata schema.
    Named(String),
}

impl GgufArchitecture {
    pub fn name(&self) -> &str {
        match self {
            Self::Qwen35(_) => "qwen35",
            Self::Named(name) => name,
        }
    }

    pub fn qwen35(&self) -> Option<&Qwen35Architecture> {
        match self {
            Self::Qwen35(metadata) => Some(metadata),
            Self::Named(_) => None,
        }
    }
}

pub(super) fn parse(
    name: &str,
    metadata: &HashMap<String, MetaValue>,
    total_layers: u32,
) -> Result<GgufArchitecture> {
    if name != "qwen35" {
        return Ok(GgufArchitecture::Named(name.to_string()));
    }

    let rope_sections =
        required_non_negative_u32_array::<4>(metadata, "qwen35.rope.dimension_sections")?;
    let recurrent_layers = optional_bool_array(metadata, "qwen35.attention.recurrent_layers")?;

    Ok(GgufArchitecture::Qwen35(Qwen35Architecture::new(
        Qwen35ArchitectureArgs {
            total_layers,
            attention_key_length: required_u32(metadata, "qwen35.attention.key_length")?,
            attention_value_length: required_u32(metadata, "qwen35.attention.value_length")?,
            rope_sections,
            ssm_conv_kernel: required_u32(metadata, "qwen35.ssm.conv_kernel")?,
            ssm_state_size: required_u32(metadata, "qwen35.ssm.state_size")?,
            ssm_group_count: required_u32(metadata, "qwen35.ssm.group_count")?,
            ssm_time_step_rank: required_u32(metadata, "qwen35.ssm.time_step_rank")?,
            ssm_inner_size: required_u32(metadata, "qwen35.ssm.inner_size")?,
            nextn_predict_layers: optional_u32(metadata, "qwen35.nextn_predict_layers")?
                .unwrap_or(0),
            recurrent_layers,
            full_attention_interval: optional_u32(metadata, "qwen35.full_attention_interval")?,
        },
    )?))
}

fn required_u32(metadata: &HashMap<String, MetaValue>, key: &str) -> Result<u32> {
    optional_u32(metadata, key)?.ok_or_else(|| {
        PowerError::InvalidFormat(format!("GGUF: missing required metadata field {key}"))
    })
}

fn optional_u32(metadata: &HashMap<String, MetaValue>, key: &str) -> Result<Option<u32>> {
    match metadata.get(key) {
        Some(value) => value.as_u32().map(Some).ok_or_else(|| {
            PowerError::InvalidFormat(format!("GGUF: {key} must be a u32-compatible integer"))
        }),
        None => Ok(None),
    }
}

fn required_non_negative_u32_array<const N: usize>(
    metadata: &HashMap<String, MetaValue>,
    key: &str,
) -> Result<[u32; N]> {
    let values = metadata
        .get(key)
        .and_then(MetaValue::as_i32_array)
        .ok_or_else(|| {
            PowerError::InvalidFormat(format!(
                "GGUF: {key} must be an array of {N} non-negative integers"
            ))
        })?;
    if values.len() != N {
        return Err(PowerError::InvalidFormat(format!(
            "GGUF: {key} must contain exactly {N} entries, got {}",
            values.len()
        )));
    }
    let converted = values
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                PowerError::InvalidFormat(format!("GGUF: {key} entries must be non-negative"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    converted.try_into().map_err(|_| {
        PowerError::InvalidFormat(format!("GGUF: {key} must contain exactly {N} entries"))
    })
}

fn optional_bool_array(
    metadata: &HashMap<String, MetaValue>,
    key: &str,
) -> Result<Option<Vec<bool>>> {
    match metadata.get(key) {
        Some(value) => value.as_bool_array().map(Some).ok_or_else(|| {
            PowerError::InvalidFormat(format!("GGUF: {key} must be an array of booleans"))
        }),
        None => Ok(None),
    }
}

/// Native execution role of one Qwen3.5 block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    /// Recurrent Gated DeltaNet block.
    Recurrent,
    /// Full gated attention block in the target trunk.
    FullAttention,
    /// Extra trained NextN block used only by MTP draft inference.
    Mtp,
}

/// Validated Qwen3.5 hyperparameters and block plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Architecture {
    pub attention_key_length: u32,
    pub attention_value_length: u32,
    pub rope_sections: [u32; 4],
    pub ssm_conv_kernel: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_inner_size: u32,
    pub nextn_predict_layers: u32,
    layer_kinds: Vec<Qwen35LayerKind>,
}

/// Arguments kept as a struct so architecture validation cannot accidentally
/// swap adjacent dimension fields.
pub(crate) struct Qwen35ArchitectureArgs {
    pub total_layers: u32,
    pub attention_key_length: u32,
    pub attention_value_length: u32,
    pub rope_sections: [u32; 4],
    pub ssm_conv_kernel: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_inner_size: u32,
    pub nextn_predict_layers: u32,
    pub recurrent_layers: Option<Vec<bool>>,
    pub full_attention_interval: Option<u32>,
}

impl Qwen35Architecture {
    pub(crate) fn new(args: Qwen35ArchitectureArgs) -> Result<Self> {
        let Qwen35ArchitectureArgs {
            total_layers,
            attention_key_length,
            attention_value_length,
            rope_sections,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
            nextn_predict_layers,
            recurrent_layers,
            full_attention_interval,
        } = args;

        for (name, value) in [
            ("block_count", total_layers),
            ("attention.key_length", attention_key_length),
            ("attention.value_length", attention_value_length),
            ("ssm.conv_kernel", ssm_conv_kernel),
            ("ssm.state_size", ssm_state_size),
            ("ssm.group_count", ssm_group_count),
            ("ssm.time_step_rank", ssm_time_step_rank),
            ("ssm.inner_size", ssm_inner_size),
        ] {
            if value == 0 {
                return Err(PowerError::InvalidFormat(format!(
                    "GGUF: qwen35.{name} must be greater than zero"
                )));
            }
        }

        if nextn_predict_layers >= total_layers {
            return Err(PowerError::InvalidFormat(format!(
                "GGUF: qwen35.nextn_predict_layers ({nextn_predict_layers}) must be less than block_count ({total_layers})"
            )));
        }

        if ssm_inner_size % ssm_time_step_rank != 0 {
            return Err(PowerError::InvalidFormat(format!(
                "GGUF: qwen35.ssm.inner_size ({ssm_inner_size}) must be divisible by ssm.time_step_rank ({ssm_time_step_rank})"
            )));
        }
        if ssm_time_step_rank % ssm_group_count != 0 {
            return Err(PowerError::InvalidFormat(format!(
                "GGUF: qwen35.ssm.time_step_rank ({ssm_time_step_rank}) must be divisible by ssm.group_count ({ssm_group_count})"
            )));
        }

        let rope_pairs = rope_sections.iter().try_fold(0u32, |sum, section| {
            sum.checked_add(*section).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "GGUF: qwen35.rope.dimension_sections overflows u32".to_string(),
                )
            })
        })?;
        if rope_pairs == 0 {
            return Err(PowerError::InvalidFormat(
                "GGUF: qwen35.rope.dimension_sections must not be all zero".to_string(),
            ));
        }

        let trunk_layers = total_layers - nextn_predict_layers;
        let trunk_len = usize::try_from(trunk_layers).map_err(|_| {
            PowerError::InvalidFormat("GGUF: qwen35 trunk layer count exceeds usize".to_string())
        })?;
        let total_len = usize::try_from(total_layers).map_err(|_| {
            PowerError::InvalidFormat("GGUF: qwen35 block_count exceeds usize".to_string())
        })?;

        let recurrent = match recurrent_layers {
            Some(mut flags) => {
                if flags.len() == total_len {
                    if flags[trunk_len..].iter().any(|flag| *flag) {
                        return Err(PowerError::InvalidFormat(
                            "GGUF: qwen35 MTP blocks must not be marked recurrent".to_string(),
                        ));
                    }
                    flags.truncate(trunk_len);
                    flags
                } else if flags.len() == trunk_len {
                    flags
                } else {
                    return Err(PowerError::InvalidFormat(format!(
                        "GGUF: qwen35.attention.recurrent_layers has {} entries; expected {trunk_len} trunk entries or {total_len} total entries",
                        flags.len()
                    )));
                }
            }
            None => {
                // This is the architecture default used by the reference model
                // when the explicit recurrent-layer bitmap is absent.
                let interval = full_attention_interval.unwrap_or(4);
                if interval == 0 {
                    return Err(PowerError::InvalidFormat(
                        "GGUF: qwen35.full_attention_interval must be greater than zero"
                            .to_string(),
                    ));
                }
                (0..trunk_layers)
                    .map(|layer| (layer + 1) % interval != 0)
                    .collect()
            }
        };

        if recurrent.iter().all(|flag| *flag) {
            return Err(PowerError::InvalidFormat(
                "GGUF: qwen35 trunk must contain at least one full-attention block".to_string(),
            ));
        }

        let mut layer_kinds = Vec::with_capacity(total_len);
        layer_kinds.extend(recurrent.into_iter().map(|is_recurrent| {
            if is_recurrent {
                Qwen35LayerKind::Recurrent
            } else {
                Qwen35LayerKind::FullAttention
            }
        }));
        layer_kinds.extend(std::iter::repeat_n(
            Qwen35LayerKind::Mtp,
            nextn_predict_layers as usize,
        ));

        Ok(Self {
            attention_key_length,
            attention_value_length,
            rope_sections,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
            nextn_predict_layers,
            layer_kinds,
        })
    }

    pub fn total_layer_count(&self) -> usize {
        self.layer_kinds.len()
    }

    pub fn trunk_layer_count(&self) -> usize {
        self.layer_kinds.len() - self.nextn_predict_layers as usize
    }

    pub fn layer_kind(&self, layer: usize) -> Option<Qwen35LayerKind> {
        self.layer_kinds.get(layer).copied()
    }

    pub fn layer_kinds(&self) -> &[Qwen35LayerKind] {
        &self.layer_kinds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> HashMap<String, MetaValue> {
        HashMap::from([
            (
                "qwen35.attention.key_length".to_string(),
                MetaValue::U32(256),
            ),
            (
                "qwen35.attention.value_length".to_string(),
                MetaValue::U32(256),
            ),
            (
                "qwen35.rope.dimension_sections".to_string(),
                MetaValue::Array(vec![
                    MetaValue::I32(11),
                    MetaValue::I32(11),
                    MetaValue::I32(10),
                    MetaValue::I32(0),
                ]),
            ),
            ("qwen35.ssm.conv_kernel".to_string(), MetaValue::U32(4)),
            ("qwen35.ssm.state_size".to_string(), MetaValue::U32(128)),
            ("qwen35.ssm.group_count".to_string(), MetaValue::U32(16)),
            ("qwen35.ssm.time_step_rank".to_string(), MetaValue::U32(16)),
            ("qwen35.ssm.inner_size".to_string(), MetaValue::U32(2048)),
            (
                "qwen35.full_attention_interval".to_string(),
                MetaValue::U32(4),
            ),
        ])
    }

    #[test]
    fn parses_qwen35_default_layer_plan() {
        let parsed = parse("qwen35", &metadata(), 24).unwrap();
        let qwen35 = parsed.qwen35().unwrap();

        assert_eq!(qwen35.attention_key_length, 256);
        assert_eq!(qwen35.rope_sections, [11, 11, 10, 0]);
        assert_eq!(qwen35.trunk_layer_count(), 24);
        assert_eq!(qwen35.layer_kind(2), Some(Qwen35LayerKind::Recurrent));
        assert_eq!(qwen35.layer_kind(3), Some(Qwen35LayerKind::FullAttention));
    }

    #[test]
    fn parses_explicit_recurrent_bitmap_and_mtp_layer() {
        let mut values = metadata();
        values.insert("qwen35.nextn_predict_layers".to_string(), MetaValue::U32(1));
        values.insert(
            "qwen35.attention.recurrent_layers".to_string(),
            MetaValue::Array(vec![
                MetaValue::Bool(true),
                MetaValue::Bool(true),
                MetaValue::Bool(true),
                MetaValue::Bool(false),
                MetaValue::Bool(false),
            ]),
        );

        let parsed = parse("qwen35", &values, 5).unwrap();
        let qwen35 = parsed.qwen35().unwrap();

        assert_eq!(qwen35.trunk_layer_count(), 4);
        assert_eq!(qwen35.layer_kind(4), Some(Qwen35LayerKind::Mtp));
    }

    #[test]
    fn rejects_recurrent_mtp_layer() {
        let mut values = metadata();
        values.insert("qwen35.nextn_predict_layers".to_string(), MetaValue::U32(1));
        values.insert(
            "qwen35.attention.recurrent_layers".to_string(),
            MetaValue::Array(vec![
                MetaValue::Bool(true),
                MetaValue::Bool(true),
                MetaValue::Bool(true),
                MetaValue::Bool(false),
                MetaValue::Bool(true),
            ]),
        );

        let err = parse("qwen35", &values, 5).unwrap_err();

        assert!(err
            .to_string()
            .contains("MTP blocks must not be marked recurrent"));
    }
}
