use super::attention::Qwen35AttentionConfig;
use crate::backend::gguf_stream::GgufMeta;
use crate::backend::picolm_ops::norm::rms_norm_f32;
use crate::error::{PowerError, Result};

/// Qwen3.5 interleaved multimodal RoPE configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35MropeConfig {
    sections: [usize; 4],
    interleaved_limits: [usize; 3],
    rope_dim: usize,
    theta: f32,
}

impl Qwen35MropeConfig {
    pub fn new(sections: [u32; 4], rope_dim: usize, theta: f32) -> Result<Self> {
        if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 MRoPE dimension must be positive and even, got {rope_dim}"
            )));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 MRoPE theta must be finite and positive, got {theta}"
            )));
        }

        let sections = sections.map(|section| section as usize);
        if sections[..3].iter().all(|section| *section == 0) {
            return Err(PowerError::InvalidFormat(
                "qwen35 MRoPE must define a temporal, height, or width section".to_string(),
            ));
        }
        let section_pairs = sections.iter().try_fold(0usize, |sum, section| {
            sum.checked_add(*section)
                .ok_or_else(|| dimension_overflow("section sum"))
        })?;
        let section_dim = section_pairs
            .checked_mul(2)
            .ok_or_else(|| dimension_overflow("section dimension"))?;
        if section_dim != rope_dim {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 MRoPE sections describe {section_dim} dimensions, expected {rope_dim}"
            )));
        }
        let interleaved_limits = [
            sections[0]
                .checked_mul(3)
                .ok_or_else(|| dimension_overflow("temporal interleave limit"))?,
            sections[1]
                .checked_mul(3)
                .ok_or_else(|| dimension_overflow("height interleave limit"))?,
            sections[2]
                .checked_mul(3)
                .ok_or_else(|| dimension_overflow("width interleave limit"))?,
        ];

        Ok(Self {
            sections,
            interleaved_limits,
            rope_dim,
            theta,
        })
    }

    pub fn from_metadata(meta: &GgufMeta) -> Result<Self> {
        let architecture = meta.architecture.qwen35().ok_or_else(|| {
            PowerError::InvalidFormat(format!(
                "qwen35 MRoPE requires architecture 'qwen35', got '{}'",
                meta.architecture.name()
            ))
        })?;
        let rope_dim = meta.rope_dim.ok_or_else(|| {
            PowerError::InvalidFormat("qwen35 MRoPE dimension_count is missing".to_string())
        })?;
        Self::new(
            architecture.rope_sections,
            usize::try_from(rope_dim).map_err(|_| dimension_overflow("dimension"))?,
            meta.rope_theta,
        )
    }

    pub fn sections(self) -> [usize; 4] {
        self.sections
    }

    pub fn rope_dim(self) -> usize {
        self.rope_dim
    }

    pub fn theta(self) -> f32 {
        self.theta
    }

    /// Apply Qwen3.5's `IMROPE` mapping with NeoX-style half-offset pairs.
    pub fn apply(
        &self,
        values: &mut [f32],
        heads: usize,
        head_dim: usize,
        positions: [i32; 4],
    ) -> Result<()> {
        self.validate_shape(values.len(), heads, head_dim)?;

        let pair_count = self.rope_dim / 2;
        let section_pairs = self.sections.iter().sum::<usize>();
        let theta_scale = self.theta.powf(-2.0 / self.rope_dim as f32);
        for head in values.chunks_exact_mut(head_dim) {
            let mut frequency = 1.0f32;
            for pair in 0..pair_count {
                let sector = pair % section_pairs;
                let axis = self.position_axis(sector);
                let angle = positions[axis] as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let first = head[pair];
                let second = head[pair + pair_count];
                head[pair] = first * cos - second * sin;
                head[pair + pair_count] = first * sin + second * cos;
                frequency *= theta_scale;
            }
        }
        Ok(())
    }

    fn validate_shape(&self, actual: usize, heads: usize, head_dim: usize) -> Result<()> {
        if heads == 0 || head_dim < self.rope_dim {
            return Err(PowerError::InferenceFailed(format!(
                "qwen35 MRoPE requires non-zero heads and head dimension at least {}, got {heads} heads of dimension {head_dim}",
                self.rope_dim
            )));
        }
        let expected = heads.checked_mul(head_dim).ok_or_else(|| {
            PowerError::InferenceFailed("qwen35 MRoPE shape overflows usize".to_string())
        })?;
        validate_len("MRoPE input", actual, expected)
    }

    fn position_axis(&self, sector: usize) -> usize {
        if sector % 3 == 1 && sector < self.interleaved_limits[1] {
            1
        } else if sector % 3 == 2 && sector < self.interleaved_limits[2] {
            2
        } else if sector.is_multiple_of(3) && sector < self.interleaved_limits[0] {
            0
        } else {
            3
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35AttentionProjectionInputs<'a> {
    /// Per-head `[query, gate]` blocks from the joint Q/G projection.
    pub q_gate: &'a [f32],
    pub k: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35AttentionProjectionWeights<'a> {
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

#[derive(Debug)]
pub struct Qwen35AttentionProjectionOutputs<'a> {
    pub q: &'a mut [f32],
    pub k: &'a mut [f32],
    pub gate: &'a mut [f32],
}

/// Split a joint Q/G projection, apply per-head Q/K RMSNorm, then IMRoPE.
pub fn qwen35_prepare_attention(
    config: Qwen35AttentionConfig,
    mrope: &Qwen35MropeConfig,
    positions: [i32; 4],
    inputs: Qwen35AttentionProjectionInputs<'_>,
    weights: Qwen35AttentionProjectionWeights<'_>,
    norm_eps: f32,
    outputs: Qwen35AttentionProjectionOutputs<'_>,
) -> Result<()> {
    if !norm_eps.is_finite() || norm_eps < 0.0 {
        return Err(PowerError::InferenceFailed(format!(
            "qwen35 attention norm epsilon must be finite and non-negative, got {norm_eps}"
        )));
    }
    if mrope.rope_dim > config.head_dim() {
        return Err(PowerError::InferenceFailed(format!(
            "qwen35 MRoPE dimension {} exceeds attention head dimension {}",
            mrope.rope_dim,
            config.head_dim()
        )));
    }
    let joint_width = config.query_width().checked_mul(2).ok_or_else(|| {
        PowerError::InferenceFailed("qwen35 joint Q/G width overflows usize".to_string())
    })?;
    validate_len("joint Q/G projection", inputs.q_gate.len(), joint_width)?;
    validate_len("K projection", inputs.k.len(), config.kv_width())?;
    validate_len("Q norm weights", weights.q_norm.len(), config.head_dim())?;
    validate_len("K norm weights", weights.k_norm.len(), config.head_dim())?;
    validate_len("prepared query", outputs.q.len(), config.query_width())?;
    validate_len("prepared key", outputs.k.len(), config.kv_width())?;
    validate_len("prepared gate", outputs.gate.len(), config.query_width())?;

    let Qwen35AttentionProjectionOutputs { q, k, gate } = outputs;
    let dim = config.head_dim();
    for head in 0..config.query_heads() {
        let source_start = head * dim * 2;
        let target_start = head * dim;
        q[target_start..target_start + dim]
            .copy_from_slice(&inputs.q_gate[source_start..source_start + dim]);
        gate[target_start..target_start + dim]
            .copy_from_slice(&inputs.q_gate[source_start + dim..source_start + dim * 2]);
    }
    k.copy_from_slice(inputs.k);

    for head in q.chunks_exact_mut(dim) {
        rms_norm_f32(head, weights.q_norm, norm_eps);
    }
    for head in k.chunks_exact_mut(dim) {
        rms_norm_f32(head, weights.k_norm, norm_eps);
    }
    mrope.apply(q, config.query_heads(), dim, positions)?;
    mrope.apply(k, config.kv_heads(), dim, positions)?;
    Ok(())
}

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(PowerError::InferenceFailed(format!(
            "qwen35 attention {name} has {actual} elements, expected {expected}"
        )))
    }
}

fn dimension_overflow(label: &str) -> PowerError {
    PowerError::InvalidFormat(format!("qwen35 MRoPE {label} overflows usize"))
}
