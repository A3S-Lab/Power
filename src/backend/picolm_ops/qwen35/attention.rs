use crate::backend::gguf_stream::GgufMeta;
use crate::error::{PowerError, Result};

/// Checked dimensions for one Qwen3.5 full-attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35AttentionConfig {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_sequence_length: usize,
    query_width: usize,
    kv_width: usize,
}

impl Qwen35AttentionConfig {
    pub fn new(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        max_sequence_length: usize,
    ) -> Result<Self> {
        for (name, value) in [
            ("query head count", query_heads),
            ("KV head count", kv_heads),
            ("head dimension", head_dim),
            ("maximum sequence length", max_sequence_length),
        ] {
            if value == 0 {
                return Err(PowerError::InvalidFormat(format!(
                    "qwen35 attention {name} must be greater than zero"
                )));
            }
        }
        if !query_heads.is_multiple_of(kv_heads) {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 attention query head count ({query_heads}) must be divisible by KV head count ({kv_heads})"
            )));
        }

        let query_width = checked_product(query_heads, head_dim, "query width")?;
        let kv_width = checked_product(kv_heads, head_dim, "KV width")?;
        checked_product(max_sequence_length, kv_width, "maximum KV cache elements")?;

        Ok(Self {
            query_heads,
            kv_heads,
            head_dim,
            max_sequence_length,
            query_width,
            kv_width,
        })
    }

    pub fn from_metadata(meta: &GgufMeta, max_sequence_length: usize) -> Result<Self> {
        let architecture = meta.architecture.qwen35().ok_or_else(|| {
            PowerError::InvalidFormat(format!(
                "qwen35 attention state requires architecture 'qwen35', got '{}'",
                meta.architecture.name()
            ))
        })?;
        if architecture.attention_key_length != architecture.attention_value_length {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 attention reference requires equal key/value lengths, got {}/{}",
                architecture.attention_key_length, architecture.attention_value_length
            )));
        }
        let model_context = usize::try_from(meta.context_length)
            .map_err(|_| dimension_overflow("model context length"))?;
        if max_sequence_length > model_context {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 attention cache limit ({max_sequence_length}) exceeds model context length ({model_context})"
            )));
        }

        Self::new(
            usize::try_from(meta.n_heads).map_err(|_| dimension_overflow("query head count"))?,
            usize::try_from(meta.n_kv_heads).map_err(|_| dimension_overflow("KV head count"))?,
            usize::try_from(architecture.attention_key_length)
                .map_err(|_| dimension_overflow("head dimension"))?,
            max_sequence_length,
        )
    }

    pub fn query_heads(self) -> usize {
        self.query_heads
    }

    pub fn kv_heads(self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(self) -> usize {
        self.head_dim
    }

    pub fn max_sequence_length(self) -> usize {
        self.max_sequence_length
    }

    pub fn query_width(self) -> usize {
        self.query_width
    }

    pub fn kv_width(self) -> usize {
        self.kv_width
    }
}

/// Pure-f32 KV state for one Qwen3.5 full-attention layer.
///
/// Queries and keys passed to [`Self::step`] must already have Q/K RMSNorm and
/// MRoPE applied. The method performs causal GQA, sigmoid output gating, and
/// appends the current K/V pair to this bounded state.
#[derive(Debug, Clone)]
pub struct Qwen35AttentionState {
    config: Qwen35AttentionConfig,
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

impl Qwen35AttentionState {
    pub fn new(config: Qwen35AttentionConfig) -> Self {
        Self {
            config,
            keys: Vec::new(),
            values: Vec::new(),
            len: 0,
        }
    }

    pub fn config(&self) -> Qwen35AttentionConfig {
        self.config
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        validate_len("query", q.len(), self.config.query_width)?;
        validate_len("key", k.len(), self.config.kv_width)?;
        validate_len("value", v.len(), self.config.kv_width)?;
        validate_len("output gate", gate.len(), self.config.query_width)?;
        validate_len("output", output.len(), self.config.query_width)?;
        if self.len == self.config.max_sequence_length {
            return Err(PowerError::InferenceFailed(format!(
                "qwen35 attention context limit {} reached",
                self.config.max_sequence_length
            )));
        }

        self.keys.extend_from_slice(k);
        self.values.extend_from_slice(v);
        self.len += 1;
        output.fill(0.0);

        let dim = self.config.head_dim;
        let heads_per_kv = self.config.query_heads / self.config.kv_heads;
        let scale = 1.0 / (dim as f32).sqrt();
        let mut scores = vec![0.0; self.len];

        for query_head in 0..self.config.query_heads {
            let kv_head = query_head / heads_per_kv;
            let q_head = &q[query_head * dim..(query_head + 1) * dim];
            for (position, score) in scores.iter_mut().enumerate() {
                let key_start = position * self.config.kv_width + kv_head * dim;
                *score = dot(q_head, &self.keys[key_start..key_start + dim]) * scale;
            }
            softmax(&mut scores);

            let output_start = query_head * dim;
            let output_head = &mut output[output_start..output_start + dim];
            for (position, score) in scores.iter().copied().enumerate() {
                let value_start = position * self.config.kv_width + kv_head * dim;
                let value_head = &self.values[value_start..value_start + dim];
                for index in 0..dim {
                    output_head[index] += score * value_head[index];
                }
            }
            for index in 0..dim {
                output_head[index] *= sigmoid(gate[output_start + index]);
            }
        }
        Ok(())
    }

    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.len {
            return Err(PowerError::InferenceFailed(format!(
                "cannot extend qwen35 attention state from {} to {new_len} tokens",
                self.len
            )));
        }
        let elements = new_len * self.config.kv_width;
        self.keys[elements..].fill(0.0);
        self.values[elements..].fill(0.0);
        self.keys.truncate(elements);
        self.values.truncate(elements);
        self.len = new_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.keys.clear();
        self.values.clear();
        self.len = 0;
    }

    pub fn memory_bytes(&self) -> usize {
        self.keys
            .capacity()
            .saturating_add(self.values.capacity())
            .saturating_mul(std::mem::size_of::<f32>())
    }
}

impl Drop for Qwen35AttentionState {
    fn drop(&mut self) {
        self.clear();
    }
}

fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    if sum > 0.0 {
        let inverse = 1.0 / sum;
        values.iter_mut().for_each(|value| *value *= inverse);
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
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

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| dimension_overflow(label))
}

fn dimension_overflow(label: &str) -> PowerError {
    PowerError::InvalidFormat(format!("qwen35 attention {label} overflows usize"))
}
