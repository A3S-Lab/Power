use crate::backend::gguf_stream::Qwen35Architecture;
use crate::error::{PowerError, Result};

/// Checked dimensions for one Qwen3.5 Gated DeltaNet layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35RecurrentConfig {
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    conv_kernel: usize,
    key_width: usize,
    value_width: usize,
    conv_width: usize,
    conv_kernel_elements: usize,
    conv_state_elements: usize,
    delta_state_elements: usize,
}

impl Qwen35RecurrentConfig {
    pub fn new(
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
        conv_kernel: usize,
    ) -> Result<Self> {
        for (name, value) in [
            ("key head count", key_heads),
            ("value head count", value_heads),
            ("head dimension", head_dim),
            ("convolution kernel", conv_kernel),
        ] {
            if value == 0 {
                return Err(PowerError::InvalidFormat(format!(
                    "qwen35 recurrent {name} must be greater than zero"
                )));
            }
        }
        if !value_heads.is_multiple_of(key_heads) {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 recurrent value head count ({value_heads}) must be divisible by key head count ({key_heads})"
            )));
        }

        let key_width = checked_product(&[key_heads, head_dim], "key width")?;
        let value_width = checked_product(&[value_heads, head_dim], "value width")?;
        let conv_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| dimension_overflow("convolution width"))?;
        let conv_kernel_elements =
            checked_product(&[conv_width, conv_kernel], "convolution weights")?;
        let conv_state_elements =
            checked_product(&[conv_width, conv_kernel - 1], "convolution state")?;
        let delta_state_elements =
            checked_product(&[value_heads, head_dim, head_dim], "Gated DeltaNet state")?;

        Ok(Self {
            key_heads,
            value_heads,
            head_dim,
            conv_kernel,
            key_width,
            value_width,
            conv_width,
            conv_kernel_elements,
            conv_state_elements,
            delta_state_elements,
        })
    }

    pub fn from_architecture(architecture: &Qwen35Architecture) -> Result<Self> {
        let key_heads = usize::try_from(architecture.ssm_group_count)
            .map_err(|_| dimension_overflow("key head count"))?;
        let value_heads = usize::try_from(architecture.ssm_time_step_rank)
            .map_err(|_| dimension_overflow("value head count"))?;
        let head_dim = usize::try_from(architecture.ssm_state_size)
            .map_err(|_| dimension_overflow("head dimension"))?;
        let conv_kernel = usize::try_from(architecture.ssm_conv_kernel)
            .map_err(|_| dimension_overflow("convolution kernel"))?;
        let config = Self::new(key_heads, value_heads, head_dim, conv_kernel)?;
        let inner_size = usize::try_from(architecture.ssm_inner_size)
            .map_err(|_| dimension_overflow("inner size"))?;
        if config.value_width != inner_size {
            return Err(PowerError::InvalidFormat(format!(
                "qwen35 recurrent value width ({}) does not match ssm.inner_size ({inner_size})",
                config.value_width
            )));
        }
        Ok(config)
    }

    pub fn key_heads(self) -> usize {
        self.key_heads
    }

    pub fn value_heads(self) -> usize {
        self.value_heads
    }

    pub fn head_dim(self) -> usize {
        self.head_dim
    }

    pub fn conv_kernel(self) -> usize {
        self.conv_kernel
    }

    pub fn key_width(self) -> usize {
        self.key_width
    }

    pub fn value_width(self) -> usize {
        self.value_width
    }

    pub fn conv_width(self) -> usize {
        self.conv_width
    }

    pub fn conv_kernel_elements(self) -> usize {
        self.conv_kernel_elements
    }

    pub fn conv_state_elements(self) -> usize {
        self.conv_state_elements
    }

    pub fn delta_state_elements(self) -> usize {
        self.delta_state_elements
    }
}

/// Per-channel causal convolution history.
///
/// History and GGUF kernels are channel-major. For a kernel of width `K`, a
/// channel owns `K - 1` history values ordered oldest to newest and `K`
/// consecutive weights ordered from the oldest sample to the current sample.
#[derive(Debug, Clone)]
pub struct CausalConv1dState {
    channels: usize,
    kernel_size: usize,
    kernel_elements: usize,
    history: Vec<f32>,
}

impl CausalConv1dState {
    pub fn new(channels: usize, kernel_size: usize) -> Result<Self> {
        if channels == 0 || kernel_size == 0 {
            return Err(PowerError::InvalidFormat(
                "qwen35 causal convolution dimensions must be greater than zero".to_string(),
            ));
        }
        let kernel_elements = checked_product(&[channels, kernel_size], "convolution weights")?;
        let history_elements = checked_product(&[channels, kernel_size - 1], "convolution state")?;
        Ok(Self {
            channels,
            kernel_size,
            kernel_elements,
            history: vec![0.0; history_elements],
        })
    }

    pub fn step(&mut self, input: &[f32], kernel: &[f32], output: &mut [f32]) -> Result<()> {
        validate_len("causal convolution input", input.len(), self.channels)?;
        validate_len(
            "causal convolution kernel",
            kernel.len(),
            self.kernel_elements,
        )?;
        validate_len("causal convolution output", output.len(), self.channels)?;

        let history_len = self.kernel_size - 1;
        for channel in 0..self.channels {
            let history_start = channel * history_len;
            let kernel_start = channel * self.kernel_size;
            let mut sum = input[channel] * kernel[kernel_start + history_len];
            for tap in 0..history_len {
                sum += self.history[history_start + tap] * kernel[kernel_start + tap];
            }
            output[channel] = sum;

            if history_len > 1 {
                self.history.copy_within(
                    history_start + 1..history_start + history_len,
                    history_start,
                );
            }
            if history_len > 0 {
                self.history[history_start + history_len - 1] = input[channel];
            }
        }
        Ok(())
    }

    pub fn history(&self) -> &[f32] {
        &self.history
    }

    pub fn clear(&mut self) {
        self.history.fill(0.0);
    }

    pub fn memory_bytes(&self) -> usize {
        self.history
            .capacity()
            .saturating_mul(std::mem::size_of::<f32>())
    }
}

impl Drop for CausalConv1dState {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Mutable single-sequence state for one Qwen3.5 recurrent layer.
#[derive(Debug, Clone)]
pub struct Qwen35RecurrentState {
    config: Qwen35RecurrentConfig,
    conv: CausalConv1dState,
    /// Transposed state layout `[value_head, output_dim, input_dim]`, matching
    /// GGML's contiguous dot-product representation.
    delta: Vec<f32>,
}

impl Qwen35RecurrentState {
    pub fn new(config: Qwen35RecurrentConfig) -> Result<Self> {
        let conv = CausalConv1dState::new(config.conv_width, config.conv_kernel)?;
        Ok(Self {
            config,
            conv,
            delta: vec![0.0; config.delta_state_elements],
        })
    }

    pub fn config(&self) -> Qwen35RecurrentConfig {
        self.config
    }

    pub fn conv_history(&self) -> &[f32] {
        self.conv.history()
    }

    pub fn delta_state(&self) -> &[f32] {
        &self.delta
    }

    pub fn conv_step(&mut self, input: &[f32], kernel: &[f32], output: &mut [f32]) -> Result<()> {
        self.conv.step(input, kernel, output)
    }

    /// Apply one autoregressive Gated DeltaNet update.
    ///
    /// `gate` contains log-decay values and `beta` contains already-sigmoided
    /// update strengths, one scalar per value head.
    pub fn gated_delta_step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        validate_len("Gated DeltaNet query", q.len(), self.config.key_width)?;
        validate_len("Gated DeltaNet key", k.len(), self.config.key_width)?;
        validate_len("Gated DeltaNet value", v.len(), self.config.value_width)?;
        validate_len("Gated DeltaNet gate", gate.len(), self.config.value_heads)?;
        validate_len("Gated DeltaNet beta", beta.len(), self.config.value_heads)?;
        validate_len(
            "Gated DeltaNet output",
            output.len(),
            self.config.value_width,
        )?;

        let dim = self.config.head_dim;
        let matrix_elements = dim * dim;
        let scale = 1.0 / (dim as f32).sqrt();

        for value_head in 0..self.config.value_heads {
            // llama.cpp's Qwen3.5 GGUF converter stores value heads in tiled
            // order so GGML can repeat Q/K heads without an extra transpose.
            let key_head = value_head % self.config.key_heads;
            let q_head = &q[key_head * dim..(key_head + 1) * dim];
            let k_head = &k[key_head * dim..(key_head + 1) * dim];
            let v_head = &v[value_head * dim..(value_head + 1) * dim];
            let output_head = &mut output[value_head * dim..(value_head + 1) * dim];
            let state_start = value_head * matrix_elements;
            let state = &mut self.delta[state_start..state_start + matrix_elements];
            let decay = gate[value_head].exp();

            for output_dim in 0..dim {
                let row = &mut state[output_dim * dim..(output_dim + 1) * dim];
                for value in row.iter_mut() {
                    *value *= decay;
                }

                let prediction = dot(row, k_head);
                let correction = (v_head[output_dim] - prediction) * beta[value_head];
                for input_dim in 0..dim {
                    row[input_dim] += k_head[input_dim] * correction;
                }
                output_head[output_dim] = dot(row, q_head) * scale;
            }
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.conv.clear();
        self.delta.fill(0.0);
    }

    pub fn memory_bytes(&self) -> usize {
        self.conv.memory_bytes().saturating_add(
            self.delta
                .capacity()
                .saturating_mul(std::mem::size_of::<f32>()),
        )
    }
}

impl Drop for Qwen35RecurrentState {
    fn drop(&mut self) {
        self.delta.fill(0.0);
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
            "qwen35 {name} has {actual} elements, expected {expected}"
        )))
    }
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| dimension_overflow(label))
    })
}

fn dimension_overflow(label: &str) -> PowerError {
    PowerError::InvalidFormat(format!("qwen35 recurrent {label} overflows usize"))
}
