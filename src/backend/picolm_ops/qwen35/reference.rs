use super::state::Qwen35RecurrentState;
use crate::backend::picolm_ops::norm::rms_norm_f32;
use crate::error::{PowerError, Result};

/// Pre-projected inputs for one Qwen3.5 recurrent token.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35RecurrentInputs<'a> {
    /// Joint Q/K/V projection in that order.
    pub qkv: &'a [f32],
    /// Per-value-channel output gate projection.
    pub z: &'a [f32],
    /// Per-value-head beta logits.
    pub beta: &'a [f32],
    /// Per-value-head alpha logits.
    pub alpha: &'a [f32],
}

/// Dequantized weights needed by the recurrent reference core.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35RecurrentWeights<'a> {
    /// Channel-major depthwise convolution weights with the kernel dimension
    /// contiguous, matching GGUF shape `[kernel, channels]`.
    pub conv: &'a [f32],
    /// Per-value-head time-step bias.
    pub dt_bias: &'a [f32],
    /// Per-value-head negative decay coefficient (`ssm_a`).
    pub decay: &'a [f32],
    /// Shared per-head RMSNorm weights.
    pub norm: &'a [f32],
}

/// Pure-f32 single-token reference for the Qwen3.5 recurrent attention core.
///
/// Linear input/output projections stay outside this function so the same
/// state transition can serve as a CPU oracle for quantized and CUDA matmuls.
/// When key and value head counts differ, all value-head projections and
/// weights must use the tiled order stored by the llama.cpp GGUF converter.
pub fn qwen35_recurrent_step(
    state: &mut Qwen35RecurrentState,
    inputs: Qwen35RecurrentInputs<'_>,
    weights: Qwen35RecurrentWeights<'_>,
    norm_eps: f32,
    output: &mut [f32],
) -> Result<()> {
    let config = state.config();
    if !norm_eps.is_finite() || norm_eps < 0.0 {
        return Err(PowerError::InferenceFailed(format!(
            "qwen35 recurrent norm epsilon must be finite and non-negative, got {norm_eps}"
        )));
    }

    validate_len("QKV projection", inputs.qkv.len(), config.conv_width())?;
    validate_len(
        "output gate projection",
        inputs.z.len(),
        config.value_width(),
    )?;
    validate_len("beta projection", inputs.beta.len(), config.value_heads())?;
    validate_len("alpha projection", inputs.alpha.len(), config.value_heads())?;
    validate_len(
        "convolution weights",
        weights.conv.len(),
        config.conv_kernel_elements(),
    )?;
    validate_len(
        "time-step bias",
        weights.dt_bias.len(),
        config.value_heads(),
    )?;
    validate_len("decay weights", weights.decay.len(), config.value_heads())?;
    validate_len("RMSNorm weights", weights.norm.len(), config.head_dim())?;
    validate_len("output", output.len(), config.value_width())?;

    let mut convolved = vec![0.0; config.conv_width()];
    state.conv_step(inputs.qkv, weights.conv, &mut convolved)?;
    convolved.iter_mut().for_each(|value| *value = silu(*value));

    let (q, key_value) = convolved.split_at_mut(config.key_width());
    let (k, v) = key_value.split_at_mut(config.key_width());
    for head in q.chunks_exact_mut(config.head_dim()) {
        l2_normalize(head, norm_eps);
    }
    for head in k.chunks_exact_mut(config.head_dim()) {
        l2_normalize(head, norm_eps);
    }

    let beta = inputs.beta.iter().copied().map(sigmoid).collect::<Vec<_>>();
    let gate = inputs
        .alpha
        .iter()
        .zip(weights.dt_bias)
        .zip(weights.decay)
        .map(|((alpha, bias), decay)| softplus(*alpha + *bias) * *decay)
        .collect::<Vec<_>>();

    let mut core_output = vec![0.0; config.value_width()];
    state.gated_delta_step(q, k, v, &gate, &beta, &mut core_output)?;

    for value_head in 0..config.value_heads() {
        let start = value_head * config.head_dim();
        let end = start + config.head_dim();
        let head = &mut core_output[start..end];
        rms_norm_f32(head, weights.norm, norm_eps);
        for index in start..end {
            output[index] = core_output[index] * silu(inputs.z[index]);
        }
    }
    Ok(())
}

pub(super) fn l2_normalize(values: &mut [f32], eps: f32) {
    let sum_squares = values.iter().map(|value| value * value).sum::<f32>();
    let scale = 1.0 / (sum_squares + eps).sqrt();
    values.iter_mut().for_each(|value| *value *= scale);
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(PowerError::InferenceFailed(format!(
            "qwen35 recurrent {name} has {actual} elements, expected {expected}"
        )))
    }
}
