use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::PowerError;
use crate::error::Result;

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;
use super::execute;
use super::gelu_erf;

mod cpu;
#[cfg(feature = "embedded-cuda")]
mod cuda;

#[cfg(feature = "embedded-cuda")]
pub(super) fn cuda_channel_bias(input: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
    cuda::bias(input, bias)
}

#[cfg(all(test, feature = "embedded-cuda"))]
pub(super) fn cuda_channel_bias_relu(input: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
    cuda::relu(input, bias)
}

#[cfg(all(test, feature = "embedded-cuda"))]
pub(super) fn cuda_channel_bias_relu_into(
    output: &Tensor,
    input: &Tensor,
    bias: &Tensor,
) -> candle_core::Result<()> {
    cuda::relu_into(output, input, bias)
}

const MAX_IDENTITY_NODES: usize = 2;

pub(super) struct FusedOutput {
    pub(super) value: GraphValue,
    pub(super) consumed_nodes: usize,
}

pub(super) struct ExecutionContext<'a> {
    pub(super) values: &'a HashMap<String, GraphValue>,
    pub(super) scalar_constants: &'a HashMap<String, f32>,
    pub(super) use_counts: &'a HashMap<String, usize>,
    pub(super) retained_output: &'a str,
    pub(super) device: &'a Device,
    pub(super) element_limit: usize,
    pub(super) cancellation: &'a CancellationToken,
}

/// Executes a reviewed Conv-to-channel-bias-to-activation window while keeping
/// the convolution itself on Candle's ordinary backend path.
///
/// Only adjacent, private intermediates and an exact contiguous NCHW channel
/// bias are eligible. Unsupported topology, devices, dtypes, layouts, or
/// shapes return `None` so the executor retains its node-by-node behavior.
pub(super) fn try_execute(
    nodes: &[GraphNode],
    context: ExecutionContext<'_>,
) -> Result<Option<FusedOutput>> {
    let ExecutionContext {
        values,
        scalar_constants,
        use_counts,
        retained_output,
        device,
        element_limit,
        cancellation,
    } = context;
    let Some(matched) = matched_window(nodes, use_counts, retained_output) else {
        return Ok(None);
    };
    if device.is_cpu() {
        return execute_cpu(
            matched,
            values,
            scalar_constants,
            device,
            element_limit,
            cancellation,
        );
    }
    if device.is_cuda() {
        #[cfg(feature = "embedded-cuda")]
        {
            return execute_cuda(
                matched,
                values,
                scalar_constants,
                device,
                element_limit,
                cancellation,
            );
        }
    }

    Ok(None)
}

fn execute_cpu(
    matched: MatchedWindow<'_>,
    values: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    device: &Device,
    element_limit: usize,
    cancellation: &CancellationToken,
) -> Result<Option<FusedOutput>> {
    let bias = tensor(values, matched.bias, matched.add)?;
    if bias.dtype() != DType::F32
        || !bias.device().is_cpu()
        || !bias.device().same_device(device)
        || !bias.is_contiguous()
    {
        return Ok(None);
    }

    if let MatchedActivation::Residual { node, residual } = matched.activation {
        let residual = tensor(values, residual, node)?;
        if residual.dtype() != DType::F32
            || !residual.device().is_cpu()
            || !residual.device().same_device(device)
            || !residual.is_contiguous()
        {
            return Ok(None);
        }
        let (Some(input_name), Some(kernel_name)) = (
            matched.convolution.inputs.first(),
            matched.convolution.inputs.get(1),
        ) else {
            return Ok(None);
        };
        let input = value(values, input_name, matched.convolution)?;
        let kernel = value(values, kernel_name, matched.convolution)?;
        let bias_value = value(values, matched.bias, matched.add)?;
        let output_channels = kernel
            .tensor(&matched.convolution.name)?
            .dims4()
            .ok()
            .map(|dimensions| dimensions.0);
        if !output_channels.is_some_and(|channels| channel_bias_shape(bias_value, channels)) {
            return Ok(None);
        }
        let Some(output) = super::spatial::try_conv_with_residual(
            matched.convolution,
            &[input, kernel, bias_value],
            device,
            residual,
        )?
        else {
            return Ok(None);
        };
        let output = output.tensor(&matched.convolution.name)?;
        if output.elem_count() == 0 || output.elem_count() > element_limit {
            return Err(PowerError::InferenceFailed(format!(
                "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
                matched.convolution.name,
                output.elem_count(),
            )));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        return Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output.clone()),
            consumed_nodes: matched.consumed_nodes,
        }));
    }

    if matches!(matched.activation, MatchedActivation::Bias) {
        let (Some(input_name), Some(kernel_name)) = (
            matched.convolution.inputs.first(),
            matched.convolution.inputs.get(1),
        ) else {
            return Ok(None);
        };
        let input = value(values, input_name, matched.convolution)?;
        let kernel = value(values, kernel_name, matched.convolution)?;
        let bias = value(values, matched.bias, matched.add)?;
        let output_channels = kernel
            .tensor(&matched.convolution.name)?
            .dims4()
            .ok()
            .map(|dimensions| dimensions.0);
        if !output_channels.is_some_and(|channels| channel_bias_shape(bias, channels)) {
            return Ok(None);
        }
        let output = super::spatial::conv(matched.convolution, &[input, kernel, bias], device)?;
        let output = output.tensor(&matched.convolution.name)?;
        if output.elem_count() == 0 || output.elem_count() > element_limit {
            return Err(PowerError::InferenceFailed(format!(
                "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
                matched.convolution.name,
                output.elem_count(),
            )));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        return Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output.clone()),
            consumed_nodes: matched.consumed_nodes,
        }));
    }

    let gelu_parameters = match matched.activation {
        MatchedActivation::Gelu {
            divisor,
            offset,
            scale,
        } => {
            let (Some(&divisor), Some(&offset), Some(&scale)) = (
                scalar_constants.get(divisor),
                scalar_constants.get(offset),
                scalar_constants.get(scale),
            ) else {
                return Ok(None);
            };
            Some((divisor, offset, scale))
        }
        _ => None,
    };
    let gated = match matched.activation {
        MatchedActivation::GatedHardSigmoid { node, multiplicand } => Some((
            tensor(values, multiplicand, node)?,
            node.float("alpha", 0.2)? as f32,
            node.float("beta", 0.5)? as f32,
        )),
        _ => None,
    };

    let convolution = execute(matched.convolution, values, scalar_constants, device)?;
    let convolution = convolution.tensor(&matched.convolution.name)?;
    if convolution.dtype() != DType::F32
        || !convolution.device().is_cpu()
        || !convolution.device().same_device(device)
        || convolution.elem_count() == 0
        || !convolution.is_contiguous()
    {
        return Ok(None);
    }
    if convolution.elem_count() > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
            matched.convolution.name,
            convolution.elem_count(),
        )));
    }
    let (_, channels, _, _) = convolution
        .dims4()
        .map_err(|error| fused_error(matched, error))?;
    if bias.dims4().ok() != Some((1, channels, 1, 1)) {
        return Ok(None);
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }

    let output = match matched.activation {
        MatchedActivation::Bias | MatchedActivation::Residual { .. } => return Ok(None),
        MatchedActivation::Relu => cpu::relu(convolution, bias),
        MatchedActivation::Gelu { .. } => {
            let Some((divisor, offset, scale)) = gelu_parameters else {
                return Ok(None);
            };
            cpu::gelu_erf(convolution, bias, divisor, offset, scale)
        }
        MatchedActivation::GatedHardSigmoid { .. } => {
            let Some((multiplicand, alpha, beta)) = gated else {
                return Ok(None);
            };
            if multiplicand.dtype() != DType::F32
                || !multiplicand.device().is_cpu()
                || !multiplicand.device().same_device(device)
                || multiplicand.elem_count() == 0
                || !multiplicand.is_contiguous()
                || !gated_nchw_shapes(convolution.dims(), multiplicand.dims())
            {
                return Ok(None);
            }
            cpu::gated_hard_sigmoid_mul(multiplicand, convolution, bias, alpha, beta)
        }
    }
    .map_err(|error| fused_error(matched, error))?;
    Ok(Some(FusedOutput {
        value: GraphValue::Tensor(output),
        consumed_nodes: matched.consumed_nodes,
    }))
}

#[cfg(feature = "embedded-cuda")]
fn execute_cuda(
    matched: MatchedWindow<'_>,
    values: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    device: &Device,
    element_limit: usize,
    cancellation: &CancellationToken,
) -> Result<Option<FusedOutput>> {
    let bias = tensor(values, matched.bias, matched.add)?;
    let gelu_parameters = match matched.activation {
        MatchedActivation::Gelu {
            divisor,
            offset,
            scale,
        } => {
            let (Some(&divisor), Some(&offset), Some(&scale)) = (
                scalar_constants.get(divisor),
                scalar_constants.get(offset),
                scalar_constants.get(scale),
            ) else {
                return Ok(None);
            };
            Some((divisor, offset, scale))
        }
        _ => None,
    };
    let gated = match matched.activation {
        MatchedActivation::GatedHardSigmoid { node, multiplicand } => Some((
            tensor(values, multiplicand, node)?,
            node.float("alpha", 0.2)? as f32,
            node.float("beta", 0.5)? as f32,
        )),
        _ => None,
    };
    if bias.dtype() != DType::F32
        || !bias.device().is_cuda()
        || !bias.device().same_device(device)
        || !bias.is_contiguous()
    {
        return Ok(None);
    }
    if let Some((multiplicand, _, _)) = gated {
        if multiplicand.dtype() != DType::F32
            || !multiplicand.device().same_device(device)
            || multiplicand.elem_count() == 0
            || u32::try_from(multiplicand.elem_count()).is_err()
            || !multiplicand.is_contiguous()
        {
            return Ok(None);
        }
    }

    if let MatchedActivation::Residual { node, residual } = matched.activation {
        let residual = tensor(values, residual, node)?;
        if residual.dtype() != DType::F32
            || !residual.device().is_cuda()
            || !residual.device().same_device(device)
            || !residual.is_contiguous()
        {
            return Ok(None);
        }
        let convolution = execute(matched.convolution, values, scalar_constants, device)?;
        let convolution = convolution.tensor(&matched.convolution.name)?;
        if convolution.dtype() != DType::F32
            || !convolution.device().is_cuda()
            || !convolution.device().same_device(device)
            || convolution.elem_count() == 0
            || u32::try_from(convolution.elem_count()).is_err()
            || !convolution.is_contiguous()
        {
            return Ok(None);
        }
        if convolution.elem_count() > element_limit {
            return Err(PowerError::InferenceFailed(format!(
                "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
                matched.convolution.name,
                convolution.elem_count(),
            )));
        }
        let (_, channels, _, _) = convolution
            .dims4()
            .map_err(|error| fused_error(matched, error))?;
        if bias.dims4().ok() != Some((1, channels, 1, 1)) || residual.dims() != convolution.dims() {
            return Ok(None);
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        let output = cuda::bias_residual(convolution, residual, bias)
            .map_err(|error| fused_error(matched, error))?;
        return Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output),
            consumed_nodes: matched.consumed_nodes,
        }));
    }

    if matches!(matched.activation, MatchedActivation::Bias) {
        let (Some(input_name), Some(kernel_name)) = (
            matched.convolution.inputs.first(),
            matched.convolution.inputs.get(1),
        ) else {
            return Ok(None);
        };
        let input = value(values, input_name, matched.convolution)?;
        let kernel = value(values, kernel_name, matched.convolution)?;
        let bias = value(values, matched.bias, matched.add)?;
        let output_channels = kernel
            .tensor(&matched.convolution.name)?
            .dims4()
            .ok()
            .map(|dimensions| dimensions.0);
        if !output_channels.is_some_and(|channels| channel_bias_shape(bias, channels)) {
            return Ok(None);
        }
        let output = super::conv(matched.convolution, &[input, kernel, bias], device)?;
        let output = output.tensor(&matched.convolution.name)?;
        if output.elem_count() == 0 || output.elem_count() > element_limit {
            return Err(PowerError::InferenceFailed(format!(
                "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
                matched.convolution.name,
                output.elem_count(),
            )));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        return Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output.clone()),
            consumed_nodes: matched.consumed_nodes,
        }));
    }

    let convolution = execute(matched.convolution, values, scalar_constants, device)?;
    let convolution = convolution.tensor(&matched.convolution.name)?;
    if convolution.dtype() != DType::F32
        || !convolution.device().is_cuda()
        || !convolution.device().same_device(device)
        || convolution.elem_count() == 0
        || u32::try_from(convolution.elem_count()).is_err()
        || !convolution.is_contiguous()
    {
        return Ok(None);
    }
    if convolution.elem_count() > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
            matched.convolution.name,
            convolution.elem_count(),
        )));
    }
    let (_, channels, _, _) = convolution
        .dims4()
        .map_err(|error| fused_error(matched, error))?;
    if bias.dims4().ok() != Some((1, channels, 1, 1)) {
        return Ok(None);
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }

    let output = match matched.activation {
        MatchedActivation::Bias | MatchedActivation::Residual { .. } => return Ok(None),
        MatchedActivation::Relu => cuda::relu(convolution, bias),
        MatchedActivation::Gelu { .. } => {
            let Some((divisor, offset, scale)) = gelu_parameters else {
                return Ok(None);
            };
            cuda::gelu_erf(convolution, bias, divisor, offset, scale)
        }
        MatchedActivation::GatedHardSigmoid { .. } => {
            let Some((multiplicand, alpha, beta)) = gated else {
                return Ok(None);
            };
            if !gated_nchw_shapes(convolution.dims(), multiplicand.dims()) {
                return Ok(None);
            }
            cuda::gated_hard_sigmoid_mul(multiplicand, convolution, bias, alpha, beta)
        }
    }
    .map_err(|error| fused_error(matched, error))?;
    Ok(Some(FusedOutput {
        value: GraphValue::Tensor(output),
        consumed_nodes: matched.consumed_nodes,
    }))
}

#[derive(Clone, Copy)]
struct MatchedWindow<'a> {
    convolution: &'a GraphNode,
    add: &'a GraphNode,
    bias: &'a str,
    activation: MatchedActivation<'a>,
    consumed_nodes: usize,
}

#[derive(Clone, Copy)]
enum MatchedActivation<'a> {
    Bias,
    Residual {
        node: &'a GraphNode,
        residual: &'a str,
    },
    Relu,
    Gelu {
        divisor: &'a str,
        offset: &'a str,
        scale: &'a str,
    },
    GatedHardSigmoid {
        node: &'a GraphNode,
        multiplicand: &'a str,
    },
}

fn matched_window<'a>(
    nodes: &'a [GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedWindow<'a>> {
    let convolution = nodes.first()?;
    let add = nodes.get(1)?;
    if convolution.op != GraphOp::Conv
        || add.op != GraphOp::Add
        || convolution.inputs.len() != 2
        || !add.attributes.is_empty()
    {
        return None;
    }
    let [convolution_output] = convolution.outputs.as_slice() else {
        return None;
    };
    let [add_left, add_right] = add.inputs.as_slice() else {
        return None;
    };
    let bias = match (
        add_left == convolution_output,
        add_right == convolution_output,
    ) {
        (true, false) => add_right.as_str(),
        (false, true) => add_left.as_str(),
        _ => return None,
    };
    if !private_intermediate(convolution_output, 1, use_counts, retained_output) {
        return None;
    }
    let [add_output] = add.outputs.as_slice() else {
        return None;
    };
    if add_output == retained_output {
        return None;
    }

    let mut activation_input = add_output.as_str();
    let mut cursor = 2;
    let mut identity_nodes = 0;
    while nodes
        .get(cursor)
        .is_some_and(|node| node.op == GraphOp::Identity)
    {
        if identity_nodes == MAX_IDENTITY_NODES {
            return None;
        }
        let identity = &nodes[cursor];
        let [input] = identity.inputs.as_slice() else {
            return None;
        };
        let [output] = identity.outputs.as_slice() else {
            return None;
        };
        if input != activation_input
            || !identity.attributes.is_empty()
            || !private_intermediate(activation_input, 1, use_counts, retained_output)
            || output == retained_output
        {
            return None;
        }
        activation_input = output;
        identity_nodes += 1;
        cursor += 1;
    }

    let activation_node = nodes.get(cursor)?;
    let activation = match activation_node.op {
        GraphOp::Add => {
            let [left, right] = activation_node.inputs.as_slice() else {
                return None;
            };
            let residual = match (left == activation_input, right == activation_input) {
                (true, false) => right.as_str(),
                (false, true) => left.as_str(),
                _ => return None,
            };
            if !activation_node.attributes.is_empty()
                || !private_intermediate(activation_input, 1, use_counts, retained_output)
            {
                return None;
            }
            MatchedWindow {
                convolution,
                add,
                bias,
                activation: MatchedActivation::Residual {
                    node: activation_node,
                    residual,
                },
                consumed_nodes: cursor + 1,
            }
        }
        GraphOp::Relu => {
            let [input] = activation_node.inputs.as_slice() else {
                return None;
            };
            if input != activation_input
                || !activation_node.attributes.is_empty()
                || !private_intermediate(activation_input, 1, use_counts, retained_output)
            {
                return None;
            }
            MatchedWindow {
                convolution,
                add,
                bias,
                activation: MatchedActivation::Relu,
                consumed_nodes: cursor + 1,
            }
        }
        GraphOp::HardSigmoid => {
            let [input] = activation_node.inputs.as_slice() else {
                return None;
            };
            let [activation_output] = activation_node.outputs.as_slice() else {
                return None;
            };
            if input != activation_input
                || !private_intermediate(activation_input, 1, use_counts, retained_output)
                || !private_intermediate(activation_output, 1, use_counts, retained_output)
            {
                return None;
            }
            let multiply = nodes.get(cursor + 1)?;
            if multiply.op != GraphOp::Mul || !multiply.attributes.is_empty() {
                return None;
            }
            let [left, right] = multiply.inputs.as_slice() else {
                return None;
            };
            let multiplicand = match (left == activation_output, right == activation_output) {
                (true, false) => right.as_str(),
                (false, true) => left.as_str(),
                _ => return None,
            };
            MatchedWindow {
                convolution,
                add,
                bias,
                activation: MatchedActivation::GatedHardSigmoid {
                    node: activation_node,
                    multiplicand,
                },
                consumed_nodes: cursor + 2,
            }
        }
        GraphOp::Div => {
            let end = cursor.checked_add(5)?;
            let gelu =
                gelu_erf::matched_inputs(nodes.get(cursor..end)?, use_counts, retained_output)?;
            if gelu.input != activation_input
                || !private_intermediate(activation_input, 2, use_counts, retained_output)
            {
                return None;
            }
            MatchedWindow {
                convolution,
                add,
                bias,
                activation: MatchedActivation::Gelu {
                    divisor: gelu.divisor,
                    offset: gelu.offset,
                    scale: gelu.scale,
                },
                consumed_nodes: end,
            }
        }
        _ => MatchedWindow {
            convolution,
            add,
            bias,
            activation: MatchedActivation::Bias,
            consumed_nodes: cursor,
        },
    };
    Some(activation)
}

fn channel_bias_shape(value: &GraphValue, channels: usize) -> bool {
    let Ok(value) = value.tensor("convolution channel-bias fusion") else {
        return false;
    };
    value.dims1().ok() == Some(channels) || value.dims4().ok() == Some((1, channels, 1, 1))
}

fn private_intermediate(
    output: &str,
    expected_uses: usize,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    output != retained_output && use_counts.get(output).copied() == Some(expected_uses)
}

fn gated_nchw_shapes(gate: &[usize], multiplicand: &[usize]) -> bool {
    let ([gate_batch, gate_channels, gate_height, gate_width], [batch, channels, _, _]) =
        (gate, multiplicand)
    else {
        return false;
    };
    gate == multiplicand
        || (gate_batch == batch
            && gate_channels == channels
            && *gate_height == 1
            && *gate_width == 1)
}

fn tensor<'a>(
    values: &'a HashMap<String, GraphValue>,
    input: &str,
    node: &GraphNode,
) -> Result<&'a Tensor> {
    values
        .get(input)
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' could not resolve input '{input}'",
                node.name
            ))
        })?
        .tensor(&node.name)
}

fn value<'a>(
    values: &'a HashMap<String, GraphValue>,
    input: &str,
    node: &GraphNode,
) -> Result<&'a GraphValue> {
    values.get(input).ok_or_else(|| {
        PowerError::InferenceFailed(format!(
            "static graph node '{}' could not resolve input '{input}'",
            node.name
        ))
    })
}

fn fused_error(matched: MatchedWindow<'_>, error: impl std::fmt::Display) -> PowerError {
    PowerError::InferenceFailed(format!(
        "static graph nodes '{}' through node offset {} fused channel-bias activation failed: {error}",
        matched.convolution.name,
        matched.consumed_nodes - 1,
    ))
}

#[cfg(test)]
mod tests;
