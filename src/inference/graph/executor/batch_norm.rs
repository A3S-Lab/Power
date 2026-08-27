use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;
use super::convolution_post::{BatchNormActivation, ConvolutionPostOperation};
#[cfg(feature = "embedded-cuda")]
use super::convolution_post::{CudaBatchNormActivation, CudaBatchNormPostOperation};
use super::gelu_erf;

mod cpu;
#[cfg(feature = "embedded-cuda")]
mod cuda;

#[cfg(feature = "embedded-cuda")]
pub(super) fn execute_cuda_post_in_place(
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    device: &candle_core::cuda_backend::CudaDevice,
    batch: usize,
    channels: usize,
    spatial: usize,
    post_operation: &CudaBatchNormPostOperation,
) -> candle_core::Result<()> {
    cuda::execute_post_in_place(output, device, batch, channels, spatial, post_operation)
}

#[cfg_attr(not(feature = "embedded-cuda"), allow(dead_code))]
pub(super) struct PreparedBatchNorm {
    scale_and_bias: Tensor,
    mean_and_stddev: Tensor,
    activation: Activation,
    consumed_nodes: usize,
    convolution_post_operation: Option<ConvolutionPostOperation>,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(feature = "embedded-cuda"), allow(dead_code))]
enum Activation {
    Identity,
    Relu,
    HardSwish {
        alpha: f32,
        beta: f32,
    },
    Sigmoid,
    Swish,
    GeluErf {
        divisor: f32,
        offset: f32,
        scale: f32,
    },
}

pub(super) struct FusedOutput {
    pub(super) value: GraphValue,
    pub(super) consumed_nodes: usize,
}

pub(super) fn prepare(
    nodes: &[GraphNode],
    constants: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> HashMap<String, PreparedBatchNorm> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            prepare_node(
                &nodes[index..],
                constants,
                scalar_constants,
                use_counts,
                retained_output,
            )
            .map(|prepared| (node.name.clone(), prepared))
        })
        .collect()
}

fn prepare_node(
    nodes: &[GraphNode],
    constants: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<PreparedBatchNorm> {
    let node = nodes.first()?;
    if node.op != GraphOp::BatchNormalization || node.inputs.len() != 5 {
        return None;
    }
    let tensors = node.inputs[1..]
        .iter()
        .map(|name| constants.get(name)?.tensor(&node.name).ok())
        .collect::<Option<Vec<_>>>()?;
    let [scale, bias, mean, variance] = tensors.as_slice() else {
        return None;
    };
    let channels = scale.dims1().ok()?;
    if channels == 0
        || tensors.iter().any(|tensor| {
            tensor.dims1().ok() != Some(channels)
                || tensor.dtype() != DType::F32
                || !tensor.device().same_device(scale.device())
                || !tensor.is_contiguous()
        })
        || !(scale.device().is_cpu() || scale.device().is_cuda())
    {
        return None;
    }
    let epsilon = node.float("epsilon", 1e-5).ok()? as f32;
    if !epsilon.is_finite() || epsilon < 0.0 {
        return None;
    }
    let scale_and_bias = Tensor::stack(&[*scale, *bias], 0).ok()?;
    let mean_and_variance = Tensor::stack(&[*mean, *variance], 0).ok()?;
    let mean_and_stddev = prepare_statistics(&mean_and_variance, epsilon).ok()?;
    let (activation, consumed_nodes) = if let Some((divisor, offset, scale)) =
        matched_gelu_erf(nodes, scalar_constants, use_counts, retained_output)
    {
        (
            Activation::GeluErf {
                divisor,
                offset,
                scale,
            },
            6,
        )
    } else if matched_relu(nodes, use_counts, retained_output) {
        (Activation::Relu, 2)
    } else if matched_swish(nodes, use_counts, retained_output) {
        (Activation::Swish, 3)
    } else if let Some((alpha, beta)) = matched_hard_swish(nodes, use_counts, retained_output) {
        (Activation::HardSwish { alpha, beta }, 3)
    } else if matched_sigmoid(nodes, use_counts, retained_output) {
        (Activation::Sigmoid, 2)
    } else {
        (Activation::Identity, 1)
    };
    let convolution_post_operation =
        convolution_post_operation(scale, bias, &scale_and_bias, &mean_and_stddev, activation);
    Some(PreparedBatchNorm {
        scale_and_bias,
        mean_and_stddev,
        activation,
        consumed_nodes,
        convolution_post_operation,
    })
}

fn convolution_post_operation(
    scale: &Tensor,
    bias: &Tensor,
    _scale_and_bias: &Tensor,
    mean_and_stddev: &Tensor,
    activation: Activation,
) -> Option<ConvolutionPostOperation> {
    if scale.device().is_cpu() {
        return cpu_convolution_post_operation(scale, bias, mean_and_stddev, activation);
    }
    #[cfg(feature = "embedded-cuda")]
    if scale.device().is_cuda() {
        let activation = match activation {
            Activation::Identity => CudaBatchNormActivation::Identity,
            Activation::Relu => CudaBatchNormActivation::Relu,
            Activation::HardSwish { alpha, beta } => {
                CudaBatchNormActivation::HardSwish { alpha, beta }
            }
            Activation::Sigmoid => return None,
            Activation::Swish => CudaBatchNormActivation::Swish,
            Activation::GeluErf {
                divisor,
                offset,
                scale,
            } => CudaBatchNormActivation::GeluErf {
                divisor,
                offset,
                scale,
            },
        };
        return ConvolutionPostOperation::cuda_batch_normalization(
            _scale_and_bias,
            mean_and_stddev,
            activation,
        );
    }
    None
}

fn cpu_convolution_post_operation(
    scale: &Tensor,
    bias: &Tensor,
    mean_and_stddev: &Tensor,
    activation: Activation,
) -> Option<ConvolutionPostOperation> {
    if !scale.device().is_cpu() {
        return None;
    }
    let scale = scale.to_vec1::<f32>().ok()?;
    let bias = bias.to_vec1::<f32>().ok()?;
    let statistics = mean_and_stddev.flatten_all().ok()?.to_vec1::<f32>().ok()?;
    let (mean, stddev) = statistics.split_at(scale.len());
    let activation = match activation {
        Activation::Identity => BatchNormActivation::Identity,
        Activation::Relu => BatchNormActivation::Relu,
        Activation::HardSwish { alpha, beta } => BatchNormActivation::HardSwish { alpha, beta },
        Activation::Sigmoid | Activation::Swish | Activation::GeluErf { .. } => return None,
    };
    ConvolutionPostOperation::batch_normalization_with_prepared_statistics(
        &scale, &bias, mean, stddev, activation,
    )
}

fn prepare_statistics(mean_and_variance: &Tensor, epsilon: f32) -> candle_core::Result<Tensor> {
    if mean_and_variance.device().is_cpu() {
        return cpu::prepare_statistics(mean_and_variance, epsilon);
    }
    #[cfg(feature = "embedded-cuda")]
    if mean_and_variance.device().is_cuda() {
        return cuda::prepare_statistics(mean_and_variance, epsilon);
    }
    candle_core::bail!("BatchNormalization statistics require a supported local device")
}

#[cfg(all(test, feature = "embedded-cuda"))]
pub(super) fn prepare_cuda_statistics(
    mean_and_variance: &Tensor,
    epsilon: f32,
) -> candle_core::Result<Tensor> {
    cuda::prepare_statistics(mean_and_variance, epsilon)
}

fn matched_gelu_erf(
    nodes: &[GraphNode],
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<(f32, f32, f32)> {
    let [batch_norm, ..] = nodes else {
        return None;
    };
    let activation_nodes = nodes.get(1..6)?;
    let matched = gelu_erf::matched_inputs(activation_nodes, use_counts, retained_output)?;
    let [batch_norm_output] = batch_norm.outputs.as_slice() else {
        return None;
    };
    if matched.input != batch_norm_output
        || batch_norm_output == retained_output
        || use_counts.get(batch_norm_output).copied() != Some(2)
    {
        return None;
    }
    let divisor = *scalar_constants.get(matched.divisor)?;
    let offset = *scalar_constants.get(matched.offset)?;
    let scale = *scalar_constants.get(matched.scale)?;
    (divisor.is_finite() && divisor != 0.0 && offset.is_finite() && scale.is_finite())
        .then_some((divisor, offset, scale))
}

fn matched_relu(
    nodes: &[GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    let [batch_norm, relu, ..] = nodes else {
        return false;
    };
    if batch_norm.op != GraphOp::BatchNormalization
        || relu.op != GraphOp::Relu
        || !relu.attributes.is_empty()
    {
        return false;
    }
    let [batch_norm_output] = batch_norm.outputs.as_slice() else {
        return false;
    };
    let [relu_input] = relu.inputs.as_slice() else {
        return false;
    };
    relu_input == batch_norm_output
        && relu.outputs.len() == 1
        && batch_norm_output != retained_output
        && use_counts.get(batch_norm_output).copied() == Some(1)
}

fn matched_sigmoid(
    nodes: &[GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    let [batch_norm, sigmoid, ..] = nodes else {
        return false;
    };
    if batch_norm.op != GraphOp::BatchNormalization
        || sigmoid.op != GraphOp::Sigmoid
        || !sigmoid.attributes.is_empty()
    {
        return false;
    }
    let [batch_norm_output] = batch_norm.outputs.as_slice() else {
        return false;
    };
    let [sigmoid_input] = sigmoid.inputs.as_slice() else {
        return false;
    };
    sigmoid_input == batch_norm_output
        && sigmoid.outputs.len() == 1
        && batch_norm_output != retained_output
        && use_counts.get(batch_norm_output).copied() == Some(1)
}

fn matched_hard_swish(
    nodes: &[GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<(f32, f32)> {
    let [batch_norm, hard_sigmoid, multiply, ..] = nodes else {
        return None;
    };
    if batch_norm.op != GraphOp::BatchNormalization
        || hard_sigmoid.op != GraphOp::HardSigmoid
        || multiply.op != GraphOp::Mul
        || !multiply.attributes.is_empty()
    {
        return None;
    }
    let [batch_norm_output] = batch_norm.outputs.as_slice() else {
        return None;
    };
    let [gate_input] = hard_sigmoid.inputs.as_slice() else {
        return None;
    };
    let [gate_output] = hard_sigmoid.outputs.as_slice() else {
        return None;
    };
    let [left, right] = multiply.inputs.as_slice() else {
        return None;
    };
    if batch_norm_output == retained_output
        || gate_output == retained_output
        || gate_input != batch_norm_output
        || use_counts.get(batch_norm_output).copied() != Some(2)
        || use_counts.get(gate_output).copied() != Some(1)
        || !((left == batch_norm_output && right == gate_output)
            || (right == batch_norm_output && left == gate_output))
    {
        return None;
    }
    Some((
        hard_sigmoid.float("alpha", 0.2).ok()? as f32,
        hard_sigmoid.float("beta", 0.5).ok()? as f32,
    ))
}

fn matched_swish(
    nodes: &[GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    let [batch_norm, sigmoid, multiply, ..] = nodes else {
        return false;
    };
    if batch_norm.op != GraphOp::BatchNormalization
        || sigmoid.op != GraphOp::Sigmoid
        || multiply.op != GraphOp::Mul
        || !sigmoid.attributes.is_empty()
        || !multiply.attributes.is_empty()
    {
        return false;
    }
    let [batch_norm_output] = batch_norm.outputs.as_slice() else {
        return false;
    };
    let [gate_input] = sigmoid.inputs.as_slice() else {
        return false;
    };
    let [gate_output] = sigmoid.outputs.as_slice() else {
        return false;
    };
    let [left, right] = multiply.inputs.as_slice() else {
        return false;
    };
    batch_norm_output != retained_output
        && gate_output != retained_output
        && gate_input == batch_norm_output
        && use_counts.get(batch_norm_output).copied() == Some(2)
        && use_counts.get(gate_output).copied() == Some(1)
        && ((left == batch_norm_output && right == gate_output)
            || (right == batch_norm_output && left == gate_output))
}

pub(super) fn try_execute(
    node: &GraphNode,
    values: &HashMap<String, GraphValue>,
    prepared: &PreparedBatchNorm,
) -> Result<Option<FusedOutput>> {
    let Some(input_name) = node.inputs.first() else {
        return Ok(None);
    };
    let input = values
        .get(input_name)
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' could not resolve input '{input_name}'",
                node.name
            ))
        })?
        .tensor(&node.name)?;
    if input.dtype() != DType::F32
        || !input.is_contiguous()
        || input.rank() < 2
        || !input.device().same_device(prepared.scale_and_bias.device())
    {
        return Ok(None);
    }

    if input.device().is_cpu() {
        return cpu::execute(
            input,
            &prepared.scale_and_bias,
            &prepared.mean_and_stddev,
            prepared.activation,
        )
        .map(|value| FusedOutput {
            value: GraphValue::Tensor(value),
            consumed_nodes: prepared.consumed_nodes,
        })
        .map(Some)
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' fused CPU BatchNormalization failed: {error}",
                node.name
            ))
        });
    }

    if !input.device().is_cuda() {
        return Ok(None);
    }

    #[cfg(feature = "embedded-cuda")]
    {
        cuda::execute(
            input,
            &prepared.scale_and_bias,
            &prepared.mean_and_stddev,
            prepared.activation,
        )
        .map(|value| FusedOutput {
            value: GraphValue::Tensor(value),
            consumed_nodes: prepared.consumed_nodes,
        })
        .map(Some)
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' fused BatchNormalization failed: {error}",
                node.name
            ))
        })
    }

    #[cfg(not(feature = "embedded-cuda"))]
    {
        let _ = prepared;
        Ok(None)
    }
}

pub(super) struct ConvolutionExecutionContext<'a> {
    pub(super) values: &'a HashMap<String, GraphValue>,
    pub(super) prepared: &'a HashMap<String, PreparedBatchNorm>,
    pub(super) use_counts: &'a HashMap<String, usize>,
    pub(super) retained_output: &'a str,
    pub(super) device: &'a Device,
    pub(super) element_limit: usize,
    pub(super) cancellation: &'a CancellationToken,
}

pub(super) fn try_execute_convolution(
    nodes: &[GraphNode],
    context: ConvolutionExecutionContext<'_>,
) -> Result<Option<FusedOutput>> {
    let ConvolutionExecutionContext {
        values,
        prepared,
        use_counts,
        retained_output,
        device,
        element_limit,
        cancellation,
    } = context;
    let (Some(convolution), Some(batch_norm)) = (nodes.first(), nodes.get(1)) else {
        return Ok(None);
    };
    #[cfg(feature = "embedded-cuda")]
    let supported_device = device.is_cpu() || device.is_cuda();
    #[cfg(not(feature = "embedded-cuda"))]
    let supported_device = device.is_cpu();
    if convolution.op != GraphOp::Conv
        || batch_norm.op != GraphOp::BatchNormalization
        || !supported_device
    {
        return Ok(None);
    }
    let Some(prepared) = prepared.get(&batch_norm.name) else {
        return Ok(None);
    };
    let Some(post_operation) = prepared.convolution_post_operation.clone() else {
        return Ok(None);
    };
    let [convolution_output] = convolution.outputs.as_slice() else {
        return Ok(None);
    };
    if convolution_output == retained_output
        || batch_norm.inputs.first() != Some(convolution_output)
        || use_counts.get(convolution_output).copied() != Some(1)
    {
        return Ok(None);
    }
    let consumed_nodes = 1_usize
        .checked_add(prepared.consumed_nodes)
        .ok_or_else(|| {
            PowerError::InferenceFailed("BatchNormalization fusion overflowed".into())
        })?;
    if nodes.len() < consumed_nodes {
        return Ok(None);
    }
    let inputs = convolution
        .inputs
        .iter()
        .map(|name| {
            values.get(name).ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "static graph node '{}' could not resolve input '{name}'",
                    convolution.name
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let Some(value) =
        super::spatial::try_conv_with_post_operation(convolution, &inputs, device, post_operation)?
    else {
        return Ok(None);
    };
    let output = value.tensor(&convolution.name)?;
    if output.elem_count() == 0 || output.elem_count() > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
            convolution.name,
            output.elem_count(),
        )));
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }
    Ok(Some(FusedOutput {
        value,
        consumed_nodes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_constants_prepare_the_fused_path() {
        let device = candle_core::Device::Cpu;
        let constants = ["scale", "bias", "mean", "variance"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    GraphValue::Tensor(Tensor::zeros(3, DType::F32, &device).unwrap()),
                )
            })
            .collect();
        let mut node = GraphNode {
            name: "batch-norm".to_string(),
            op: GraphOp::BatchNormalization,
            inputs: vec![
                "input".to_string(),
                "scale".to_string(),
                "bias".to_string(),
                "mean".to_string(),
                "variance".to_string(),
            ],
            outputs: vec!["output".to_string()],
            attributes: std::collections::BTreeMap::new(),
        };
        node.attributes
            .insert("epsilon".to_string(), serde_json::json!(0.00001));

        let prepared = prepare_node(
            &[node],
            &constants,
            &HashMap::new(),
            &HashMap::new(),
            "output",
        )
        .unwrap();
        assert_eq!(prepared.consumed_nodes, 1);
    }

    #[test]
    fn hard_swish_match_requires_private_self_gating() {
        let node = |name: &str, op: GraphOp, inputs: &[&str], output: &str| GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: std::collections::BTreeMap::new(),
        };
        let nodes = vec![
            node("batch-norm", GraphOp::BatchNormalization, &[], "normalized"),
            node("gate", GraphOp::HardSigmoid, &["normalized"], "bounded"),
            node(
                "multiply",
                GraphOp::Mul,
                &["bounded", "normalized"],
                "output",
            ),
        ];
        let uses = HashMap::from([("normalized".to_string(), 2), ("bounded".to_string(), 1)]);

        assert_eq!(
            matched_hard_swish(&nodes, &uses, "output"),
            Some((0.2, 0.5))
        );
        assert!(matched_hard_swish(
            &nodes,
            &HashMap::from([("normalized".to_string(), 3), ("bounded".to_string(), 1),]),
            "output"
        )
        .is_none());
    }

    #[test]
    fn swish_match_requires_private_self_gating() {
        let node = |name: &str, op: GraphOp, inputs: &[&str], output: &str| GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: std::collections::BTreeMap::new(),
        };
        let nodes = vec![
            node("batch-norm", GraphOp::BatchNormalization, &[], "normalized"),
            node("gate", GraphOp::Sigmoid, &["normalized"], "bounded"),
            node(
                "multiply",
                GraphOp::Mul,
                &["bounded", "normalized"],
                "output",
            ),
        ];
        let uses = HashMap::from([("normalized".to_string(), 2), ("bounded".to_string(), 1)]);

        assert!(matched_swish(&nodes, &uses, "output"));
        assert!(!matched_swish(
            &nodes,
            &HashMap::from([("normalized".to_string(), 3), ("bounded".to_string(), 1)]),
            "output"
        ));
        assert!(!matched_swish(&nodes, &uses, "normalized"));
    }

    #[test]
    fn sigmoid_match_requires_an_exact_private_edge() {
        let node = |name: &str, op: GraphOp, inputs: &[&str], output: &str| GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: std::collections::BTreeMap::new(),
        };
        let nodes = vec![
            node("batch-norm", GraphOp::BatchNormalization, &[], "normalized"),
            node("gate", GraphOp::Sigmoid, &["normalized"], "output"),
        ];
        let private = HashMap::from([("normalized".to_string(), 1)]);

        assert!(matched_sigmoid(&nodes, &private, "output"));
        assert!(!matched_sigmoid(
            &nodes,
            &HashMap::from([("normalized".to_string(), 2)]),
            "output"
        ));
        assert!(!matched_sigmoid(&nodes, &private, "normalized"));

        let mut attributed = nodes;
        attributed[1]
            .attributes
            .insert("unexpected".to_string(), serde_json::json!(true));
        assert!(!matched_sigmoid(&attributed, &private, "output"));
    }

    #[test]
    fn relu_match_requires_an_exact_private_edge() {
        let node = |name: &str, op: GraphOp, inputs: &[&str], output: &str| GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: std::collections::BTreeMap::new(),
        };
        let nodes = vec![
            node("batch-norm", GraphOp::BatchNormalization, &[], "normalized"),
            node("relu", GraphOp::Relu, &["normalized"], "output"),
        ];

        assert!(matched_relu(
            &nodes,
            &HashMap::from([("normalized".to_string(), 1)]),
            "output"
        ));
        assert!(!matched_relu(
            &nodes,
            &HashMap::from([("normalized".to_string(), 2)]),
            "output"
        ));
        assert!(!matched_relu(
            &nodes,
            &HashMap::from([("normalized".to_string(), 1)]),
            "normalized"
        ));
    }

    #[test]
    fn gelu_erf_match_requires_exact_private_topology_and_finite_scalars() {
        let node = |name: &str, op: GraphOp, inputs: &[&str], output: &str| GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: std::collections::BTreeMap::new(),
        };
        let nodes = vec![
            node("normalization", GraphOp::BatchNormalization, &[], "n"),
            node("division", GraphOp::Div, &["n", "d"], "q"),
            node("error-function", GraphOp::Erf, &["q"], "e"),
            node("offset", GraphOp::Add, &["o", "e"], "a"),
            node("self-product", GraphOp::Mul, &["a", "n"], "p"),
            node("scale", GraphOp::Mul, &["s", "p"], "result"),
        ];
        let uses = HashMap::from([
            ("n".to_string(), 2),
            ("q".to_string(), 1),
            ("e".to_string(), 1),
            ("a".to_string(), 1),
            ("p".to_string(), 1),
        ]);
        let constants = HashMap::from([
            ("d".to_string(), std::f32::consts::SQRT_2),
            ("o".to_string(), 1.0_f32),
            ("s".to_string(), 0.5_f32),
        ]);

        assert_eq!(
            matched_gelu_erf(&nodes, &constants, &uses, "result"),
            Some((std::f32::consts::SQRT_2, 1.0, 0.5))
        );

        let mut shared = uses.clone();
        shared.insert("n".to_string(), 3);
        assert!(matched_gelu_erf(&nodes, &constants, &shared, "result").is_none());
        assert!(matched_gelu_erf(&nodes, &constants, &uses, "n").is_none());

        let mut invalid_constants = constants;
        invalid_constants.insert("d".to_string(), 0.0);
        assert!(matched_gelu_erf(&nodes, &invalid_constants, &uses, "result").is_none());
    }

    #[test]
    fn convolution_batch_norm_fusion_is_bit_exact_and_requires_a_private_edge() {
        let device = candle_core::Device::Cpu;
        let node = |name: &str,
                    op: GraphOp,
                    inputs: &[&str],
                    output: &str,
                    attributes: std::collections::BTreeMap<_, _>| GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes,
        };
        let nodes = vec![
            node(
                "convolution",
                GraphOp::Conv,
                &["input", "kernel"],
                "convolution-output",
                std::collections::BTreeMap::from([
                    ("kernel_shape".to_string(), serde_json::json!([1, 1])),
                    ("strides".to_string(), serde_json::json!([1, 1])),
                    ("dilations".to_string(), serde_json::json!([1, 1])),
                    ("pads".to_string(), serde_json::json!([0, 0, 0, 0])),
                    ("group".to_string(), serde_json::json!(1)),
                ]),
            ),
            node(
                "batch-norm",
                GraphOp::BatchNormalization,
                &["convolution-output", "scale", "bias", "mean", "variance"],
                "normalized",
                std::collections::BTreeMap::from([(
                    "epsilon".to_string(),
                    serde_json::json!(0.00001),
                )]),
            ),
            node(
                "gate",
                GraphOp::HardSigmoid,
                &["normalized"],
                "bounded",
                std::collections::BTreeMap::from([
                    ("alpha".to_string(), serde_json::json!(1.0 / 6.0)),
                    ("beta".to_string(), serde_json::json!(0.5)),
                ]),
            ),
            node(
                "multiply",
                GraphOp::Mul,
                &["normalized", "bounded"],
                "output",
                std::collections::BTreeMap::new(),
            ),
        ];
        let input = Tensor::from_iter((0..12).map(|value| (value as f32 - 5.0) / 7.0), &device)
            .unwrap()
            .reshape((1, 2, 2, 3))
            .unwrap();
        let kernel = Tensor::new(&[0.5_f32, -0.25, 1.25, 0.75, -0.5, 2.0], &device)
            .unwrap()
            .reshape((3, 2, 1, 1))
            .unwrap();
        let scale = Tensor::new(&[0.75_f32, -1.25, 2.0], &device).unwrap();
        let bias = Tensor::new(&[-0.5_f32, 0.125, 1.5], &device).unwrap();
        let mean = Tensor::new(&[0.25_f32, -0.75, 1.25], &device).unwrap();
        let variance = Tensor::new(&[0.5_f32, 1.5, 2.5], &device).unwrap();
        let constants = HashMap::from([
            ("scale".to_string(), GraphValue::Tensor(scale.clone())),
            ("bias".to_string(), GraphValue::Tensor(bias.clone())),
            ("mean".to_string(), GraphValue::Tensor(mean.clone())),
            ("variance".to_string(), GraphValue::Tensor(variance.clone())),
        ]);
        let uses = HashMap::from([
            ("convolution-output".to_string(), 1),
            ("normalized".to_string(), 2),
            ("bounded".to_string(), 1),
        ]);
        let prepared = prepare(&nodes, &constants, &HashMap::new(), &uses, "output");
        let mut values = constants;
        values.insert("input".to_string(), GraphValue::Tensor(input.clone()));
        values.insert("kernel".to_string(), GraphValue::Tensor(kernel.clone()));
        let cancellation = CancellationToken::new();
        let run = try_execute_convolution(
            &nodes,
            ConvolutionExecutionContext {
                values: &values,
                prepared: &prepared,
                use_counts: &uses,
                retained_output: "output",
                device: &device,
                element_limit: 1_000_000,
                cancellation: &cancellation,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(run.consumed_nodes, 4);

        let convolution = input.conv2d(&kernel, 0, 1, 1, 1).unwrap();
        let expected = convolution
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .chunks(6)
            .enumerate()
            .flat_map(|(channel, values)| {
                let scale = scale.to_vec1::<f32>().unwrap()[channel];
                let bias = bias.to_vec1::<f32>().unwrap()[channel];
                let mean = mean.to_vec1::<f32>().unwrap()[channel];
                let stddev = (variance.to_vec1::<f32>().unwrap()[channel] + 0.000_01).sqrt();
                values.iter().map(move |value| {
                    let normalized = (((*value - mean) / stddev) * scale) + bias;
                    normalized * ((normalized * (1.0 / 6.0)) + 0.5).clamp(0.0, 1.0)
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            run.value
                .tensor("fused test")
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected,
        );

        let mut shared_uses = uses.clone();
        shared_uses.insert("convolution-output".to_string(), 2);
        assert!(try_execute_convolution(
            &nodes,
            ConvolutionExecutionContext {
                values: &values,
                prepared: &prepared,
                use_counts: &shared_uses,
                retained_output: "output",
                device: &device,
                element_limit: 1_000_000,
                cancellation: &cancellation,
            },
        )
        .unwrap()
        .is_none());
    }
}

#[cfg(all(test, feature = "embedded-cuda"))]
mod cuda_tests;
