use std::collections::{HashMap, HashSet};

use candle_core::DType;

use crate::error::{PowerError, Result};
use crate::inference::InferenceLimits;

use super::super::super::plan::{GraphNode, GraphOp, GraphPlan};
use super::super::super::value::GraphValue;
use super::super::gelu_erf;
use super::super::tensor_geometry::{pair, quad};
use super::{Activation, Block, Padding, PreparedSegment, ResidualSource};

const MINIMUM_BLOCKS: usize = 2;
const MAXIMUM_IDENTITY_NODES: usize = 2;

pub(super) fn prepare(
    plan: &GraphPlan,
    constants: &HashMap<String, GraphValue>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    limits: &InferenceLimits,
) -> Result<HashMap<usize, PreparedSegment>> {
    let mut candidates = Vec::new();
    let mut index = 0;
    while index < plan.nodes.len() {
        let Some(candidate) =
            match_candidate(&plan.nodes, index, constants, use_counts, retained_output)?
        else {
            index += 1;
            continue;
        };
        let consumed_nodes = candidate.consumed_nodes;
        candidates.push((index, candidate));
        index += consumed_nodes;
    }
    if candidates.is_empty() {
        return Ok(HashMap::new());
    }
    let parameter_bytes = candidates
        .iter()
        .flat_map(|(_, candidate)| &candidate.blocks)
        .try_fold(0_u64, |total, block| {
            let elements = block
                .weights
                .len()
                .checked_add(block.bias.len())
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "optimized CPU graph segment parameter size overflowed".to_string(),
                    )
                })?;
            total.checked_add(elements).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "optimized CPU graph segment parameter inventory overflowed".to_string(),
                )
            })
        })?;
    let persistent_budget = limits.max_state_bytes / 2;
    if parameter_bytes >= persistent_budget {
        return Ok(HashMap::new());
    }
    let cache_budget = (persistent_budget - parameter_bytes)
        / u64::try_from(candidates.len()).unwrap_or(u64::MAX).max(1);
    let execution_budget = limits
        .max_state_bytes
        .saturating_sub(persistent_budget)
        .max(1);
    let context_cache_budget =
        execution_budget / u64::try_from(candidates.len()).unwrap_or(u64::MAX).max(1);
    if cache_budget == 0 || context_cache_budget == 0 {
        return Ok(HashMap::new());
    }
    let mut prepared = HashMap::with_capacity(candidates.len());
    for (index, candidate) in candidates {
        prepared.insert(
            index,
            PreparedSegment::new(
                candidate.input,
                candidate.consumed_nodes,
                candidate.blocks,
                cache_budget,
                context_cache_budget,
                execution_budget,
            )?,
        );
    }
    Ok(prepared)
}

struct Candidate {
    input: String,
    consumed_nodes: usize,
    blocks: Vec<Block>,
}

struct MatchedBlock {
    block: Block,
    input: String,
    output: String,
    consumed_nodes: usize,
}

fn match_candidate(
    nodes: &[GraphNode],
    start: usize,
    constants: &HashMap<String, GraphValue>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Result<Option<Candidate>> {
    let Some(first) = match_block(nodes, start, constants, use_counts, retained_output)? else {
        return Ok(None);
    };
    let input = first.input.clone();
    let mut blocks = vec![first.block];
    let mut consumed_nodes = first.consumed_nodes;
    let mut preceding_output = first.output;
    let mut sources = HashMap::from([(input.clone(), ResidualSource::Input)]);
    sources.insert(preceding_output.clone(), ResidualSource::Block(0));
    let mut best = None;
    loop {
        let next_index = start + consumed_nodes;
        let closed_residual = if let Some(residual) = match_residual(
            nodes,
            next_index,
            &preceding_output,
            &sources,
            use_counts,
            retained_output,
        ) {
            let last = blocks.len() - 1;
            if residual.source == ResidualSource::Input && last == 0 {
                break;
            }
            blocks[last].residual = Some(residual.source);
            sources.remove(&preceding_output);
            preceding_output = residual.output;
            sources.insert(preceding_output.clone(), ResidualSource::Block(last));
            consumed_nodes += residual.consumed_nodes;
            true
        } else {
            false
        };

        if blocks.len() >= MINIMUM_BLOCKS
            && candidate_window_is_closed(nodes, start, consumed_nodes, use_counts, retained_output)
        {
            best = Some((blocks.len(), consumed_nodes, preceding_output.clone()));
            // A closed residual dependency region is a natural pipeline
            // boundary: no value inside it is live outside the window except
            // its terminal output. Keeping later regions separate bounds the
            // activation lifetime and avoids serializing independent graph
            // executions behind one oversized native segment.
            if closed_residual {
                break;
            }
        }

        let next_index = start + consumed_nodes;
        let Some(next) = match_block(nodes, next_index, constants, use_counts, retained_output)?
        else {
            break;
        };
        if next.input != preceding_output {
            break;
        }
        consumed_nodes += next.consumed_nodes;
        preceding_output = next.output;
        blocks.push(next.block);
        sources.insert(
            preceding_output.clone(),
            ResidualSource::Block(blocks.len() - 1),
        );
    }
    let Some((block_count, consumed_nodes, _output)) = best else {
        return Ok(None);
    };
    blocks.truncate(block_count);
    Ok(Some(Candidate {
        input,
        consumed_nodes,
        blocks,
    }))
}

struct MatchedResidual {
    source: ResidualSource,
    output: String,
    consumed_nodes: usize,
}

fn match_residual(
    nodes: &[GraphNode],
    start: usize,
    main: &str,
    sources: &HashMap<String, ResidualSource>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedResidual> {
    let add = nodes.get(start)?;
    if add.op != GraphOp::Add || !add.attributes.is_empty() {
        return None;
    }
    let [left, right] = add.inputs.as_slice() else {
        return None;
    };
    let source_name = match (left == main, right == main) {
        (true, false) => right,
        (false, true) => left,
        _ => return None,
    };
    let source = *sources.get(source_name)?;
    let [add_output] = add.outputs.as_slice() else {
        return None;
    };
    let mut output = add_output.clone();
    let mut consumed_nodes = 1;
    while let Some(identity) = nodes
        .get(start + consumed_nodes)
        .filter(|node| node.op == GraphOp::Identity)
    {
        if consumed_nodes > MAXIMUM_IDENTITY_NODES {
            return None;
        }
        let ([identity_input], [identity_output]) =
            (identity.inputs.as_slice(), identity.outputs.as_slice())
        else {
            return None;
        };
        if identity_input != &output
            || !identity.attributes.is_empty()
            || !private_value(&output, use_counts, retained_output)
        {
            return None;
        }
        output = identity_output.clone();
        consumed_nodes += 1;
    }
    Some(MatchedResidual {
        source,
        output,
        consumed_nodes,
    })
}

fn candidate_window_is_closed(
    nodes: &[GraphNode],
    start: usize,
    consumed_nodes: usize,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    let Some(window) = nodes.get(start..start + consumed_nodes) else {
        return false;
    };
    let Some(terminal) = window.last().and_then(|node| node.outputs.first()) else {
        return false;
    };
    let produced = window
        .iter()
        .flat_map(|node| node.outputs.iter())
        .collect::<HashSet<_>>();
    let mut covered_uses = HashMap::new();
    for input in window.iter().flat_map(|node| node.inputs.iter()) {
        if produced.contains(input) {
            *covered_uses.entry(input).or_insert(0_usize) += 1;
        }
    }
    produced.into_iter().all(|value| {
        value == terminal
            || (value != retained_output
                && covered_uses.get(value).copied().unwrap_or(0)
                    == use_counts.get(value).copied().unwrap_or(0))
    })
}

fn match_block(
    nodes: &[GraphNode],
    start: usize,
    constants: &HashMap<String, GraphValue>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Result<Option<MatchedBlock>> {
    let Some(convolution) = nodes.get(start) else {
        return Ok(None);
    };
    let Some(add) = nodes.get(start + 1) else {
        return Ok(None);
    };
    if convolution.op != GraphOp::Conv
        || add.op != GraphOp::Add
        || convolution.inputs.len() != 2
        || !add.attributes.is_empty()
    {
        return Ok(None);
    }
    if convolution.attributes.keys().any(|name| {
        !matches!(
            name.as_str(),
            "auto_pad" | "dilations" | "group" | "kernel_shape" | "pads" | "strides"
        )
    }) {
        return Ok(None);
    }
    let [input, weight_name] = convolution.inputs.as_slice() else {
        return Ok(None);
    };
    let [convolution_output] = convolution.outputs.as_slice() else {
        return Ok(None);
    };
    let [add_left, add_right] = add.inputs.as_slice() else {
        return Ok(None);
    };
    let bias_name = match (
        add_left == convolution_output,
        add_right == convolution_output,
    ) {
        (true, false) => add_right,
        (false, true) => add_left,
        _ => return Ok(None),
    };
    if !private_value(convolution_output, use_counts, retained_output) {
        return Ok(None);
    }
    let [add_output] = add.outputs.as_slice() else {
        return Ok(None);
    };
    let mut activation_input = add_output.as_str();
    let mut cursor = start + 2;
    let mut identities = 0;
    while nodes
        .get(cursor)
        .is_some_and(|node| node.op == GraphOp::Identity)
    {
        if identities == MAXIMUM_IDENTITY_NODES {
            return Ok(None);
        }
        let identity = &nodes[cursor];
        let ([identity_input], [identity_output]) =
            (identity.inputs.as_slice(), identity.outputs.as_slice())
        else {
            return Ok(None);
        };
        if identity_input != activation_input
            || !identity.attributes.is_empty()
            || !private_value(activation_input, use_counts, retained_output)
        {
            return Ok(None);
        }
        activation_input = identity_output;
        identities += 1;
        cursor += 1;
    }
    let (activation, output, consumed_nodes) =
        if let Some(relu) = nodes.get(cursor).filter(|node| node.op == GraphOp::Relu) {
            let ([relu_input], [relu_output]) = (relu.inputs.as_slice(), relu.outputs.as_slice())
            else {
                return Ok(None);
            };
            if relu_input != activation_input
                || !relu.attributes.is_empty()
                || !private_value(activation_input, use_counts, retained_output)
            {
                return Ok(None);
            }
            (Activation::Relu, relu_output.clone(), cursor - start + 1)
        } else if let Some(window) = nodes.get(cursor..cursor + 5) {
            let Some(gelu) = gelu_erf::matched_inputs(window, use_counts, retained_output) else {
                return biased_block(
                    convolution,
                    input,
                    weight_name,
                    bias_name,
                    activation_input,
                    cursor - start,
                    constants,
                );
            };
            if gelu.input != activation_input
                || activation_input == retained_output
                || use_counts.get(activation_input).copied() != Some(2)
                || window.iter().any(|node| !node.attributes.is_empty())
                || !standard_gelu_constants(gelu, constants)
            {
                return biased_block(
                    convolution,
                    input,
                    weight_name,
                    bias_name,
                    activation_input,
                    cursor - start,
                    constants,
                );
            }
            let [gelu_output] = window[4].outputs.as_slice() else {
                return Ok(None);
            };
            (Activation::GeluErf, gelu_output.clone(), cursor - start + 5)
        } else {
            (
                Activation::Bias,
                activation_input.to_string(),
                cursor - start,
            )
        };

    build_block(
        convolution,
        input,
        weight_name,
        bias_name,
        activation,
        output,
        consumed_nodes,
        constants,
    )
}

fn biased_block(
    convolution: &GraphNode,
    input: &str,
    weight_name: &str,
    bias_name: &str,
    output: &str,
    consumed_nodes: usize,
    constants: &HashMap<String, GraphValue>,
) -> Result<Option<MatchedBlock>> {
    build_block(
        convolution,
        input,
        weight_name,
        bias_name,
        Activation::Bias,
        output.to_string(),
        consumed_nodes,
        constants,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_block(
    convolution: &GraphNode,
    input: &str,
    weight_name: &str,
    bias_name: &str,
    activation: Activation,
    output: String,
    consumed_nodes: usize,
    constants: &HashMap<String, GraphValue>,
) -> Result<Option<MatchedBlock>> {
    let Some(weight) = f32_tensor(constants.get(weight_name)) else {
        return Ok(None);
    };
    let Ok((output_channels, kernel_input_channels, kernel_height, kernel_width)) = weight.dims4()
    else {
        return Ok(None);
    };
    let groups = usize::try_from(convolution.int("group", 1)?).ok();
    let Some(groups) = groups.filter(|groups| *groups != 0) else {
        return Ok(None);
    };
    let Some(input_channels) = kernel_input_channels.checked_mul(groups) else {
        return Ok(None);
    };
    if !output_channels.is_multiple_of(groups) {
        return Ok(None);
    }
    let kernel = pair(
        &convolution.ints("kernel_shape", &[])?,
        "kernel_shape",
        convolution,
    )?;
    if kernel != (kernel_height, kernel_width) {
        return Ok(None);
    }
    let Some(bias) = f32_tensor(constants.get(bias_name)) else {
        return Ok(None);
    };
    if !(bias.dims1().ok() == Some(output_channels)
        || bias.dims4().ok() == Some((1, output_channels, 1, 1)))
    {
        return Ok(None);
    }
    let strides = pair(
        &convolution.ints("strides", &[1, 1])?,
        "strides",
        convolution,
    )?;
    let dilations = pair(
        &convolution.ints("dilations", &[1, 1])?,
        "dilations",
        convolution,
    )?;
    let padding = match convolution.string("auto_pad", "NOTSET")? {
        "NOTSET" => Padding::Explicit(quad(
            &convolution.ints("pads", &[0, 0, 0, 0])?,
            "pads",
            convolution,
        )?),
        "SAME_UPPER" => Padding::SameUpper,
        _ => return Ok(None),
    };
    let weights = weight
        .flatten_all()
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(|error| invalid_tensor(convolution, error))?;
    let bias = bias
        .flatten_all()
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(|error| invalid_tensor(convolution, error))?;
    Ok(Some(MatchedBlock {
        block: Block {
            input_channels,
            output_channels,
            groups,
            kernel,
            strides,
            dilations,
            padding,
            activation,
            residual: None,
            weights,
            bias,
        },
        input: input.to_string(),
        output,
        consumed_nodes,
    }))
}

fn standard_gelu_constants(
    matched: gelu_erf::MatchedInputs<'_>,
    constants: &HashMap<String, GraphValue>,
) -> bool {
    let values = [matched.divisor, matched.offset, matched.scale]
        .map(|name| scalar_f32(constants.get(name)));
    matches!(
        values,
        [Some(divisor), Some(offset), Some(scale)]
            if divisor.to_bits() == f32::from_bits(0x3fb5_04f3).to_bits()
                && offset.to_bits() == 1.0_f32.to_bits()
                && scale.to_bits() == 0.5_f32.to_bits()
    )
}

fn scalar_f32(value: Option<&GraphValue>) -> Option<f32> {
    let tensor = f32_tensor(value)?;
    if tensor.elem_count() != 1 {
        return None;
    }
    tensor
        .flatten_all()
        .ok()?
        .to_vec1::<f32>()
        .ok()?
        .first()
        .copied()
}

fn f32_tensor(value: Option<&GraphValue>) -> Option<&candle_core::Tensor> {
    let GraphValue::Tensor(tensor) = value? else {
        return None;
    };
    (tensor.dtype() == DType::F32 && tensor.device().is_cpu() && tensor.is_contiguous())
        .then_some(tensor)
}

fn private_value(value: &str, use_counts: &HashMap<String, usize>, retained_output: &str) -> bool {
    value != retained_output && use_counts.get(value).copied() == Some(1)
}

fn invalid_tensor(node: &GraphNode, error: candle_core::Error) -> PowerError {
    PowerError::InvalidFormat(format!(
        "optimized CPU graph segment could not read node '{}' parameters: {error}",
        node.name
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use candle_core::{Device, Tensor};
    use serde_json::json;

    use super::*;

    #[test]
    fn matches_two_private_blocks_from_operator_topology() {
        let nodes = blocks(2);
        let constants = constants(2);
        let use_counts = use_counts(&nodes);

        let candidate = match_candidate(
            &nodes,
            0,
            &constants,
            &use_counts,
            nodes.last().unwrap().outputs.first().unwrap(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(candidate.blocks.len(), 2);
        assert_eq!(candidate.consumed_nodes, 8);
    }

    #[test]
    fn shared_intermediate_starts_a_later_private_segment() {
        let mut nodes = blocks(3);
        let shared = nodes[3].outputs[0].clone();
        nodes.push(node(
            "branch",
            GraphOp::Identity,
            vec![shared],
            vec!["branch-output".to_string()],
            BTreeMap::new(),
        ));
        let constants = constants(3);
        let use_counts = use_counts(&nodes);

        assert!(
            match_candidate(&nodes, 0, &constants, &use_counts, "branch-output")
                .unwrap()
                .is_none()
        );
        let candidate = match_candidate(&nodes, 4, &constants, &use_counts, "branch-output")
            .unwrap()
            .unwrap();
        assert_eq!(candidate.blocks.len(), 2);
        assert_eq!(candidate.input, "relu-0");
    }

    #[test]
    fn standard_private_erf_gelu_extends_a_convolution_segment() {
        let nodes = gelu_block_pair();
        let constants = gelu_constants();
        let use_counts = use_counts(&nodes);

        let candidate = match_candidate(
            &nodes,
            0,
            &constants,
            &use_counts,
            nodes.last().unwrap().outputs.first().unwrap(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(candidate.blocks.len(), 2);
        assert!(matches!(
            candidate.blocks[0].activation,
            Activation::GeluErf
        ));
        assert_eq!(candidate.consumed_nodes, 12);
    }

    #[test]
    fn input_residual_is_proven_inside_the_candidate_window() {
        let mut nodes = blocks(2);
        nodes.push(node(
            "residual",
            GraphOp::Add,
            vec!["relu-1".into(), "input".into()],
            vec!["residual-output".into()],
            BTreeMap::new(),
        ));
        let constants = constants(2);
        let use_counts = use_counts(&nodes);

        let candidate = match_candidate(&nodes, 0, &constants, &use_counts, "residual-output")
            .unwrap()
            .unwrap();

        assert_eq!(candidate.blocks.len(), 2);
        assert_eq!(candidate.blocks[1].residual, Some(ResidualSource::Input));
        assert_eq!(candidate.consumed_nodes, 9);
    }

    #[test]
    fn closed_residual_region_is_a_pipeline_boundary() {
        let mut nodes = blocks(2);
        nodes.push(node(
            "residual",
            GraphOp::Add,
            vec!["relu-1".into(), "input".into()],
            vec!["residual-output".into()],
            BTreeMap::new(),
        ));
        append_block(&mut nodes, 2, "residual-output");
        append_block(&mut nodes, 3, "relu-2");
        let constants = constants(4);
        let use_counts = use_counts(&nodes);

        let candidate = match_candidate(
            &nodes,
            0,
            &constants,
            &use_counts,
            nodes.last().unwrap().outputs.first().unwrap(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(candidate.blocks.len(), 2);
        assert_eq!(candidate.consumed_nodes, 9);
        assert_eq!(candidate.blocks[1].residual, Some(ResidualSource::Input));
    }

    fn blocks(count: usize) -> Vec<GraphNode> {
        let mut nodes = Vec::new();
        for index in 0..count {
            let input = if index == 0 {
                "input".to_string()
            } else {
                format!("relu-{}", index - 1)
            };
            append_block(&mut nodes, index, &input);
        }
        nodes
    }

    fn append_block(nodes: &mut Vec<GraphNode>, index: usize, input: &str) {
        nodes.push(node(
            &format!("conv-{index}"),
            GraphOp::Conv,
            vec![input.to_string(), format!("weight-{index}")],
            vec![format!("conv-output-{index}")],
            BTreeMap::from([
                ("group".to_string(), json!(1)),
                ("kernel_shape".to_string(), json!([1, 1])),
                ("pads".to_string(), json!([0, 0, 0, 0])),
                ("strides".to_string(), json!([1, 1])),
            ]),
        ));
        nodes.push(node(
            &format!("add-{index}"),
            GraphOp::Add,
            vec![format!("conv-output-{index}"), format!("bias-{index}")],
            vec![format!("add-output-{index}")],
            BTreeMap::new(),
        ));
        nodes.push(node(
            &format!("identity-{index}"),
            GraphOp::Identity,
            vec![format!("add-output-{index}")],
            vec![format!("identity-output-{index}")],
            BTreeMap::new(),
        ));
        nodes.push(node(
            &format!("relu-{index}"),
            GraphOp::Relu,
            vec![format!("identity-output-{index}")],
            vec![format!("relu-{index}")],
            BTreeMap::new(),
        ));
    }

    fn constants(count: usize) -> HashMap<String, GraphValue> {
        let mut values = HashMap::new();
        for index in 0..count {
            values.insert(
                format!("weight-{index}"),
                GraphValue::Tensor(
                    Tensor::from_vec(vec![0.25_f32; 4], (2, 2, 1, 1), &Device::Cpu).unwrap(),
                ),
            );
            values.insert(
                format!("bias-{index}"),
                GraphValue::Tensor(Tensor::from_vec(vec![0.0_f32; 2], 2, &Device::Cpu).unwrap()),
            );
        }
        values
    }

    fn gelu_block_pair() -> Vec<GraphNode> {
        let mut nodes = blocks(1);
        nodes.truncate(3);
        nodes.extend([
            node(
                "divide",
                GraphOp::Div,
                vec!["identity-output-0".into(), "sqrt-two".into()],
                vec!["divided".into()],
                BTreeMap::new(),
            ),
            node(
                "erf",
                GraphOp::Erf,
                vec!["divided".into()],
                vec!["erf-output".into()],
                BTreeMap::new(),
            ),
            node(
                "offset",
                GraphOp::Add,
                vec!["erf-output".into(), "one".into()],
                vec!["shifted".into()],
                BTreeMap::new(),
            ),
            node(
                "gate",
                GraphOp::Mul,
                vec!["identity-output-0".into(), "shifted".into()],
                vec!["product".into()],
                BTreeMap::new(),
            ),
            node(
                "scale",
                GraphOp::Mul,
                vec!["product".into(), "half".into()],
                vec!["gelu-0".into()],
                BTreeMap::new(),
            ),
            node(
                "conv-1",
                GraphOp::Conv,
                vec!["gelu-0".into(), "weight-1".into()],
                vec!["conv-output-1".into()],
                BTreeMap::from([
                    ("group".into(), json!(1)),
                    ("kernel_shape".into(), json!([1, 1])),
                    ("pads".into(), json!([0, 0, 0, 0])),
                    ("strides".into(), json!([1, 1])),
                ]),
            ),
            node(
                "add-1",
                GraphOp::Add,
                vec!["conv-output-1".into(), "bias-1".into()],
                vec!["add-output-1".into()],
                BTreeMap::new(),
            ),
            node(
                "identity-1",
                GraphOp::Identity,
                vec!["add-output-1".into()],
                vec!["identity-output-1".into()],
                BTreeMap::new(),
            ),
            node(
                "relu-1",
                GraphOp::Relu,
                vec!["identity-output-1".into()],
                vec!["relu-1".into()],
                BTreeMap::new(),
            ),
        ]);
        nodes
    }

    fn gelu_constants() -> HashMap<String, GraphValue> {
        let mut values = constants(2);
        for (name, value) in [
            ("sqrt-two", f32::from_bits(0x3fb5_04f3)),
            ("one", 1.0_f32),
            ("half", 0.5_f32),
        ] {
            values.insert(
                name.to_string(),
                GraphValue::Tensor(Tensor::new(value, &Device::Cpu).unwrap()),
            );
        }
        values
    }

    fn use_counts(nodes: &[GraphNode]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for input in nodes.iter().flat_map(|node| &node.inputs) {
            *counts.entry(input.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn node(
        name: &str,
        op: GraphOp,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: BTreeMap<String, serde_json::Value>,
    ) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            op,
            inputs,
            outputs,
            attributes,
        }
    }
}
