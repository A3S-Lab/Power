use std::collections::HashMap;

use candle_core::DType;
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Executes an exact Div-to-Erf-to-Add-to-Mul-to-Mul error-function activation
/// in one CUDA kernel while preserving every original f32 rounding boundary.
pub(super) fn try_execute(
    nodes: &[GraphNode],
    values: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    _cancellation: &CancellationToken,
) -> Result<Option<GraphValue>> {
    let Some(matched) = matched_inputs(nodes, use_counts, retained_output) else {
        return Ok(None);
    };
    let input = tensor(values, matched.input, &nodes[0])?;
    let (Some(&_divisor), Some(&_offset), Some(&_scale)) = (
        scalar_constants.get(matched.divisor),
        scalar_constants.get(matched.offset),
        scalar_constants.get(matched.scale),
    ) else {
        return Ok(None);
    };
    if input.dtype() != DType::F32
        || !input.device().is_cuda()
        || input.elem_count() == 0
        || u32::try_from(input.elem_count()).is_err()
        || !input.is_contiguous()
    {
        return Ok(None);
    }

    #[cfg(feature = "embedded-cuda")]
    {
        if _cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        let output = cuda::execute(input, _divisor, _offset, _scale).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused error-function activation failed: {error}",
                nodes[0].name, nodes[4].name
            ))
        })?;
        Ok(Some(GraphValue::Tensor(output)))
    }

    #[cfg(not(feature = "embedded-cuda"))]
    Ok(None)
}

struct MatchedInputs<'a> {
    input: &'a str,
    divisor: &'a str,
    offset: &'a str,
    scale: &'a str,
}

fn matched_inputs<'a>(
    nodes: &'a [GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedInputs<'a>> {
    let [divide, erf, add, multiply, scale] = nodes else {
        return None;
    };
    if [divide.op, erf.op, add.op, multiply.op, scale.op]
        != [
            GraphOp::Div,
            GraphOp::Erf,
            GraphOp::Add,
            GraphOp::Mul,
            GraphOp::Mul,
        ]
    {
        return None;
    }
    let [input, divisor] = divide.inputs.as_slice() else {
        return None;
    };
    let [divide_output] = divide.outputs.as_slice() else {
        return None;
    };
    let [erf_input] = erf.inputs.as_slice() else {
        return None;
    };
    let [erf_output] = erf.outputs.as_slice() else {
        return None;
    };
    if erf_input != divide_output {
        return None;
    }
    let [add_left, add_right] = add.inputs.as_slice() else {
        return None;
    };
    let offset = match (add_left == erf_output, add_right == erf_output) {
        (true, false) => add_right,
        (false, true) => add_left,
        _ => return None,
    };
    let [add_output] = add.outputs.as_slice() else {
        return None;
    };
    let [multiply_left, multiply_right] = multiply.inputs.as_slice() else {
        return None;
    };
    if !matches!(
        (multiply_left.as_str(), multiply_right.as_str()),
        (left, right) if (left == input && right == add_output)
            || (left == add_output && right == input)
    ) {
        return None;
    }
    let [multiply_output] = multiply.outputs.as_slice() else {
        return None;
    };
    let [scale_left, scale_right] = scale.inputs.as_slice() else {
        return None;
    };
    let scale = match (
        scale_left == multiply_output,
        scale_right == multiply_output,
    ) {
        (true, false) => scale_right,
        (false, true) => scale_left,
        _ => return None,
    };
    for output in [divide_output, erf_output, add_output, multiply_output] {
        if output == retained_output || use_counts.get(output).copied() != Some(1) {
            return None;
        }
    }
    Some(MatchedInputs {
        input,
        divisor,
        offset,
        scale,
    })
}

fn tensor<'a>(
    values: &'a HashMap<String, GraphValue>,
    input: &str,
    node: &GraphNode,
) -> Result<&'a candle_core::Tensor> {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn node(name: &str, op: GraphOp, inputs: &[&str], output: &str) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: BTreeMap::new(),
        }
    }

    fn activation() -> Vec<GraphNode> {
        vec![
            node("divide", GraphOp::Div, &["input", "divisor"], "divided"),
            node("erf", GraphOp::Erf, &["divided"], "activated"),
            node("add", GraphOp::Add, &["offset", "activated"], "shifted"),
            node("multiply", GraphOp::Mul, &["shifted", "input"], "product"),
            node("scale", GraphOp::Mul, &["scale", "product"], "output"),
        ]
    }

    #[test]
    fn matches_only_adjacent_single_consumer_activation_prefixes() {
        let nodes = activation();
        let one_use = HashMap::from([
            ("divided".to_string(), 1),
            ("activated".to_string(), 1),
            ("shifted".to_string(), 1),
            ("product".to_string(), 1),
        ]);
        let matched = matched_inputs(&nodes, &one_use, "graph-output").unwrap();
        assert_eq!(matched.input, "input");
        assert_eq!(matched.divisor, "divisor");
        assert_eq!(matched.offset, "offset");
        assert_eq!(matched.scale, "scale");

        let mut shared = one_use.clone();
        shared.insert("activated".to_string(), 2);
        assert!(matched_inputs(&nodes, &shared, "graph-output").is_none());
        assert!(matched_inputs(&nodes, &one_use, "product").is_none());

        let mut wrong = nodes;
        wrong[3].inputs[1] = "another-input".to_string();
        assert!(matched_inputs(&wrong, &one_use, "graph-output").is_none());
    }
}
