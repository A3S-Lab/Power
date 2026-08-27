use std::collections::HashMap;

use candle_core::DType;
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

mod cpu;
#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Executes an adjacent, single-consumer HardSigmoid-to-Mul pair without
/// materializing the four HardSigmoid elementwise passes.
///
/// The public graph remains unchanged. Unsupported devices, dtypes, layouts,
/// shapes, and broadcast forms return `None` so the executor retains the
/// ordinary node-by-node path.
pub(super) fn try_mul(
    hard_sigmoid: &GraphNode,
    multiply: &GraphNode,
    values: &HashMap<String, GraphValue>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    cancellation: &CancellationToken,
) -> Result<Option<GraphValue>> {
    let Some((gate_name, multiplicand_name)) =
        matched_inputs(hard_sigmoid, multiply, use_counts, retained_output)
    else {
        return Ok(None);
    };
    let gate = values
        .get(gate_name)
        .ok_or_else(|| missing_input(hard_sigmoid, gate_name))?
        .tensor(&hard_sigmoid.name)?;
    let multiplicand = values
        .get(multiplicand_name)
        .ok_or_else(|| missing_input(multiply, multiplicand_name))?
        .tensor(&multiply.name)?;
    if gate.dtype() != DType::F32
        || multiplicand.dtype() != DType::F32
        || !(gate.device().is_cpu() || gate.device().is_cuda())
        || !multiplicand.device().same_device(gate.device())
        || gate.elem_count() == 0
        || multiplicand.elem_count() == 0
        || u32::try_from(multiplicand.elem_count()).is_err()
        || !gate.is_contiguous()
        || !multiplicand.is_contiguous()
        || !fused_nchw_shapes(gate.dims(), multiplicand.dims())
    {
        return Ok(None);
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }
    let alpha = hard_sigmoid.float("alpha", 0.2)? as f32;
    let beta = hard_sigmoid.float("beta", 0.5)? as f32;
    if gate.device().is_cpu() {
        let output = cpu::mul(multiplicand, gate, alpha, beta).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' and '{}' fused gated HardSigmoid failed: {error}",
                hard_sigmoid.name, multiply.name
            ))
        })?;
        return Ok(Some(GraphValue::Tensor(output)));
    }

    #[cfg(feature = "embedded-cuda")]
    {
        let output = cuda::mul(multiplicand, gate, alpha, beta).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' and '{}' fused gated HardSigmoid failed: {error}",
                hard_sigmoid.name, multiply.name
            ))
        })?;
        Ok(Some(GraphValue::Tensor(output)))
    }

    #[cfg(not(feature = "embedded-cuda"))]
    Ok(None)
}

fn fused_nchw_shapes(gate: &[usize], multiplicand: &[usize]) -> bool {
    let ([gate_batch, gate_channels, gate_height, gate_width], [batch, channels, _height, _width]) =
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

fn matched_inputs<'a>(
    hard_sigmoid: &'a GraphNode,
    multiply: &'a GraphNode,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<(&'a str, &'a str)> {
    if hard_sigmoid.op != GraphOp::HardSigmoid || multiply.op != GraphOp::Mul {
        return None;
    }
    let [gate_name] = hard_sigmoid.inputs.as_slice() else {
        return None;
    };
    let [hard_sigmoid_output] = hard_sigmoid.outputs.as_slice() else {
        return None;
    };
    if hard_sigmoid_output == retained_output
        || use_counts.get(hard_sigmoid_output).copied() != Some(1)
    {
        return None;
    }
    let [left, right] = multiply.inputs.as_slice() else {
        return None;
    };
    let multiplicand = match (left == hard_sigmoid_output, right == hard_sigmoid_output) {
        (true, false) => right,
        (false, true) => left,
        _ => return None,
    };
    Some((gate_name, multiplicand))
}

fn missing_input(node: &GraphNode, input: &str) -> PowerError {
    PowerError::InferenceFailed(format!(
        "static graph node '{}' could not resolve input '{input}'",
        node.name
    ))
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

    #[test]
    fn matches_only_adjacent_single_consumer_non_output_pairs() {
        let hard_sigmoid = node(
            "activation",
            GraphOp::HardSigmoid,
            &["gate"],
            "bounded-gate",
        );
        let right_multiply = node(
            "multiply",
            GraphOp::Mul,
            &["features", "bounded-gate"],
            "output",
        );
        let left_multiply = node(
            "multiply",
            GraphOp::Mul,
            &["bounded-gate", "features"],
            "output",
        );
        let one_use = HashMap::from([("bounded-gate".to_string(), 1)]);

        assert_eq!(
            matched_inputs(&hard_sigmoid, &right_multiply, &one_use, "output"),
            Some(("gate", "features"))
        );
        assert_eq!(
            matched_inputs(&hard_sigmoid, &left_multiply, &one_use, "output"),
            Some(("gate", "features"))
        );
        assert!(matched_inputs(
            &hard_sigmoid,
            &right_multiply,
            &HashMap::from([("bounded-gate".to_string(), 2)]),
            "output"
        )
        .is_none());
        assert!(matched_inputs(&hard_sigmoid, &right_multiply, &one_use, "bounded-gate").is_none());
    }

    #[test]
    fn accepts_only_reviewed_rank_four_shapes() {
        assert!(fused_nchw_shapes(&[1, 2, 3, 4], &[1, 2, 3, 4]));
        assert!(fused_nchw_shapes(&[1, 2, 1, 1], &[1, 2, 3, 4]));
        assert!(!fused_nchw_shapes(&[1, 2, 1, 1], &[1, 2, 4]));
        assert!(!fused_nchw_shapes(&[1, 2, 1, 1], &[1, 3, 3, 4]));
        assert!(!fused_nchw_shapes(&[1, 2, 1, 2], &[1, 2, 3, 4]));
    }
}
