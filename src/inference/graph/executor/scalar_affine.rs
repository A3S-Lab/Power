use std::collections::HashMap;

use candle_core::DType;
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

pub(super) struct FusedOutput {
    pub(super) value: GraphValue,
    pub(super) consumed_nodes: usize,
}

/// Executes a private `Mul(scalar) -> Identity* -> Add(scalar)` chain as one
/// affine pass. Matching depends only on graph topology, static scalar shape,
/// and liveness; unsupported dtypes, layouts, devices, or values retain the
/// ordinary node-by-node path.
pub(super) fn try_execute(
    nodes: &[GraphNode],
    values: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    cancellation: &CancellationToken,
) -> Result<Option<FusedOutput>> {
    let Some(matched) = matched_window(nodes, scalar_constants, use_counts, retained_output) else {
        return Ok(None);
    };
    let input = values
        .get(matched.input)
        .ok_or_else(|| missing_input(matched.multiply, matched.input))?
        .tensor(&matched.multiply.name)?;
    if input.dtype() != DType::F32
        || !(input.device().is_cpu() || input.device().is_cuda())
        || !input.is_contiguous()
        || input.elem_count() == 0
        || !matched.scale.is_finite()
        || !matched.bias.is_finite()
    {
        return Ok(None);
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }
    let output = input
        .affine(f64::from(matched.scale), f64::from(matched.bias))
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph scalar-affine fusion failed at nodes '{}' through '{}': {error}",
                matched.multiply.name, matched.add.name
            ))
        })?;
    Ok(Some(FusedOutput {
        value: GraphValue::Tensor(output),
        consumed_nodes: matched.consumed_nodes,
    }))
}

struct MatchedWindow<'a> {
    multiply: &'a GraphNode,
    add: &'a GraphNode,
    input: &'a str,
    scale: f32,
    bias: f32,
    consumed_nodes: usize,
}

fn matched_window<'a>(
    nodes: &'a [GraphNode],
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedWindow<'a>> {
    let multiply = nodes.first()?;
    if multiply.op != GraphOp::Mul || !multiply.attributes.is_empty() {
        return None;
    }
    let [left, right] = multiply.inputs.as_slice() else {
        return None;
    };
    let (input, scale) = scalar_operand(left, right, scalar_constants)?;
    let [multiply_output] = multiply.outputs.as_slice() else {
        return None;
    };
    if !private_intermediate(multiply_output, use_counts, retained_output) {
        return None;
    }

    let mut affine_output = multiply_output.as_str();
    let mut cursor = 1;
    while let Some(identity) = nodes
        .get(cursor)
        .filter(|node| node.op == GraphOp::Identity)
    {
        let [identity_input] = identity.inputs.as_slice() else {
            return None;
        };
        let [identity_output] = identity.outputs.as_slice() else {
            return None;
        };
        if identity_input != affine_output
            || !identity.attributes.is_empty()
            || !private_intermediate(identity_output, use_counts, retained_output)
        {
            return None;
        }
        affine_output = identity_output;
        cursor += 1;
    }

    let add = nodes.get(cursor)?;
    if add.op != GraphOp::Add || !add.attributes.is_empty() {
        return None;
    }
    let [left, right] = add.inputs.as_slice() else {
        return None;
    };
    let (add_input, bias) = scalar_operand(left, right, scalar_constants)?;
    if add_input != affine_output || add.outputs.len() != 1 {
        return None;
    }
    Some(MatchedWindow {
        multiply,
        add,
        input,
        scale,
        bias,
        consumed_nodes: cursor + 1,
    })
}

fn scalar_operand<'a>(
    left: &'a str,
    right: &'a str,
    scalar_constants: &HashMap<String, f32>,
) -> Option<(&'a str, f32)> {
    match (
        scalar_constants.get(left).copied(),
        scalar_constants.get(right).copied(),
    ) {
        (Some(scalar), None) => Some((right, scalar)),
        (None, Some(scalar)) => Some((left, scalar)),
        _ => None,
    }
}

fn private_intermediate(
    output: &str,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    output != retained_output && use_counts.get(output).copied() == Some(1)
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

    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn matches_commuted_scalars_and_any_private_identity_chain() {
        let nodes = vec![
            node("scale", GraphOp::Mul, &["scale", "input"], "scaled"),
            node("rename-1", GraphOp::Identity, &["scaled"], "renamed-1"),
            node("rename-2", GraphOp::Identity, &["renamed-1"], "renamed-2"),
            node("bias", GraphOp::Add, &["renamed-2", "bias"], "output"),
        ];
        let constants = HashMap::from([("scale".to_string(), 2.0), ("bias".to_string(), -1.0)]);
        let uses = HashMap::from([
            ("scaled".to_string(), 1),
            ("renamed-1".to_string(), 1),
            ("renamed-2".to_string(), 1),
        ]);

        let matched = matched_window(&nodes, &constants, &uses, "graph-output").unwrap();

        assert_eq!(matched.input, "input");
        assert_eq!(matched.scale, 2.0);
        assert_eq!(matched.bias, -1.0);
        assert_eq!(matched.consumed_nodes, 4);
    }

    #[test]
    fn rejects_dynamic_scalars_shared_intermediates_and_retained_intermediates() {
        let nodes = vec![
            node("scale", GraphOp::Mul, &["scale", "input"], "scaled"),
            node("bias", GraphOp::Add, &["scaled", "bias"], "output"),
        ];
        let constants = HashMap::from([("scale".to_string(), 2.0), ("bias".to_string(), 1.0)]);

        assert!(matched_window(
            &nodes,
            &HashMap::from([("scale".to_string(), 2.0)]),
            &HashMap::from([("scaled".to_string(), 1)]),
            "graph-output"
        )
        .is_none());
        assert!(matched_window(
            &nodes,
            &constants,
            &HashMap::from([("scaled".to_string(), 2)]),
            "graph-output"
        )
        .is_none());
        assert!(matched_window(
            &nodes,
            &constants,
            &HashMap::from([("scaled".to_string(), 1)]),
            "scaled"
        )
        .is_none());
    }

    #[test]
    fn fused_float32_affine_matches_separate_scalar_broadcasts() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            vec![-1.0e20_f32, -3.5, -0.0, 0.1, 7.0, 1.0e20],
            (1, 1, 2, 3),
            &device,
        )
        .unwrap();
        let scale = Tensor::new(0.8125_f32, &device).unwrap();
        let bias = Tensor::new(-0.375_f32, &device).unwrap();
        let explicit = input
            .broadcast_mul(&scale)
            .and_then(|value| value.broadcast_add(&bias))
            .unwrap();
        let fused = input.affine(0.8125, -0.375).unwrap();

        assert_eq!(
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            explicit.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    fn node(name: &str, op: GraphOp, inputs: &[&str], output: &str) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: BTreeMap::new(),
        }
    }
}
