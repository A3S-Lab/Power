use std::collections::HashMap;

use candle_core::DType;
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

mod cpu;

pub(super) struct FusedOutput {
    pub(super) value: GraphValue,
    pub(super) consumed_nodes: usize,
}

/// Executes a private scalar-affine -> HardSwish -> scalar-affine topology in
/// one CPU pass. Eligibility is defined solely by operators, static scalar
/// inputs, exact data flow, and liveness. Every unsupported case retains the
/// ordinary graph path.
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
        .ok_or_else(|| missing_input(matched.first_multiply, matched.input))?
        .tensor(&matched.first_multiply.name)?;
    if input.dtype() != DType::F32
        || !input.device().is_cpu()
        || !input.is_contiguous()
        || input.elem_count() == 0
        || ![
            matched.pre_scale,
            matched.pre_bias,
            matched.alpha,
            matched.beta,
            matched.post_scale,
            matched.post_bias,
        ]
        .into_iter()
        .all(f32::is_finite)
    {
        return Ok(None);
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }
    let output = cpu::execute(
        input,
        matched.pre_scale,
        matched.pre_bias,
        matched.alpha,
        matched.beta,
        matched.post_scale,
        matched.post_bias,
    )
    .map_err(|error| {
        PowerError::InferenceFailed(format!(
            "static graph affine-HardSwish-affine fusion failed at nodes '{}' through '{}': {error}",
            matched.first_multiply.name, matched.final_add.name
        ))
    })?;
    Ok(Some(FusedOutput {
        value: GraphValue::Tensor(output),
        consumed_nodes: matched.consumed_nodes,
    }))
}

struct MatchedWindow<'a> {
    first_multiply: &'a GraphNode,
    final_add: &'a GraphNode,
    input: &'a str,
    pre_scale: f32,
    pre_bias: f32,
    alpha: f32,
    beta: f32,
    post_scale: f32,
    post_bias: f32,
    consumed_nodes: usize,
}

fn matched_window<'a>(
    nodes: &'a [GraphNode],
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedWindow<'a>> {
    let first_multiply = nodes.first()?;
    let (input, pre_scale, mut value) =
        scalar_binary(first_multiply, GraphOp::Mul, scalar_constants)?;
    if !private_intermediate(value, 1, use_counts, retained_output) {
        return None;
    }

    let mut cursor = 1;
    value = skip_private_identities(nodes, &mut cursor, value, use_counts, retained_output)?;
    let first_add = nodes.get(cursor)?;
    let (add_input, pre_bias, add_output) =
        scalar_binary(first_add, GraphOp::Add, scalar_constants)?;
    if add_input != value || add_output == retained_output {
        return None;
    }
    value = add_output;
    cursor += 1;
    value = skip_private_identities(nodes, &mut cursor, value, use_counts, retained_output)?;
    if !private_intermediate(value, 2, use_counts, retained_output) {
        return None;
    }
    let activation_input = value;

    let hard_sigmoid = nodes.get(cursor)?;
    if hard_sigmoid.op != GraphOp::HardSigmoid
        || hard_sigmoid.inputs.as_slice() != [activation_input]
    {
        return None;
    }
    let [hard_sigmoid_output] = hard_sigmoid.outputs.as_slice() else {
        return None;
    };
    if !private_intermediate(hard_sigmoid_output, 1, use_counts, retained_output) {
        return None;
    }
    let alpha = hard_sigmoid.float("alpha", 0.2).ok()? as f32;
    let beta = hard_sigmoid.float("beta", 0.5).ok()? as f32;
    cursor += 1;

    let hard_swish_multiply = nodes.get(cursor)?;
    if hard_swish_multiply.op != GraphOp::Mul || !hard_swish_multiply.attributes.is_empty() {
        return None;
    }
    let [left, right] = hard_swish_multiply.inputs.as_slice() else {
        return None;
    };
    if !matches!(
        (left.as_str(), right.as_str()),
        (left, right)
            if (left == activation_input && right == hard_sigmoid_output)
                || (right == activation_input && left == hard_sigmoid_output)
    ) {
        return None;
    }
    let [hard_swish_output] = hard_swish_multiply.outputs.as_slice() else {
        return None;
    };
    if !private_intermediate(hard_swish_output, 1, use_counts, retained_output) {
        return None;
    }
    value = hard_swish_output;
    cursor += 1;
    value = skip_private_identities(nodes, &mut cursor, value, use_counts, retained_output)?;

    let second_multiply = nodes.get(cursor)?;
    let (second_input, post_scale, second_output) =
        scalar_binary(second_multiply, GraphOp::Mul, scalar_constants)?;
    if second_input != value || !private_intermediate(second_output, 1, use_counts, retained_output)
    {
        return None;
    }
    value = second_output;
    cursor += 1;
    value = skip_private_identities(nodes, &mut cursor, value, use_counts, retained_output)?;

    let final_add = nodes.get(cursor)?;
    let (final_input, post_bias, _) = scalar_binary(final_add, GraphOp::Add, scalar_constants)?;
    if final_input != value {
        return None;
    }
    Some(MatchedWindow {
        first_multiply,
        final_add,
        input,
        pre_scale,
        pre_bias,
        alpha,
        beta,
        post_scale,
        post_bias,
        consumed_nodes: cursor + 1,
    })
}

fn scalar_binary<'a>(
    node: &'a GraphNode,
    operation: GraphOp,
    scalar_constants: &HashMap<String, f32>,
) -> Option<(&'a str, f32, &'a str)> {
    if node.op != operation || !node.attributes.is_empty() {
        return None;
    }
    let [left, right] = node.inputs.as_slice() else {
        return None;
    };
    let (input, scalar) = match (
        scalar_constants.get(left).copied(),
        scalar_constants.get(right).copied(),
    ) {
        (Some(scalar), None) => (right.as_str(), scalar),
        (None, Some(scalar)) => (left.as_str(), scalar),
        _ => return None,
    };
    let [output] = node.outputs.as_slice() else {
        return None;
    };
    Some((input, scalar, output))
}

fn skip_private_identities<'a>(
    nodes: &'a [GraphNode],
    cursor: &mut usize,
    mut input: &'a str,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<&'a str> {
    while let Some(identity) = nodes
        .get(*cursor)
        .filter(|node| node.op == GraphOp::Identity)
    {
        let [identity_input] = identity.inputs.as_slice() else {
            return None;
        };
        let [identity_output] = identity.outputs.as_slice() else {
            return None;
        };
        if identity_input != input
            || !identity.attributes.is_empty()
            || !private_intermediate(input, 1, use_counts, retained_output)
            || identity_output == retained_output
        {
            return None;
        }
        input = identity_output;
        *cursor += 1;
    }
    Some(input)
}

fn private_intermediate(
    output: &str,
    expected_uses: usize,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    output != retained_output && use_counts.get(output).copied() == Some(expected_uses)
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
    fn matches_only_exact_private_operator_topology() {
        let nodes = hard_swish_window();
        let constants = scalar_constants();
        let uses = use_counts();

        let matched = matched_window(&nodes, &constants, &uses, "graph-output").unwrap();

        assert_eq!(matched.input, "input");
        assert_eq!(matched.pre_scale, 0.875);
        assert_eq!(matched.pre_bias, -0.125);
        assert_eq!(matched.post_scale, 1.125);
        assert_eq!(matched.post_bias, 0.0625);
        assert_eq!(matched.consumed_nodes, nodes.len());

        let mut shared = uses.clone();
        shared.insert("activated-input".to_string(), 3);
        assert!(matched_window(&nodes, &constants, &shared, "graph-output").is_none());
        assert!(matched_window(&nodes, &constants, &uses, "activated-input").is_none());
    }

    #[test]
    fn fused_cpu_pass_is_bitwise_equal_to_separate_float32_nodes() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            vec![-9.25_f32, -3.0, -0.0, 0.1, 2.75, 8.5],
            (1, 1, 2, 3),
            &device,
        )
        .unwrap();
        let pre_scale = 0.8123457_f32;
        let pre_bias = -0.137531_f32;
        let alpha = 1.0_f32 / 6.0;
        let beta = 0.5_f32;
        let post_scale = 1.071337_f32;
        let post_bias = 0.031337_f32;
        let scalar = |value| Tensor::new(value, &device).unwrap();
        let activated_input = input
            .broadcast_mul(&scalar(pre_scale))
            .and_then(|value| value.broadcast_add(&scalar(pre_bias)))
            .unwrap();
        let gate = (&activated_input * f64::from(alpha))
            .and_then(|value| value.affine(1.0, f64::from(beta)))
            .and_then(|value| value.clamp(0.0, 1.0))
            .unwrap();
        let explicit = activated_input
            .broadcast_mul(&gate)
            .and_then(|value| value.broadcast_mul(&scalar(post_scale)))
            .and_then(|value| value.broadcast_add(&scalar(post_bias)))
            .unwrap();
        let fused = cpu::execute(
            &input, pre_scale, pre_bias, alpha, beta, post_scale, post_bias,
        )
        .unwrap();

        assert_eq!(
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            explicit.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    fn hard_swish_window() -> Vec<GraphNode> {
        let mut hard_sigmoid = node(
            "gate",
            GraphOp::HardSigmoid,
            &["activated-input"],
            "bounded-gate",
        );
        hard_sigmoid
            .attributes
            .insert("alpha".to_string(), serde_json::json!(1.0 / 6.0));
        hard_sigmoid
            .attributes
            .insert("beta".to_string(), serde_json::json!(0.5));
        vec![
            node("pre-scale", GraphOp::Mul, &["pre-scale", "input"], "scaled"),
            node(
                "rename-scale",
                GraphOp::Identity,
                &["scaled"],
                "scaled-renamed",
            ),
            node(
                "pre-bias",
                GraphOp::Add,
                &["scaled-renamed", "pre-bias"],
                "biased",
            ),
            node(
                "rename-bias",
                GraphOp::Identity,
                &["biased"],
                "activated-input",
            ),
            hard_sigmoid,
            node(
                "activate",
                GraphOp::Mul,
                &["activated-input", "bounded-gate"],
                "activated",
            ),
            node(
                "post-scale",
                GraphOp::Mul,
                &["post-scale", "activated"],
                "post-scaled",
            ),
            node(
                "rename-post-scale",
                GraphOp::Identity,
                &["post-scaled"],
                "post-scaled-renamed",
            ),
            node(
                "post-bias",
                GraphOp::Add,
                &["post-scaled-renamed", "post-bias"],
                "output",
            ),
        ]
    }

    fn scalar_constants() -> HashMap<String, f32> {
        HashMap::from([
            ("pre-scale".to_string(), 0.875),
            ("pre-bias".to_string(), -0.125),
            ("post-scale".to_string(), 1.125),
            ("post-bias".to_string(), 0.0625),
        ])
    }

    fn use_counts() -> HashMap<String, usize> {
        HashMap::from([
            ("scaled".to_string(), 1),
            ("scaled-renamed".to_string(), 1),
            ("biased".to_string(), 1),
            ("activated-input".to_string(), 2),
            ("bounded-gate".to_string(), 1),
            ("activated".to_string(), 1),
            ("post-scaled".to_string(), 1),
            ("post-scaled-renamed".to_string(), 1),
        ])
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
