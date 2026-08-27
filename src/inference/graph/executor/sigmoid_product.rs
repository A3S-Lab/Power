use std::collections::HashMap;

use candle_core::{DType, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

#[cfg(feature = "embedded-cuda")]
mod cuda;

pub(super) struct FusedOutput {
    pub(super) value: GraphValue,
    pub(super) consumed_nodes: usize,
}

/// Removes private sigmoid/product intermediates without changing the graph's
/// arithmetic. Equal-shape dual sigmoids can execute in one CUDA pass. When a
/// preceding broadcast gate has already been materialized, the second full-
/// shape sigmoid and its terminal product execute in one pass so the smaller
/// gate's exponential is not redundantly recomputed for every output element.
/// Eligibility depends only on topology, ownership, tensor layout, device,
/// dtype, cancellation, and resource bounds.
pub(super) fn try_execute(
    nodes: &[GraphNode],
    values: &HashMap<String, GraphValue>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    element_limit: usize,
    cancellation: &CancellationToken,
) -> Result<Option<FusedOutput>> {
    if let Some(matched) = matched_dual_window(nodes, use_counts, retained_output) {
        let left = tensor(values, matched.left_input, matched.first)?;
        let right = tensor(values, matched.right_input, matched.second)?;
        let output_elements = left.elem_count();
        if left.dims() == right.dims() && execution_eligible(left, right, output_elements) {
            validate_execution(
                matched.product,
                output_elements,
                element_limit,
                cancellation,
            )?;
            #[cfg(feature = "embedded-cuda")]
            {
                let output = cuda::execute_product(left, right).map_err(|error| {
                    PowerError::InferenceFailed(format!(
                        "static graph nodes '{}' through '{}' fused sigmoid product failed: {error}",
                        matched.first.name, matched.product.name
                    ))
                })?;
                return Ok(Some(FusedOutput {
                    value: GraphValue::Tensor(output),
                    consumed_nodes: 3,
                }));
            }
        }
    }

    let Some(matched) = matched_sigmoid_mul(nodes, use_counts, retained_output) else {
        return Ok(None);
    };
    let input = tensor(values, matched.input, matched.sigmoid)?;
    let multiplier = tensor(values, matched.multiplier, matched.product)?;
    if multiplier_layout(input.dims(), multiplier.dims()).is_none()
        || !execution_eligible(input, multiplier, input.elem_count())
    {
        return Ok(None);
    }
    validate_execution(
        matched.product,
        input.elem_count(),
        element_limit,
        cancellation,
    )?;

    #[cfg(feature = "embedded-cuda")]
    {
        let output = cuda::execute_sigmoid_mul(input, multiplier).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused sigmoid multiplication failed: {error}",
                matched.sigmoid.name, matched.product.name
            ))
        })?;
        Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output),
            consumed_nodes: 2,
        }))
    }

    #[cfg(not(feature = "embedded-cuda"))]
    Ok(None)
}

fn execution_eligible(left: &Tensor, right: &Tensor, output_elements: usize) -> bool {
    left.dtype() == DType::F32
        && right.dtype() == DType::F32
        && left.device().is_cuda()
        && right.device().same_device(left.device())
        && left.is_contiguous()
        && right.is_contiguous()
        && output_elements != 0
        && u32::try_from(output_elements).is_ok()
}

fn validate_execution(
    product: &GraphNode,
    output_elements: usize,
    element_limit: usize,
    cancellation: &CancellationToken,
) -> Result<()> {
    if output_elements > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
            product.name, output_elements,
        )));
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct MatchedDualWindow<'a> {
    first: &'a GraphNode,
    second: &'a GraphNode,
    product: &'a GraphNode,
    left_input: &'a str,
    right_input: &'a str,
}

#[derive(Clone, Copy, Debug)]
struct MatchedSigmoidMul<'a> {
    sigmoid: &'a GraphNode,
    product: &'a GraphNode,
    input: &'a str,
    multiplier: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MultiplierLayout {
    SameShape,
    NchwPerChannel,
    NchwPerSpatialPosition,
}

fn multiplier_layout(input: &[usize], multiplier: &[usize]) -> Option<MultiplierLayout> {
    if input == multiplier {
        return Some(MultiplierLayout::SameShape);
    }
    if input.len() != 4 || multiplier.len() != 4 || input[0] != multiplier[0] {
        return None;
    }
    if input[1] == multiplier[1] && multiplier[2] == 1 && multiplier[3] == 1 {
        return Some(MultiplierLayout::NchwPerChannel);
    }
    if multiplier[1] == 1 && input[2] == multiplier[2] && input[3] == multiplier[3] {
        return Some(MultiplierLayout::NchwPerSpatialPosition);
    }
    None
}

fn matched_dual_window<'a>(
    nodes: &'a [GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedDualWindow<'a>> {
    let [first, second, product, ..] = nodes else {
        return None;
    };
    if [first.op, second.op, product.op] != [GraphOp::Sigmoid, GraphOp::Sigmoid, GraphOp::Mul]
        || [first, second, product]
            .iter()
            .any(|node| !node.attributes.is_empty() || node.outputs.len() != 1)
    {
        return None;
    }
    let [left_input] = first.inputs.as_slice() else {
        return None;
    };
    let [right_input] = second.inputs.as_slice() else {
        return None;
    };
    let [left_output] = first.outputs.as_slice() else {
        return None;
    };
    let [right_output] = second.outputs.as_slice() else {
        return None;
    };
    let [product_left, product_right] = product.inputs.as_slice() else {
        return None;
    };
    if left_output == right_output
        || left_output == retained_output
        || right_output == retained_output
        || use_counts.get(left_output).copied() != Some(1)
        || use_counts.get(right_output).copied() != Some(1)
        || !((product_left == left_output && product_right == right_output)
            || (product_left == right_output && product_right == left_output))
    {
        return None;
    }
    Some(MatchedDualWindow {
        first,
        second,
        product,
        left_input,
        right_input,
    })
}

fn matched_sigmoid_mul<'a>(
    nodes: &'a [GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedSigmoidMul<'a>> {
    let [sigmoid, product, ..] = nodes else {
        return None;
    };
    if [sigmoid.op, product.op] != [GraphOp::Sigmoid, GraphOp::Mul]
        || [sigmoid, product]
            .iter()
            .any(|node| !node.attributes.is_empty() || node.outputs.len() != 1)
    {
        return None;
    }
    let [input] = sigmoid.inputs.as_slice() else {
        return None;
    };
    let [sigmoid_output] = sigmoid.outputs.as_slice() else {
        return None;
    };
    let [product_left, product_right] = product.inputs.as_slice() else {
        return None;
    };
    if sigmoid_output == retained_output || use_counts.get(sigmoid_output).copied() != Some(1) {
        return None;
    }
    let multiplier = if product_left == sigmoid_output && product_right != sigmoid_output {
        product_right
    } else if product_right == sigmoid_output && product_left != sigmoid_output {
        product_left
    } else {
        return None;
    };
    Some(MatchedSigmoidMul {
        sigmoid,
        product,
        input,
        multiplier,
    })
}

fn tensor<'a>(
    values: &'a HashMap<String, GraphValue>,
    name: &str,
    node: &GraphNode,
) -> Result<&'a Tensor> {
    values
        .get(name)
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' could not resolve input '{name}'",
                node.name
            ))
        })?
        .tensor(&node.name)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use candle_core::{Device, Tensor};

    use super::*;

    fn node(name: &str, op: GraphOp, inputs: &[&str], output: &str) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|input| (*input).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: BTreeMap::new(),
        }
    }

    fn dual_window() -> Vec<GraphNode> {
        vec![
            node("left-gate", GraphOp::Sigmoid, &["left"], "left-gated"),
            node("right-gate", GraphOp::Sigmoid, &["right"], "right-gated"),
            node(
                "product",
                GraphOp::Mul,
                &["right-gated", "left-gated"],
                "output",
            ),
        ]
    }

    fn sigmoid_mul_window() -> Vec<GraphNode> {
        vec![
            node("gate", GraphOp::Sigmoid, &["input"], "gated"),
            node("product", GraphOp::Mul, &["multiplier", "gated"], "output"),
        ]
    }

    #[test]
    fn dual_matcher_accepts_only_the_exact_private_product() {
        let nodes = dual_window();
        let uses = HashMap::from([
            ("left-gated".to_string(), 1),
            ("right-gated".to_string(), 1),
        ]);

        let matched = matched_dual_window(&nodes, &uses, "output").unwrap();

        assert_eq!(matched.left_input, "left");
        assert_eq!(matched.right_input, "right");
        assert_eq!(matched.product.name, "product");
    }

    #[test]
    fn dual_matcher_rejects_shared_retained_and_unrelated_intermediates() {
        let nodes = dual_window();
        let shared = HashMap::from([
            ("left-gated".to_string(), 2),
            ("right-gated".to_string(), 1),
        ]);
        assert!(matched_dual_window(&nodes, &shared, "output").is_none());

        let private = HashMap::from([
            ("left-gated".to_string(), 1),
            ("right-gated".to_string(), 1),
        ]);
        assert!(matched_dual_window(&nodes, &private, "right-gated").is_none());

        let mut unrelated = nodes;
        unrelated[2].inputs[1] = "unrelated".to_string();
        assert!(matched_dual_window(&unrelated, &private, "output").is_none());
    }

    #[test]
    fn sigmoid_mul_matcher_accepts_commuted_private_products_only() {
        let nodes = sigmoid_mul_window();
        let uses = HashMap::from([("gated".to_string(), 1)]);
        let matched = matched_sigmoid_mul(&nodes, &uses, "output").unwrap();
        assert_eq!(matched.input, "input");
        assert_eq!(matched.multiplier, "multiplier");

        let mut direct = nodes.clone();
        direct[1].inputs.swap(0, 1);
        assert!(matched_sigmoid_mul(&direct, &uses, "output").is_some());

        let shared = HashMap::from([("gated".to_string(), 2)]);
        assert!(matched_sigmoid_mul(&nodes, &shared, "output").is_none());
        assert!(matched_sigmoid_mul(&nodes, &uses, "gated").is_none());
    }

    #[test]
    fn runtime_retains_the_ordinary_broadcast_path_off_cuda() {
        let nodes = sigmoid_mul_window();
        let uses = HashMap::from([("gated".to_string(), 1)]);
        let values = HashMap::from([
            (
                "input".to_string(),
                GraphValue::Tensor(Tensor::zeros((2, 3, 5, 7), DType::F32, &Device::Cpu).unwrap()),
            ),
            (
                "multiplier".to_string(),
                GraphValue::Tensor(Tensor::ones((2, 1, 5, 7), DType::F32, &Device::Cpu).unwrap()),
            ),
        ]);

        let fused = try_execute(
            &nodes,
            &values,
            &uses,
            "output",
            1_000,
            &CancellationToken::new(),
        )
        .unwrap();

        assert!(fused.is_none());
    }

    #[test]
    fn multiplier_layouts_are_generic_and_unsupported_broadcasts_fall_back() {
        assert_eq!(
            multiplier_layout(&[2, 3, 5], &[2, 3, 5]),
            Some(MultiplierLayout::SameShape)
        );
        assert_eq!(
            multiplier_layout(&[7, 11, 13, 17], &[7, 11, 1, 1]),
            Some(MultiplierLayout::NchwPerChannel)
        );
        assert_eq!(
            multiplier_layout(&[3, 5, 19, 23], &[3, 1, 19, 23]),
            Some(MultiplierLayout::NchwPerSpatialPosition)
        );
        assert!(multiplier_layout(&[2, 3, 5, 7], &[1, 3, 5, 7]).is_none());
        assert!(multiplier_layout(&[2, 3, 5, 7], &[1]).is_none());
    }
}
