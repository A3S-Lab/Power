use std::collections::{BTreeSet, HashMap};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

/// Resolves private, constant, contiguous tensor reshapes once while building
/// an executor.
///
/// A contiguous reshape changes only tensor metadata. Removing that pure node
/// avoids repeating the same view construction for every inference and can
/// expose adjacent generic fusion windows. Unsupported values remain in the
/// executable plan so their established runtime validation and fallback are
/// unchanged.
pub(super) fn fold_private_constants(
    nodes: &mut Vec<GraphNode>,
    retained_outputs: &BTreeSet<String>,
    constants: &mut HashMap<String, GraphValue>,
    scalar_constants: &mut HashMap<String, f32>,
    element_limit: usize,
) -> usize {
    let mut folded = 0;
    let mut executable = Vec::with_capacity(nodes.len());
    for node in std::mem::take(nodes) {
        let Some((output_name, value, scalar)) = foldable_value(
            &node,
            retained_outputs,
            constants,
            scalar_constants,
            element_limit,
        ) else {
            executable.push(node);
            continue;
        };

        constants.insert(output_name.clone(), value);
        if let Some(scalar) = scalar {
            scalar_constants.insert(output_name, scalar);
        }
        folded += 1;
    }
    *nodes = executable;
    folded
}

fn foldable_value(
    node: &GraphNode,
    retained_outputs: &BTreeSet<String>,
    constants: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    element_limit: usize,
) -> Option<(String, GraphValue, Option<f32>)> {
    if node.op != GraphOp::Reshape || node.inputs.len() != 2 || node.outputs.len() != 1 {
        return None;
    }
    let output_name = node.outputs.first()?;
    if retained_outputs.contains(output_name) {
        return None;
    }

    let source_name = node.inputs.first()?;
    let source = constants.get(source_name)?;
    let source_tensor = source.tensor(&node.name).ok()?;
    if !source_tensor.is_contiguous()
        || source_tensor.elem_count() == 0
        || source_tensor.elem_count() > element_limit
    {
        return None;
    }
    let requested = constants.get(node.inputs.get(1)?)?.ints(&node.name).ok()?;
    let shape = super::resolve_reshape(source_tensor.dims(), requested, node).ok()?;
    let value = source_tensor.reshape(shape.as_slice()).ok()?;
    if value.elem_count() == 0 || value.elem_count() > element_limit {
        return None;
    }

    Some((
        output_name.clone(),
        GraphValue::Tensor(value),
        scalar_constants.get(source_name).copied(),
    ))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    fn node(name: &str, inputs: &[&str], output: &str) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            op: GraphOp::Reshape,
            inputs: inputs.iter().map(|input| (*input).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: Default::default(),
        }
    }

    #[test]
    fn folds_a_private_constant_contiguous_reshape() {
        let mut nodes = vec![
            GraphNode {
                name: "convolution".to_string(),
                op: GraphOp::Conv,
                inputs: vec!["input".to_string(), "kernel".to_string()],
                outputs: vec!["convolution-output".to_string()],
                attributes: Default::default(),
            },
            node("bias-view", &["bias", "bias-shape"], "viewed-bias"),
            GraphNode {
                name: "add-bias".to_string(),
                op: GraphOp::Add,
                inputs: vec!["convolution-output".to_string(), "viewed-bias".to_string()],
                outputs: vec!["output".to_string()],
                attributes: Default::default(),
            },
        ];
        let mut constants = HashMap::from([
            (
                "bias".to_string(),
                GraphValue::Tensor(Tensor::new(&[2.0_f32, 3.0], &Device::Cpu).unwrap()),
            ),
            (
                "bias-shape".to_string(),
                GraphValue::Ints {
                    values: vec![1, 2, 1, 1],
                    shape: vec![4],
                },
            ),
        ]);

        let folded = fold_private_constants(
            &mut nodes,
            &BTreeSet::from(["output".to_string()]),
            &mut constants,
            &mut HashMap::new(),
            16,
        );

        assert_eq!(folded, 1);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].op, GraphOp::Conv);
        assert_eq!(nodes[1].op, GraphOp::Add);
        assert_eq!(constants["viewed-bias"].shape(), [1, 2, 1, 1]);
    }

    #[test]
    fn retains_a_runtime_dependent_reshape() {
        let mut nodes = vec![node("runtime-view", &["input", "shape"], "view")];
        let mut constants = HashMap::from([(
            "shape".to_string(),
            GraphValue::Ints {
                values: vec![2, 2],
                shape: vec![2],
            },
        )]);

        let folded = fold_private_constants(
            &mut nodes,
            &BTreeSet::new(),
            &mut constants,
            &mut HashMap::new(),
            16,
        );

        assert_eq!(folded, 0);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn retains_a_published_constant_reshape() {
        let mut nodes = vec![node("published-view", &["source", "shape"], "output")];
        let mut constants = constant_values();

        let folded = fold_private_constants(
            &mut nodes,
            &BTreeSet::from(["output".to_string()]),
            &mut constants,
            &mut HashMap::new(),
            16,
        );

        assert_eq!(folded, 0);
        assert_eq!(nodes.len(), 1);
        assert!(!constants.contains_key("output"));
    }

    #[test]
    fn retains_a_noncontiguous_constant_reshape() {
        let source = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert!(!source.is_contiguous());
        let mut nodes = vec![node("strided-view", &["source", "shape"], "view")];
        let mut constants = HashMap::from([
            ("source".to_string(), GraphValue::Tensor(source)),
            (
                "shape".to_string(),
                GraphValue::Ints {
                    values: vec![4],
                    shape: vec![1],
                },
            ),
        ]);

        let folded = fold_private_constants(
            &mut nodes,
            &BTreeSet::new(),
            &mut constants,
            &mut HashMap::new(),
            16,
        );

        assert_eq!(folded, 0);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn propagates_a_scalar_constant_through_reshape() {
        let mut nodes = vec![node("scalar-view", &["source", "shape"], "view")];
        let mut constants = HashMap::from([
            (
                "source".to_string(),
                GraphValue::Tensor(Tensor::new(&[2.0_f32], &Device::Cpu).unwrap()),
            ),
            (
                "shape".to_string(),
                GraphValue::Ints {
                    values: vec![1, 1],
                    shape: vec![2],
                },
            ),
        ]);
        let mut scalars = HashMap::from([("source".to_string(), 2.0_f32)]);

        let folded = fold_private_constants(
            &mut nodes,
            &BTreeSet::new(),
            &mut constants,
            &mut scalars,
            16,
        );

        assert_eq!(folded, 1);
        assert_eq!(scalars["view"], 2.0);
    }

    fn constant_values() -> HashMap<String, GraphValue> {
        HashMap::from([
            (
                "source".to_string(),
                GraphValue::Tensor(Tensor::new(&[1.0_f32, 2.0], &Device::Cpu).unwrap()),
            ),
            (
                "shape".to_string(),
                GraphValue::Ints {
                    values: vec![1, 2],
                    shape: vec![2],
                },
            ),
        ])
    }
}
