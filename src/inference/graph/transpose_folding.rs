use std::collections::{BTreeMap, BTreeSet};

use super::plan::{GraphNode, GraphOp};

/// Composes private producer-consumer `Transpose` pairs in a validated graph.
///
/// Only explicit, complete permutations are folded. An omitted permutation
/// depends on the runtime rank, while an invalid permutation must keep flowing
/// to the executor's existing validation error. A producer also remains
/// explicit when its value is published or has more than one consumer.
pub(super) fn fold_private_transposes(
    nodes: &mut Vec<GraphNode>,
    retained_outputs: &BTreeSet<String>,
) {
    let mut use_counts = BTreeMap::<String, usize>::new();
    let mut producer_by_output = BTreeMap::<String, usize>::new();
    for (index, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            *use_counts.entry(input.clone()).or_default() += 1;
        }
        if let Some(output) = node.outputs.first() {
            producer_by_output.insert(output.clone(), index);
        }
    }

    let mut removed = vec![false; nodes.len()];
    for consumer_index in 0..nodes.len() {
        let Some(consumer_input) = transpose_input(&nodes[consumer_index]) else {
            continue;
        };
        if retained_outputs.contains(consumer_input)
            || use_counts.get(consumer_input).copied() != Some(1)
        {
            continue;
        }
        let Some(&producer_index) = producer_by_output.get(consumer_input) else {
            continue;
        };
        if producer_index >= consumer_index || removed[producer_index] {
            continue;
        }
        let Some(producer_input) = transpose_input(&nodes[producer_index]).map(str::to_owned)
        else {
            continue;
        };
        let Some(producer_permutation) = explicit_permutation(&nodes[producer_index]) else {
            continue;
        };
        let Some(consumer_permutation) = explicit_permutation(&nodes[consumer_index]) else {
            continue;
        };
        let Some(composed) = compose(&producer_permutation, &consumer_permutation) else {
            continue;
        };

        let consumer = &mut nodes[consumer_index];
        consumer.inputs[0] = producer_input;
        consumer.attributes.insert(
            "perm".to_string(),
            serde_json::Value::Array(composed.into_iter().map(serde_json::Value::from).collect()),
        );
        removed[producer_index] = true;
    }

    let mut index = 0usize;
    nodes.retain(|_| {
        let retain = !removed[index];
        index += 1;
        retain
    });
}

fn transpose_input(node: &GraphNode) -> Option<&str> {
    (node.op == GraphOp::Transpose && node.inputs.len() == 1 && node.outputs.len() == 1)
        .then(|| node.inputs[0].as_str())
}

fn explicit_permutation(node: &GraphNode) -> Option<Vec<i64>> {
    let values = node.attributes.get("perm")?.as_array()?;
    let permutation = values
        .iter()
        .map(serde_json::Value::as_i64)
        .collect::<Option<Vec<_>>>()?;
    let mut reviewed = permutation.clone();
    reviewed.sort_unstable();
    reviewed
        .iter()
        .enumerate()
        .all(|(axis, value)| usize::try_from(*value).ok() == Some(axis))
        .then_some(permutation)
}

/// ONNX Transpose defines output axis `i` as input axis `perm[i]`.
fn compose(producer: &[i64], consumer: &[i64]) -> Option<Vec<i64>> {
    if producer.len() != consumer.len() {
        return None;
    }
    consumer
        .iter()
        .map(|axis| producer.get(usize::try_from(*axis).ok()?).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_pair_becomes_one_validated_identity_permutation() {
        let mut nodes = vec![
            transpose("first", "input", "middle", &[0, 2, 1]),
            transpose("second", "middle", "published", &[0, 2, 1]),
        ];

        fold_private_transposes(&mut nodes, &BTreeSet::from(["published".to_string()]));

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "second");
        assert_eq!(nodes[0].op, GraphOp::Transpose);
        assert_eq!(nodes[0].inputs, ["input"]);
        assert_eq!(nodes[0].outputs, ["published"]);
        assert_eq!(nodes[0].ints("perm", &[]).unwrap(), [0, 1, 2]);
    }

    #[test]
    fn non_identity_pair_becomes_one_composed_transpose() {
        let mut nodes = vec![
            transpose("first", "input", "middle", &[1, 0, 2]),
            transpose("second", "middle", "published", &[0, 2, 1]),
        ];

        fold_private_transposes(&mut nodes, &BTreeSet::from(["published".to_string()]));

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].op, GraphOp::Transpose);
        assert_eq!(nodes[0].inputs, ["input"]);
        assert_eq!(nodes[0].ints("perm", &[]).unwrap(), [1, 2, 0]);
    }

    #[test]
    fn complete_private_chain_is_composed_to_one_identity_permutation() {
        let mut nodes = vec![
            transpose("first", "input", "one", &[1, 0, 2]),
            transpose("second", "one", "two", &[0, 2, 1]),
            transpose("third", "two", "published", &[2, 0, 1]),
        ];

        fold_private_transposes(&mut nodes, &BTreeSet::from(["published".to_string()]));

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].op, GraphOp::Transpose);
        assert_eq!(nodes[0].inputs, ["input"]);
        assert_eq!(nodes[0].outputs, ["published"]);
        assert_eq!(nodes[0].ints("perm", &[]).unwrap(), [0, 1, 2]);
    }

    #[test]
    fn fan_out_keeps_the_producer_and_consumer_unchanged() {
        let mut nodes = vec![
            transpose("first", "input", "middle", &[0, 2, 1]),
            transpose("second", "middle", "published", &[0, 2, 1]),
            node("other-consumer", GraphOp::Add, &["middle", "bias"], "other"),
        ];
        let original = nodes.clone();

        fold_private_transposes(&mut nodes, &BTreeSet::from(["published".to_string()]));

        assert_nodes_unchanged(&nodes, &original);
    }

    #[test]
    fn retained_intermediate_keeps_the_pair_unchanged() {
        let mut nodes = vec![
            transpose("first", "input", "middle", &[0, 2, 1]),
            transpose("second", "middle", "downstream", &[0, 2, 1]),
        ];
        let original = nodes.clone();

        fold_private_transposes(&mut nodes, &BTreeSet::from(["middle".to_string()]));

        assert_nodes_unchanged(&nodes, &original);
    }

    #[test]
    fn missing_or_invalid_permutation_keeps_runtime_validation_authority() {
        let mut missing = node("first", GraphOp::Transpose, &["input"], "middle");
        missing.attributes.clear();
        let valid = transpose("second", "middle", "published", &[0, 2, 1]);
        let invalid = transpose("invalid", "middle", "published", &[0, 0, 1]);
        for mut nodes in [
            vec![missing, valid.clone()],
            vec![transpose("first", "input", "middle", &[0, 2, 1]), invalid],
        ] {
            let original = nodes.clone();

            fold_private_transposes(&mut nodes, &BTreeSet::from(["published".to_string()]));

            assert_nodes_unchanged(&nodes, &original);
        }
    }

    fn transpose(name: &str, input: &str, output: &str, permutation: &[i64]) -> GraphNode {
        let mut node = node(name, GraphOp::Transpose, &[input], output);
        node.attributes.insert(
            "perm".to_string(),
            serde_json::Value::Array(
                permutation
                    .iter()
                    .copied()
                    .map(serde_json::Value::from)
                    .collect(),
            ),
        );
        node
    }

    fn node(name: &str, op: GraphOp, inputs: &[&str], output: &str) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|input| (*input).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: BTreeMap::new(),
        }
    }

    fn assert_nodes_unchanged(actual: &[GraphNode], expected: &[GraphNode]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.op, expected.op);
            assert_eq!(actual.inputs, expected.inputs);
            assert_eq!(actual.outputs, expected.outputs);
            assert_eq!(actual.attributes, expected.attributes);
        }
    }
}
