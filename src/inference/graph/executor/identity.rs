use std::collections::HashMap;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;
use super::liveness;

/// Commits an Identity node as a value rename. A last-use input moves without
/// cloning its tensor handle; shared or retained inputs preserve their alias.
pub(super) fn try_commit(
    node: &GraphNode,
    retained_output: &str,
    remaining_uses: &mut HashMap<String, usize>,
    values: &mut HashMap<String, GraphValue>,
) -> Result<bool> {
    if node.op != GraphOp::Identity {
        return Ok(false);
    }
    let input = node.inputs.first().ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "static graph node '{}' is missing input 0",
            node.name
        ))
    })?;
    let output = node.outputs.first().ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "static graph node '{}' is missing output 0",
            node.name
        ))
    })?;
    let move_value = input != retained_output && remaining_uses.get(input) == Some(&1);
    let value = if move_value {
        let value = values.remove(input);
        if value.is_some() {
            // The match above proves this is the final use. Update the counter
            // directly because the value has already been moved out.
            if let Some(remaining) = remaining_uses.get_mut(input) {
                *remaining = 0;
            }
        }
        value
    } else {
        values.get(input).cloned()
    }
    .ok_or_else(|| {
        PowerError::InferenceFailed(format!(
            "static graph node '{}' could not resolve input '{input}'",
            node.name
        ))
    })?;
    if !move_value {
        liveness::release_consumed_values(&node.inputs, retained_output, remaining_uses, values);
    }
    values.insert(output.clone(), value);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_a_last_use_value_and_preserves_exact_shape() {
        let node = identity_node();
        let mut remaining = HashMap::from([("input".to_string(), 1)]);
        let mut values = HashMap::from([(
            "input".to_string(),
            GraphValue::Ints {
                values: vec![3, 5],
                shape: vec![2],
            },
        )]);

        assert!(try_commit(&node, "graph-output", &mut remaining, &mut values).unwrap());

        assert!(!values.contains_key("input"));
        assert_eq!(remaining["input"], 0);
        assert_eq!(values["output"].shape(), [2]);
    }

    #[test]
    fn retains_a_shared_input_alias() {
        let node = identity_node();
        let mut remaining = HashMap::from([("input".to_string(), 2)]);
        let mut values = HashMap::from([(
            "input".to_string(),
            GraphValue::Ints {
                values: vec![7],
                shape: vec![1],
            },
        )]);

        assert!(try_commit(&node, "graph-output", &mut remaining, &mut values).unwrap());

        assert!(values.contains_key("input"));
        assert!(values.contains_key("output"));
        assert_eq!(remaining["input"], 1);
    }

    #[test]
    fn rejects_an_identity_without_an_output() {
        let mut node = identity_node();
        node.outputs.clear();
        let mut remaining = HashMap::from([("input".to_string(), 1)]);
        let mut values = HashMap::from([(
            "input".to_string(),
            GraphValue::Ints {
                values: vec![11],
                shape: vec![1],
            },
        )]);

        let error = try_commit(&node, "graph-output", &mut remaining, &mut values).unwrap_err();

        assert!(matches!(error, PowerError::InvalidFormat(_)));
        assert!(values.contains_key("input"));
        assert_eq!(remaining["input"], 1);
    }

    fn identity_node() -> GraphNode {
        GraphNode {
            name: "identity".to_string(),
            op: GraphOp::Identity,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attributes: Default::default(),
        }
    }
}
