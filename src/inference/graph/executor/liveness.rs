use std::collections::HashMap;

use super::{GraphPlan, GraphValue};

/// Counts every graph input occurrence so eager intermediate tensors can be
/// released immediately after their last consumer has finished.
pub(super) fn value_use_counts(plan: &GraphPlan) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for input in plan
        .nodes
        .iter()
        .flat_map(|node| node.inputs.iter())
        .filter(|input| !input.is_empty())
    {
        *counts.entry(input.clone()).or_insert(0) += 1;
    }
    counts
}

pub(super) fn release_consumed_values(
    inputs: &[String],
    retained_value: &str,
    remaining_uses: &mut HashMap<String, usize>,
    values: &mut HashMap<String, GraphValue>,
) {
    release_consumed(inputs, retained_value, remaining_uses, values);
}

fn release_consumed<T>(
    inputs: &[String],
    retained_value: &str,
    remaining_uses: &mut HashMap<String, usize>,
    values: &mut HashMap<String, T>,
) {
    for input in inputs.iter().filter(|input| !input.is_empty()) {
        let Some(remaining) = remaining_uses.get_mut(input) else {
            continue;
        };
        if *remaining == 0 {
            continue;
        }
        *remaining -= 1;
        if *remaining == 0 && input != retained_value {
            values.remove(input);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::release_consumed;

    #[test]
    fn releases_a_value_only_after_its_last_input_occurrence() {
        let mut remaining = HashMap::from([("shared".to_string(), 2)]);
        let mut values = HashMap::from([("shared".to_string(), 7_u8)]);

        release_consumed(
            &["shared".to_string()],
            "output",
            &mut remaining,
            &mut values,
        );
        assert_eq!(remaining["shared"], 1);
        assert_eq!(values["shared"], 7);

        release_consumed(
            &["shared".to_string()],
            "output",
            &mut remaining,
            &mut values,
        );
        assert_eq!(remaining["shared"], 0);
        assert!(!values.contains_key("shared"));
    }

    #[test]
    fn retains_the_declared_graph_output_after_its_last_consumer() {
        let mut remaining = HashMap::from([("output".to_string(), 1)]);
        let mut values = HashMap::from([("output".to_string(), 9_u8)]);

        release_consumed(
            &["output".to_string()],
            "output",
            &mut remaining,
            &mut values,
        );

        assert_eq!(remaining["output"], 0);
        assert_eq!(values["output"], 9);
    }

    #[test]
    fn ignores_optional_empty_inputs_and_unknown_values() {
        let mut remaining = HashMap::new();
        let mut values = HashMap::from([("live".to_string(), 1_u8)]);

        release_consumed(
            &[String::new(), "unknown".to_string()],
            "output",
            &mut remaining,
            &mut values,
        );

        assert_eq!(values["live"], 1);
    }
}
