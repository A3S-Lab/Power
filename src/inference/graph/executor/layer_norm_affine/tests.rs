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

fn tail() -> Vec<GraphNode> {
    vec![
        node(
            "epsilon",
            GraphOp::Add,
            &["variance", "epsilon-constant"],
            "shifted-variance",
        ),
        node("sqrt", GraphOp::Sqrt, &["shifted-variance"], "denominator"),
        node(
            "normalize",
            GraphOp::Div,
            &["centered", "denominator"],
            "normalized",
        ),
        node("scale", GraphOp::Mul, &["gamma", "normalized"], "scaled"),
        node("bias", GraphOp::Add, &["scaled", "beta"], "output"),
    ]
}

fn private_uses() -> HashMap<String, usize> {
    HashMap::from([
        ("shifted-variance".to_string(), 1),
        ("denominator".to_string(), 1),
        ("normalized".to_string(), 1),
        ("scaled".to_string(), 1),
    ])
}

#[test]
fn matches_only_the_exact_private_layer_norm_tail() {
    let nodes = tail();
    let scalars = HashMap::from([("epsilon-constant".to_string(), 0.00001_f32)]);
    let uses = private_uses();
    let matched = matched_inputs(&nodes, &scalars, &uses, "graph-output").unwrap();
    assert_eq!(
        matched,
        MatchedInputs {
            centered: "centered",
            variance: "variance",
            epsilon: "epsilon-constant",
            scale: "gamma",
            bias: "beta",
        }
    );

    let mut shared = uses.clone();
    shared.insert("denominator".to_string(), 2);
    assert!(matched_inputs(&nodes, &scalars, &shared, "graph-output").is_none());
    assert!(matched_inputs(&nodes, &scalars, &uses, "scaled").is_none());

    let mut reversed_divide = nodes.clone();
    reversed_divide[2].inputs.swap(0, 1);
    assert!(matched_inputs(&reversed_divide, &scalars, &uses, "graph-output").is_none());

    let mut attributed = nodes;
    attributed[1]
        .attributes
        .insert("unreviewed".to_string(), serde_json::json!(true));
    assert!(matched_inputs(&attributed, &scalars, &uses, "graph-output").is_none());
}

#[test]
fn accepts_only_exact_last_axis_broadcast_shapes() {
    assert!(last_axis_layer_norm_shapes(
        &[2, 7, 120],
        &[2, 7, 1],
        &[120],
        &[120]
    ));
    assert!(!last_axis_layer_norm_shapes(
        &[2, 7, 120],
        &[2, 1, 1],
        &[120],
        &[120]
    ));
    assert!(!last_axis_layer_norm_shapes(
        &[2, 7, 120],
        &[2, 7, 1],
        &[1, 120],
        &[120]
    ));
    assert!(!last_axis_layer_norm_shapes(&[], &[], &[], &[]));
}
