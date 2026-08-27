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

fn mean_node(name: &str, input: &str, output: &str) -> GraphNode {
    let mut node = node(name, GraphOp::ReduceMean, &[input], output);
    node.attributes
        .insert("axes".to_string(), serde_json::json!([-1]));
    node
}

fn full_layer_norm() -> Vec<GraphNode> {
    vec![
        mean_node("mean", "input", "mean-output"),
        node(
            "center",
            GraphOp::Sub,
            &["input", "mean-output"],
            "centered",
        ),
        node(
            "square",
            GraphOp::Pow,
            &["centered", "square-exponent"],
            "squared",
        ),
        mean_node("variance", "squared", "variance-output"),
        node(
            "epsilon",
            GraphOp::Add,
            &["variance-output", "epsilon-constant"],
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

fn full_private_uses() -> HashMap<String, usize> {
    HashMap::from([
        ("mean-output".to_string(), 1),
        ("centered".to_string(), 2),
        ("squared".to_string(), 1),
        ("variance-output".to_string(), 1),
        ("shifted-variance".to_string(), 1),
        ("denominator".to_string(), 1),
        ("normalized".to_string(), 1),
        ("scaled".to_string(), 1),
    ])
}

#[test]
fn matches_only_the_exact_private_full_layer_norm() {
    let nodes = full_layer_norm();
    let scalars = HashMap::from([
        ("square-exponent".to_string(), 2.0_f32),
        ("epsilon-constant".to_string(), 0.00001_f32),
    ]);
    let uses = full_private_uses();
    assert_eq!(
        matched_full_inputs(&nodes, &scalars, &uses, "graph-output"),
        Some(MatchedFullInputs {
            input: "input",
            mean_axis: -1,
            variance_axis: -1,
            exponent: "square-exponent",
            epsilon: "epsilon-constant",
            scale: "gamma",
            bias: "beta",
        })
    );

    let mut shared = uses.clone();
    shared.insert("centered".to_string(), 3);
    assert!(matched_full_inputs(&nodes, &scalars, &shared, "graph-output").is_none());

    let mut non_square = scalars.clone();
    non_square.insert("square-exponent".to_string(), 3.0);
    assert!(matched_full_inputs(&nodes, &non_square, &uses, "graph-output").is_none());

    let mut wrong_axis = nodes.clone();
    wrong_axis[3]
        .attributes
        .insert("axes".to_string(), serde_json::json!([-2]));
    let matched = matched_full_inputs(&wrong_axis, &scalars, &uses, "graph-output").unwrap();
    assert!(!is_last_axis(matched.variance_axis, 3));

    let mut reversed_divide = nodes;
    reversed_divide[6].inputs.swap(0, 1);
    assert!(matched_full_inputs(&reversed_divide, &scalars, &uses, "graph-output").is_none());
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
