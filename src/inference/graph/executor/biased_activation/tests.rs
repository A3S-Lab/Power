use std::collections::{BTreeMap, HashMap};

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

fn private_uses(values: &[&str]) -> HashMap<String, usize> {
    values
        .iter()
        .map(|value| ((*value).to_string(), 1))
        .collect()
}

#[test]
fn matches_adjacent_private_relu_and_gated_windows() {
    let relu = vec![
        node("conv", GraphOp::Conv, &["input", "weights"], "convolved"),
        node(
            "bias",
            GraphOp::Add,
            &["channel-bias", "convolved"],
            "biased",
        ),
        node(
            "identity",
            GraphOp::Identity,
            &["biased"],
            "activation-input",
        ),
        node("relu", GraphOp::Relu, &["activation-input"], "output"),
    ];
    let uses = private_uses(&["convolved", "biased", "activation-input"]);
    let matched = matched_window(&relu, &uses, "output").unwrap();
    assert_eq!(matched.bias, "channel-bias");
    assert_eq!(matched.consumed_nodes, 4);
    assert!(matches!(matched.activation, MatchedActivation::Relu));

    let mut gated = relu[..3].to_vec();
    gated.push(node(
        "gate",
        GraphOp::HardSigmoid,
        &["activation-input"],
        "bounded-gate",
    ));
    gated.push(node(
        "multiply",
        GraphOp::Mul,
        &["features", "bounded-gate"],
        "output",
    ));
    let uses = private_uses(&["convolved", "biased", "activation-input", "bounded-gate"]);
    let matched = matched_window(&gated, &uses, "output").unwrap();
    assert_eq!(matched.consumed_nodes, 5);
    assert!(matches!(
        matched.activation,
        MatchedActivation::GatedHardSigmoid {
            multiplicand: "features",
            ..
        }
    ));
}

#[test]
fn matches_the_exact_error_function_activation_after_two_identities() {
    let nodes = vec![
        node("conv", GraphOp::Conv, &["input", "weights"], "convolved"),
        node(
            "bias",
            GraphOp::Add,
            &["convolved", "channel-bias"],
            "biased",
        ),
        node(
            "identity-1",
            GraphOp::Identity,
            &["biased"],
            "identity-1-out",
        ),
        node(
            "identity-2",
            GraphOp::Identity,
            &["identity-1-out"],
            "activation-input",
        ),
        node(
            "divide",
            GraphOp::Div,
            &["activation-input", "divisor"],
            "divided",
        ),
        node("erf", GraphOp::Erf, &["divided"], "activated"),
        node("offset", GraphOp::Add, &["activated", "offset"], "shifted"),
        node(
            "multiply-input",
            GraphOp::Mul,
            &["activation-input", "shifted"],
            "product",
        ),
        node("scale", GraphOp::Mul, &["product", "scale"], "output"),
    ];
    let mut uses = private_uses(&[
        "convolved",
        "biased",
        "identity-1-out",
        "activation-input",
        "divided",
        "activated",
        "shifted",
        "product",
    ]);
    uses.insert("activation-input".to_string(), 2);

    let matched = matched_window(&nodes, &uses, "output").unwrap();

    assert_eq!(matched.consumed_nodes, 9);
    assert!(matches!(
        matched.activation,
        MatchedActivation::Gelu {
            divisor: "divisor",
            offset: "offset",
            scale: "scale"
        }
    ));
}

#[test]
fn matches_an_exact_private_residual_add_after_channel_bias() {
    let nodes = vec![
        node("conv", GraphOp::Conv, &["input", "weights"], "convolved"),
        node(
            "bias",
            GraphOp::Add,
            &["convolved", "channel-bias"],
            "biased",
        ),
        node("identity", GraphOp::Identity, &["biased"], "branch"),
        node("residual", GraphOp::Add, &["skip", "branch"], "output"),
    ];
    let uses = private_uses(&["convolved", "biased", "branch"]);

    let matched = matched_window(&nodes, &uses, "graph-output").unwrap();

    assert_eq!(matched.consumed_nodes, 4);
    assert!(matches!(
        matched.activation,
        MatchedActivation::Residual {
            residual: "skip",
            ..
        }
    ));
}

#[test]
fn rejects_shared_intermediates_and_unreviewed_identity_depth() {
    let mut nodes = vec![
        node("conv", GraphOp::Conv, &["input", "weights"], "convolved"),
        node(
            "bias",
            GraphOp::Add,
            &["convolved", "channel-bias"],
            "biased",
        ),
        node(
            "identity-1",
            GraphOp::Identity,
            &["biased"],
            "identity-1-out",
        ),
        node(
            "identity-2",
            GraphOp::Identity,
            &["identity-1-out"],
            "identity-2-out",
        ),
        node(
            "identity-3",
            GraphOp::Identity,
            &["identity-2-out"],
            "activation-input",
        ),
        node("relu", GraphOp::Relu, &["activation-input"], "output"),
    ];
    let uses = private_uses(&[
        "convolved",
        "biased",
        "identity-1-out",
        "identity-2-out",
        "activation-input",
    ]);
    assert!(matched_window(&nodes, &uses, "output").is_none());

    nodes.remove(4);
    nodes[4].inputs[0] = "identity-2-out".to_string();
    let mut shared = uses;
    shared.insert("convolved".to_string(), 2);
    assert!(matched_window(&nodes, &shared, "output").is_none());
}

#[test]
#[cfg(feature = "embedded-cuda")]
fn accepts_only_same_shape_or_nchw_channel_gates() {
    assert!(gated_nchw_shapes(&[1, 2, 3, 4], &[1, 2, 3, 4]));
    assert!(gated_nchw_shapes(&[1, 2, 1, 1], &[1, 2, 3, 4]));
    assert!(!gated_nchw_shapes(&[1, 2, 1, 1], &[1, 3, 3, 4]));
    assert!(!gated_nchw_shapes(&[1, 2, 1], &[1, 2, 3, 4]));
}

#[test]
fn matches_private_convolution_bias_without_an_activation() {
    let mut convolution = node(
        "depthwise",
        GraphOp::Conv,
        &["input", "weights"],
        "convolved",
    );
    convolution
        .attributes
        .insert("group".to_string(), serde_json::json!(4));
    let nodes = vec![
        convolution.clone(),
        node(
            "bias",
            GraphOp::Add,
            &["convolved", "channel-bias"],
            "biased",
        ),
        node("identity", GraphOp::Identity, &["biased"], "features"),
        node(
            "next",
            GraphOp::Conv,
            &["features", "next-weights"],
            "output",
        ),
    ];
    let uses = private_uses(&["convolved", "biased", "features"]);

    let matched = matched_window(&nodes, &uses, "output").unwrap();

    assert_eq!(matched.consumed_nodes, 3);
    assert!(matches!(matched.activation, MatchedActivation::Bias));

    convolution.attributes.clear();
    let mut ungrouped = nodes;
    ungrouped[0] = convolution;
    let matched = matched_window(&ungrouped, &uses, "output").unwrap();
    assert_eq!(matched.consumed_nodes, 3);
    assert!(matches!(matched.activation, MatchedActivation::Bias));
}

#[test]
fn channel_bias_shape_accepts_only_exact_channel_broadcasts() {
    let device = Device::Cpu;
    let vector = GraphValue::Tensor(Tensor::zeros(7, DType::F32, &device).unwrap());
    let nchw = GraphValue::Tensor(Tensor::zeros((1, 7, 1, 1), DType::F32, &device).unwrap());
    let residual = GraphValue::Tensor(Tensor::zeros((2, 7, 3, 5), DType::F32, &device).unwrap());
    let wrong_channels =
        GraphValue::Tensor(Tensor::zeros((1, 8, 1, 1), DType::F32, &device).unwrap());

    assert!(channel_bias_shape(&vector, 7));
    assert!(channel_bias_shape(&nchw, 7));
    assert!(!channel_bias_shape(&residual, 7));
    assert!(!channel_bias_shape(&wrong_channels, 7));
}
