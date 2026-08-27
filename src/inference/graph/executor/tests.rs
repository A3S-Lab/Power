use super::*;

fn node() -> GraphNode {
    GraphNode {
        name: "test".to_string(),
        op: GraphOp::Reshape,
        inputs: Vec::new(),
        outputs: vec!["out".to_string()],
        attributes: Default::default(),
    }
}

#[test]
fn reshape_resolves_zero_and_inferred_dimensions() {
    assert_eq!(
        resolve_reshape(&[2, 3, 4], &[0, -1], &node()).unwrap(),
        [2, 12]
    );
    assert!(resolve_reshape(&[2, 3], &[-1, -1], &node()).is_err());
}

#[test]
fn same_upper_padding_puts_odd_pixel_at_end() {
    assert_eq!(same_upper_padding(48, 2, 1, 1), (0, 1));
    assert_eq!(same_upper_padding(48, 3, 2, 1), (0, 1));
}

#[test]
fn fused_window_rejects_empty_and_out_of_bounds_ranges() {
    let nodes = vec![node()];

    assert!(fused_node_window(&nodes, 0, 0, "empty").is_err());
    assert!(fused_node_window(&nodes, 0, 2, "overflow").is_err());

    let (consumed, terminal) = fused_node_window(&nodes, 0, 1, "single").unwrap();
    assert!(consumed.is_empty());
    assert_eq!(terminal.name, "test");
}

#[test]
fn matmul_materializes_a_transposed_rhs_view() {
    let mut node = node();
    node.op = GraphOp::MatMul;
    let left = GraphValue::Tensor(
        Tensor::zeros((3, 8, 41, 15), candle_core::DType::F32, &Device::Cpu).unwrap(),
    );
    let right = Tensor::zeros((3, 8, 41, 15), candle_core::DType::F32, &Device::Cpu)
        .unwrap()
        .transpose(2, 3)
        .unwrap();
    assert!(!right.is_contiguous());
    let right = GraphValue::Tensor(right);

    let output = matmul(&node, &[&left, &right]).unwrap();

    assert_eq!(output.shape(), [3, 8, 41, 41]);
}

#[test]
fn identity_transpose_validates_rank_and_returns_the_input_value() {
    let mut node = node();
    node.op = GraphOp::Transpose;
    node.attributes
        .insert("perm".to_string(), serde_json::json!([0, 1, 2]));
    let input =
        GraphValue::Tensor(Tensor::new(&[[[1.0_f32, 2.0], [3.0, 4.0]]], &Device::Cpu).unwrap());

    let output = transpose(&node, &[&input]).unwrap();

    assert_eq!(output.shape(), [1, 2, 2]);
    assert_eq!(
        output
            .tensor(&node.name)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );

    let wrong_rank = GraphValue::Tensor(
        Tensor::zeros((1, 2, 2, 1), candle_core::DType::F32, &Device::Cpu).unwrap(),
    );
    assert!(transpose(&node, &[&wrong_rank]).is_err());
}

#[test]
fn pow_uses_the_host_cached_scalar_initializer() {
    let mut node = node();
    node.op = GraphOp::Pow;
    node.inputs = vec!["base".to_string(), "exponent".to_string()];
    let base = GraphValue::Tensor(Tensor::new(&[2.0_f32, 3.0], &Device::Cpu).unwrap());
    let device_only_exponent = GraphValue::Ints {
        values: vec![2],
        shape: vec![1],
    };
    let scalar_constants = HashMap::from([("exponent".to_string(), 2.0)]);

    let output = pow(&node, &[&base, &device_only_exponent], &scalar_constants).unwrap();

    assert_eq!(
        output.tensor(&node.name).unwrap().to_vec1::<f32>().unwrap(),
        [4.0, 9.0]
    );
}

#[test]
fn pow_rejects_a_cached_non_square_exponent() {
    let mut node = node();
    node.op = GraphOp::Pow;
    node.inputs = vec!["base".to_string(), "exponent".to_string()];
    let base = GraphValue::Tensor(Tensor::new(&[2.0_f32], &Device::Cpu).unwrap());
    let exponent = GraphValue::Tensor(Tensor::new(&[3.0_f32], &Device::Cpu).unwrap());
    let scalar_constants = HashMap::from([("exponent".to_string(), 3.0)]);

    let error = match pow(&node, &[&base, &exponent], &scalar_constants) {
        Ok(_) => panic!("a non-square exponent must be rejected"),
        Err(error) => error,
    };

    assert!(matches!(error, PowerError::InferenceFailed(_)));
}

#[test]
fn batch_normalization_broadcasts_parameters_across_the_input_rank() {
    let mut node = node();
    node.op = GraphOp::BatchNormalization;
    node.attributes
        .insert("epsilon".to_string(), serde_json::json!(0.0));
    let input =
        GraphValue::Tensor(Tensor::new(&[[[1.0_f32, 3.0], [2.0, 5.0]]], &Device::Cpu).unwrap());
    let scale = GraphValue::Tensor(Tensor::new(&[2.0_f32, 3.0], &Device::Cpu).unwrap());
    let bias = GraphValue::Tensor(Tensor::zeros(2, candle_core::DType::F32, &Device::Cpu).unwrap());
    let mean = GraphValue::Tensor(Tensor::new(&[1.0_f32, 2.0], &Device::Cpu).unwrap());
    let variance = GraphValue::Tensor(Tensor::new(&[4.0_f32, 9.0], &Device::Cpu).unwrap());

    let output = batch_norm_fallback(&node, &[&input, &scale, &bias, &mean, &variance]).unwrap();

    assert_eq!(output.shape(), [1, 2, 2]);
    assert_eq!(
        output
            .tensor(&node.name)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        [0.0, 2.0, 0.0, 3.0]
    );
}
