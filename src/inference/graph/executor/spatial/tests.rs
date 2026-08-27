use std::collections::BTreeMap;

use candle_core::{Device, Tensor};

use super::{conv, symmetric_padding, try_conv_with_post_operation, GraphNode, GraphValue};
use crate::inference::graph::executor::convolution_post::ConvolutionPostOperation;
use crate::inference::graph::plan::GraphOp;

#[test]
fn native_symmetric_padding_matches_materialized_zero_padding_bits() {
    let device = Device::Cpu;
    let input = Tensor::from_iter(
        (0..2 * 3 * 7 * 11).map(|value| ((value * 17 % 251) as f32 - 125.0) / 61.0),
        &device,
    )
    .unwrap()
    .reshape((2, 3, 7, 11))
    .unwrap();
    let kernel = Tensor::from_iter(
        (0..5 * 3 * 3 * 3).map(|value| ((value * 29 % 257) as f32 - 128.0) / 67.0),
        &device,
    )
    .unwrap()
    .reshape((5, 3, 3, 3))
    .unwrap();
    let materialized = input
        .pad_with_zeros(2, 1, 1)
        .unwrap()
        .pad_with_zeros(3, 1, 1)
        .unwrap()
        .conv2d(&kernel, 0, 1, 1, 1)
        .unwrap();
    let native = input.conv2d(&kernel, 1, 1, 1, 1).unwrap();

    assert_eq!(native.dims(), materialized.dims());
    assert_eq!(
        native.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        materialized
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    );
    assert_eq!(symmetric_padding((1, 1, 1, 1)), Some(1));
    assert_eq!(symmetric_padding((0, 0, 0, 1)), None);
}

#[test]
fn graph_spatial_convolution_applies_declared_padding_once() {
    let device = Device::Cpu;
    let input = Tensor::from_iter(
        (0..2 * 3 * 7 * 11).map(|value| ((value * 17 % 251) as f32 - 125.0) / 61.0),
        &device,
    )
    .unwrap()
    .reshape((2, 3, 7, 11))
    .unwrap();
    let kernel = Tensor::from_iter(
        (0..5 * 3 * 3 * 3).map(|value| ((value * 29 % 257) as f32 - 128.0) / 67.0),
        &device,
    )
    .unwrap()
    .reshape((5, 3, 3, 3))
    .unwrap();
    let expected = input.conv2d(&kernel, 1, 1, 1, 1).unwrap();
    let node = GraphNode {
        name: "spatial-convolution".to_string(),
        op: GraphOp::Conv,
        inputs: vec!["input".to_string(), "kernel".to_string()],
        outputs: vec!["output".to_string()],
        attributes: BTreeMap::from([
            ("kernel_shape".to_string(), serde_json::json!([3, 3])),
            ("pads".to_string(), serde_json::json!([1, 1, 1, 1])),
        ]),
    };
    let input = GraphValue::Tensor(input);
    let kernel = GraphValue::Tensor(kernel);

    let actual = conv(&node, &[&input, &kernel], &device)
        .unwrap()
        .tensor(&node.name)
        .unwrap()
        .clone();

    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
}

#[test]
fn graph_pointwise_convolution_applies_declared_padding_once() {
    let device = Device::Cpu;
    let input = Tensor::from_iter(
        (0..2 * 3 * 7 * 11).map(|value| ((value * 17 % 251) as f32 - 125.0) / 61.0),
        &device,
    )
    .unwrap()
    .reshape((2, 3, 7, 11))
    .unwrap();
    let kernel = Tensor::from_iter(
        (0..5 * 3).map(|value| ((value * 29 % 257) as f32 - 128.0) / 67.0),
        &device,
    )
    .unwrap()
    .reshape((5, 3, 1, 1))
    .unwrap();
    let bias = Tensor::from_iter((0..5).map(|value| (value as f32 - 4.0) / 19.0), &device).unwrap();
    let expected = input
        .pad_with_zeros(2, 1, 1)
        .unwrap()
        .pad_with_zeros(3, 1, 1)
        .unwrap()
        .conv2d(&kernel, 0, 1, 1, 1)
        .unwrap()
        .broadcast_add(&bias.reshape((1, 5, 1, 1)).unwrap())
        .unwrap();
    let node = GraphNode {
        name: "pointwise-convolution".to_string(),
        op: GraphOp::Conv,
        inputs: vec![
            "input".to_string(),
            "kernel".to_string(),
            "bias".to_string(),
        ],
        outputs: vec!["output".to_string()],
        attributes: BTreeMap::from([
            ("kernel_shape".to_string(), serde_json::json!([1, 1])),
            ("pads".to_string(), serde_json::json!([1, 1, 1, 1])),
        ]),
    };
    let input = GraphValue::Tensor(input);
    let kernel = GraphValue::Tensor(kernel);
    let bias = GraphValue::Tensor(bias);

    let actual = conv(&node, &[&input, &kernel, &bias], &device)
        .unwrap()
        .tensor(&node.name)
        .unwrap()
        .clone();

    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
}

#[test]
fn graph_pointwise_post_operation_preserves_padding_and_bias_bits() {
    let device = Device::Cpu;
    let input = Tensor::from_iter(
        (0..3 * 7 * 11).map(|value| ((value * 17 % 251) as f32 - 125.0) / 61.0),
        &device,
    )
    .unwrap()
    .reshape((1, 3, 7, 11))
    .unwrap();
    let kernel = Tensor::from_iter(
        (0..5 * 3).map(|value| ((value * 29 % 257) as f32 - 128.0) / 67.0),
        &device,
    )
    .unwrap()
    .reshape((5, 3, 1, 1))
    .unwrap();
    let bias = Tensor::from_iter((0..5).map(|value| (value as f32 - 4.0) / 19.0), &device)
        .unwrap()
        .reshape((1, 5, 1, 1))
        .unwrap();
    let expected = input
        .pad_with_zeros(2, 1, 1)
        .unwrap()
        .pad_with_zeros(3, 1, 1)
        .unwrap()
        .conv2d(&kernel, 0, 1, 1, 1)
        .unwrap()
        .broadcast_add(&bias)
        .unwrap()
        .relu()
        .unwrap();
    let node = GraphNode {
        name: "pointwise-convolution-with-post-operation".to_string(),
        op: GraphOp::Conv,
        inputs: vec!["input".to_string(), "kernel".to_string()],
        outputs: vec!["output".to_string()],
        attributes: BTreeMap::from([
            ("kernel_shape".to_string(), serde_json::json!([1, 1])),
            ("pads".to_string(), serde_json::json!([1, 1, 1, 1])),
        ]),
    };
    let input = GraphValue::Tensor(input);
    let kernel = GraphValue::Tensor(kernel);
    let bias = GraphValue::Tensor(bias);

    let actual = try_conv_with_post_operation(
        &node,
        &[&input, &kernel, &bias],
        &device,
        ConvolutionPostOperation::Relu,
    )
    .unwrap()
    .unwrap()
    .tensor(&node.name)
    .unwrap()
    .clone();

    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
}

#[cfg(feature = "embedded-cuda")]
#[test]
#[ignore = "requires an explicit CUDA device"]
fn graph_cuda_pointwise_batch_norm_keeps_the_two_output_path() {
    use crate::inference::graph::executor::convolution_post::CudaBatchNormActivation;

    let device = Device::new_cuda_with_stream(0).unwrap();
    let input = Tensor::from_iter(
        (0..2 * 3 * 5 * 7).map(|value| ((value * 17 % 251) as f32 - 125.0) / 127.0),
        &device,
    )
    .unwrap()
    .reshape((2, 3, 5, 7))
    .unwrap();
    let kernel = Tensor::from_iter(
        (0..5 * 3).map(|value| ((value * 29 % 257) as f32 - 128.0) / 131.0),
        &device,
    )
    .unwrap()
    .reshape((5, 3, 1, 1))
    .unwrap();
    let scale_and_bias = Tensor::new(
        &[
            [0.75_f32, -1.25, 2.0, 0.5, 1.5],
            [-0.5_f32, 0.125, 1.5, -0.25, 0.75],
        ],
        &device,
    )
    .unwrap();
    let mean_and_stddev = Tensor::new(
        &[
            [0.25_f32, -0.75, 1.25, 0.0, 0.5],
            [0.75_f32, 1.25, 1.75, 2.25, 2.75],
        ],
        &device,
    )
    .unwrap();
    let post_operation = ConvolutionPostOperation::cuda_batch_normalization(
        &scale_and_bias,
        &mean_and_stddev,
        CudaBatchNormActivation::Identity,
    )
    .unwrap();
    let node = GraphNode {
        name: "cuda-pointwise-convolution".to_string(),
        op: GraphOp::Conv,
        inputs: vec!["input".to_string(), "kernel".to_string()],
        outputs: vec!["output".to_string()],
        attributes: BTreeMap::from([("kernel_shape".to_string(), serde_json::json!([1, 1]))]),
    };
    let input = GraphValue::Tensor(input);
    let kernel = GraphValue::Tensor(kernel);

    let output =
        try_conv_with_post_operation(&node, &[&input, &kernel], &device, post_operation).unwrap();

    assert!(
        output.is_none(),
        "CUDA pointwise BatchNormalization must remain on the ordinary two-output path"
    );
}
