use candle_core::{Device, Tensor};

use super::super::{conv2d, conv2d_with_post_operation};
use crate::inference::graph::executor::convolution_post::ConvolutionPostOperation;

fn bits(tensor: &Tensor) -> Vec<u32> {
    tensor
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
#[ignore = "requires an explicit CUDA device"]
fn fused_layout_bias_activation_preserves_explicit_graph_bits_and_offsets() {
    let device = Device::new_cuda_with_stream(0).unwrap();
    let input = Tensor::from_iter(
        (0..2 * 3 * 7 * 9).map(|value| ((value * 17 % 101) as f32 - 50.0) / 53.0),
        &device,
    )
    .unwrap()
    .reshape((2, 3, 7, 9))
    .unwrap();
    let kernel = Tensor::from_iter(
        (0..4 * 3 * 3 * 3).map(|value| ((value * 13 % 67) as f32 - 33.0) / 31.0),
        &device,
    )
    .unwrap()
    .reshape((4, 3, 3, 3))
    .unwrap();
    let bias_storage = Tensor::new(&[91.0_f32, -0.25, 0.125, 0.375, -0.5, -92.0], &device).unwrap();
    let bias = bias_storage.narrow(0, 1, 4).unwrap();

    for convolution_bias in [None, Some(&bias)] {
        let convolution = conv2d(&input, &kernel, convolution_bias, (1, 2, 0, 1), 2, 1).unwrap();
        let expected_relu = convolution.relu().unwrap();
        let actual_relu = conv2d_with_post_operation(
            &input,
            &kernel,
            convolution_bias,
            (1, 2, 0, 1),
            2,
            1,
            ConvolutionPostOperation::Relu,
        )
        .unwrap();
        actual_relu.device().synchronize().unwrap();
        assert_eq!(actual_relu.dims(), expected_relu.dims());
        assert_eq!(
            bits(&actual_relu),
            bits(&expected_relu),
            "ReLU convolution_bias={}",
            convolution_bias.is_some(),
        );

        let divisor = std::f32::consts::SQRT_2;
        let offset = 1.0_f32;
        let scale = 0.5_f32;
        let divisor_tensor = Tensor::new(&[divisor], &device).unwrap();
        let offset_tensor = Tensor::new(&[offset], &device).unwrap();
        let scale_tensor = Tensor::new(&[scale], &device).unwrap();
        let expected_gelu = convolution
            .broadcast_div(&divisor_tensor)
            .and_then(|value| value.erf())
            .and_then(|value| value.broadcast_add(&offset_tensor))
            .and_then(|value| convolution.broadcast_mul(&value))
            .and_then(|value| value.broadcast_mul(&scale_tensor))
            .unwrap();
        let actual_gelu = conv2d_with_post_operation(
            &input,
            &kernel,
            convolution_bias,
            (1, 2, 0, 1),
            2,
            1,
            ConvolutionPostOperation::GeluErf {
                divisor,
                offset,
                scale,
            },
        )
        .unwrap();
        actual_gelu.device().synchronize().unwrap();
        assert_eq!(actual_gelu.dims(), convolution.dims());
        assert_eq!(
            bits(&actual_gelu),
            bits(&expected_gelu),
            "GELU convolution_bias={}",
            convolution_bias.is_some(),
        );
    }
}
