use std::hint::black_box;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use super::{conv2d, conv2d_with_post_operation};
use crate::inference::graph::executor::convolution_post::ConvolutionPostOperation;

#[test]
fn direct_spatial_convolution_matches_candle_bits() {
    for (batch, input_channels, output_channels, height, width, kernel, stride, pads, dilation) in [
        (1, 3, 5, 9, 13, 2, 1, (0, 0, 0, 0), 1),
        (2, 4, 7, 11, 15, 3, 2, (1, 1, 1, 1), 1),
        (3, 5, 6, 8, 17, 2, 1, (1, 2, 0, 1), 1),
        (2, 3, 4, 12, 19, 3, 1, (2, 1, 1, 2), 2),
    ] {
        let input = Tensor::from_iter(
            (0..batch * input_channels * height * width)
                .map(|value| ((value * 17 % 251) as f32 - 125.0) / 127.0),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((batch, input_channels, height, width))
        .unwrap();
        let weights = Tensor::from_iter(
            (0..output_channels * input_channels * kernel * kernel)
                .map(|value| ((value * 29 % 257) as f32 - 128.0) / 131.0),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((output_channels, input_channels, kernel, kernel))
        .unwrap();
        let bias = Tensor::from_iter(
            (0..output_channels).map(|value| (value as f32 - 3.0) / 37.0),
            &Device::Cpu,
        )
        .unwrap();

        let padded = input
            .pad_with_zeros(2, pads.0, pads.2)
            .unwrap()
            .pad_with_zeros(3, pads.1, pads.3)
            .unwrap();
        let expected = padded
            .conv2d(&weights, 0, stride, dilation, 1)
            .unwrap()
            .broadcast_add(&bias.reshape((1, output_channels, 1, 1)).unwrap())
            .unwrap();
        let actual = conv2d(&input, &weights, Some(&bias), pads, stride, dilation).unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "geometry {batch}x{input_channels}x{height}x{width} -> {output_channels}, kernel={kernel}, stride={stride}, pads={pads:?}, dilation={dilation}",
        );

        let expected_relu = expected.relu().unwrap();
        let actual_relu = conv2d_with_post_operation(
            &input,
            &weights,
            Some(&bias),
            pads,
            stride,
            dilation,
            ConvolutionPostOperation::Relu,
        )
        .unwrap();
        assert_eq!(
            actual_relu.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected_relu.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "ReLU geometry {batch}x{input_channels}x{height}x{width} -> {output_channels}, kernel={kernel}, stride={stride}, pads={pads:?}, dilation={dilation}",
        );

        let divisor = std::f32::consts::SQRT_2;
        let offset = 1.0_f32;
        let scale = 0.5_f32;
        let expected_gelu = expected
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(|value| {
                let activated = candle_core::cpu::erf::erf_f32(value / divisor);
                (value * (activated + offset)) * scale
            })
            .collect::<Vec<_>>();
        let actual_gelu = conv2d_with_post_operation(
            &input,
            &weights,
            Some(&bias),
            pads,
            stride,
            dilation,
            ConvolutionPostOperation::GeluErf {
                divisor,
                offset,
                scale,
            },
        )
        .unwrap();
        assert_eq!(
            actual_gelu.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected_gelu,
            "GELU geometry {batch}x{input_channels}x{height}x{width} -> {output_channels}, kernel={kernel}, stride={stride}, pads={pads:?}, dilation={dilation}",
        );

        let batch_norm_scale = (0..output_channels)
            .map(|channel| (channel as f32 + 3.0) / 11.0)
            .collect::<Vec<_>>();
        let batch_norm_bias = (0..output_channels)
            .map(|channel| (channel as f32 - 5.0) / 17.0)
            .collect::<Vec<_>>();
        let batch_norm_mean = (0..output_channels)
            .map(|channel| (channel as f32 - 2.0) / 13.0)
            .collect::<Vec<_>>();
        let batch_norm_variance = (0..output_channels)
            .map(|channel| (channel as f32 + 7.0) / 19.0)
            .collect::<Vec<_>>();
        let epsilon = 0.000_01_f32;
        let alpha = 1.0_f32 / 6.0;
        let beta = 0.5_f32;
        let output_spatial = actual.dims4().unwrap().2 * actual.dims4().unwrap().3;
        let expected_batch_norm = expected
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .chunks(output_spatial)
            .enumerate()
            .flat_map(|(batch_channel, values)| {
                let channel = batch_channel % output_channels;
                let stddev = (batch_norm_variance[channel] + epsilon).sqrt();
                let mean = batch_norm_mean[channel];
                let scale = batch_norm_scale[channel];
                let bias = batch_norm_bias[channel];
                values.iter().map(move |value| {
                    let normalized = (((*value - mean) / stddev) * scale) + bias;
                    normalized * ((normalized * alpha) + beta).clamp(0.0, 1.0)
                })
            })
            .collect::<Vec<_>>();
        let post_operation = ConvolutionPostOperation::batch_normalization(
            &batch_norm_scale,
            &batch_norm_bias,
            &batch_norm_mean,
            &batch_norm_variance,
            epsilon,
            Some((alpha, beta)),
        )
        .unwrap();
        let actual_batch_norm = conv2d_with_post_operation(
            &input,
            &weights,
            Some(&bias),
            pads,
            stride,
            dilation,
            post_operation,
        )
        .unwrap();
        assert_eq!(
            actual_batch_norm
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected_batch_norm,
            "BatchNormalization geometry {batch}x{input_channels}x{height}x{width} -> {output_channels}, kernel={kernel}, stride={stride}, pads={pads:?}, dilation={dilation}",
        );
    }
}

#[test]
fn direct_spatial_convolution_matches_a_geometry_grid_without_content_policy() {
    let geometries = [
        (1, 1, 1, 6, 9, 2, 3, 1, (0, 0, 0, 0), 1),
        (2, 5, 1, 9, 19, 3, 5, 1, (1, 2, 1, 2), 1),
        (2, 3, 1, 12, 23, 2, 3, 2, (1, 2, 0, 1), 2),
        (2, 1, 4, 7, 10, 3, 2, 2, (1, 0, 2, 1), 1),
        (1, 3, 2, 8, 11, 2, 3, 1, (0, 2, 1, 0), 2),
        (2, 3, 5, 9, 12, 3, 2, 2, (2, 1, 0, 2), 2),
    ];
    for (
        batch,
        input_channels,
        output_channels,
        height,
        width,
        kernel_height,
        kernel_width,
        stride,
        pads,
        dilation,
    ) in geometries
    {
        let input = Tensor::from_iter(
            (0..batch * input_channels * height * width)
                .map(|value| ((value * 31 % 263) as f32 - 131.0) / 137.0),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((batch, input_channels, height, width))
        .unwrap();
        let weights = Tensor::from_iter(
            (0..output_channels * input_channels * kernel_height * kernel_width)
                .map(|value| ((value * 43 % 269) as f32 - 134.0) / 139.0),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((output_channels, input_channels, kernel_height, kernel_width))
        .unwrap();
        let bias = Tensor::from_iter(
            (0..output_channels).map(|value| (value as f32 - 2.0) / 41.0),
            &Device::Cpu,
        )
        .unwrap();
        let padded = input
            .pad_with_zeros(2, pads.0, pads.2)
            .unwrap()
            .pad_with_zeros(3, pads.1, pads.3)
            .unwrap();

        for bias in [None, Some(&bias)] {
            let expected = padded.conv2d(&weights, 0, stride, dilation, 1).unwrap();
            let expected = match bias {
                Some(bias) => expected
                    .broadcast_add(&bias.reshape((1, output_channels, 1, 1)).unwrap())
                    .unwrap(),
                None => expected,
            };
            let actual = conv2d(&input, &weights, bias, pads, stride, dilation).unwrap();

            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "geometry {batch}x{input_channels}x{height}x{width} -> {output_channels}, kernel={kernel_height}x{kernel_width}, stride={stride}, pads={pads:?}, dilation={dilation}, bias={}",
                bias.is_some(),
            );
        }
    }
}

#[test]
fn direct_spatial_convolution_reads_contiguous_offset_layouts() {
    let input = Tensor::from_iter(
        (0..4 * 3 * 11 * 13).map(|value| ((value * 17 % 251) as f32 - 125.0) / 61.0),
        &Device::Cpu,
    )
    .unwrap()
    .reshape((4, 3, 11, 13))
    .unwrap()
    .narrow(0, 1, 2)
    .unwrap();
    let weights = Tensor::from_iter(
        (0..7 * 3 * 3 * 3).map(|value| ((value * 29 % 257) as f32 - 128.0) / 67.0),
        &Device::Cpu,
    )
    .unwrap()
    .reshape((7, 3, 3, 3))
    .unwrap()
    .narrow(0, 1, 6)
    .unwrap();
    let bias = Tensor::from_iter(
        (0..7).map(|value| (value as f32 - 4.0) / 19.0),
        &Device::Cpu,
    )
    .unwrap()
    .narrow(0, 1, 6)
    .unwrap();
    assert!(input.is_contiguous());
    assert!(weights.is_contiguous());
    assert!(bias.is_contiguous());
    assert_ne!(input.layout().start_offset(), 0);
    assert_ne!(weights.layout().start_offset(), 0);
    assert_ne!(bias.layout().start_offset(), 0);

    let expected = input
        .conv2d(&weights, 1, 2, 1, 1)
        .unwrap()
        .broadcast_add(&bias.reshape((1, 6, 1, 1)).unwrap())
        .unwrap();
    let actual = conv2d(&input, &weights, Some(&bias), (1, 1, 1, 1), 2, 1).unwrap();

    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
}

#[test]
fn direct_spatial_convolution_rejects_invalid_geometry() {
    let input = Tensor::zeros((1, 3, 4, 4), DType::F32, &Device::Cpu).unwrap();
    let weights = Tensor::zeros((5, 3, 7, 7), DType::F32, &Device::Cpu).unwrap();

    assert!(conv2d(&input, &weights, None, (0, 0, 0, 0), 1, 1).is_err());
}

#[test]
#[ignore = "diagnostic CPU spatial convolution comparison"]
fn compare_direct_spatial_convolution() {
    for (batch, input_channels, output_channels, height, width, kernel, stride, padding) in [
        (1, 3, 48, 48, 320, 3, 2, 1),
        (8, 3, 48, 48, 320, 3, 2, 1),
        (19, 3, 48, 48, 320, 3, 2, 1),
        (1, 24, 48, 12, 160, 2, 1, 0),
        (8, 24, 48, 12, 160, 2, 1, 0),
        (19, 24, 48, 12, 80, 2, 1, 0),
        (1, 48, 24, 13, 161, 2, 1, 0),
        (8, 48, 24, 13, 161, 2, 1, 0),
        (19, 48, 24, 13, 81, 2, 1, 0),
        (1, 96, 48, 12, 160, 3, 2, 1),
        (8, 96, 48, 12, 160, 3, 2, 1),
        (19, 96, 48, 12, 80, 3, 2, 1),
    ] {
        let input = Tensor::zeros(
            (batch, input_channels, height, width),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let weights = Tensor::zeros(
            (output_channels, input_channels, kernel, kernel),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        black_box(input.conv2d(&weights, padding, stride, 1, 1).unwrap());
        black_box(
            conv2d(
                &input,
                &weights,
                None,
                (padding, padding, padding, padding),
                stride,
                1,
            )
            .unwrap(),
        );
        let iterations = 5_u32;
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(input.conv2d(&weights, padding, stride, 1, 1).unwrap());
        }
        let candle = started.elapsed();
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(
                conv2d(
                    &input,
                    &weights,
                    None,
                    (padding, padding, padding, padding),
                    stride,
                    1,
                )
                .unwrap(),
            );
        }
        let direct = started.elapsed();
        eprintln!(
            "batch={batch} channels={input_channels} outputs={output_channels} input={height}x{width} kernel={kernel} stride={stride} candle_ms={:.3} direct_ms={:.3}",
            candle.as_secs_f64() * 1_000.0 / f64::from(iterations),
            direct.as_secs_f64() * 1_000.0 / f64::from(iterations),
        );
    }
}
