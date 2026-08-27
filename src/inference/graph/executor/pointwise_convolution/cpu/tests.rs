use candle_core::{Device, Tensor};

use super::super::conv2d_with_residual;
use super::should_parallelize_batches;

#[test]
fn batch_parallelism_uses_only_geometry_and_live_worker_capacity() {
    assert!(!should_parallelize_batches(1, 192, 3_600, 16));
    assert!(!should_parallelize_batches(8, 192, 3_600, 16));
    assert!(should_parallelize_batches(16, 192, 3_600, 16));
    assert!(should_parallelize_batches(2, 192, 32, 16));
    assert!(!should_parallelize_batches(16, 192, 3_600, 1));
}

#[test]
fn residual_fusion_matches_explicit_convolution_bias_and_add_bits() {
    let device = Device::Cpu;
    for (batch, channels, output_channels, height, width) in [
        (1, 3, 5, 2, 7),
        (2, 48, 96, 6, 19),
        (8, 96, 192, 3, 41),
        (16, 192, 384, 1, 17),
    ] {
        let input = Tensor::from_iter(
            (0..batch * channels * height * width)
                .map(|value| ((value * 17 % 251) as f32 - 125.0) / 127.0),
            &device,
        )
        .unwrap()
        .reshape((batch, channels, height, width))
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..output_channels * channels)
                .map(|value| ((value * 29 % 257) as f32 - 128.0) / 131.0),
            &device,
        )
        .unwrap()
        .reshape((output_channels, channels, 1, 1))
        .unwrap();
        let bias = Tensor::from_iter(
            (0..output_channels).map(|value| (value as f32 - 17.0) / 37.0),
            &device,
        )
        .unwrap();
        let residual = Tensor::from_iter(
            (0..batch * output_channels * height * width)
                .map(|value| ((value * 31 % 263) as f32 - 131.0) / 137.0),
            &device,
        )
        .unwrap()
        .reshape((batch, output_channels, height, width))
        .unwrap();

        for bias in [None, Some(&bias)] {
            let convolution = input.conv2d(&kernel, 0, 1, 1, 1).unwrap();
            let biased = match bias {
                Some(bias) => convolution
                    .broadcast_add(&bias.reshape((1, output_channels, 1, 1)).unwrap())
                    .unwrap(),
                None => convolution,
            };
            let expected = (biased + &residual).unwrap();
            let actual = conv2d_with_residual(&input, &kernel, bias, &residual).unwrap();

            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "geometry {batch}x{channels}x{height}x{width} -> {output_channels} bias={}",
                bias.is_some()
            );
        }
    }
}

#[test]
fn residual_fusion_reads_a_contiguous_offset_addend() {
    let device = Device::Cpu;
    let input = Tensor::zeros((2, 3, 2, 5), candle_core::DType::F32, &device).unwrap();
    let kernel = Tensor::zeros((7, 3, 1, 1), candle_core::DType::F32, &device).unwrap();
    let residual = Tensor::from_iter(
        (0..3 * 7 * 2 * 5).map(|value| (value as f32 - 11.0) / 23.0),
        &device,
    )
    .unwrap()
    .reshape((3, 7, 2, 5))
    .unwrap()
    .narrow(0, 1, 2)
    .unwrap();
    assert!(residual.is_contiguous());
    assert_ne!(residual.layout().start_offset(), 0);

    let actual = conv2d_with_residual(&input, &kernel, None, &residual).unwrap();

    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        residual.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );
}
