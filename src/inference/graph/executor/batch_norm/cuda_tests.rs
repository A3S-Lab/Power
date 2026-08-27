use super::super::convolution_post::CudaBatchNormActivation;
use super::*;

#[test]
#[ignore = "requires an explicit CUDA device"]
fn cuda_depthwise_batch_norm_fusion_matches_the_two_kernel_path_bit_for_bit() {
    let device = Device::new_cuda_with_stream(0).unwrap();
    let kernel = Tensor::from_iter(
        (0..3 * 3 * 3).map(|value| ((value * 13 % 29) as f32 - 14.0) / 31.0),
        &device,
    )
    .unwrap()
    .reshape((3, 1, 3, 3))
    .unwrap();
    let convolution_bias = Tensor::new(&[-0.25_f32, 0.125, 0.375], &device).unwrap();
    let scale = Tensor::new(&[0.75_f32, -1.25, 2.0], &device).unwrap();
    let bias = Tensor::new(&[-0.5_f32, 0.125, 1.5], &device).unwrap();
    let mean = Tensor::new(&[0.25_f32, -0.75, 1.25], &device).unwrap();
    let variance = Tensor::new(&[0.5_f32, 1.5, 2.5], &device).unwrap();
    let scale_and_bias = Tensor::stack(&[&scale, &bias], 0).unwrap();
    let mean_and_variance = Tensor::stack(&[&mean, &variance], 0).unwrap();
    let epsilon = 0.000_01_f32;
    let mean_and_stddev = prepare_cuda_statistics(&mean_and_variance, epsilon).unwrap();
    let activations = [
        (Activation::Identity, CudaBatchNormActivation::Identity),
        (Activation::Relu, CudaBatchNormActivation::Relu),
        (
            Activation::HardSwish {
                alpha: 1.0 / 6.0,
                beta: 0.5,
            },
            CudaBatchNormActivation::HardSwish {
                alpha: 1.0 / 6.0,
                beta: 0.5,
            },
        ),
        (Activation::Swish, CudaBatchNormActivation::Swish),
        (
            Activation::GeluErf {
                divisor: std::f32::consts::SQRT_2,
                offset: 1.0,
                scale: 0.5,
            },
            CudaBatchNormActivation::GeluErf {
                divisor: std::f32::consts::SQRT_2,
                offset: 1.0,
                scale: 0.5,
            },
        ),
    ];

    for (height, width) in [(5, 7), (16, 32)] {
        let input = Tensor::from_iter(
            (0..2 * 3 * height * width).map(|value| ((value * 17 % 101) as f32 - 50.0) / 53.0),
            &device,
        )
        .unwrap()
        .reshape((2, 3, height, width))
        .unwrap();
        for convolution_bias in [None, Some(&convolution_bias)] {
            for (activation, cuda_activation) in activations {
                let convolution = super::super::depthwise::conv2d(
                    &input,
                    &kernel,
                    convolution_bias,
                    (1, 1, 1, 1),
                    (1, 1),
                    1,
                )
                .unwrap();
                let expected = super::cuda::execute(
                    &convolution,
                    &scale_and_bias,
                    &mean_and_stddev,
                    activation,
                )
                .unwrap();
                let post_operation = ConvolutionPostOperation::cuda_batch_normalization(
                    &scale_and_bias,
                    &mean_and_stddev,
                    cuda_activation,
                )
                .unwrap();
                let actual = super::super::depthwise::try_conv2d_with_post_operation(
                    &input,
                    &kernel,
                    convolution_bias,
                    (1, 1, 1, 1),
                    (1, 1),
                    1,
                    post_operation,
                )
                .unwrap()
                .unwrap();
                actual.device().synchronize().unwrap();

                let expected = expected
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let actual = actual
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "geometry={height}x{width} activation={activation:?} convolution_bias={}",
                    convolution_bias.is_some(),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an explicit CUDA device"]
fn cuda_pointwise_batch_norm_in_place_matches_the_two_kernel_path_bit_for_bit() {
    use crate::inference::graph::executor::cuda_reproducibility::REPRODUCIBLE_BATCH_ITEMS;

    let device = Device::new_cuda_with_stream(0).unwrap();
    let activations = [
        (Activation::Identity, CudaBatchNormActivation::Identity),
        (Activation::Relu, CudaBatchNormActivation::Relu),
        (
            Activation::HardSwish {
                alpha: 1.0 / 6.0,
                beta: 0.5,
            },
            CudaBatchNormActivation::HardSwish {
                alpha: 1.0 / 6.0,
                beta: 0.5,
            },
        ),
        (Activation::Swish, CudaBatchNormActivation::Swish),
        (
            Activation::GeluErf {
                divisor: std::f32::consts::SQRT_2,
                offset: 1.0,
                scale: 0.5,
            },
            CudaBatchNormActivation::GeluErf {
                divisor: std::f32::consts::SQRT_2,
                offset: 1.0,
                scale: 0.5,
            },
        ),
    ];

    for (batch, input_channels, output_channels, height, width) in [
        (1, 3, 5, 2, 7),
        (2 * REPRODUCIBLE_BATCH_ITEMS + 1, 3, 5, 1, 7),
        (2, 48, 96, 3, 19),
    ] {
        let input_base = Tensor::from_iter(
            (0..(batch + 1) * input_channels * height * width)
                .map(|value| ((value * 17 % 251) as f32 - 125.0) / 127.0),
            &device,
        )
        .unwrap()
        .reshape((batch + 1, input_channels, height, width))
        .unwrap();
        let input = input_base.narrow(0, 1, batch).unwrap();
        let kernel_base = Tensor::from_iter(
            (0..(output_channels + 1) * input_channels)
                .map(|value| ((value * 29 % 257) as f32 - 128.0) / 131.0),
            &device,
        )
        .unwrap()
        .reshape((output_channels + 1, input_channels, 1, 1))
        .unwrap();
        let kernel = kernel_base.narrow(0, 1, output_channels).unwrap();

        let mut scale_and_bias_values = Vec::with_capacity(2 * output_channels + 1);
        scale_and_bias_values.push(17.0_f32);
        scale_and_bias_values
            .extend((0..output_channels).map(|channel| (channel as f32 + 3.0) / 11.0));
        scale_and_bias_values
            .extend((0..output_channels).map(|channel| (channel as f32 - 5.0) / 17.0));
        let scale_and_bias =
            Tensor::from_vec(scale_and_bias_values, 2 * output_channels + 1, &device)
                .unwrap()
                .narrow(0, 1, 2 * output_channels)
                .unwrap()
                .reshape((2, output_channels))
                .unwrap();

        let mut mean_and_stddev_values = Vec::with_capacity(2 * output_channels + 1);
        mean_and_stddev_values.push(19.0_f32);
        mean_and_stddev_values
            .extend((0..output_channels).map(|channel| (channel as f32 - 2.0) / 13.0));
        mean_and_stddev_values
            .extend((0..output_channels).map(|channel| (channel as f32 + 7.0) / 19.0));
        let mean_and_stddev =
            Tensor::from_vec(mean_and_stddev_values, 2 * output_channels + 1, &device)
                .unwrap()
                .narrow(0, 1, 2 * output_channels)
                .unwrap()
                .reshape((2, output_channels))
                .unwrap();

        assert_ne!(input.layout().start_offset(), 0);
        assert_ne!(kernel.layout().start_offset(), 0);
        assert_ne!(scale_and_bias.layout().start_offset(), 0);
        assert_ne!(mean_and_stddev.layout().start_offset(), 0);

        let convolution =
            super::super::pointwise_convolution::conv2d(&input, &kernel, None).unwrap();
        for (activation, cuda_activation) in activations {
            let expected =
                super::cuda::execute(&convolution, &scale_and_bias, &mean_and_stddev, activation)
                    .unwrap();
            let post_operation = ConvolutionPostOperation::cuda_batch_normalization(
                &scale_and_bias,
                &mean_and_stddev,
                cuda_activation,
            )
            .unwrap();
            let actual = super::super::pointwise_convolution::conv2d_with_post_operation(
                &input,
                &kernel,
                None,
                post_operation,
            )
            .unwrap();
            actual.device().synchronize().unwrap();

            let expected = expected
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = actual
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "geometry={batch}x{input_channels}x{height}x{width}->{output_channels} activation={activation:?}",
            );
        }
    }
}

#[test]
#[ignore = "requires an explicit CUDA device"]
fn cuda_pointwise_batch_norm_in_place_records_a_cross_stream_write_dependency() {
    const ELEMENTS: usize = 256 * 1024;
    const DELAY_BYTES: usize = 64 * 1024 * 1024;
    const DELAY_PASSES: usize = 128;

    let producer_device = Device::new_cuda_with_stream(0).unwrap();
    let consumer_device = Device::new_cuda_with_stream(0).unwrap();
    let producer = producer_device.as_cuda_device().unwrap();
    let consumer = consumer_device.as_cuda_device().unwrap();
    assert_eq!(
        producer.cuda_stream().context().as_ref(),
        consumer.cuda_stream().context().as_ref(),
        "the regression requires two streams in the same primary CUDA context",
    );

    let source = (0..ELEMENTS)
        .map(|index| ((index * 17 % 251) as f32 - 125.0) / 127.0)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(source.clone(), (1, 1, 1, ELEMENTS), &producer_device).unwrap();
    let scale_and_bias = Tensor::new(&[[0.75_f32], [-0.375_f32]], &producer_device).unwrap();
    let mean_and_stddev = Tensor::new(&[[0.125_f32], [1.375_f32]], &producer_device).unwrap();
    let expected = super::cuda::execute(
        &input,
        &scale_and_bias,
        &mean_and_stddev,
        Activation::Identity,
    )
    .unwrap()
    .to_device(&Device::Cpu)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();

    let post_operation = ConvolutionPostOperation::cuda_batch_normalization(
        &scale_and_bias,
        &mean_and_stddev,
        CudaBatchNormActivation::Identity,
    )
    .unwrap();
    let parameters = post_operation
        .cuda_batch_normalization_parameters()
        .unwrap();
    let mut output = unsafe { producer.alloc::<f32>(ELEMENTS).unwrap() };
    producer.memcpy_htod(&source, &mut output).unwrap();

    // Keep the producer stream occupied between the preceding output write and
    // BatchNormalization. A consumer that observes only the old write event
    // will copy the unnormalized source before BatchNormalization can run.
    let mut delay = unsafe { producer.alloc::<u8>(DELAY_BYTES).unwrap() };
    let producer_stream = producer.cuda_stream();
    for _ in 0..DELAY_PASSES {
        producer_stream.memset_zeros(&mut delay).unwrap();
    }
    super::cuda::execute_post_in_place(&mut output, producer, 1, 1, ELEMENTS, parameters).unwrap();

    let actual = consumer.clone_dtoh(&output).unwrap();
    producer_device.synchronize().unwrap();
    let mismatch =
        actual
            .iter()
            .zip(&expected)
            .enumerate()
            .find_map(|(index, (actual, expected))| {
                (actual.to_bits() != expected.to_bits()).then_some((
                    index,
                    actual.to_bits(),
                    expected.to_bits(),
                ))
            });
    assert_eq!(mismatch, None, "cross-stream output mismatch");
}
