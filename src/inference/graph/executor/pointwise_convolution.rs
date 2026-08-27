use candle_core::{Result, Tensor};
#[cfg(test)]
use rayon::prelude::*;

use super::convolution_post::ConvolutionPostOperation;

mod cpu;
#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Executes a contiguous CPU 1x1 convolution as matrix products over NCHW.
///
/// NCHW already stores each batch as a contiguous `[channels, spatial]`
/// matrix. Broadcasting the shared `[outputs, channels]` kernel avoids the
/// generic convolution path's per-batch input repacking.
pub(super) fn conv2d(input: &Tensor, kernel: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    validate_inputs(input, kernel, bias)?;

    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        let output = cuda::conv2d(input, kernel)?;
        return match bias {
            Some(bias) => {
                let output_channels = kernel.dim(0)?;
                let bias = bias.reshape((1, output_channels, 1, 1))?;
                super::biased_activation::cuda_channel_bias(&output, &bias)
            }
            None => Ok(output),
        };
    }

    cpu::conv2d(input, kernel, bias, ConvolutionPostOperation::Identity)
}

/// Executes a pointwise convolution and one topology-proven private
/// post-operation.
pub(super) fn conv2d_with_post_operation(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    post_operation: ConvolutionPostOperation,
) -> Result<Tensor> {
    validate_inputs(input, kernel, bias)?;

    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        if bias.is_some() {
            candle_core::bail!(
                "CUDA pointwise BatchNormalization fusion requires a bias-free convolution"
            )
        }
        return cuda::conv2d_with_post_operation(input, kernel, post_operation);
    }

    cpu::conv2d(input, kernel, bias, post_operation)
}

/// Executes a pointwise convolution and its private exact-shape residual add
/// without materializing the intermediate convolution output.
pub(super) fn conv2d_with_residual(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    validate_inputs(input, kernel, bias)?;
    let (batch, _, height, width) = input.dims4()?;
    let output_channels = kernel.dim(0)?;
    if !residual.is_contiguous()
        || residual.dims4().ok() != Some((batch, output_channels, height, width))
    {
        candle_core::bail!(
            "CPU pointwise residual fusion requires an exact contiguous output-shaped addend"
        )
    }
    cpu::conv2d_with_residual(input, kernel, bias, residual)
}

fn validate_inputs(input: &Tensor, kernel: &Tensor, bias: Option<&Tensor>) -> Result<()> {
    let (batch, channels, _, _) = input.dims4()?;
    let (output_channels, kernel_channels, kernel_height, kernel_width) = kernel.dims4()?;
    if batch == 0
        || channels == 0
        || output_channels == 0
        || kernel_channels != channels
        || kernel_height != 1
        || kernel_width != 1
        || !input.is_contiguous()
        || !kernel.is_contiguous()
        || !input.device().same_device(kernel.device())
    {
        candle_core::bail!(
            "batched pointwise convolution requires non-empty contiguous NCHW tensors on one device"
        )
    }
    if bias.is_some_and(|bias| !bias.is_contiguous() || !input.device().same_device(bias.device()))
    {
        candle_core::bail!("batched pointwise convolution requires a contiguous co-located bias")
    }
    if bias.is_some_and(|bias| bias.dims1().ok() != Some(output_channels)) {
        candle_core::bail!("batched pointwise convolution requires one bias per output channel")
    }
    Ok(())
}

#[cfg(test)]
fn expanded_kernel_matmul(
    input: &Tensor,
    kernel: &Tensor,
    batch: usize,
    output_channels: usize,
    channels: usize,
) -> Result<Tensor> {
    let kernel = kernel.unsqueeze(0)?;
    let kernel = if batch == 1 {
        kernel
    } else {
        kernel.broadcast_as((batch, output_channels, channels))?
    };
    kernel.contiguous()?.matmul(input)
}

/// Reuses one pointwise kernel across batch slots while retaining the exact
/// per-slot GEMM geometry and accumulation order.
///
/// Concatenating the outputs copies `B * O * S` elements. The ordinary path
/// materializes `B * O * C` broadcast kernel elements. This path is selected
/// only when its required copy is strictly smaller.
#[cfg(test)]
fn shared_kernel_matmul(
    input: &Tensor,
    kernel: &Tensor,
    batch: usize,
    channels: usize,
    spatial: usize,
) -> Result<Tensor> {
    let outputs = input
        .chunk(batch, 0)?
        .into_par_iter()
        .map(|input| {
            kernel
                .matmul(&input.reshape((channels, spatial))?)?
                .unsqueeze(0)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&outputs, 0)
}

#[cfg(test)]
fn should_share_kernel(batch: usize, channels: usize, spatial: usize) -> bool {
    batch > 1 && spatial < channels
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use candle_core::Device;

    use super::*;

    #[test]
    fn batched_pointwise_matches_convolution() {
        let device = Device::Cpu;
        let input = Tensor::from_iter(
            (0..3 * 5 * 7 * 9).map(|value| (value as f32 - 100.0) / 31.0),
            &device,
        )
        .unwrap()
        .reshape((3, 5, 7, 9))
        .unwrap();
        let kernel =
            Tensor::from_iter((0..4 * 5).map(|value| (value as f32 - 7.0) / 23.0), &device)
                .unwrap()
                .reshape((4, 5, 1, 1))
                .unwrap();
        let bias = Tensor::new(&[0.125_f32, -0.25, 0.5, 1.0], &device).unwrap();
        let expected = input
            .conv2d(&kernel, 0, 1, 1, 1)
            .unwrap()
            .broadcast_add(&bias.reshape((1, 4, 1, 1)).unwrap())
            .unwrap();
        let actual = conv2d(&input, &kernel, Some(&bias)).unwrap();

        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn shared_kernel_batches_match_batched_pointwise_bits() {
        let device = Device::Cpu;
        for (batch, channels, output_channels, height, width) in [
            (1, 3, 5, 2, 7),
            (2, 48, 96, 6, 19),
            (8, 96, 192, 3, 41),
            (24, 192, 384, 1, 73),
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

            for bias in [None, Some(&bias)] {
                let expected = input.conv2d(&kernel, 0, 1, 1, 1).unwrap();
                let expected = match bias {
                    Some(bias) => expected
                        .broadcast_add(&bias.reshape((1, output_channels, 1, 1)).unwrap())
                        .unwrap(),
                    None => expected,
                };
                let actual = conv2d(&input, &kernel, bias).unwrap();

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
    fn outer_parallel_pointwise_matches_standalone_bits() {
        let device = Device::Cpu;
        let input = Tensor::from_iter(
            (0..8 * 96 * 3 * 41).map(|value| ((value * 17 % 251) as f32 - 125.0) / 127.0),
            &device,
        )
        .unwrap()
        .reshape((8, 96, 3, 41))
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..192 * 96).map(|value| ((value * 29 % 257) as f32 - 128.0) / 131.0),
            &device,
        )
        .unwrap()
        .reshape((192, 96, 1, 1))
        .unwrap();
        let bias =
            Tensor::from_iter((0..192).map(|value| (value as f32 - 17.0) / 37.0), &device).unwrap();

        let standalone = conv2d(&input, &kernel, Some(&bias))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let outer_parallel = pool.install(|| {
            assert!(rayon::current_thread_index().is_some());
            conv2d(&input, &kernel, Some(&bias))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        });

        assert_eq!(outer_parallel, standalone);
    }

    #[test]
    fn direct_pointwise_reads_contiguous_offset_layouts() {
        let device = Device::Cpu;
        let input = Tensor::from_iter(
            (0..4 * 7 * 5 * 9).map(|value| ((value * 17 % 251) as f32 - 125.0) / 61.0),
            &device,
        )
        .unwrap()
        .reshape((4, 7, 5, 9))
        .unwrap()
        .narrow(0, 1, 3)
        .unwrap();
        let kernel = Tensor::from_iter(
            (0..12 * 7).map(|value| ((value * 29 % 257) as f32 - 128.0) / 67.0),
            &device,
        )
        .unwrap()
        .reshape((12, 7, 1, 1))
        .unwrap()
        .narrow(0, 1, 11)
        .unwrap();
        let bias = Tensor::from_iter((0..13).map(|value| (value as f32 - 4.0) / 19.0), &device)
            .unwrap()
            .narrow(0, 1, 11)
            .unwrap();
        assert!(input.is_contiguous());
        assert!(kernel.is_contiguous());
        assert!(bias.is_contiguous());
        assert_ne!(input.layout().start_offset(), 0);
        assert_ne!(kernel.layout().start_offset(), 0);
        assert_ne!(bias.layout().start_offset(), 0);

        let expected = input
            .conv2d(&kernel, 0, 1, 1, 1)
            .unwrap()
            .broadcast_add(&bias.reshape((1, 11, 1, 1)).unwrap())
            .unwrap();
        let actual = conv2d(&input, &kernel, Some(&bias)).unwrap();

        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_pointwise_direct_nchw_obeys_f32_reduction_error_bound() {
        let device = Device::new_cuda(0).unwrap();
        for (batch, channels, output_channels, height, width) in
            [(3, 5, 4, 7, 9), (8, 96, 192, 3, 41), (24, 192, 384, 1, 73)]
        {
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
            let expected = input
                .conv2d(&kernel, 0, 1, 1, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let direct = conv2d(&input, &kernel, None).unwrap();
            let actual = direct.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut changed = 0_usize;
            let mut maximum_absolute_difference = 0.0_f32;
            for (expected, actual) in expected.iter().zip(&actual) {
                changed += usize::from(expected.to_bits() != actual.to_bits());
                maximum_absolute_difference =
                    maximum_absolute_difference.max((expected - actual).abs());
            }
            // Both GEMMs evaluate the same length-C dot product with F32
            // fused multiply-adds but use different output tilings. For this
            // fixture |x| and |w| are bounded by one. The standard forward
            // error bound for either reduction is gamma_C * sum(|x*w|), so
            // their pairwise difference is bounded by twice that value.
            let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
            let reduction = channels as f64 * unit_roundoff;
            let gamma = reduction / (1.0 - reduction);
            let forward_error_bound = (2.0 * gamma * channels as f64) as f32;
            eprintln!(
                "CUDA_POINTWISE_PARITY geometry={batch}x{channels}x{height}x{width}->{output_channels} changed={changed} max_abs={maximum_absolute_difference:.9e} forward_error_bound={forward_error_bound:.9e}"
            );
            assert!(maximum_absolute_difference <= forward_error_bound);

            let bias = Tensor::from_iter(
                (0..output_channels).map(|value| (value as f32 - 17.0) / 37.0),
                &device,
            )
            .unwrap();
            let expected_biased = direct
                .broadcast_add(&bias.reshape((1, output_channels, 1, 1)).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual_biased = conv2d(&input, &kernel, Some(&bias))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(actual_biased, expected_biased);
        }
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_pointwise_large_batch_matches_explicit_reproducible_partitions() {
        use crate::inference::graph::executor::cuda_reproducibility::REPRODUCIBLE_BATCH_ITEMS;

        let device = Device::new_cuda_with_stream(0).unwrap();
        let batch = 2 * REPRODUCIBLE_BATCH_ITEMS + 1;
        let channels = 24;
        let output_channels = 96;
        let input = Tensor::from_iter(
            (0..batch * channels).map(|value| ((value * 17 % 251) as f32 - 125.0) / 127.0),
            &device,
        )
        .unwrap()
        .reshape((batch, channels, 1, 1))
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

        let expected = (0..batch)
            .step_by(REPRODUCIBLE_BATCH_ITEMS)
            .map(|offset| {
                let chunk_items = (batch - offset).min(REPRODUCIBLE_BATCH_ITEMS);
                let chunk = input.narrow(0, offset, chunk_items)?;
                conv2d(&chunk, &kernel, Some(&bias))
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|outputs| Tensor::cat(&outputs, 0))
            .unwrap();
        let actual = conv2d(&input, &kernel, Some(&bias)).unwrap();
        actual.device().synchronize().unwrap();

        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(
            actual
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            expected
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fused_pointwise_post_operations_match_explicit_graph_bits() {
        let device = Device::Cpu;
        for (batch, channels, output_channels, height, width) in [
            (1, 3, 5, 2, 7),
            (2, 48, 96, 6, 19),
            (8, 96, 192, 3, 41),
            (24, 192, 384, 1, 73),
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
            let convolution = conv2d(&input, &kernel, None).unwrap();
            let biased = convolution
                .broadcast_add(&bias.reshape((1, output_channels, 1, 1)).unwrap())
                .unwrap();

            let expected_relu = biased.relu().unwrap();
            let actual_relu = conv2d_with_post_operation(
                &input,
                &kernel,
                Some(&bias),
                ConvolutionPostOperation::Relu,
            )
            .unwrap();
            assert_eq!(
                actual_relu.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected_relu
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                "ReLU geometry {batch}x{channels}x{height}x{width} -> {output_channels}",
            );

            let divisor = std::f32::consts::SQRT_2;
            let offset = 1.0_f32;
            let scale = 0.5_f32;
            let expected_gelu = biased
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
                &kernel,
                Some(&bias),
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
                "GELU geometry {batch}x{channels}x{height}x{width} -> {output_channels}",
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
            let spatial = height * width;
            let expected_batch_norm = biased
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .chunks(spatial)
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
            let actual_batch_norm =
                conv2d_with_post_operation(&input, &kernel, Some(&bias), post_operation).unwrap();
            assert_eq!(
                actual_batch_norm
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                expected_batch_norm,
                "BatchNormalization geometry {batch}x{channels}x{height}x{width} -> {output_channels}",
            );
        }
    }

    #[test]
    fn shared_kernel_selection_follows_copy_complexity() {
        assert!(!should_share_kernel(1, 384, 16));
        assert!(should_share_kernel(2, 384, 16));
        assert!(!should_share_kernel(2, 384, 384));
        assert!(!should_share_kernel(2, 384, 512));
    }

    #[test]
    #[ignore = "diagnostic CPU kernel comparison"]
    fn compare_shared_kernel_scalar_batches() {
        let device = Device::Cpu;
        for (batch, channels, output_channels, spatial) in [
            (2, 96, 192, 32),
            (4, 96, 192, 32),
            (8, 96, 192, 32),
            (24, 96, 192, 32),
            (2, 96, 192, 128),
            (8, 96, 192, 128),
            (24, 96, 192, 128),
            (2, 192, 384, 16),
            (4, 192, 384, 16),
            (8, 192, 384, 16),
            (2, 192, 384, 32),
            (4, 192, 384, 32),
            (8, 192, 384, 32),
            (24, 192, 384, 32),
            (2, 384, 768, 16),
            (4, 384, 768, 16),
            (8, 384, 768, 16),
            (2, 384, 768, 32),
            (4, 384, 768, 32),
            (8, 384, 768, 32),
        ] {
            let input = Tensor::zeros(
                (batch, channels, 1, spatial),
                candle_core::DType::F32,
                &device,
            )
            .unwrap();
            let kernel = Tensor::zeros(
                (output_channels, channels, 1, 1),
                candle_core::DType::F32,
                &device,
            )
            .unwrap();
            let input_matrix = input.reshape((batch, channels, spatial)).unwrap();
            let kernel_matrix = kernel.reshape((output_channels, channels)).unwrap();
            black_box(
                expanded_kernel_matmul(
                    &input_matrix,
                    &kernel_matrix,
                    batch,
                    output_channels,
                    channels,
                )
                .unwrap(),
            );
            black_box(
                shared_kernel_matmul(&input_matrix, &kernel_matrix, batch, channels, spatial)
                    .unwrap(),
            );
            black_box(conv2d(&input, &kernel, None).unwrap());
            let iterations = 10_u32;
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(
                    expanded_kernel_matmul(
                        &input_matrix,
                        &kernel_matrix,
                        batch,
                        output_channels,
                        channels,
                    )
                    .unwrap(),
                );
            }
            let batched = started.elapsed();
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(
                    shared_kernel_matmul(&input_matrix, &kernel_matrix, batch, channels, spatial)
                        .unwrap(),
                );
            }
            let shared = started.elapsed();
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(conv2d(&input, &kernel, None).unwrap());
            }
            let direct = started.elapsed();
            eprintln!(
                "batch={batch} channels={channels} outputs={output_channels} spatial={spatial} batched_ms={:.3} shared_ms={:.3} direct_ms={:.3}",
                batched.as_secs_f64() * 1_000.0 / f64::from(iterations),
                shared.as_secs_f64() * 1_000.0 / f64::from(iterations),
                direct.as_secs_f64() * 1_000.0 / f64::from(iterations),
            );
        }
    }
}
