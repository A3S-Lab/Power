use candle_core::{DType, Tensor};

use crate::error::{PowerError, Result};

mod cpu;
#[cfg(feature = "embedded-cuda")]
mod cuda;
mod matmul_cpu;

/// Projects F32 logits along their last axis to
/// `[last-top-index-on-tie, Softmax probability, all-logits-finite]`.
///
/// The operation is model-neutral and remains on the input device. Model
/// crates own the meaning of the classes and must bind use of this projection
/// into their execution identity.
pub fn row_softmax_top1_last_finite(input: &Tensor) -> Result<Tensor> {
    validate_input(input)?;

    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        return cuda::execute(input).map_err(candle_error);
    }

    if input.device().is_cpu() {
        return cpu::execute(input).map_err(candle_error);
    }

    candle_projection(input)
}

/// Adds one F32 last-axis bias and projects the resulting rows to
/// `[last-top-index-on-tie, Softmax probability, all-logits-finite]`.
///
/// CPU execution fuses the bias into the bounded row scan so an intermediate
/// tensor the size of the classifier output is not materialized. Other
/// devices retain the explicit broadcast addition and reviewed projection.
pub fn row_bias_softmax_top1_last_finite(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let classes = validate_input(input)?;
    if bias.dtype() != DType::F32 || bias.dims() != [classes] {
        return Err(projection_error(format!(
            "row Softmax top-1 bias requires exact F32 [classes] shape, found {:?} {:?}",
            bias.dtype(),
            bias.dims()
        )));
    }
    if !input.device().same_device(bias.device()) {
        return Err(projection_error(
            "row Softmax top-1 input and bias must use the same device",
        ));
    }

    if input.device().is_cpu() {
        return cpu::execute_with_bias(input, bias).map_err(candle_error);
    }

    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        return cuda::execute_with_bias(input, bias).map_err(candle_error);
    }

    let biased = input.broadcast_add(bias).map_err(candle_error)?;
    row_softmax_top1_last_finite(&biased)
}

/// Multiplies rows by one F32 classifier matrix, adds its last-axis bias, and
/// projects to `[last-top-index-on-tie, Softmax probability,
/// all-logits-finite]` without retaining the complete classifier tensor.
///
/// CPU execution uses bounded row tiles while preserving the same GEMM,
/// bias-addition, exponential, and reduction arithmetic as the explicit
/// operators. Other devices retain the explicit reviewed graph.
pub fn row_matmul_bias_softmax_top1_last_finite(
    input: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    if input.dtype() != DType::F32 || weights.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        return Err(projection_error(
            "row classifier projection requires F32 input, weights, and bias",
        ));
    }
    let features = input.dims().last().copied().unwrap_or_default();
    let (weight_features, classes) = weights.dims2().map_err(candle_error)?;
    if input.rank() < 2
        || features == 0
        || features != weight_features
        || classes == 0
        || classes > (1 << 24)
        || bias.dims() != [classes]
    {
        return Err(projection_error(
            "row classifier projection received incompatible bounded shapes",
        ));
    }
    if !input.device().same_device(weights.device()) || !input.device().same_device(bias.device()) {
        return Err(projection_error(
            "row classifier input, weights, and bias must use the same device",
        ));
    }

    if input.device().is_cpu()
        && input.is_contiguous()
        && weights.is_contiguous()
        && bias.is_contiguous()
    {
        return matmul_cpu::execute(input, weights, bias).map_err(candle_error);
    }

    let logits = super::matrix_multiplication::broadcast(input, weights).map_err(candle_error)?;
    row_bias_softmax_top1_last_finite(&logits, bias)
}

fn validate_input(input: &Tensor) -> Result<usize> {
    if input.dtype() != DType::F32 {
        return Err(projection_error(format!(
            "row Softmax top-1 projection requires F32 input, found {:?}",
            input.dtype()
        )));
    }
    let classes = input.dims().last().copied().unwrap_or_default();
    if input.rank() < 2 || classes == 0 || classes > (1 << 24) {
        return Err(projection_error(
            "row Softmax top-1 projection received an invalid bounded shape",
        ));
    }
    Ok(classes)
}

fn candle_projection(input: &Tensor) -> Result<Tensor> {
    let axis = input.rank() - 1;
    let probabilities = candle_nn::ops::softmax(input, axis).map_err(candle_error)?;
    super::row_top1_last_finite(&probabilities)
}

fn candle_error(error: candle_core::Error) -> PowerError {
    projection_error(format!("row Softmax top-1 projection failed: {error}"))
}

fn projection_error(message: impl Into<String>) -> PowerError {
    PowerError::InferenceFailed(message.into())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use candle_core::Device;

    use super::*;

    #[test]
    fn cpu_projection_matches_explicit_softmax_then_top1() {
        let input = Tensor::from_vec(
            vec![0.1_f32, 0.8, 0.8, 0.2, 0.9, -0.3, 0.1, 0.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let expected = candle_projection(&input).unwrap().to_vec3::<f32>().unwrap();
        let actual = row_softmax_top1_last_finite(&input)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual[0][0][0], 2.0);
        assert_eq!(actual[0][1][0], 0.0);
    }

    #[test]
    fn cpu_projection_preserves_explicit_probability_bits_across_rows() {
        for (rows, classes) in [(1, 3), (2, 17), (7, 257), (3, 18_710)] {
            let input = Tensor::from_iter(
                (0..rows * classes).map(|index| ((index * 131 % 997) as f32 - 498.0) / 83.0),
                &Device::Cpu,
            )
            .unwrap()
            .reshape((1, rows, classes))
            .unwrap();

            let expected = candle_projection(&input)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = row_softmax_top1_last_finite(&input)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            assert_eq!(actual, expected, "rows={rows}, classes={classes}");
        }
    }

    #[test]
    fn cpu_projection_preserves_post_softmax_ties_from_adjacent_logits() {
        let input = Tensor::new(&[[0.0_f32, -f32::from_bits(1)]], &Device::Cpu).unwrap();

        let expected = candle_projection(&input).unwrap().to_vec2::<f32>().unwrap();
        let actual = row_softmax_top1_last_finite(&input)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual[0][0], 1.0);
    }

    #[test]
    fn cpu_bias_projection_matches_explicit_add_softmax_and_top1_bits() {
        for (rows, classes) in [(1, 3), (2, 17), (7, 257), (3, 18_710)] {
            let input = Tensor::from_iter(
                (0..rows * classes).map(|index| ((index * 131 % 997) as f32 - 498.0) / 83.0),
                &Device::Cpu,
            )
            .unwrap()
            .reshape((1, rows, classes))
            .unwrap();
            let bias = Tensor::from_iter(
                (0..classes).map(|index| ((index * 43 % 271) as f32 - 135.0) / 97.0),
                &Device::Cpu,
            )
            .unwrap();
            let expected = candle_projection(&input.broadcast_add(&bias).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = row_bias_softmax_top1_last_finite(&input, &bias)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            assert_eq!(actual, expected, "rows={rows}, classes={classes}");
        }
    }

    #[test]
    fn cpu_row_projection_evaluates_each_source_value_once() {
        let values = [0.25_f32, -0.5, 1.0, 1.0, -2.0];
        let calls = Cell::new(0_usize);
        let mut exponentials = vec![0.0_f32; values.len()];
        let mut output = [0.0_f32; 3];

        cpu::project_row(values.len(), &mut exponentials, &mut output, |index| {
            calls.set(calls.get() + 1);
            values[index]
        });

        assert_eq!(calls.get(), values.len());
        assert_eq!(output[0], 3.0);
        assert_eq!(output[2], 1.0);
    }

    #[test]
    fn cpu_projection_honors_contiguous_layout_offsets() {
        let source = Tensor::from_iter(
            (0..4 * 19).map(|index| ((index * 41 % 101) as f32 - 50.0) / 13.0),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((4, 19))
        .unwrap();
        let input = source.narrow(0, 1, 2).unwrap();
        assert!(input.is_contiguous());
        assert_ne!(input.layout().start_offset(), 0);

        let expected = candle_projection(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = row_softmax_top1_last_finite(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn cpu_projection_marks_non_finite_rows_without_publishing_non_finite_output() {
        let input = Tensor::from_vec(
            vec![0.1_f32, 0.8, 0.2, 0.9, f32::NAN, 0.1],
            (1, 2, 3),
            &Device::Cpu,
        )
        .unwrap();

        let actual = row_softmax_top1_last_finite(&input)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert_eq!(actual[0][0][0], 1.0);
        assert!(actual[0][0][1].is_finite());
        assert_eq!(actual[0][0][2], 1.0);
        assert_eq!(actual[0][1], [2.0, 0.0, 0.0]);
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn cuda_projection_matches_the_reviewed_cpu_contract() {
        let values = vec![
            0.1_f32, 0.8, 0.8, 0.2, 0.9, -0.3, 0.1, 0.0, -1.0, -1.0, -2.0, -3.0,
        ];
        let cpu = Tensor::from_vec(values.clone(), (1, 3, 4), &Device::Cpu).unwrap();
        let cuda_device = Device::new_cuda(0).unwrap();
        let cuda = Tensor::from_vec(values, (1, 3, 4), &cuda_device).unwrap();
        let expected = candle_projection(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = row_softmax_top1_last_finite(&cuda)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (actual, expected) in actual.chunks_exact(3).zip(expected.chunks_exact(3)) {
            assert_eq!(actual[0], expected[0]);
            assert!((actual[1] - expected[1]).abs() <= 1e-6);
            assert_eq!(actual[2], expected[2]);
        }
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn cuda_bias_fusion_matches_explicit_addition_bits_with_offsets() {
        let device = Device::new_cuda_with_stream(0).unwrap();
        let classes = 257_usize;
        let input = Tensor::from_iter(
            (0..2 * 5 * classes).map(|index| ((index * 131 % 997) as f32 - 498.0) / 83.0),
            &device,
        )
        .unwrap()
        .reshape((2, 5, classes))
        .unwrap();
        let bias_storage = Tensor::from_iter(
            (0..classes + 2).map(|index| ((index * 43 % 271) as f32 - 135.0) / 97.0),
            &device,
        )
        .unwrap();
        let bias = bias_storage.narrow(0, 1, classes).unwrap();
        assert_ne!(bias.layout().start_offset(), 0);

        let expected = cuda::execute(&input.broadcast_add(&bias).unwrap()).unwrap();
        let actual = cuda::execute_with_bias(&input, &bias).unwrap();
        device.synchronize().unwrap();
        let bits = |tensor: Tensor| {
            tensor
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };

        assert_eq!(actual.dims(), [2, 5, 3]);
        assert_eq!(bits(actual), bits(expected));
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "profiles an explicit CUDA device"]
    fn cuda_projection_profiles_reviewed_recognition_shape() {
        let device = Device::new_cuda(0).unwrap();
        let input = Tensor::zeros((32, 80, 18_710), DType::F32, &device).unwrap();
        let iterations = 5_u32;

        let profile = |label: &str, project: &dyn Fn() -> Result<Tensor>| {
            let warmup = project().unwrap();
            warmup.device().synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                let output = project().unwrap();
                output.device().synchronize().unwrap();
            }
            eprintln!(
                "row Softmax top-1 CUDA profile: implementation={label} mean_ms={:.3}",
                started.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations),
            );
        };

        profile("candle-softmax-plus-top1", &|| candle_projection(&input));
        profile("fused", &|| cuda::execute(&input).map_err(candle_error));
    }
}
