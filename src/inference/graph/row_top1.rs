use candle_core::{DType, Tensor};

use crate::error::{PowerError, Result};

#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Projects the last axis of a contiguous F32 tensor to
/// `[last-index-on-tie, score, all-source-values-finite]`.
///
/// The operation is model-neutral and remains on the input device. Model
/// crates own the meaning of the classes and must bind use of this projection
/// into their execution identity.
pub fn row_top1_last_finite(input: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::F32 {
        return Err(projection_error(format!(
            "row top-1 projection requires F32 input, found {:?}",
            input.dtype()
        )));
    }
    let classes = input.dims().last().copied().unwrap_or_default();
    if input.rank() < 2 || classes == 0 || classes > (1 << 24) {
        return Err(projection_error(
            "row top-1 projection received an invalid bounded shape",
        ));
    }

    #[cfg(feature = "embedded-cuda")]
    if input.device().is_cuda() {
        return cuda::execute(input).map_err(candle_error);
    }

    candle_projection(input, classes)
}

fn candle_projection(input: &Tensor, classes: usize) -> Result<Tensor> {
    let reversed = input.flip(&[input.rank() - 1]).map_err(candle_error)?;
    let axis = input.rank() - 1;
    let reversed_indices = reversed.argmax_keepdim(axis).map_err(candle_error)?;
    let scores = reversed
        .gather(&reversed_indices, axis)
        .map_err(candle_error)?;
    let indices = reversed_indices
        .to_dtype(DType::F32)
        .and_then(|indices| indices.affine(-1.0, (classes - 1) as f64))
        .map_err(candle_error)?;
    let finite = input
        .abs()
        .and_then(|values| values.le(f32::MAX))
        .and_then(|values| values.min_keepdim(axis))
        .and_then(|values| values.to_dtype(DType::F32))
        .map_err(candle_error)?;
    Tensor::cat(&[&indices, &scores, &finite], axis).map_err(candle_error)
}

fn candle_error(error: candle_core::Error) -> PowerError {
    projection_error(format!("row top-1 projection failed: {error}"))
}

fn projection_error(message: impl Into<String>) -> PowerError {
    PowerError::InferenceFailed(message.into())
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn cpu_projection_preserves_last_ties_and_source_finiteness() {
        let input = Tensor::from_vec(
            vec![0.1_f32, 0.8, 0.8, 0.2, 0.9, f32::NAN, 0.1, 0.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();

        let projected = row_top1_last_finite(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(projected[..3], [2.0, 0.8, 1.0]);
        assert_eq!(projected[3], 0.0);
        assert_eq!(projected[4], 0.9);
        assert_eq!(projected[5], 0.0);
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn cuda_projection_matches_the_reviewed_cpu_contract() {
        let values = vec![
            0.1_f32,
            0.8,
            0.8,
            0.2,
            0.9,
            f32::NAN,
            0.1,
            0.0,
            -1.0,
            -1.0,
            -2.0,
            -3.0,
        ];
        let cpu = Tensor::from_vec(values.clone(), (1, 3, 4), &Device::Cpu).unwrap();
        let cuda_device = Device::new_cuda(0).unwrap();
        let cuda = Tensor::from_vec(values, (1, 3, 4), &cuda_device).unwrap();
        let expected = candle_projection(&cpu, 4)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = row_top1_last_finite(&cuda)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
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
                "row top-1 CUDA profile: implementation={label} mean_ms={:.3}",
                started.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations),
            );
        };

        profile("candle", &|| candle_projection(&input, 18_710));
        for threads in [128, 256, 512, 1024] {
            profile(&format!("fused-{threads}"), &|| {
                cuda::execute_with_threads(&input, threads).map_err(candle_error)
            });
        }
    }
}
