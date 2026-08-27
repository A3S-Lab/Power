use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use super::{GraphExecutor, TensorInput, TensorOutput};
use crate::error::{PowerError, Result};
use crate::inference::{ExecutionPermit, InferenceLimits};

impl GraphExecutor {
    /// Executes heterogeneous input shapes through one terminal classifier by
    /// coalescing only its independent feature rows.
    ///
    /// Every reviewed graph prefix retains its exact input shape and dynamic
    /// width. Power flattens the resulting `[... , F]` features to rows,
    /// concatenates those rows, applies the model-owned last-axis classifier
    /// projection once, and restores each original leading shape. The supplied
    /// projection must therefore be row-independent: its output may depend on
    /// each feature row, weights, and bias, but not on neighboring rows or the
    /// temporary aggregate row count.
    pub fn run_many_with_row_coalesced_terminal_matmul_bias_softmax_projection<F>(
        &self,
        inputs: Vec<TensorInput>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<Vec<TensorOutput>>
    where
        F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        if inputs.is_empty() {
            return Err(PowerError::InvalidRequest(
                "row-coalesced terminal classifier execution requires at least one input"
                    .to_string(),
            ));
        }
        self.validate_projection_execution(permit, cancellation)?;
        let boundary = self.terminal_classifier_boundary()?;
        let (inputs, upload_guard) = TensorInput::into_candle_many(
            inputs,
            self.runtime.device().tensor_device(),
            self.runtime.limits(),
            permit.input_upload_pool(),
        )?;
        if inputs.len() == 1 {
            let projected = self.enqueue_terminal_classifier(
                inputs.into_iter().next().ok_or_else(|| {
                    PowerError::InferenceFailed(
                        "terminal classifier window lost its only input".to_string(),
                    )
                })?,
                &boundary,
                cancellation,
                projection,
            )?;
            let output = TensorOutput::from_candle_many(vec![projected], self.runtime.limits());
            upload_guard.complete();
            return output;
        }

        let feature_width = boundary.weights.dims2().map_err(candle_error)?.0;
        let mut rows = Vec::with_capacity(inputs.len());
        let mut row_counts = Vec::with_capacity(inputs.len());
        let mut leading_shapes = Vec::with_capacity(inputs.len());
        let mut total_rows = 0_usize;
        for input in inputs {
            let features =
                self.enqueue_terminal_classifier_features(input, &boundary, cancellation)?;
            let dimensions = features.dims();
            if features.dtype() != candle_core::DType::F32
                || dimensions.len() < 2
                || dimensions.last().copied() != Some(feature_width)
            {
                return Err(PowerError::InferenceFailed(format!(
                    "terminal classifier row coalescing requires F32 [..., {feature_width}] features, found {:?} {dimensions:?}",
                    features.dtype(),
                )));
            }
            let row_count = dimensions[..dimensions.len() - 1]
                .iter()
                .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| {
                    PowerError::InferenceFailed(
                        "terminal classifier feature row count overflowed".to_string(),
                    )
                })?;
            total_rows = total_rows.checked_add(row_count).ok_or_else(|| {
                PowerError::InferenceFailed(
                    "terminal classifier aggregate row count overflowed".to_string(),
                )
            })?;
            leading_shapes.push(dimensions[..dimensions.len() - 1].to_vec());
            row_counts.push(row_count);
            rows.push(
                features
                    .reshape((row_count, feature_width))
                    .map_err(candle_error)?,
            );
        }
        self.runtime.limits().checked_elements(
            &[total_rows, feature_width],
            "coalesced terminal classifier features",
        )?;
        let row_refs = rows.iter().collect::<Vec<_>>();
        let coalesced = Tensor::cat(&row_refs, 0).map_err(candle_error)?;
        let projected = projection(&coalesced, boundary.weights, boundary.bias)?;
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        if !projected.device().same_device(coalesced.device())
            || projected.dtype() != candle_core::DType::F32
            || projected.rank() != 2
            || projected.dims().first().copied() != Some(total_rows)
        {
            return Err(PowerError::InferenceFailed(format!(
                "row-coalesced terminal classifier projection must return F32 [{total_rows}, P] on the graph device, found {:?} {:?}",
                projected.dtype(),
                projected.dims(),
            )));
        }
        let projected_width = projected.dims()[1];
        if projected_width == 0 {
            return Err(PowerError::InferenceFailed(
                "row-coalesced terminal classifier projection returned an empty row".to_string(),
            ));
        }
        self.runtime.limits().checked_elements(
            &[total_rows, projected_width],
            "coalesced terminal classifier projection",
        )?;

        let projected = TensorOutput::from_candle(&projected, self.runtime.limits())?;
        upload_guard.complete();
        partition_projected_output(
            projected,
            row_counts,
            leading_shapes,
            total_rows,
            projected_width,
            self.runtime.limits(),
        )
    }
}

fn partition_projected_output(
    projected: TensorOutput,
    row_counts: Vec<usize>,
    leading_shapes: Vec<Vec<usize>>,
    total_rows: usize,
    projected_width: usize,
    limits: &InferenceLimits,
) -> Result<Vec<TensorOutput>> {
    if projected.shape != [total_rows, projected_width] {
        return Err(PowerError::InferenceFailed(format!(
            "row-coalesced terminal classifier output shape changed before partitioning: expected [{total_rows}, {projected_width}], found {:?}",
            projected.shape,
        )));
    }
    if row_counts.len() != leading_shapes.len() {
        return Err(PowerError::InferenceFailed(
            "row-coalesced terminal classifier lost a partition shape".to_string(),
        ));
    }

    let mut values = projected.values.into_iter();
    let mut outputs = Vec::with_capacity(row_counts.len());
    let mut offset = 0_usize;
    for (row_count, mut shape) in row_counts.into_iter().zip(leading_shapes) {
        offset = offset.checked_add(row_count).ok_or_else(|| {
            PowerError::InferenceFailed(
                "terminal classifier output row offset overflowed".to_string(),
            )
        })?;
        let expected_values = row_count.checked_mul(projected_width).ok_or_else(|| {
            PowerError::InferenceFailed(
                "row-coalesced terminal classifier partition size overflowed".to_string(),
            )
        })?;
        shape.push(projected_width);
        let value_count =
            limits.checked_elements(&shape, "row-coalesced terminal classifier partition")?;
        if value_count != expected_values {
            return Err(PowerError::InferenceFailed(
                "row-coalesced terminal classifier leading shape changed its row count".to_string(),
            ));
        }
        let output_values = values.by_ref().take(value_count).collect::<Vec<_>>();
        if output_values.len() != value_count {
            return Err(PowerError::InferenceFailed(
                "row-coalesced terminal classifier output ended before every partition was materialized"
                    .to_string(),
            ));
        }
        outputs.push(TensorOutput {
            shape,
            values: output_values,
        });
    }
    if offset != total_rows || values.next().is_some() {
        return Err(PowerError::InferenceFailed(
            "terminal classifier output partitions did not cover every row".to_string(),
        ));
    }
    Ok(outputs)
}

fn candle_error(error: candle_core::Error) -> PowerError {
    PowerError::InferenceFailed(format!(
        "terminal classifier tensor operation failed: {error}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn one_copy_projection_materialization_matches_individual_cpu_slices() {
        assert_one_copy_projection_materialization(Device::Cpu);
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn one_copy_projection_materialization_matches_individual_cuda_slices() {
        assert_one_copy_projection_materialization(Device::new_cuda_with_stream(0).unwrap());
    }

    fn assert_one_copy_projection_materialization(device: Device) {
        let limits = InferenceLimits::default();
        let projected = Tensor::from_vec(
            vec![0.125_f32, 0.25, 0.625, 0.5, 0.25, 0.25, 0.75, 0.125, 0.125],
            (3, 3),
            &device,
        )
        .unwrap();
        let first = projected.narrow(0, 0, 1).unwrap().reshape((1, 3)).unwrap();
        let second = projected
            .narrow(0, 1, 2)
            .unwrap()
            .reshape((1, 2, 3))
            .unwrap();
        let individually_materialized =
            TensorOutput::from_candle_many(vec![first, second], &limits).unwrap();

        let once_materialized = partition_projected_output(
            TensorOutput::from_candle(&projected, &limits).unwrap(),
            vec![1, 2],
            vec![vec![1], vec![1, 2]],
            3,
            3,
            &limits,
        )
        .unwrap();

        assert_eq!(once_materialized, individually_materialized);
    }
}
