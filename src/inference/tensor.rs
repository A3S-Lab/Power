use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::InferenceLimits;

/// Provider-neutral owned F32 tensor accepted by an embedded session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorInput {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl TensorInput {
    pub fn new(shape: Vec<usize>, values: Vec<f32>, limits: &InferenceLimits) -> Result<Self> {
        let expected = limits.checked_elements(&shape, "input tensor")?;
        if values.len() != expected {
            return Err(PowerError::InvalidRequest(format!(
                "input tensor has {} values but shape {shape:?} requires {expected}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PowerError::InvalidRequest(
                "input tensor contains a non-finite value".to_string(),
            ));
        }
        Ok(Self { shape, values })
    }

    /// Concatenates compatible tensors along their leading axis.
    ///
    /// Model crates retain ownership of padding, bucketing, and slot geometry.
    /// Power validates only the provider-neutral tensor contract, exact caller
    /// order, and the shared inference limits.
    pub fn stack_leading(items: Vec<Self>, limits: &InferenceLimits) -> Result<Self> {
        let first = items.first().ok_or_else(|| {
            PowerError::InvalidRequest(
                "a leading-axis tensor batch requires at least one item".to_string(),
            )
        })?;
        if first.shape.is_empty() {
            return Err(PowerError::InvalidRequest(
                "a leading-axis tensor batch requires ranked items".to_string(),
            ));
        }
        let trailing_shape = first.shape[1..].to_vec();
        let mut leading = 0_usize;
        let mut value_count = 0_usize;
        for item in &items {
            if item.shape.is_empty() {
                return Err(PowerError::InvalidRequest(
                    "leading-axis tensor batch items must have ranked shapes".to_string(),
                ));
            }
            let expected = limits.checked_elements(&item.shape, "batched input tensor item")?;
            if item.shape[1..] != trailing_shape
                || item.values.len() != expected
                || item.values.iter().any(|value| !value.is_finite())
            {
                return Err(PowerError::InvalidRequest(
                    "leading-axis tensor batch items must have identical trailing shapes and valid finite values"
                        .to_string(),
                ));
            }
            leading = leading.checked_add(item.shape[0]).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "leading-axis tensor batch dimension overflowed".to_string(),
                )
            })?;
            value_count = value_count.checked_add(item.values.len()).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "leading-axis tensor batch value count overflowed".to_string(),
                )
            })?;
        }
        let mut shape = Vec::with_capacity(trailing_shape.len() + 1);
        shape.push(leading);
        shape.extend(trailing_shape);
        let expected = limits.checked_elements(&shape, "batched input tensor")?;
        if value_count != expected {
            return Err(PowerError::InvalidRequest(
                "leading-axis tensor batch values do not match the combined shape".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(value_count);
        for item in items {
            values.extend(item.values);
        }
        Ok(Self { shape, values })
    }

    pub(crate) fn into_candle(self, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(self.values, self.shape.as_slice(), device).map_err(|error| {
            PowerError::InferenceFailed(format!("failed to materialize input tensor: {error}"))
        })
    }
}

/// Provider-neutral owned F32 tensor returned by an embedded session.
///
/// Static graphs must produce F32 explicitly. The runtime refuses other
/// dtypes instead of silently changing model precision at the API boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorOutput {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl TensorOutput {
    pub(crate) fn from_candle(tensor: &Tensor, limits: &InferenceLimits) -> Result<Self> {
        if tensor.dtype() != DType::F32 {
            return Err(PowerError::InvalidFormat(format!(
                "static graph output must be F32, found {:?}",
                tensor.dtype()
            )));
        }
        let shape = tensor.dims().to_vec();
        let expected = limits.checked_elements(&shape, "output tensor")?;
        let values = tensor
            .flatten_all()
            .and_then(|value| value.to_vec1::<f32>())
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "failed to copy the output tensor from the execution device: {error}"
                ))
            })?;
        if values.len() != expected {
            return Err(PowerError::InferenceFailed(
                format!(
                    "embedded inference returned {} output values for shape {shape:?}, expected {expected}",
                    values.len()
                ),
            ));
        }
        if let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(PowerError::InferenceFailed(format!(
                "embedded inference returned non-finite output value {value} at flat index {index}"
            )));
        }
        Ok(Self { shape, values })
    }

    /// Splits a tensor into exact, ordered leading-axis partitions.
    ///
    /// The partition sizes must be positive and cover the complete leading
    /// axis. No model-specific slot or padding meaning is interpreted here.
    pub fn split_leading(
        self,
        leading_partitions: &[usize],
        limits: &InferenceLimits,
    ) -> Result<Vec<Self>> {
        if self.shape.is_empty() || leading_partitions.is_empty() || leading_partitions.contains(&0)
        {
            return Err(PowerError::InvalidRequest(
                "leading-axis output partitions require a ranked tensor and positive partition sizes"
                    .to_string(),
            ));
        }
        let expected = limits.checked_elements(&self.shape, "batched output tensor")?;
        if self.values.len() != expected || self.values.iter().any(|value| !value.is_finite()) {
            return Err(PowerError::InvalidRequest(
                "batched output tensor values do not match its finite declared shape".to_string(),
            ));
        }
        let covered = leading_partitions
            .iter()
            .try_fold(0_usize, |total, size| total.checked_add(*size))
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "leading-axis output partition count overflowed".to_string(),
                )
            })?;
        if covered != self.shape[0] {
            return Err(PowerError::InvalidRequest(format!(
                "leading-axis output partitions cover {covered} rows but the tensor contains {}",
                self.shape[0]
            )));
        }
        let row_elements = self.shape[1..]
            .iter()
            .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "leading-axis output row dimensions overflowed".to_string(),
                )
            })?;
        let mut values = self.values.into_iter();
        let mut outputs = Vec::with_capacity(leading_partitions.len());
        for partition in leading_partitions {
            let value_count = partition.checked_mul(row_elements).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "leading-axis output partition dimensions overflowed".to_string(),
                )
            })?;
            let mut shape = self.shape.clone();
            shape[0] = *partition;
            limits.checked_elements(&shape, "output tensor partition")?;
            outputs.push(Self {
                shape,
                values: values.by_ref().take(value_count).collect(),
            });
        }
        if values.next().is_some() || outputs.iter().any(|output| output.values.is_empty()) {
            return Err(PowerError::InvalidRequest(
                "leading-axis output partitioning did not consume the exact tensor".to_string(),
            ));
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_shape_and_values_must_agree() {
        let limits = InferenceLimits::default();
        assert!(TensorInput::new(vec![1, 2], vec![1.0], &limits).is_err());
        assert!(TensorInput::new(vec![1, 2], vec![1.0, f32::NAN], &limits).is_err());
        assert!(TensorInput::new(vec![1, 2], vec![1.0, 2.0], &limits).is_ok());
    }

    #[test]
    fn output_precision_is_never_silently_changed() {
        let limits = InferenceLimits::default();
        let tensor = Tensor::new(&[1_f32], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        assert!(TensorOutput::from_candle(&tensor, &limits).is_err());
    }

    #[test]
    fn leading_axis_batches_preserve_exact_item_order() {
        let limits = InferenceLimits::default();
        let first = TensorInput::new(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0], &limits).unwrap();
        let second = TensorInput::new(vec![1, 1, 2, 2], vec![5.0, 6.0, 7.0, 8.0], &limits).unwrap();

        let batch = TensorInput::stack_leading(vec![first, second], &limits).unwrap();

        assert_eq!(batch.shape, [2, 1, 2, 2]);
        assert_eq!(batch.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn leading_axis_batches_reject_incompatible_items_and_limits() {
        let limits = InferenceLimits::default();
        let first = TensorInput::new(vec![1, 2], vec![1.0, 2.0], &limits).unwrap();
        let incompatible = TensorInput::new(vec![1, 3], vec![3.0, 4.0, 5.0], &limits).unwrap();
        assert!(TensorInput::stack_leading(vec![first, incompatible], &limits).is_err());
        let valid = TensorInput::new(vec![1, 2], vec![1.0, 2.0], &limits).unwrap();
        let malformed = TensorInput {
            shape: Vec::new(),
            values: Vec::new(),
        };
        assert!(TensorInput::stack_leading(vec![valid, malformed], &limits).is_err());

        let tight = InferenceLimits {
            max_tensor_elements: 3,
            ..InferenceLimits::default()
        };
        let first = TensorInput::new(vec![1, 2], vec![1.0, 2.0], &tight).unwrap();
        let second = TensorInput::new(vec![1, 2], vec![3.0, 4.0], &tight).unwrap();
        assert!(TensorInput::stack_leading(vec![first, second], &tight).is_err());
    }

    #[test]
    fn leading_axis_output_slices_preserve_shapes_and_values() {
        let limits = InferenceLimits::default();
        let output = TensorOutput {
            shape: vec![3, 1, 2],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let slices = output.split_leading(&[1, 2], &limits).unwrap();

        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].shape, [1, 1, 2]);
        assert_eq!(slices[0].values, [1.0, 2.0]);
        assert_eq!(slices[1].shape, [2, 1, 2]);
        assert_eq!(slices[1].values, [3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn leading_axis_output_slices_reject_invalid_partitions() {
        let limits = InferenceLimits::default();
        let output = TensorOutput {
            shape: vec![2, 1, 2],
            values: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert!(output.clone().split_leading(&[1], &limits).is_err());
        assert!(output.clone().split_leading(&[1, 0, 1], &limits).is_err());

        let malformed = TensorOutput {
            shape: vec![2, 1, 2],
            values: vec![1.0],
        };
        assert!(malformed.split_leading(&[1, 1], &limits).is_err());
    }
}
