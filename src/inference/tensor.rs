use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::InferenceLimits;

mod input_upload;

use input_upload::InputUploadGuard;
pub(crate) use input_upload::InputUploadPool;

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

    pub(crate) fn into_candle(
        self,
        device: &Device,
        limits: &InferenceLimits,
        upload_pool: &InputUploadPool,
    ) -> Result<(Tensor, InputUploadGuard)> {
        self.validate(limits, "input tensor")?;
        input_upload::materialize(
            self.values,
            self.shape,
            device,
            upload_pool,
            limits.max_input_bytes,
            "input tensor",
        )
    }

    /// Materializes a bounded execution window before any graph work is
    /// enqueued. This ordering is important for accelerators whose ordinary
    /// host buffers require a stream synchronization when an upload returns:
    /// all uploads finish first, so a later upload cannot fence an earlier
    /// graph execution.
    pub(crate) fn into_candle_many(
        inputs: Vec<Self>,
        device: &Device,
        limits: &InferenceLimits,
        upload_pool: &InputUploadPool,
    ) -> Result<(Vec<Tensor>, InputUploadGuard)> {
        Self::validate_many(&inputs, limits)?;
        if device.is_cuda() && inputs.iter().all(|input| !input.values.is_empty()) {
            return Self::into_one_cuda_storage(inputs, device, limits, upload_pool);
        }
        let mut tensors = Vec::with_capacity(inputs.len());
        let mut uploads = InputUploadGuard::default();
        for input in inputs {
            let (tensor, upload) = input.into_candle(device, limits, upload_pool)?;
            tensors.push(tensor);
            uploads.append(upload);
        }
        Ok((tensors, uploads))
    }

    fn into_one_cuda_storage(
        inputs: Vec<Self>,
        device: &Device,
        limits: &InferenceLimits,
        upload_pool: &InputUploadPool,
    ) -> Result<(Vec<Tensor>, InputUploadGuard)> {
        let total_elements = inputs.iter().try_fold(0_usize, |total, input| {
            total.checked_add(input.values.len()).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "tensor execution window input element count overflowed".to_string(),
                )
            })
        })?;

        let mut values = Vec::with_capacity(total_elements);
        let mut partitions = Vec::with_capacity(inputs.len());
        for input in inputs {
            partitions.push((input.shape, input.values.len()));
            values.extend(input.values);
        }
        let (aggregate, upload) = input_upload::materialize(
            values,
            vec![total_elements],
            device,
            upload_pool,
            limits.max_input_bytes,
            "the CUDA input window",
        )?;
        let tensors = restore_cuda_input_partitions(aggregate, partitions, total_elements)?;
        Ok((tensors, upload))
    }

    fn validate(&self, limits: &InferenceLimits, label: &str) -> Result<()> {
        let elements = limits.checked_elements(&self.shape, label)?;
        if self.values.len() != elements || self.values.iter().any(|value| !value.is_finite()) {
            return Err(PowerError::InvalidRequest(format!(
                "{label} values do not match its finite declared shape"
            )));
        }
        Ok(())
    }

    pub(crate) fn validate_many(inputs: &[Self], limits: &InferenceLimits) -> Result<()> {
        if inputs.is_empty() {
            return Err(PowerError::InvalidRequest(
                "a tensor execution window requires at least one input".to_string(),
            ));
        }
        let mut total_elements = 0_usize;
        for input in inputs {
            input.validate(limits, "window input tensor")?;
            let elements = input.values.len();
            total_elements = total_elements.checked_add(elements).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "tensor execution window input element count overflowed".to_string(),
                )
            })?;
        }
        if total_elements > limits.max_tensor_elements {
            return Err(PowerError::InvalidRequest(format!(
                "tensor execution window contains {total_elements} input elements, exceeding the {} element limit",
                limits.max_tensor_elements
            )));
        }
        let total_bytes = total_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "tensor execution window input byte count overflowed".to_string(),
                )
            })?;
        if total_bytes > limits.max_input_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "tensor execution window contains {total_bytes} input bytes, exceeding the {} byte limit",
                limits.max_input_bytes
            )));
        }
        Ok(())
    }
}

fn restore_cuda_input_partitions(
    aggregate: Tensor,
    partitions: Vec<(Vec<usize>, usize)>,
    total_elements: usize,
) -> Result<Vec<Tensor>> {
    let mut offset = 0_usize;
    let mut tensors = Vec::with_capacity(partitions.len());
    for (shape, elements) in partitions {
        let end = offset.checked_add(elements).ok_or_else(|| {
            PowerError::InferenceFailed("CUDA input window partition offset overflowed".to_string())
        })?;
        let tensor = aggregate
            .narrow(0, offset, elements)
            .and_then(|tensor| tensor.reshape(shape.as_slice()))
            .map_err(|error| {
                PowerError::InferenceFailed(format!(
                    "failed to restore a CUDA input window tensor: {error}"
                ))
            })?;
        tensors.push(tensor);
        offset = end;
    }
    if offset != total_elements {
        return Err(PowerError::InferenceFailed(
            "CUDA input window partitions did not cover every value".to_string(),
        ));
    }
    Ok(tensors)
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
    pub(crate) fn from_f32_values(
        shape: Vec<usize>,
        values: Vec<f32>,
        limits: &InferenceLimits,
    ) -> Result<Self> {
        let expected = limits.checked_elements(&shape, "output tensor")?;
        if values.len() != expected {
            return Err(PowerError::InferenceFailed(format!(
                "embedded inference returned {} output values for shape {shape:?}, expected {expected}",
                values.len(),
            )));
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
        debug_assert_eq!(values.len(), expected);
        Self::from_f32_values(shape, values, limits)
    }

    /// Materializes a bounded set of device-resident F32 outputs only after
    /// the complete execution window has been submitted, preserving exact
    /// tensor shapes and order.
    pub(crate) fn from_candle_many(
        tensors: Vec<Tensor>,
        limits: &InferenceLimits,
    ) -> Result<Vec<Self>> {
        let first = tensors.first().ok_or_else(|| {
            PowerError::InvalidRequest(
                "a tensor execution window requires at least one output".to_string(),
            )
        })?;
        let mut shapes = Vec::with_capacity(tensors.len());
        let mut total_elements = 0_usize;
        for tensor in &tensors {
            if tensor.dtype() != DType::F32 {
                return Err(PowerError::InvalidFormat(format!(
                    "static graph output must be F32, found {:?}",
                    tensor.dtype()
                )));
            }
            if !tensor.device().same_device(first.device()) {
                return Err(PowerError::InferenceFailed(
                    "window output tensors use different devices".to_string(),
                ));
            }
            let shape = tensor.dims().to_vec();
            let elements = limits.checked_elements(&shape, "window output tensor")?;
            total_elements = total_elements.checked_add(elements).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "tensor execution window output element count overflowed".to_string(),
                )
            })?;
            shapes.push((shape, elements));
        }
        if total_elements > limits.max_tensor_elements {
            return Err(PowerError::InvalidRequest(format!(
                "tensor execution window contains {total_elements} output elements, exceeding the {} element limit",
                limits.max_tensor_elements
            )));
        }
        debug_assert_eq!(shapes.len(), tensors.len());
        tensors
            .iter()
            .map(|tensor| Self::from_candle(tensor, limits))
            .collect()
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

    #[test]
    fn execution_window_inputs_obey_aggregate_byte_and_element_limits() {
        let limits = InferenceLimits {
            max_input_bytes: 16,
            max_tensor_elements: 8,
            ..InferenceLimits::default()
        };
        let upload_pool = InputUploadPool::new(limits.max_input_bytes);
        let inputs = vec![
            TensorInput::new(vec![1, 2], vec![1.0, 2.0], &limits).unwrap(),
            TensorInput::new(vec![1, 2], vec![3.0, 4.0], &limits).unwrap(),
        ];
        let (tensors, upload_guard) =
            TensorInput::into_candle_many(inputs, &Device::Cpu, &limits, &upload_pool).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(upload_guard.pinned_upload_count(), 0);

        let inputs = vec![
            TensorInput::new(vec![1, 2], vec![1.0, 2.0], &limits).unwrap(),
            TensorInput::new(vec![1, 3], vec![3.0, 4.0, 5.0], &limits).unwrap(),
        ];
        assert!(
            TensorInput::into_candle_many(inputs, &Device::Cpu, &limits, &upload_pool).is_err()
        );
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn cuda_pinned_input_upload_is_bounded_stream_safe_and_exact() {
        let limits = InferenceLimits::default();
        let upload_pool = InputUploadPool::new(limits.max_input_bytes);
        let device = Device::new_cuda_with_stream(0).unwrap();
        let Device::Cuda(cuda) = &device else {
            panic!("explicit CUDA device resolved another backend");
        };
        // SAFETY: this test owns the device and its one stream. Disabling
        // cudarc's optional tensor event tracking proves that the pinned
        // host allocation's own completion event is sufficient on its own.
        unsafe { cuda.disable_event_tracking() };

        let (single, single_upload) = TensorInput::new(vec![1, 3], vec![7.0, 8.0, 9.0], &limits)
            .unwrap()
            .into_candle(&device, &limits, &upload_pool)
            .unwrap();
        assert_eq!(single_upload.pinned_upload_count(), 1);
        drop(single_upload);
        assert_eq!(
            single.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [7.0, 8.0, 9.0]
        );

        let inputs = vec![
            TensorInput::new(vec![1, 2], vec![1.0, 2.0], &limits).unwrap(),
            TensorInput::new(vec![1, 1, 3], vec![3.0, 4.0, 5.0], &limits).unwrap(),
        ];

        let (tensors, upload_guard) =
            TensorInput::into_candle_many(inputs, &device, &limits, &upload_pool).unwrap();

        assert_eq!(upload_guard.pinned_upload_count(), 1);
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].dims(), [1, 2]);
        assert_eq!(tensors[1].dims(), [1, 1, 3]);
        assert_eq!(
            tensors[0].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [1.0, 2.0]
        );
        assert_eq!(
            tensors[1].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [3.0, 4.0, 5.0]
        );
        assert_eq!(tensors[0].layout().start_offset(), 0);
        assert_eq!(tensors[1].layout().start_offset(), 2);
        assert!(tensors[0].device().same_device(tensors[1].device()));

        let tight = InferenceLimits {
            max_input_bytes: std::mem::size_of::<f32>(),
            ..limits
        };
        let (pageable, pageable_upload) = TensorInput::new(vec![1, 2], vec![10.0, 11.0], &tight)
            .unwrap()
            .into_candle(&device, &tight, &upload_pool)
            .unwrap();
        assert_eq!(pageable_upload.pinned_upload_count(), 0);
        assert_eq!(pageable.to_vec2::<f32>().unwrap(), [[10.0, 11.0]]);

        let allocations_before_reuse = upload_pool.allocation_count();
        drop(upload_guard);
        let reuse_limits = InferenceLimits::default();
        let repeated_inputs = vec![
            TensorInput::new(vec![1, 2], vec![6.0, 7.0], &reuse_limits).unwrap(),
            TensorInput::new(vec![1, 1, 3], vec![8.0, 9.0, 10.0], &reuse_limits).unwrap(),
        ];
        let (repeated, repeated_upload) =
            TensorInput::into_candle_many(repeated_inputs, &device, &reuse_limits, &upload_pool)
                .unwrap();
        assert_eq!(repeated[0].to_vec2::<f32>().unwrap(), [[6.0, 7.0]]);
        assert_eq!(repeated[1].to_vec3::<f32>().unwrap(), [[[8.0, 9.0, 10.0]]]);
        drop(repeated_upload);
        assert_eq!(upload_pool.allocation_count(), allocations_before_reuse);
        assert!(upload_pool.retained_bytes() <= reuse_limits.max_input_bytes);
        cuda.cuda_stream().synchronize().unwrap();
        cuda.cuda_stream().context().check_err().unwrap();
    }

    #[test]
    fn execution_window_outputs_preserve_exact_shapes_values_and_order() {
        let limits = InferenceLimits::default();
        let tensors = vec![
            Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &Device::Cpu).unwrap(),
            Tensor::from_vec(vec![3.0_f32, 4.0, 5.0], (1, 1, 3), &Device::Cpu).unwrap(),
        ];

        let outputs = TensorOutput::from_candle_many(tensors, &limits).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape, [1, 2]);
        assert_eq!(outputs[0].values, [1.0, 2.0]);
        assert_eq!(outputs[1].shape, [1, 1, 3]);
        assert_eq!(outputs[1].values, [3.0, 4.0, 5.0]);
    }
}
