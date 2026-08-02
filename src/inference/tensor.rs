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
}
