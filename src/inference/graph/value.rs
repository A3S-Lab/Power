use candle_core::{DType, Device, Tensor};

use crate::error::{PowerError, Result};

use super::super::WeightStore;
use super::plan::Initializer;

#[derive(Clone)]
pub(super) enum GraphValue {
    Tensor(Tensor),
    Ints { values: Vec<i64>, shape: Vec<usize> },
}

impl GraphValue {
    pub(super) fn load(
        initializer: &Initializer,
        store: &WeightStore,
        device: &Device,
    ) -> Result<(Self, Option<f32>)> {
        let tensor = store.load(&initializer.name, &Device::Cpu)?;
        let scalar_f32 = f32_scalar(&tensor)?;
        let value = match tensor.dtype() {
            DType::I64 => Self::Ints {
                values: tensor
                    .flatten_all()
                    .and_then(|value| value.to_vec1::<i64>())
                    .map_err(value_error)?,
                shape: tensor.dims().to_vec(),
            },
            DType::I32 => Self::Ints {
                values: tensor
                    .flatten_all()
                    .and_then(|value| value.to_vec1::<i32>())
                    .map_err(value_error)?
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                shape: tensor.dims().to_vec(),
            },
            _ => Self::Tensor(tensor.to_device(device).map_err(value_error)?),
        };
        Ok((value, scalar_f32))
    }

    pub(super) fn tensor(&self, node: &str) -> Result<&Tensor> {
        match self {
            Self::Tensor(value) => Ok(value),
            Self::Ints { .. } => Err(PowerError::InvalidFormat(format!(
                "static graph node '{node}' expected a tensor value"
            ))),
        }
    }

    pub(super) fn ints(&self, node: &str) -> Result<&[i64]> {
        match self {
            Self::Ints { values, .. } => Ok(values),
            Self::Tensor(_) => Err(PowerError::InvalidFormat(format!(
                "static graph node '{node}' expected an integer control value"
            ))),
        }
    }

    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(value) => value.dims(),
            Self::Ints { shape, .. } => shape,
        }
    }
}

fn f32_scalar(tensor: &Tensor) -> Result<Option<f32>> {
    if tensor.dtype() != DType::F32 || tensor.elem_count() != 1 {
        return Ok(None);
    }
    tensor
        .reshape(())
        .and_then(|value| value.to_scalar::<f32>())
        .map(Some)
        .map_err(value_error)
}

fn value_error(error: candle_core::Error) -> PowerError {
    PowerError::InvalidFormat(format!("failed to load static graph initializer: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_ranked_single_element_f32_initializers_as_scalars() {
        let scalar = Tensor::new(&[0.5_f32], &Device::Cpu).unwrap();
        let vector = Tensor::new(&[0.5_f32, 1.0], &Device::Cpu).unwrap();
        let integer = Tensor::new(&[1_i64], &Device::Cpu).unwrap();

        assert_eq!(f32_scalar(&scalar).unwrap(), Some(0.5));
        assert_eq!(f32_scalar(&vector).unwrap(), None);
        assert_eq!(f32_scalar(&integer).unwrap(), None);
    }
}
