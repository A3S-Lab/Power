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
    ) -> Result<Self> {
        let tensor = store.load(&initializer.name, &Device::Cpu)?;
        match tensor.dtype() {
            DType::I64 => Ok(Self::Ints {
                values: tensor
                    .flatten_all()
                    .and_then(|value| value.to_vec1::<i64>())
                    .map_err(value_error)?,
                shape: tensor.dims().to_vec(),
            }),
            DType::I32 => Ok(Self::Ints {
                values: tensor
                    .flatten_all()
                    .and_then(|value| value.to_vec1::<i32>())
                    .map_err(value_error)?
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                shape: tensor.dims().to_vec(),
            }),
            _ => Ok(Self::Tensor(tensor.to_device(device).map_err(value_error)?)),
        }
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

fn value_error(error: candle_core::Error) -> PowerError {
    PowerError::InvalidFormat(format!("failed to load static graph initializer: {error}"))
}
