use std::collections::BTreeMap;

use crate::error::{PowerError, Result};
use crate::inference::InferenceLimits;

use super::{validate_name, GraphTensor};

pub(crate) type GraphShapeBindings = BTreeMap<String, usize>;

impl GraphTensor {
    pub(super) fn validate(&self, limits: &InferenceLimits) -> Result<()> {
        validate_name(&self.name, limits)?;
        if self.shape.is_empty() {
            return Err(PowerError::InvalidFormat(
                "static graph tensor shape must contain at least one dimension".to_string(),
            ));
        }
        let mut fixed_elements = 1_usize;
        for dimension in &self.shape {
            match dimension {
                serde_json::Value::Number(_) => {
                    let fixed = fixed_dimension(dimension)?;
                    fixed_elements = fixed_elements.checked_mul(fixed).ok_or_else(|| {
                        PowerError::InvalidFormat(
                            "static graph fixed tensor dimensions overflowed".to_string(),
                        )
                    })?;
                    if fixed_elements > limits.max_tensor_elements {
                        return Err(PowerError::InvalidFormat(
                            "static graph fixed tensor dimensions exceed the tensor element limit"
                                .to_string(),
                        ));
                    }
                }
                serde_json::Value::String(symbol) => {
                    validate_name(symbol, limits).map_err(|_| {
                        PowerError::InvalidFormat(
                            "static graph tensor shape contains an invalid symbolic dimension"
                                .to_string(),
                        )
                    })?
                }
                serde_json::Value::Null => {}
                _ => return Err(PowerError::InvalidFormat(
                    "static graph tensor dimensions must be positive integers, symbols, or null"
                        .to_string(),
                )),
            }
        }
        Ok(())
    }

    pub(super) fn bind_shape(
        &self,
        actual: &[usize],
        bindings: &mut GraphShapeBindings,
        output: bool,
    ) -> Result<()> {
        if actual.len() != self.shape.len() {
            return Err(shape_mismatch(output));
        }
        for (expected, actual) in self.shape.iter().zip(actual) {
            match expected {
                serde_json::Value::Number(_) => {
                    if fixed_dimension(expected)? != *actual {
                        return Err(shape_mismatch(output));
                    }
                }
                serde_json::Value::String(symbol) => match bindings.get(symbol) {
                    Some(bound) if *bound != *actual => return Err(shape_mismatch(output)),
                    Some(_) => {}
                    None => {
                        bindings.insert(symbol.clone(), *actual);
                    }
                },
                serde_json::Value::Null => {}
                _ => {
                    return Err(PowerError::InvalidFormat(
                        "validated static graph tensor lost its shape contract".to_string(),
                    ))
                }
            }
        }
        Ok(())
    }
}

fn fixed_dimension(value: &serde_json::Value) -> Result<usize> {
    let value = value.as_u64().ok_or_else(|| {
        PowerError::InvalidFormat(
            "static graph fixed tensor dimensions must be positive integers".to_string(),
        )
    })?;
    let value = usize::try_from(value).map_err(|_| {
        PowerError::InvalidFormat(
            "static graph fixed tensor dimension exceeds the host address space".to_string(),
        )
    })?;
    if value == 0 {
        return Err(PowerError::InvalidFormat(
            "static graph fixed tensor dimensions must be positive".to_string(),
        ));
    }
    Ok(value)
}

fn shape_mismatch(output: bool) -> PowerError {
    let message = "static graph tensor shape does not match its reviewed shape contract";
    if output {
        PowerError::InferenceFailed(message.to_string())
    } else {
        PowerError::InvalidRequest(message.to_string())
    }
}
