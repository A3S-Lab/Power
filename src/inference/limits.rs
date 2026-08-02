use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

/// Hard resource bounds applied before embedded model allocation or execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InferenceLimits {
    pub max_model_files: usize,
    pub max_model_bytes: u64,
    pub max_input_bytes: usize,
    pub max_image_pixels: u64,
    pub max_tensor_elements: usize,
    pub max_context_tokens: usize,
    pub max_generated_tokens: usize,
    pub max_concurrent_requests: usize,
}

impl Default for InferenceLimits {
    fn default() -> Self {
        Self {
            max_model_files: 512,
            max_model_bytes: 16 * 1024 * 1024 * 1024,
            max_input_bytes: 64 * 1024 * 1024,
            max_image_pixels: 64 * 1024 * 1024,
            max_tensor_elements: 256 * 1024 * 1024,
            max_context_tokens: 32_768,
            max_generated_tokens: 32_768,
            max_concurrent_requests: 1,
        }
    }
}

impl InferenceLimits {
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("max_model_files", self.max_model_files),
            ("max_input_bytes", self.max_input_bytes),
            ("max_tensor_elements", self.max_tensor_elements),
            ("max_context_tokens", self.max_context_tokens),
            ("max_generated_tokens", self.max_generated_tokens),
            ("max_concurrent_requests", self.max_concurrent_requests),
        ];
        if let Some((name, _)) = positive.into_iter().find(|(_, value)| *value == 0) {
            return Err(PowerError::Config(format!(
                "embedded inference {name} must be greater than zero"
            )));
        }
        if self.max_model_bytes == 0 || self.max_image_pixels == 0 {
            return Err(PowerError::Config(
                "embedded inference byte and pixel limits must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn checked_elements(&self, shape: &[usize], label: &str) -> Result<usize> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(PowerError::InvalidRequest(format!(
                "{label} must have a non-empty shape with positive dimensions"
            )));
        }
        let elements = shape.iter().try_fold(1_usize, |product, dimension| {
            product.checked_mul(*dimension)
        });
        let elements = elements
            .ok_or_else(|| PowerError::InvalidRequest(format!("{label} dimensions overflowed")))?;
        if elements > self.max_tensor_elements {
            return Err(PowerError::InvalidRequest(format!(
                "{label} contains {elements} elements, exceeding the {} element limit",
                self.max_tensor_elements
            )));
        }
        Ok(elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_element_arithmetic_is_bounded() {
        let limits = InferenceLimits {
            max_tensor_elements: 12,
            ..InferenceLimits::default()
        };
        assert_eq!(limits.checked_elements(&[1, 3, 4], "input").unwrap(), 12);
        assert!(limits.checked_elements(&[1, 3, 5], "input").is_err());
        assert!(limits.checked_elements(&[usize::MAX, 2], "input").is_err());
    }

    #[test]
    fn zero_concurrency_is_rejected() {
        let limits = InferenceLimits {
            max_concurrent_requests: 0,
            ..InferenceLimits::default()
        };
        assert!(limits.validate().is_err());
    }
}
