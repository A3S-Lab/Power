use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::error::{PowerError, Result};

/// Hard resource bounds applied before embedded model allocation or execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InferenceLimits {
    pub max_model_files: usize,
    #[serde(default = "default_max_weight_sources")]
    pub max_weight_sources: usize,
    pub max_model_bytes: u64,
    pub max_resident_weight_bytes: u64,
    pub max_state_bytes: u64,
    pub max_input_bytes: usize,
    pub max_image_pixels: u64,
    pub max_tensor_elements: usize,
    pub max_graph_plan_bytes: usize,
    pub max_graph_nodes: usize,
    pub max_graph_initializers: usize,
    pub max_graph_name_bytes: usize,
    pub max_context_tokens: usize,
    pub max_generated_tokens: usize,
    pub max_concurrent_requests: usize,
    #[serde(default = "default_max_queued_requests")]
    pub max_queued_requests: usize,
}

impl Default for InferenceLimits {
    fn default() -> Self {
        Self {
            max_model_files: 512,
            max_weight_sources: default_max_weight_sources(),
            max_model_bytes: 16 * 1024 * 1024 * 1024,
            max_resident_weight_bytes: 16 * 1024 * 1024 * 1024,
            max_state_bytes: 4 * 1024 * 1024 * 1024,
            max_input_bytes: 64 * 1024 * 1024,
            max_image_pixels: 64 * 1024 * 1024,
            max_tensor_elements: 256 * 1024 * 1024,
            max_graph_plan_bytes: 32 * 1024 * 1024,
            max_graph_nodes: 16_384,
            max_graph_initializers: 65_536,
            max_graph_name_bytes: 1_024,
            max_context_tokens: 32_768,
            max_generated_tokens: 32_768,
            max_concurrent_requests: 1,
            max_queued_requests: default_max_queued_requests(),
        }
    }
}

const fn default_max_weight_sources() -> usize {
    8
}

const fn default_max_queued_requests() -> usize {
    32
}

impl InferenceLimits {
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("max_model_files", self.max_model_files),
            ("max_weight_sources", self.max_weight_sources),
            ("max_input_bytes", self.max_input_bytes),
            ("max_tensor_elements", self.max_tensor_elements),
            ("max_graph_plan_bytes", self.max_graph_plan_bytes),
            ("max_graph_nodes", self.max_graph_nodes),
            ("max_graph_initializers", self.max_graph_initializers),
            ("max_graph_name_bytes", self.max_graph_name_bytes),
            ("max_context_tokens", self.max_context_tokens),
            ("max_generated_tokens", self.max_generated_tokens),
            ("max_concurrent_requests", self.max_concurrent_requests),
        ];
        if let Some((name, _)) = positive.into_iter().find(|(_, value)| *value == 0) {
            return Err(PowerError::Config(format!(
                "embedded inference {name} must be greater than zero"
            )));
        }
        if self.max_model_bytes == 0
            || self.max_resident_weight_bytes == 0
            || self.max_state_bytes == 0
            || self.max_image_pixels == 0
        {
            return Err(PowerError::Config(
                "embedded inference model, residency, state, and pixel limits must be greater than zero"
                    .to_string(),
            ));
        }
        if self.max_concurrent_requests > Semaphore::MAX_PERMITS
            || self.max_queued_requests > Semaphore::MAX_PERMITS
        {
            return Err(PowerError::Config(format!(
                "embedded inference active and waiting request limits cannot exceed {}",
                Semaphore::MAX_PERMITS
            )));
        }
        Ok(())
    }

    pub fn checked_elements(&self, shape: &[usize], label: &str) -> Result<usize> {
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

    pub fn checked_state_bytes(&self, bytes: u64, label: &str) -> Result<u64> {
        if bytes > self.max_state_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "{label} requires {bytes} bytes, exceeding the {} byte state limit",
                self.max_state_bytes
            )));
        }
        Ok(bytes)
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

    #[test]
    fn zero_waiting_capacity_keeps_fail_fast_admission_valid() {
        let limits = InferenceLimits {
            max_queued_requests: 0,
            ..InferenceLimits::default()
        };
        limits.validate().unwrap();
    }

    #[test]
    fn admission_capacity_must_fit_the_runtime_semaphore() {
        let active = InferenceLimits {
            max_concurrent_requests: Semaphore::MAX_PERMITS + 1,
            ..InferenceLimits::default()
        };
        assert!(active.validate().is_err());

        let waiting = InferenceLimits {
            max_queued_requests: Semaphore::MAX_PERMITS + 1,
            ..InferenceLimits::default()
        };
        assert!(waiting.validate().is_err());
    }

    #[test]
    fn older_serialized_limits_receive_the_bounded_queue_default() {
        let expected = InferenceLimits::default();
        let mut encoded = serde_json::to_value(&expected).unwrap();
        encoded.as_object_mut().unwrap().remove("maxQueuedRequests");
        let decoded: InferenceLimits = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.max_queued_requests, expected.max_queued_requests);
    }

    #[test]
    fn state_bytes_are_bounded() {
        let limits = InferenceLimits {
            max_state_bytes: 128,
            ..InferenceLimits::default()
        };
        assert_eq!(limits.checked_state_bytes(128, "KV cache").unwrap(), 128);
        assert!(limits.checked_state_bytes(129, "KV cache").is_err());
    }
}
