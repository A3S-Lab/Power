use std::time::Duration;

use crate::error::{PowerError, Result};

pub(super) fn tensor_bytes(elements: usize, label: &str) -> Result<u64> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph {label} tensor byte count overflowed"
            ))
        })?;
    u64::try_from(bytes).map_err(|_| {
        PowerError::InferenceFailed(format!(
            "static graph {label} tensor byte count exceeds u64"
        ))
    })
}

pub(super) fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
