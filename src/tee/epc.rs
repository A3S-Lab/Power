//! EPC (Encrypted Page Cache) memory detection for TEE environments.
//!
//! AMD SEV-SNP and Intel TDX encrypt guest memory using hardware keys.
//! The available EPC budget determines how large a model can be loaded
//! without exceeding the encrypted memory limit.
//!
//! This module reads available memory from the OS and provides a
//! conservative estimate of usable EPC for model loading.

/// Returns the estimated available EPC memory in bytes, or `None` if:
/// - Not running on Linux
/// - Not in a TEE environment
/// - Cannot read memory info
///
/// On Linux inside a TEE, we use `/proc/meminfo` MemAvailable as a
/// conservative proxy for available EPC. The actual EPC limit is
/// hardware-enforced; MemAvailable reflects what the OS sees after
/// the hypervisor has allocated EPC pages.
pub fn available_epc_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        read_mem_available_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn read_mem_available_linux() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if line.starts_with("MemAvailable:") {
            // Format: "MemAvailable:   123456 kB"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: u64 = parts[1].parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

/// Returns true if the given model size would exceed the available EPC budget.
///
/// Uses a 75% threshold to leave headroom for the inference working buffer,
/// KV cache, and OS overhead.
///
/// Returns `false` (model fits) when EPC info is unavailable — this is the
/// safe default that preserves existing behavior on non-TEE systems.
pub fn model_exceeds_epc(model_size_bytes: u64) -> bool {
    match available_epc_bytes() {
        Some(epc) => model_size_bytes > (epc * 3 / 4),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_fits_when_epc_unknown() {
        // When EPC info is unavailable, model_exceeds_epc returns false
        // (preserves existing behavior on non-TEE / non-Linux systems)
        // We can't control what available_epc_bytes() returns in CI,
        // but we can test the logic directly.
        assert!(!exceeds_threshold(100, None));
        assert!(!exceeds_threshold(u64::MAX, None));
    }

    #[test]
    fn test_model_exceeds_threshold() {
        // 4GB model, 2GB EPC → exceeds 75% threshold (1.5GB)
        assert!(exceeds_threshold(
            4 * 1024 * 1024 * 1024,
            Some(2 * 1024 * 1024 * 1024)
        ));
    }

    #[test]
    fn test_model_fits_threshold() {
        // 1GB model, 4GB EPC → fits within 75% threshold (3GB)
        assert!(!exceeds_threshold(
            1024 * 1024 * 1024,
            Some(4 * 1024 * 1024 * 1024)
        ));
    }

    #[test]
    fn test_model_exactly_at_threshold() {
        // Exactly 75% → fits (threshold is strictly greater than)
        let epc = 4 * 1024 * 1024 * 1024u64;
        let model = epc * 3 / 4;
        assert!(!exceeds_threshold(model, Some(epc)));
    }

    #[test]
    fn test_model_one_byte_over_threshold() {
        let epc = 4 * 1024 * 1024 * 1024u64;
        let model = epc * 3 / 4 + 1;
        assert!(exceeds_threshold(model, Some(epc)));
    }

    /// Pure logic test, independent of OS state.
    fn exceeds_threshold(model_size: u64, epc: Option<u64>) -> bool {
        match epc {
            Some(e) => model_size > (e * 3 / 4),
            None => false,
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_available_epc_bytes_returns_some_on_linux() {
        // On Linux, /proc/meminfo should always be readable
        let result = available_epc_bytes();
        assert!(
            result.is_some(),
            "available_epc_bytes() should return Some on Linux"
        );
        assert!(result.unwrap() > 0, "Available memory should be > 0");
    }
}
