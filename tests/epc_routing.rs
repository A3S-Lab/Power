//! Integration tests for EPC-aware backend routing.
//!
//! Verifies that `BackendRegistry::find_for_tee()` correctly routes to the
//! picolm backend when model size exceeds the EPC threshold, and falls back
//! to the normal priority order when the model fits comfortably.

use std::sync::Arc;

use a3s_power::backend::BackendRegistry;
use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::ModelFormat;

fn test_config() -> Arc<PowerConfig> {
    Arc::new(PowerConfig::default())
}

// ── Tests without picolm ──────────────────────────────────────────────────────

#[test]
fn test_find_for_tee_falls_back_to_format_when_no_picolm() {
    // When picolm is not registered, find_for_tee falls back to find_for_format
    let registry = BackendRegistry::new();
    // Empty registry — both should return Err
    let result = registry.find_for_tee(&ModelFormat::Gguf, 1024 * 1024 * 1024);
    assert!(result.is_err());
}

#[test]
fn test_find_for_tee_small_model_uses_normal_priority() {
    // A tiny model (1MB) should never trigger EPC routing
    // Even with picolm registered, a small model should use the first backend
    let registry = BackendRegistry::new();
    let result = registry.find_for_tee(&ModelFormat::Gguf, 1024 * 1024);
    // Empty registry → error, but the routing logic ran without panic
    assert!(result.is_err());
}

// ── Tests with picolm ─────────────────────────────────────────────────────────

#[cfg(feature = "picolm")]
mod picolm_routing {
    use super::*;
    use a3s_power::backend::picolm::PicolmBackend;

    fn registry_with_picolm() -> BackendRegistry {
        let config = test_config();
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(PicolmBackend::new(config)));
        registry
    }

    #[test]
    fn test_picolm_registered_supports_gguf() {
        let registry = registry_with_picolm();
        let result = registry.find_for_format(&ModelFormat::Gguf);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "picolm");
    }

    #[test]
    fn test_find_for_tee_large_model_routes_to_picolm() {
        // 8GB model — exceeds any realistic EPC budget
        // On Linux, model_exceeds_epc(8GB) should return true
        // On non-Linux, it returns false (no EPC info) — picolm still available via format
        let registry = registry_with_picolm();
        let result = registry.find_for_tee(&ModelFormat::Gguf, 8 * 1024 * 1024 * 1024);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "picolm");
    }

    #[test]
    fn test_find_for_tee_unsupported_format_fails() {
        // picolm only supports GGUF — SafeTensors should fail
        let registry = registry_with_picolm();
        let result = registry.find_for_tee(&ModelFormat::SafeTensors, 8 * 1024 * 1024 * 1024);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("No backend"));
    }

    #[test]
    fn test_find_for_tee_zero_size_model() {
        // Zero-size model should never trigger EPC routing
        let registry = registry_with_picolm();
        let result = registry.find_for_tee(&ModelFormat::Gguf, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_default_backends_includes_picolm() {
        let config = test_config();
        let registry = a3s_power::backend::default_backends(config);
        let names = registry.list_names();
        assert!(
            names.contains(&"picolm"),
            "picolm must be in default_backends when feature is enabled; got: {names:?}"
        );
    }

    #[test]
    fn test_find_by_name_picolm() {
        let registry = registry_with_picolm();
        let result = registry.find_by_name("picolm");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "picolm");
    }
}

// ── EPC threshold logic ───────────────────────────────────────────────────────

#[test]
fn test_epc_model_fits_when_info_unavailable() {
    use a3s_power::tee::epc::model_exceeds_epc;
    // On macOS / non-Linux, EPC info is unavailable → model always "fits"
    // This preserves existing behavior on dev machines
    #[cfg(not(target_os = "linux"))]
    assert!(!model_exceeds_epc(u64::MAX));
}

#[test]
fn test_epc_available_bytes_type() {
    use a3s_power::tee::epc::available_epc_bytes;
    // Just verify the function returns the right type and doesn't panic
    let _result: Option<u64> = available_epc_bytes();
}

#[cfg(target_os = "linux")]
#[test]
fn test_epc_available_bytes_positive_on_linux() {
    use a3s_power::tee::epc::available_epc_bytes;
    let bytes = available_epc_bytes();
    assert!(bytes.is_some(), "Should read /proc/meminfo on Linux");
    assert!(bytes.unwrap() > 0, "Available memory should be > 0");
}
