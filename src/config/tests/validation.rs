use super::super::*;
use super::valid_model_signing_key_hex;
use serial_test::serial;

// --- validate() tests ---

#[test]
fn test_validate_default_config_no_warnings() {
    // Default config is valid — validate() should not panic
    let config = PowerConfig::default();
    config.validate().unwrap(); // must not panic
}

#[test]
fn test_validate_keep_alive_valid_formats() {
    // All valid formats should pass without warnings
    for ka in &["0", "-1", "5m", "1h", "30s", "300"] {
        let config = PowerConfig {
            keep_alive: ka.to_string(),
            ..Default::default()
        };
        config.validate().unwrap(); // must not panic
    }
}

#[test]
fn test_validate_rejects_invalid_keep_alive() {
    let config = PowerConfig {
        keep_alive: "later".to_string(),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("invalid keep_alive"));
}

#[test]
fn test_validate_keep_alive_overflow_returns_error() {
    let config = PowerConfig {
        keep_alive: "18446744073709551615h".to_string(),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("invalid keep_alive"));
}

#[test]
fn test_validate_model_signing_key_valid_hex() {
    let config = PowerConfig {
        model_signing_key: Some(valid_model_signing_key_hex()),
        ..Default::default()
    };
    config.validate().unwrap(); // must not panic
}

#[test]
fn test_validate_rejects_model_signing_key_wrong_length() {
    let config = PowerConfig {
        model_signing_key: Some("deadbeef".repeat(4)),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("model_signing_key"));
}

#[test]
fn test_validate_rejects_model_signing_key_non_hex() {
    let config = PowerConfig {
        model_signing_key: Some("z".repeat(64)),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("model_signing_key"));
}

#[test]
fn test_validate_rejects_ra_tls_without_tls_port() {
    let config = PowerConfig {
        ra_tls: true,
        tee_mode: true,
        tls_port: None,
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("tls_port"));
}

#[test]
fn test_validate_rejects_ra_tls_without_tee_mode() {
    let config = PowerConfig {
        ra_tls: true,
        tls_port: Some(11435),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("tee_mode"));
}

#[cfg(not(feature = "tls"))]
#[test]
fn test_validate_rejects_tls_port_without_tls_feature() {
    let config = PowerConfig {
        tls_port: Some(11435),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("tls feature"));
}

#[cfg(feature = "tls")]
#[test]
fn test_validate_ra_tls_with_tls_port_and_tee_mode_is_valid() {
    let config = PowerConfig {
        ra_tls: true,
        tee_mode: true,
        tls_port: Some(11435),
        ..Default::default()
    };
    config.validate().unwrap();
}

#[test]
fn test_validate_rejects_rotating_provider_empty_sources() {
    let config = PowerConfig {
        key_provider: "rotating".to_string(),
        key_rotation_sources: vec![],
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("key_rotation_sources"));
}

#[test]
fn test_validate_rotating_provider_with_sources_is_valid() {
    let config = PowerConfig {
        key_provider: "rotating".to_string(),
        key_rotation_sources: vec![crate::tee::encrypted_model::KeySource::Env(
            "TEST_MODEL_KEY".to_string(),
        )],
        ..Default::default()
    };

    config.validate().unwrap();
}

#[test]
fn test_validate_rejects_unknown_key_provider() {
    let config = PowerConfig {
        key_provider: "vault-ish".to_string(),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("key_provider"));
}

#[test]
fn test_validate_rejects_ambiguous_gpu_evidence_sources() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            evidence_hex: Some("00".to_string()),
            evidence_path: Some(PathBuf::from("/run/a3s/gpu.evidence")),
            ..Default::default()
        },
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("mutually exclusive"));
}

#[test]
fn test_validate_rejects_ambiguous_gpu_verdict_sources() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            evidence_hex: Some("00".to_string()),
            verdict_hex: Some("11".to_string()),
            verdict_path: Some(PathBuf::from("/run/a3s/nras.verdict")),
            ..Default::default()
        },
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("mutually exclusive"));
}

#[test]
fn test_validate_rejects_nvattest_zero_timeout() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_timeout_secs: 0,
            ..Default::default()
        },
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nvattest_timeout_secs"));
}

#[test]
fn test_validate_rejects_invalid_nvattest_verifier() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_verifier: "maybe".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nvattest_verifier"));
}

#[test]
fn test_validate_rejects_invalid_nvattest_gpu_evidence_source() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_gpu_evidence_source: "driver".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nvattest_gpu_evidence_source"));
}

#[test]
fn test_validate_rejects_corelib_nvattest_without_architecture() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_gpu_evidence_source: "corelib".to_string(),
            nvattest_gpu_architecture: None,
            ..Default::default()
        },
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nvattest_gpu_architecture"));
}

#[test]
fn test_validate_accepts_corelib_nvattest_with_architecture() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_gpu_evidence_source: "corelib".to_string(),
            nvattest_gpu_architecture: Some("HOPPER".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    config.validate().unwrap();
}

fn nras_rest_config() -> PowerConfig {
    PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_hex: Some(hex::encode(br#"{"evidence":"ZXZpZGVuY2U"}"#)),
            nras_gpu_architecture: Some("HOPPER".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn test_validate_rejects_nras_rest_without_evidence() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            nras_gpu_architecture: Some("HOPPER".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("evidence"));
}

#[test]
fn test_validate_rejects_nras_rest_with_configured_verdict() {
    let mut config = nras_rest_config();
    config.gpu_attestation.verdict_hex = Some("00".to_string());

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("verdict"));
}

#[test]
fn test_validate_rejects_nras_rest_without_architecture() {
    let mut config = nras_rest_config();
    config.gpu_attestation.nras_gpu_architecture = None;

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nras_gpu_architecture"));
}

#[test]
fn test_validate_rejects_nras_rest_invalid_architecture() {
    let mut config = nras_rest_config();
    config.gpu_attestation.nras_gpu_architecture = Some("ADA".to_string());

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nras_gpu_architecture"));
}

#[test]
fn test_validate_rejects_nras_rest_invalid_claims_version() {
    let mut config = nras_rest_config();
    config.gpu_attestation.nras_claims_version = "4.0".to_string();

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nras_claims_version"));
}

#[test]
fn test_validate_rejects_nras_rest_zero_timeout() {
    let mut config = nras_rest_config();
    config.gpu_attestation.nras_timeout_secs = 0;

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("nras_timeout_secs"));
}

#[test]
fn test_validate_accepts_valid_nras_rest_config() {
    let config = nras_rest_config();

    config.validate().unwrap();
}

#[test]
fn test_validate_rejects_audit_encrypt_without_key_source() {
    let config = PowerConfig {
        audit_log_encrypt: true,
        audit_key_source: None,
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("audit_key_source"));
}

#[test]
fn test_validate_audit_encrypt_with_key_source() {
    let config = PowerConfig {
        audit_log_encrypt: true,
        audit_key_source: Some(crate::tee::encrypted_model::KeySource::Env(
            "TEST_KEY".to_string(),
        )),
        ..Default::default()
    };
    config.validate().unwrap(); // must not panic, no warning
}

#[test]
fn test_validate_streaming_decrypt() {
    let config = PowerConfig {
        streaming_decrypt: true,
        ..Default::default()
    };
    config.validate().unwrap(); // must not panic; warning may be emitted if picolm not enabled
}

#[test]
fn test_validate_rejects_unknown_spec_mode() {
    let config = PowerConfig {
        spec_mode: "warp-speed".to_string(),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("unsupported spec_mode"));
}

#[test]
fn test_validate_rejects_untrained_dspark_alias() {
    let config = PowerConfig {
        spec_mode: "dspark".to_string(),
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("unsupported spec_mode"));
}

#[test]
#[serial]
fn test_load_from_rejects_unknown_spec_mode() {
    std::env::remove_var("A3S_POWER_SPEC_MODE");
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, r#"spec_mode = "warp-speed""#).unwrap();

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    assert!(err.to_string().contains("unsupported spec_mode"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_keep_alive() {
    std::env::remove_var("A3S_POWER_KEEP_ALIVE");
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, r#"keep_alive = "eventually""#).unwrap();

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    assert!(err.to_string().contains("invalid keep_alive"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_keep_alive() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_KEEP_ALIVE", "eventually");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_KEEP_ALIVE");

    assert!(err.to_string().contains("invalid keep_alive"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_model_signing_key() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, r#"model_signing_key = "not-a-public-key""#).unwrap();

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();

    assert!(err.to_string().contains("model_signing_key"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_tee_policy_mode() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_TEE_POLICY_MODE", "gpu-conf");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_TEE_POLICY_MODE");

    assert!(err.to_string().contains("A3S_POWER_TEE_POLICY_MODE"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_gpu_attestation_source() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_SOURCE", "sdk-maybe");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_SOURCE");

    assert!(err.to_string().contains("A3S_POWER_GPU_ATTESTATION_SOURCE"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_tee_mode() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_TEE_MODE", "definitely");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_TEE_MODE");

    assert!(err.to_string().contains("A3S_POWER_TEE_MODE"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_redact_logs() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_REDACT_LOGS", "sometimes");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_REDACT_LOGS");

    assert!(err.to_string().contains("A3S_POWER_REDACT_LOGS"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_proxy_effective_prompt_digest() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST", "enabled-ish");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST");

    assert!(err
        .to_string()
        .contains("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_proxy_effective_prompt_digest_required() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var(
        "A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED",
        "required-ish",
    );

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED");

    assert!(err
        .to_string()
        .contains("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_ra_tls() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_RA_TLS", "maybe");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_RA_TLS");

    assert!(err.to_string().contains("A3S_POWER_RA_TLS"));
}

#[test]
#[serial]
fn test_load_from_rejects_invalid_env_audit_log() {
    let dir = tempfile::tempdir().unwrap();
    let acl_path = dir.path().join("config.acl");
    std::fs::write(&acl_path, "").unwrap();
    std::env::set_var("A3S_POWER_AUDIT_LOG", "audit-ish");

    let err = PowerConfig::load_from(acl_path.to_str().unwrap()).unwrap_err();
    std::env::remove_var("A3S_POWER_AUDIT_LOG");

    assert!(err.to_string().contains("A3S_POWER_AUDIT_LOG"));
}
