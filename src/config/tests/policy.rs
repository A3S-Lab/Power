use super::super::*;
use super::valid_model_signing_key_hex;
use serial_test::serial;

#[test]
fn test_config_new_fields_defaults() {
    let config = PowerConfig::default();
    assert!(!config.use_mlock);
    assert!(config.num_thread.is_none());
    assert!(!config.flash_attention);
    assert_eq!(config.num_parallel, 1);
}

#[test]
fn test_config_tee_fields_from_acl() {
    let acl_str = r#"
            tee_mode = true
            tee_policy_mode = "development"
            redact_logs = true

            model_hash "llama3" {
                digest = "sha256:abc123"
            }
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert!(config.tee_mode);
    assert_eq!(config.tee_policy_mode, TeePolicyMode::Development);
    assert!(config.redact_logs);
    assert_eq!(
        config.model_hashes.get("llama3"),
        Some(&"sha256:abc123".to_string())
    );
}

#[test]
fn test_gpu_config_tensor_split_default_empty() {
    let config = GpuConfig::default();
    assert!(config.tensor_split.is_empty());
}

#[test]
fn test_gpu_config_tensor_split_from_acl() {
    let acl_str = r#"
            host = "127.0.0.1"
            port = 11434

            gpu {
                gpu_layers = -1
                tensor_split = [0.5, 0.5]
            }
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.gpu.tensor_split, vec![0.5, 0.5]);
}

#[test]
fn test_gpu_config_tensor_split_serialization_skips_empty() {
    let config = PowerConfig::default();
    let serialized = config.to_acl().unwrap();
    assert!(!serialized.contains("tensor_split"));
}

#[test]
fn test_config_acl_invalid() {
    let result: std::result::Result<PowerConfig, _> = acl::deserialize("{{{{ invalid");
    assert!(result.is_err());
}

#[test]
fn test_tls_port_defaults_to_none() {
    let config = PowerConfig::default();
    assert!(config.tls_port.is_none());
}

#[test]
fn test_ra_tls_defaults_to_false() {
    let config = PowerConfig::default();
    assert!(!config.ra_tls);
}

#[test]
fn test_tls_port_from_acl() {
    let acl_str = r#"tls_port = 8443"#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.tls_port, Some(8443));
}

#[test]
fn test_ra_tls_from_acl() {
    let acl_str = r#"ra_tls = true"#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert!(config.ra_tls);
}

#[test]
fn test_tls_port_not_serialized_when_none() {
    let config = PowerConfig::default();
    let serialized = config.to_acl().unwrap();
    assert!(!serialized.contains("tls_port"));
}

#[test]
fn test_ra_tls_not_serialized_when_false() {
    let config = PowerConfig::default();
    let serialized = config.to_acl().unwrap();
    assert!(!serialized.contains("ra_tls"));
}

#[test]
fn test_tls_port_serialized_when_set() {
    let config = PowerConfig {
        tls_port: Some(8443),
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("tls_port = 8443"));
}

#[test]
fn test_ra_tls_serialized_when_true() {
    let config = PowerConfig {
        ra_tls: true,
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("ra_tls = true"));
}

#[test]
#[serial]
fn test_env_a3s_power_tls_port() {
    std::env::set_var("A3S_POWER_TLS_PORT", "8443");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.tls_port, Some(8443));
    std::env::remove_var("A3S_POWER_TLS_PORT");
}

#[test]
#[serial]
fn test_env_a3s_power_ra_tls() {
    std::env::set_var("A3S_POWER_RA_TLS", "true");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert!(config.ra_tls);
    std::env::remove_var("A3S_POWER_RA_TLS");
}

#[test]
#[serial]
fn test_env_a3s_power_ra_tls_false_overrides_true() {
    std::env::set_var("A3S_POWER_RA_TLS", "no");
    let mut config = PowerConfig {
        ra_tls: true,
        ..Default::default()
    };
    config.apply_env_overrides().unwrap();
    assert!(!config.ra_tls);
    std::env::remove_var("A3S_POWER_RA_TLS");
}

#[test]
#[serial]
fn test_env_a3s_power_tls_port_invalid_rejected() {
    std::env::set_var("A3S_POWER_TLS_PORT", "not-a-port");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_TLS_PORT");

    assert!(err.to_string().contains("A3S_POWER_TLS_PORT"));
}

#[test]
fn test_vsock_port_defaults_to_none() {
    let config = PowerConfig::default();
    assert!(config.vsock_port.is_none());
}

#[test]
fn test_vsock_port_from_acl() {
    let acl_str = r#"vsock_port = 11434"#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.vsock_port, Some(11434));
}

#[test]
fn test_vsock_port_not_serialized_when_none() {
    let config = PowerConfig::default();
    let serialized = config.to_acl().unwrap();
    assert!(!serialized.contains("vsock_port"));
}

#[test]
fn test_vsock_port_serialized_when_set() {
    let config = PowerConfig {
        vsock_port: Some(11434),
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("vsock_port = 11434"));
}

#[test]
#[serial]
fn test_env_a3s_power_vsock_port() {
    std::env::set_var("A3S_POWER_VSOCK_PORT", "11434");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.vsock_port, Some(11434));
    std::env::remove_var("A3S_POWER_VSOCK_PORT");
}

#[test]
#[serial]
fn test_env_a3s_power_vsock_port_invalid_rejected() {
    std::env::set_var("A3S_POWER_VSOCK_PORT", "not-a-port");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_VSOCK_PORT");

    assert!(err.to_string().contains("A3S_POWER_VSOCK_PORT"));
}

#[test]
#[serial]
fn test_load_config_acl_file() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let acl_path = dir.path().join("config.acl");
    std::fs::write(
        &acl_path,
        r#"
                host = "0.0.0.0"
                port = 9090
                max_loaded_models = 2
            "#,
    )
    .unwrap();

    let config = PowerConfig::load().unwrap();
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 9090);
    assert_eq!(config.max_loaded_models, 2);

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
fn test_api_keys_defaults_to_empty() {
    let config = PowerConfig::default();
    assert!(config.api_keys.is_empty());
}

#[test]
fn test_api_keys_from_acl() {
    let acl_str = r#"api_keys = ["sha256hash1", "sha256hash2"]"#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.api_keys, vec!["sha256hash1", "sha256hash2"]);
}

#[test]
fn test_api_keys_not_serialized_when_empty() {
    let config = PowerConfig::default();
    let serialized = config.to_acl().unwrap();
    assert!(!serialized.contains("api_keys"));
}

#[test]
fn test_api_keys_serialized_when_set() {
    let config = PowerConfig {
        api_keys: vec!["key1".to_string(), "key2".to_string()],
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("api_keys"));
    assert!(serialized.contains("key1"));
    assert!(serialized.contains("key2"));
}

#[test]
#[serial]
fn test_env_a3s_power_api_keys() {
    std::env::set_var("A3S_POWER_API_KEYS", "key_a,key_b,key_c");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.api_keys, vec!["key_a", "key_b", "key_c"]);
    std::env::remove_var("A3S_POWER_API_KEYS");
}

#[test]
#[serial]
fn test_env_a3s_power_api_keys_trims_whitespace() {
    std::env::set_var("A3S_POWER_API_KEYS", " key_a , key_b ");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.api_keys, vec!["key_a", "key_b"]);
    std::env::remove_var("A3S_POWER_API_KEYS");
}

#[test]
#[serial]
fn test_env_a3s_power_api_keys_empty_ignored() {
    std::env::set_var("A3S_POWER_API_KEYS", "");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert!(config.api_keys.is_empty());
    std::env::remove_var("A3S_POWER_API_KEYS");
}

#[test]
fn test_allowed_tee_types_defaults_to_empty() {
    let config = PowerConfig::default();
    assert!(config.allowed_tee_types.is_empty());
}

#[test]
fn test_allowed_tee_types_from_acl() {
    let acl_str = r#"allowed_tee_types = ["sev-snp", "tdx"]"#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.allowed_tee_types, vec!["sev-snp", "tdx"]);
}

#[test]
fn test_expected_measurements_from_acl() {
    let acl_str = r#"
expected_measurement "sev-snp" {
  digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
}
"#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();

    assert_eq!(
            config.expected_measurements.get("sev-snp").map(String::as_str),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
}

#[test]
fn test_tee_policy_mode_from_acl() {
    let config: PowerConfig = acl::deserialize(r#"tee_policy_mode = "gpu-confidential""#).unwrap();
    assert_eq!(config.tee_policy_mode, TeePolicyMode::GpuConfidential);
    assert!(config.strict_attestation());
}

#[test]
fn test_effective_allowed_tee_types_strict_defaults_to_hardware() {
    let config = PowerConfig::default();
    assert_eq!(
        config.effective_allowed_tee_types(),
        vec!["sev-snp".to_string(), "tdx".to_string()]
    );
}

#[test]
fn test_effective_allowed_tee_types_development_remains_permissive() {
    let config = PowerConfig {
        tee_policy_mode: TeePolicyMode::Development,
        ..Default::default()
    };
    assert!(config.effective_allowed_tee_types().is_empty());
}

#[test]
fn test_effective_allowed_tee_types_strict_filters_simulated() {
    let config = PowerConfig {
        allowed_tee_types: vec![
            "sev-snp".to_string(),
            "simulated".to_string(),
            "tdx".to_string(),
        ],
        ..Default::default()
    };
    assert_eq!(
        config.effective_allowed_tee_types(),
        vec!["sev-snp".to_string(), "tdx".to_string()]
    );
}

#[test]
#[serial]
fn test_env_a3s_power_tee_policy_mode() {
    std::env::set_var("A3S_POWER_TEE_POLICY_MODE", "gpu-confidential");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.tee_policy_mode, TeePolicyMode::GpuConfidential);
    std::env::remove_var("A3S_POWER_TEE_POLICY_MODE");
}

#[test]
#[serial]
fn test_tee_strict_env_removes_simulated() {
    std::env::set_var("A3S_POWER_TEE_STRICT", "1");
    let mut config = PowerConfig {
        allowed_tee_types: vec![
            "sev-snp".to_string(),
            "simulated".to_string(),
            "tdx".to_string(),
        ],
        ..Default::default()
    };
    config.apply_env_overrides().unwrap();
    assert!(!config.allowed_tee_types.contains(&"simulated".to_string()));
    assert!(config.allowed_tee_types.contains(&"sev-snp".to_string()));
    std::env::remove_var("A3S_POWER_TEE_STRICT");
}

#[test]
#[serial]
fn test_tee_strict_env_sets_hardware_defaults_when_empty() {
    std::env::set_var("A3S_POWER_TEE_STRICT", "1");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert!(config.allowed_tee_types.contains(&"sev-snp".to_string()));
    assert!(config.allowed_tee_types.contains(&"tdx".to_string()));
    assert!(!config.allowed_tee_types.contains(&"simulated".to_string()));
    std::env::remove_var("A3S_POWER_TEE_STRICT");
}

#[test]
fn test_audit_log_defaults_to_false() {
    let config = PowerConfig::default();
    assert!(!config.audit_log);
}

#[test]
#[serial]
fn test_audit_log_env_override() {
    std::env::set_var("A3S_POWER_AUDIT_LOG", "1");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert!(config.audit_log);
    std::env::remove_var("A3S_POWER_AUDIT_LOG");
}

#[test]
#[serial]
fn test_audit_log_env_false_overrides_true() {
    std::env::set_var("A3S_POWER_AUDIT_LOG", "false");
    let mut config = PowerConfig {
        audit_log: true,
        ..Default::default()
    };
    config.apply_env_overrides().unwrap();
    assert!(!config.audit_log);
    std::env::remove_var("A3S_POWER_AUDIT_LOG");
}

#[test]
fn test_model_signing_key_defaults_to_none() {
    let config = PowerConfig::default();
    assert!(config.model_signing_key.is_none());
}

#[test]
fn test_model_signing_key_from_acl() {
    let acl_str = r#"model_signing_key = "aabbccdd""#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.model_signing_key.as_deref(), Some("aabbccdd"));
}

#[test]
fn test_to_acl_includes_policy_fields_when_set() {
    let mut measurements = HashMap::new();
    measurements.insert("sev-snp".to_string(), "deadbeef".to_string());
    let config = PowerConfig {
        allowed_tee_types: vec!["sev-snp".to_string()],
        expected_measurements: measurements,
        audit_log: true,
        model_signing_key: Some(valid_model_signing_key_hex()),
        ..Default::default()
    };
    let acl = config.to_acl().unwrap();
    assert!(acl.contains("allowed_tee_types"));
    assert!(acl.contains("sev-snp"));
    assert!(acl.contains("expected_measurement"));
    assert!(acl.contains("deadbeef"));
    assert!(acl.contains("audit_log = true"));
    assert!(acl.contains("model_signing_key"));
}

#[test]
fn test_tls_sans_defaults_to_empty() {
    let config = PowerConfig::default();
    assert!(config.tls_sans.is_empty());
}

#[test]
fn test_tls_sans_from_acl() {
    let acl_str = r#"tls_sans = ["myserver.internal", "10.0.0.1"]"#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.tls_sans, vec!["myserver.internal", "10.0.0.1"]);
}

#[cfg(feature = "tls")]
#[test]
fn test_validate_accepts_valid_tls_sans() {
    let config = PowerConfig {
        tls_sans: vec![
            "myserver.internal".to_string(),
            "*.example.com".to_string(),
            "10.0.0.1".to_string(),
            "::1".to_string(),
        ],
        ..Default::default()
    };

    config.validate().unwrap();
}

#[cfg(feature = "tls")]
#[test]
fn test_validate_rejects_invalid_tls_san() {
    let config = PowerConfig {
        tls_sans: vec!["not a valid san !!!".to_string()],
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("tls_sans"));
}
