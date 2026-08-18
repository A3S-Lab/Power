use super::super::*;
use serial_test::serial;

#[test]
fn test_default_config() {
    let config = PowerConfig::default();
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 11434);
    assert_eq!(config.max_loaded_models, 1);
    assert!(!config.tee_mode);
    assert_eq!(config.tee_policy_mode, TeePolicyMode::Strict);
    assert!(!config.redact_logs);
    assert!(config.model_hashes.is_empty());
}

#[test]
fn test_bind_address() {
    let config = PowerConfig::default();
    assert_eq!(config.bind_address(), "127.0.0.1:11434");
}

#[test]
fn test_config_deserialize_acl() {
    let acl_str = r#"
            host = "0.0.0.0"
            port = 8080
            max_loaded_models = 3
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 8080);
    assert_eq!(config.max_loaded_models, 3);
}

#[test]
fn test_config_serialize_acl() {
    let config = PowerConfig::default();
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("host"));
    assert!(serialized.contains("port"));
    assert!(serialized.contains("gpu {"));
}

#[test]
#[serial]
fn test_config_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let config = PowerConfig {
        host: "0.0.0.0".to_string(),
        port: 9999,
        data_dir: dir.path().to_path_buf(),
        max_loaded_models: 5,
        gpu: GpuConfig::default(),
        gpu_attestation: GpuAttestationConfig::default(),
        spec_mode: "prompt-lookup".to_string(),
        spec_draft_max: Some(3),
        spec_mtp_recurrent_snapshots: 7,
        spec_mtp_fr_vocab_size: Some(8192),
        spec_draft_min: 0,
        spec_draft_p_min: 0.0,
        keep_alive: "5m".to_string(),
        use_mlock: false,
        use_mmap: false,
        num_thread: None,
        flash_attention: false,
        num_parallel: 4,
        tee_mode: true,
        tee_policy_mode: TeePolicyMode::Development,
        redact_logs: true,
        model_hashes: HashMap::new(),
        model_key_source: None,
        tls_port: None,
        tls_sans: Vec::new(),
        ra_tls: false,
        vsock_port: None,
        api_keys: Vec::new(),
        allowed_tee_types: Vec::new(),
        expected_measurements: HashMap::new(),
        audit_log: false,
        audit_log_path: None,
        audit_log_encrypt: false,
        audit_key_source: None,
        model_signing_key: None,
        key_provider: "static".to_string(),
        key_rotation_sources: Vec::new(),
        in_memory_decrypt: false,
        streaming_decrypt: false,
        suppress_token_metrics: false,
        rate_limit_rps: 0,
        max_concurrent_requests: 0,
        proxy_upstreams: HashMap::new(),
        proxy_effective_prompt_digest: false,
        proxy_effective_prompt_digest_required: false,
        proxy_effective_prompt_digest_path: default_proxy_effective_prompt_digest_path(),
        timing_padding_ms: None,
    };
    config.save().unwrap();

    let loaded = PowerConfig::load().unwrap();
    assert_eq!(loaded.host, "0.0.0.0");
    assert_eq!(loaded.port, 9999);
    assert_eq!(loaded.max_loaded_models, 5);
    assert_eq!(loaded.num_parallel, 4);
    assert_eq!(loaded.spec_mtp_fr_vocab_size, Some(8192));
    assert!(!loaded.use_mmap);
    assert!(loaded.tee_mode);
    assert!(loaded.redact_logs);

    std::env::remove_var("A3S_POWER_HOME");
}

#[test]
fn test_gpu_config_defaults() {
    let config = PowerConfig::default();
    assert_eq!(config.gpu.gpu_layers, 0);
    assert_eq!(config.gpu.main_gpu, 0);
    assert_eq!(config.spec_draft_max, None);
    assert_eq!(config.spec_mtp_recurrent_snapshots, 7);
    assert_eq!(config.spec_mtp_fr_vocab_size, None);
}

#[test]
fn test_gpu_config_deserialize_acl() {
    let acl_str = r#"
            host = "127.0.0.1"
            port = 11434

            gpu {
                gpu_layers = -1
                main_gpu = 1
            }
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.gpu.gpu_layers, -1);
    assert_eq!(config.gpu.main_gpu, 1);
}

#[test]
fn test_gpu_config_missing_uses_defaults() {
    let acl_str = r#"
            host = "127.0.0.1"
            port = 11434
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.gpu.gpu_layers, 0);
    assert_eq!(config.gpu.main_gpu, 0);
}

#[test]
fn test_speculative_settings_deserialize_acl() {
    let acl_str = r#"
            spec_mode = "dspark"
            spec_draft_max = 7
            spec_mtp_recurrent_snapshots = 5
            spec_mtp_fr_vocab_size = 8192
            spec_draft_min = 2
            spec_draft_p_min = 0.6
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(config.spec_mode, "dspark");
    assert_eq!(config.spec_draft_max, Some(7));
    assert_eq!(config.spec_mtp_recurrent_snapshots, 5);
    assert_eq!(config.spec_mtp_fr_vocab_size, Some(8192));
    assert_eq!(config.spec_draft_min, 2);
    assert!((config.spec_draft_p_min - 0.6).abs() < f32::EPSILON);
}

#[test]
fn test_gpu_attestation_config_defaults() {
    let config = PowerConfig::default();
    assert_eq!(
        config.gpu_attestation.source,
        GpuAttestationSource::Configured
    );
    assert_eq!(config.gpu_attestation.provider, "nvidia-nras");
    assert_eq!(
        config.gpu_attestation.nvattest_path,
        PathBuf::from("nvattest")
    );
    assert_eq!(config.gpu_attestation.nvattest_verifier, "remote");
    assert_eq!(config.gpu_attestation.nvattest_gpu_evidence_source, "nvml");
    assert_eq!(config.gpu_attestation.nvattest_timeout_secs, 30);
    assert_eq!(config.gpu_attestation.nras_claims_version, "3.0");
    assert_eq!(config.gpu_attestation.nras_timeout_secs, 30);
    assert!(!config.gpu_attestation.evidence_configured());
    assert!(!config.gpu_attestation.verdict_configured());
}

#[test]
fn test_gpu_attestation_config_deserialize_acl() {
    let acl_str = r#"
            gpu_attestation {
                source = "configured"
                provider = "nvidia-nras"
                evidence_path = "/run/a3s/gpu.evidence"
                verdict_path = "/run/a3s/nras.verdict"
            }
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(
        config.gpu_attestation.source,
        GpuAttestationSource::Configured
    );
    assert_eq!(config.gpu_attestation.provider, "nvidia-nras");
    assert_eq!(
        config.gpu_attestation.evidence_path,
        Some(PathBuf::from("/run/a3s/gpu.evidence"))
    );
    assert_eq!(
        config.gpu_attestation.verdict_path,
        Some(PathBuf::from("/run/a3s/nras.verdict"))
    );
}

#[test]
fn test_gpu_attestation_config_serialization() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            evidence_hex: Some("0011".to_string()),
            verdict_hex: Some("2233".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("gpu_attestation {"));
    assert!(serialized.contains("evidence_hex = \"0011\""));
    assert!(serialized.contains("verdict_hex = \"2233\""));
}

#[test]
fn test_gpu_attestation_nvattest_cli_deserialize_acl() {
    let acl_str = r#"
            gpu_attestation {
                source = "nvattest-cli"
                provider = "nvidia-nras"
                nvattest_path = "/usr/local/bin/nvattest"
                nvattest_verifier = "remote"
                nvattest_gpu_evidence_source = "nvml"
                nras_url = "https://nras.attestation.nvidia.com"
                nvattest_timeout_secs = 45
            }
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(
        config.gpu_attestation.source,
        GpuAttestationSource::NvattestCli
    );
    assert_eq!(
        config.gpu_attestation.nvattest_path,
        PathBuf::from("/usr/local/bin/nvattest")
    );
    assert_eq!(config.gpu_attestation.nvattest_verifier, "remote");
    assert_eq!(
        config.gpu_attestation.nras_url.as_deref(),
        Some("https://nras.attestation.nvidia.com")
    );
    assert_eq!(config.gpu_attestation.nvattest_timeout_secs, 45);
}

#[test]
fn test_gpu_attestation_nvattest_cli_serialization() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nras_url: Some("https://nras.attestation.nvidia.com".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("source = \"nvattest-cli\""));
    assert!(serialized.contains("nvattest_path = \"nvattest\""));
    assert!(serialized.contains("nvattest_verifier = \"remote\""));
    assert!(serialized.contains("nras_url = \"https://nras.attestation.nvidia.com\""));
}

#[test]
fn test_gpu_attestation_nras_rest_deserialize_acl() {
    let acl_str = r#"
            gpu_attestation {
                source = "nras-rest"
                provider = "nvidia-nras"
                evidence_path = "/run/a3s/gpu-evidence.json"
                nras_url = "https://nras.attestation.nvidia.com"
                nras_gpu_architecture = "HOPPER"
                nras_claims_version = "3.0"
                nras_bearer_token_env = "NRAS_TOKEN"
                nras_timeout_secs = 45
            }
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(
        config.gpu_attestation.source,
        GpuAttestationSource::NrasRest
    );
    assert_eq!(
        config.gpu_attestation.evidence_path,
        Some(PathBuf::from("/run/a3s/gpu-evidence.json"))
    );
    assert_eq!(
        config.gpu_attestation.nras_gpu_architecture.as_deref(),
        Some("HOPPER")
    );
    assert_eq!(config.gpu_attestation.nras_claims_version, "3.0");
    assert_eq!(
        config.gpu_attestation.nras_bearer_token_env.as_deref(),
        Some("NRAS_TOKEN")
    );
    assert_eq!(config.gpu_attestation.nras_timeout_secs, 45);
}

#[test]
fn test_gpu_attestation_nras_rest_serialization() {
    let config = PowerConfig {
        gpu_attestation: GpuAttestationConfig {
            source: GpuAttestationSource::NrasRest,
            evidence_hex: Some("0011".to_string()),
            nras_url: Some("https://nras.attestation.nvidia.com".to_string()),
            nras_gpu_architecture: Some("HOPPER".to_string()),
            nras_bearer_token_env: Some("NRAS_TOKEN".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("source = \"nras-rest\""));
    assert!(serialized.contains("evidence_hex = \"0011\""));
    assert!(serialized.contains("nras_url = \"https://nras.attestation.nvidia.com\""));
    assert!(serialized.contains("nras_gpu_architecture = \"HOPPER\""));
    assert!(serialized.contains("nras_claims_version = \"3.0\""));
    assert!(serialized.contains("nras_bearer_token_env = \"NRAS_TOKEN\""));
    assert!(serialized.contains("nras_timeout_secs = 30"));
}

#[test]
fn test_proxy_effective_prompt_digest_defaults() {
    let config = PowerConfig::default();
    assert!(!config.proxy_effective_prompt_digest);
    assert!(!config.proxy_effective_prompt_digest_required);
    assert_eq!(
        config.proxy_effective_prompt_digest_path,
        "/v1/chat/effective-prompt-digest"
    );
}

#[test]
fn test_proxy_effective_prompt_digest_deserialize_acl() {
    let acl_str = r#"
            proxy_upstream "llama-70b" {
                url = "http://vllm:8000"
            }
            proxy_effective_prompt_digest = true
            proxy_effective_prompt_digest_required = true
            proxy_effective_prompt_digest_path = "/v1/rendered-prompt-digest"
        "#;
    let config: PowerConfig = acl::deserialize(acl_str).unwrap();
    assert_eq!(
        config.proxy_upstreams.get("llama-70b").map(String::as_str),
        Some("http://vllm:8000")
    );
    assert!(config.proxy_effective_prompt_digest);
    assert!(config.proxy_effective_prompt_digest_required);
    assert_eq!(
        config.proxy_effective_prompt_digest_path,
        "/v1/rendered-prompt-digest"
    );
}

#[test]
fn test_proxy_effective_prompt_digest_serialization() {
    let mut proxy_upstreams = HashMap::new();
    proxy_upstreams.insert("llama-70b".to_string(), "http://vllm:8000".to_string());
    let config = PowerConfig {
        proxy_upstreams,
        proxy_effective_prompt_digest: true,
        proxy_effective_prompt_digest_required: true,
        proxy_effective_prompt_digest_path: "/v1/rendered-prompt-digest".to_string(),
        ..Default::default()
    };
    let serialized = config.to_acl().unwrap();
    assert!(serialized.contains("proxy_upstream \"llama-70b\" {"));
    assert!(serialized.contains("url = \"http://vllm:8000\""));
    assert!(serialized.contains("proxy_effective_prompt_digest = true"));
    assert!(serialized.contains("proxy_effective_prompt_digest_required = true"));
    assert!(
        serialized.contains("proxy_effective_prompt_digest_path = \"/v1/rendered-prompt-digest\"")
    );
}

#[test]
fn test_default_keep_alive() {
    let config = PowerConfig::default();
    assert_eq!(config.keep_alive, "5m");
}

#[test]
fn test_parse_keep_alive_minutes() {
    assert_eq!(
        parse_keep_alive("5m").unwrap(),
        std::time::Duration::from_secs(300)
    );
}

#[test]
fn test_parse_keep_alive_hours() {
    assert_eq!(
        parse_keep_alive("1h").unwrap(),
        std::time::Duration::from_secs(3600)
    );
}

#[test]
fn test_parse_keep_alive_seconds() {
    assert_eq!(
        parse_keep_alive("30s").unwrap(),
        std::time::Duration::from_secs(30)
    );
}

#[test]
fn test_parse_keep_alive_zero() {
    assert_eq!(parse_keep_alive("0").unwrap(), std::time::Duration::ZERO);
}

#[test]
fn test_parse_keep_alive_never() {
    assert_eq!(parse_keep_alive("-1").unwrap(), std::time::Duration::MAX);
}

#[test]
fn test_parse_keep_alive_raw_number() {
    assert_eq!(
        parse_keep_alive("120").unwrap(),
        std::time::Duration::from_secs(120)
    );
}

#[test]
fn test_parse_keep_alive_invalid_returns_error() {
    let err = parse_keep_alive("abc").unwrap_err();
    assert!(err.contains("invalid keep_alive"));
}

#[test]
fn test_parse_keep_alive_overflow_returns_error() {
    let err = parse_keep_alive("18446744073709551615m").unwrap_err();
    assert!(err.contains("invalid keep_alive"));

    let err = parse_keep_alive("18446744073709551615h").unwrap_err();
    assert!(err.contains("invalid keep_alive"));
}

// ---------------------------------------------------------------
// Environment variable override tests
// ---------------------------------------------------------------

#[test]
#[serial]
fn test_env_a3s_power_host() {
    std::env::set_var("A3S_POWER_HOST", "0.0.0.0");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.host, "0.0.0.0");
    std::env::remove_var("A3S_POWER_HOST");
}

#[test]
#[serial]
fn test_env_a3s_power_port() {
    std::env::set_var("A3S_POWER_PORT", "8080");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.port, 8080);
    std::env::remove_var("A3S_POWER_PORT");
}

#[test]
#[serial]
fn test_env_a3s_power_port_invalid_rejected() {
    std::env::set_var("A3S_POWER_PORT", "not-a-port");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_PORT");

    assert!(err.to_string().contains("A3S_POWER_PORT"));
}

#[test]
#[serial]
fn test_env_a3s_power_data_dir() {
    std::env::set_var("A3S_POWER_DATA_DIR", "/tmp/my-models");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.data_dir, PathBuf::from("/tmp/my-models"));
    std::env::remove_var("A3S_POWER_DATA_DIR");
}

#[test]
#[serial]
fn test_env_a3s_power_max_models() {
    std::env::set_var("A3S_POWER_MAX_MODELS", "4");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.max_loaded_models, 4);
    std::env::remove_var("A3S_POWER_MAX_MODELS");
}

#[test]
#[serial]
fn test_env_a3s_power_max_models_invalid_rejected() {
    std::env::set_var("A3S_POWER_MAX_MODELS", "not-a-number");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_MAX_MODELS");

    assert!(err.to_string().contains("A3S_POWER_MAX_MODELS"));
}

#[test]
#[serial]
fn test_env_a3s_power_keep_alive() {
    std::env::set_var("A3S_POWER_KEEP_ALIVE", "10m");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.keep_alive, "10m");
    std::env::remove_var("A3S_POWER_KEEP_ALIVE");
}

#[test]
#[serial]
fn test_env_a3s_power_speculative_settings() {
    std::env::set_var("A3S_POWER_SPEC_MODE", "mtp");
    std::env::set_var("A3S_POWER_SPEC_DRAFT_MAX", "5");
    std::env::set_var("A3S_POWER_SPEC_MTP_RECURRENT_SNAPSHOTS", "4");
    std::env::set_var("A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE", "8192");
    std::env::set_var("A3S_POWER_SPEC_DRAFT_MIN", "1");
    std::env::set_var("A3S_POWER_SPEC_DRAFT_P_MIN", "0.75");

    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();

    std::env::remove_var("A3S_POWER_SPEC_MODE");
    std::env::remove_var("A3S_POWER_SPEC_DRAFT_MAX");
    std::env::remove_var("A3S_POWER_SPEC_MTP_RECURRENT_SNAPSHOTS");
    std::env::remove_var("A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE");
    std::env::remove_var("A3S_POWER_SPEC_DRAFT_MIN");
    std::env::remove_var("A3S_POWER_SPEC_DRAFT_P_MIN");

    assert_eq!(config.spec_mode, "mtp");
    assert_eq!(config.spec_draft_max, Some(5));
    assert_eq!(config.spec_mtp_recurrent_snapshots, 4);
    assert_eq!(config.spec_mtp_fr_vocab_size, Some(8192));
    assert_eq!(config.spec_draft_min, 1);
    assert!((config.spec_draft_p_min - 0.75).abs() < f32::EPSILON);
}

#[test]
#[serial]
fn test_env_a3s_power_spec_draft_max_invalid_rejected() {
    std::env::set_var("A3S_POWER_SPEC_DRAFT_MAX", "not-a-number");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_SPEC_DRAFT_MAX");

    assert!(err.to_string().contains("A3S_POWER_SPEC_DRAFT_MAX"));
}

#[test]
#[serial]
fn test_env_a3s_power_gpu_layers() {
    std::env::set_var("A3S_POWER_GPU_LAYERS", "-1");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.gpu.gpu_layers, -1);
    std::env::remove_var("A3S_POWER_GPU_LAYERS");
}

#[test]
#[serial]
fn test_env_a3s_power_gpu_layers_invalid_rejected() {
    std::env::set_var("A3S_POWER_GPU_LAYERS", "abc");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_GPU_LAYERS");

    assert!(err.to_string().contains("A3S_POWER_GPU_LAYERS"));
}

#[test]
#[serial]
fn test_env_a3s_power_gpu_attestation_paths() {
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_PROVIDER", "nvidia-nras");
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_SOURCE", "configured");
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_EVIDENCE_PATH",
        "/tmp/gpu.evidence",
    );
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_VERDICT_PATH",
        "/tmp/nras.verdict",
    );
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(
        config.gpu_attestation.source,
        GpuAttestationSource::Configured
    );
    assert_eq!(config.gpu_attestation.provider, "nvidia-nras");
    assert_eq!(
        config.gpu_attestation.evidence_path,
        Some(PathBuf::from("/tmp/gpu.evidence"))
    );
    assert_eq!(
        config.gpu_attestation.verdict_path,
        Some(PathBuf::from("/tmp/nras.verdict"))
    );
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_PROVIDER");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_SOURCE");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_EVIDENCE_PATH");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_VERDICT_PATH");
}

#[test]
#[serial]
fn test_env_a3s_power_gpu_attestation_nvattest_cli() {
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_SOURCE", "nvattest-cli");
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_PATH", "/opt/nvattest");
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_VERIFIER", "remote");
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_NVATTEST_GPU_EVIDENCE_SOURCE",
        "nvml",
    );
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_NRAS_URL",
        "https://nras.attestation.nvidia.com",
    );
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS", "45");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(
        config.gpu_attestation.source,
        GpuAttestationSource::NvattestCli
    );
    assert_eq!(
        config.gpu_attestation.nvattest_path,
        PathBuf::from("/opt/nvattest")
    );
    assert_eq!(config.gpu_attestation.nvattest_verifier, "remote");
    assert_eq!(
        config.gpu_attestation.nras_url.as_deref(),
        Some("https://nras.attestation.nvidia.com")
    );
    assert_eq!(config.gpu_attestation.nvattest_timeout_secs, 45);
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_SOURCE");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_PATH");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_VERIFIER");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_GPU_EVIDENCE_SOURCE");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NRAS_URL");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS");
}

#[test]
#[serial]
fn test_env_a3s_power_gpu_attestation_nvattest_timeout_invalid_rejected() {
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS", "soon");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS");

    assert!(err
        .to_string()
        .contains("A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS"));
}

#[test]
#[serial]
fn test_env_a3s_power_gpu_attestation_nras_rest() {
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_SOURCE", "nras-rest");
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_EVIDENCE_PATH",
        "/tmp/gpu-evidence.json",
    );
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_NRAS_URL",
        "https://nras.attestation.nvidia.com",
    );
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_NRAS_GPU_ARCHITECTURE",
        "BLACKWELL",
    );
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_NRAS_CLAIMS_VERSION", "3.0");
    std::env::set_var(
        "A3S_POWER_GPU_ATTESTATION_NRAS_BEARER_TOKEN_ENV",
        "NRAS_TOKEN",
    );
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS", "45");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(
        config.gpu_attestation.source,
        GpuAttestationSource::NrasRest
    );
    assert_eq!(
        config.gpu_attestation.evidence_path,
        Some(PathBuf::from("/tmp/gpu-evidence.json"))
    );
    assert_eq!(
        config.gpu_attestation.nras_url.as_deref(),
        Some("https://nras.attestation.nvidia.com")
    );
    assert_eq!(
        config.gpu_attestation.nras_gpu_architecture.as_deref(),
        Some("BLACKWELL")
    );
    assert_eq!(config.gpu_attestation.nras_claims_version, "3.0");
    assert_eq!(
        config.gpu_attestation.nras_bearer_token_env.as_deref(),
        Some("NRAS_TOKEN")
    );
    assert_eq!(config.gpu_attestation.nras_timeout_secs, 45);
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_SOURCE");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_EVIDENCE_PATH");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NRAS_URL");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NRAS_GPU_ARCHITECTURE");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NRAS_CLAIMS_VERSION");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NRAS_BEARER_TOKEN_ENV");
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS");
}

#[test]
#[serial]
fn test_env_a3s_power_gpu_attestation_nras_timeout_invalid_rejected() {
    std::env::set_var("A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS", "eventually");
    let mut config = PowerConfig::default();
    let err = config.apply_env_overrides().unwrap_err();
    std::env::remove_var("A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS");

    assert!(err
        .to_string()
        .contains("A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS"));
}

#[test]
#[serial]
fn test_env_a3s_power_proxy_effective_prompt_digest() {
    std::env::set_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST", "true");
    std::env::set_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED", "1");
    std::env::set_var(
        "A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_PATH",
        "/v1/rendered-prompt-digest",
    );
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert!(config.proxy_effective_prompt_digest);
    assert!(config.proxy_effective_prompt_digest_required);
    assert_eq!(
        config.proxy_effective_prompt_digest_path,
        "/v1/rendered-prompt-digest"
    );
    std::env::remove_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST");
    std::env::remove_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED");
    std::env::remove_var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_PATH");
}

#[test]
#[serial]
fn test_env_a3s_power_tee_mode() {
    std::env::set_var("A3S_POWER_TEE_MODE", "true");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert!(config.tee_mode);
    assert!(config.redact_logs); // auto-enabled when tee_mode
    std::env::remove_var("A3S_POWER_TEE_MODE");
}

#[test]
#[serial]
fn test_env_a3s_power_tee_mode_false_overrides_true() {
    std::env::set_var("A3S_POWER_TEE_MODE", "false");
    let mut config = PowerConfig {
        tee_mode: true,
        ..Default::default()
    };
    config.apply_env_overrides().unwrap();
    assert!(!config.tee_mode);
    std::env::remove_var("A3S_POWER_TEE_MODE");
}

#[test]
#[serial]
fn test_env_a3s_power_redact_logs() {
    std::env::set_var("A3S_POWER_REDACT_LOGS", "1");
    let mut config = PowerConfig::default();
    config.apply_env_overrides().unwrap();
    assert!(config.redact_logs);
    std::env::remove_var("A3S_POWER_REDACT_LOGS");
}

#[test]
#[serial]
fn test_env_a3s_power_redact_logs_false_overrides_true() {
    std::env::set_var("A3S_POWER_REDACT_LOGS", "0");
    let mut config = PowerConfig {
        redact_logs: true,
        ..Default::default()
    };
    config.apply_env_overrides().unwrap();
    assert!(!config.redact_logs);
    std::env::remove_var("A3S_POWER_REDACT_LOGS");
}
