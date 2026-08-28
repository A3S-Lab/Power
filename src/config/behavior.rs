use super::*;

impl GpuConfig {
    /// Validate bounded exact tensor-placement names shared by config and
    /// execution-policy digest callers.
    pub fn validate(&self) -> Result<()> {
        let cpu_tensor_names = validate_exact_tensor_names(
            "gpu.cpu_tensors",
            &self.cpu_tensors,
            MAX_CPU_TENSOR_OVERRIDES,
            MAX_CPU_TENSOR_NAME_BYTES,
        )?;
        let gpu_tensor_names = validate_exact_tensor_names(
            "gpu.gpu_tensors",
            &self.gpu_tensors,
            MAX_GPU_TENSOR_OVERRIDES,
            MAX_GPU_TENSOR_NAME_BYTES,
        )?;
        if let Some(name) = self
            .gpu_tensors
            .iter()
            .find(|name| cpu_tensor_names.contains(name.as_str()))
        {
            return Err(PowerError::Config(format!(
                "exact tensor name {name:?} cannot appear in both gpu.cpu_tensors and gpu.gpu_tensors"
            )));
        }
        debug_assert_eq!(gpu_tensor_names.len(), self.gpu_tensors.len());
        Ok(())
    }
}

fn validate_exact_tensor_names<'a>(
    field: &str,
    names: &'a [String],
    maximum_count: usize,
    maximum_name_bytes: usize,
) -> Result<std::collections::HashSet<&'a str>> {
    if names.len() > maximum_count {
        return Err(PowerError::Config(format!(
            "{field} accepts at most {maximum_count} exact tensor names"
        )));
    }

    let mut unique = std::collections::HashSet::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        if name.is_empty()
            || name.len() > maximum_name_bytes
            || name.trim() != name
            || name.chars().any(char::is_control)
        {
            return Err(PowerError::Config(format!(
                "{field}[{index}] must be a non-empty tensor name of at most {maximum_name_bytes} bytes without surrounding whitespace or control characters"
            )));
        }
        if !unique.insert(name.as_str()) {
            return Err(PowerError::Config(format!(
                "{field} contains duplicate exact tensor name {name:?}"
            )));
        }
    }
    Ok(unique)
}

impl PowerConfig {
    /// Load configuration from a specific file path (A3S ACL format).
    ///
    /// After loading from file, applies `A3S_POWER_*` environment variable overrides.
    pub fn load_from(path: &str) -> Result<Self> {
        let path = std::path::Path::new(path);
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::error::PowerError::Config(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;
        let mut config = acl::deserialize(&content).map_err(|error| {
            crate::error::PowerError::Acl(format!("failed to load {}: {error}", path.display()))
        })?;
        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from the default config file path (A3S ACL format).
    /// Returns default config if the file does not exist.
    ///
    /// After loading from file, applies `A3S_POWER_*` environment variable overrides.
    pub fn load() -> Result<Self> {
        let path = dirs::config_path();
        let mut config = if path.exists() {
            let content = std::fs::read_to_string(&path).map_err(|e| {
                crate::error::PowerError::Config(format!(
                    "Failed to read config file {}: {}",
                    path.display(),
                    e
                ))
            })?;
            acl::deserialize(&content).map_err(|error| {
                crate::error::PowerError::Acl(format!("failed to load {}: {error}", path.display()))
            })?
        } else {
            Self::default()
        };

        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Validate fatal policy settings and emit warnings for soft misconfiguration patterns.
    ///
    /// Some legacy operational warnings remain non-fatal, but policy-bearing
    /// settings that would otherwise be silently ignored return an error.
    pub fn validate(&self) -> Result<()> {
        self.gpu.validate()?;
        self.serving_execution.validate()?;
        if !self.serving_execution.is_aggregated() && self.api_keys.is_empty() {
            return Err(PowerError::Config(
                "prefill-decode serving requires api_keys for authenticated internal phase routes"
                    .to_string(),
            ));
        }

        if self.prompt_cache_max_entries == 0
            || self.prompt_cache_max_entries > MAX_PROMPT_CACHE_ENTRIES
        {
            return Err(PowerError::Config(format!(
                "prompt_cache_max_entries must be between 1 and {MAX_PROMPT_CACHE_ENTRIES}, got {}",
                self.prompt_cache_max_entries
            )));
        }
        if self.prompt_cache_ttl_seconds == 0
            || self.prompt_cache_ttl_seconds > MAX_PROMPT_CACHE_TTL_SECONDS
        {
            return Err(PowerError::Config(format!(
                "prompt_cache_ttl_seconds must be between 1 and {MAX_PROMPT_CACHE_TTL_SECONDS}, got {}",
                self.prompt_cache_ttl_seconds
            )));
        }
        if self.worker_observation_ttl_seconds == 0
            || self.worker_observation_ttl_seconds > MAX_WORKER_OBSERVATION_TTL_SECONDS
        {
            return Err(PowerError::Config(format!(
                "worker_observation_ttl_seconds must be between 1 and {MAX_WORKER_OBSERVATION_TTL_SECONDS}, got {}",
                self.worker_observation_ttl_seconds
            )));
        }

        if !is_valid_spec_mode(&self.spec_mode) {
            return Err(PowerError::Config(format!(
                "unsupported spec_mode '{}'; expected one of: auto, off, prompt-lookup, ngram-context, draft-model, mtp, dflash, dflash2, dspark",
                self.spec_mode
            )));
        }
        if let Some(spec_draft_max) = self.spec_draft_max {
            if spec_draft_max == 0 || spec_draft_max > 64 {
                return Err(PowerError::Config(format!(
                    "spec_draft_max must be between 1 and 64, got {spec_draft_max}"
                )));
            }
            if self.spec_draft_min > spec_draft_max {
                return Err(PowerError::Config(format!(
                    "spec_draft_min ({}) must not exceed spec_draft_max ({spec_draft_max})",
                    self.spec_draft_min
                )));
            }
        } else if self.spec_draft_min > 64 {
            return Err(PowerError::Config(format!(
                "spec_draft_min must not exceed 64, got {}",
                self.spec_draft_min
            )));
        }
        if self.spec_mtp_recurrent_snapshots == 0 || self.spec_mtp_recurrent_snapshots > 64 {
            return Err(PowerError::Config(format!(
                "spec_mtp_recurrent_snapshots must be between 1 and 64, got {}",
                self.spec_mtp_recurrent_snapshots
            )));
        }
        if let Some(vocab_size) = self.spec_mtp_fr_vocab_size {
            if !(1024..=1_048_576).contains(&vocab_size) {
                return Err(PowerError::Config(format!(
                    "spec_mtp_fr_vocab_size must be between 1024 and 1048576, got {vocab_size}"
                )));
            }
        }
        if !self.spec_draft_p_min.is_finite() || !(0.0..=1.0).contains(&self.spec_draft_p_min) {
            return Err(PowerError::Config(format!(
                "spec_draft_p_min must be finite and between 0 and 1, got {}",
                self.spec_draft_p_min
            )));
        }

        parse_keep_alive(&self.keep_alive).map_err(PowerError::Config)?;

        if let Some(ref key) = self.model_signing_key {
            validate_model_signing_key(key).map_err(PowerError::Config)?;
        }

        if self.ra_tls {
            if self.tls_port.is_none() {
                return Err(PowerError::Config(
                    "ra_tls = true requires tls_port so the RA-TLS listener is started".to_string(),
                ));
            }
            if !self.tee_mode {
                return Err(PowerError::Config(
                    "ra_tls = true requires tee_mode = true so the TLS certificate contains attestation"
                        .to_string(),
                ));
            }
        }

        #[cfg(not(feature = "tls"))]
        if self.tls_port.is_some() {
            return Err(PowerError::Config(
                "tls_port requires building a3s-power with the tls feature enabled".to_string(),
            ));
        }

        #[cfg(feature = "tls")]
        for san in &self.tls_sans {
            crate::tee::cert::validate_tls_san(san)?;
        }

        match self.key_provider.as_str() {
            "static" => {}
            "rotating" => {
                if self.key_rotation_sources.is_empty() {
                    return Err(PowerError::Config(
                        "key_provider = \"rotating\" requires at least one key_rotation_sources entry"
                            .to_string(),
                    ));
                }
            }
            other => {
                return Err(PowerError::Config(format!(
                    "unsupported key_provider '{other}'; expected one of: static, rotating"
                )));
            }
        }

        if self.audit_log_encrypt && self.audit_key_source.is_none() {
            return Err(PowerError::Config(
                "audit_log_encrypt = true requires audit_key_source to encrypt audit entries"
                    .to_string(),
            ));
        }

        if self.in_memory_decrypt {
            tracing::warn!(
                "in_memory_decrypt = true requires a backend that explicitly supports locked \
                 in-memory plaintext loading. Unsupported backends fail closed before load."
            );
        }

        if self.streaming_decrypt {
            tracing::warn!(
                "streaming_decrypt = true requires a backend that explicitly supports \
                 layer-streaming decrypted plaintext loading. Unsupported backends fail closed \
                 before load."
            );
            #[cfg(not(feature = "picolm"))]
            tracing::warn!(
                "streaming_decrypt = true but the picolm feature is not enabled. \
                 The current GGUF streaming-decrypt backend path requires --features picolm."
            );
        }

        if self.gpu_attestation.evidence_hex.is_some()
            && self.gpu_attestation.evidence_path.is_some()
        {
            return Err(PowerError::Config(
                "gpu_attestation.evidence_hex and gpu_attestation.evidence_path are mutually exclusive"
                    .to_string(),
            ));
        }

        if self.gpu_attestation.verdict_hex.is_some() && self.gpu_attestation.verdict_path.is_some()
        {
            return Err(PowerError::Config(
                "gpu_attestation.verdict_hex and gpu_attestation.verdict_path are mutually exclusive"
                    .to_string(),
            ));
        }

        if self.gpu_attestation.source == GpuAttestationSource::NvattestCli
            && self.gpu_attestation.nvattest_timeout_secs == 0
        {
            return Err(PowerError::Config(
                "gpu_attestation.nvattest_timeout_secs must be greater than zero".to_string(),
            ));
        }

        if self.gpu_attestation.source == GpuAttestationSource::NvattestCli {
            let verifier = self
                .gpu_attestation
                .nvattest_verifier
                .trim()
                .to_ascii_lowercase();
            if !matches!(verifier.as_str(), "local" | "remote") {
                return Err(PowerError::Config(format!(
                    "gpu_attestation.nvattest_verifier must be \"remote\" or \"local\", got {:?}",
                    self.gpu_attestation.nvattest_verifier
                )));
            }

            let evidence_source = self
                .gpu_attestation
                .nvattest_gpu_evidence_source
                .trim()
                .to_ascii_lowercase();
            if !matches!(evidence_source.as_str(), "nvml" | "corelib") {
                return Err(PowerError::Config(format!(
                    "gpu_attestation.nvattest_gpu_evidence_source must be \"nvml\" or \"corelib\", got {:?}",
                    self.gpu_attestation.nvattest_gpu_evidence_source
                )));
            }
            if evidence_source == "corelib"
                && self.gpu_attestation.nvattest_gpu_architecture.is_none()
            {
                return Err(PowerError::Config(
                    "gpu_attestation.nvattest_gpu_architecture is required when nvattest_gpu_evidence_source = \"corelib\"".to_string(),
                ));
            }
        }

        if self.gpu_attestation.source == GpuAttestationSource::NrasRest {
            if !self.gpu_attestation.evidence_configured() {
                return Err(PowerError::Config(
                    "gpu_attestation.source = \"nras-rest\" requires evidence_hex or evidence_path"
                        .to_string(),
                ));
            }
            if self.gpu_attestation.verdict_configured() {
                return Err(PowerError::Config(
                    "gpu_attestation.source = \"nras-rest\" obtains the verdict from NRAS; verdict_hex/verdict_path must not be configured".to_string(),
                ));
            }
            let architecture = self
                .gpu_attestation
                .nras_gpu_architecture
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    PowerError::Config(
                        "gpu_attestation.nras_gpu_architecture is required when source = \"nras-rest\""
                            .to_string(),
                    )
                })?
                .to_ascii_uppercase();
            if !matches!(architecture.as_str(), "HOPPER" | "BLACKWELL") {
                return Err(PowerError::Config(format!(
                    "gpu_attestation.nras_gpu_architecture must be \"HOPPER\" or \"BLACKWELL\", got {:?}",
                    self.gpu_attestation.nras_gpu_architecture
                )));
            }
            if !matches!(
                self.gpu_attestation.nras_claims_version.trim(),
                "2.0" | "3.0"
            ) {
                return Err(PowerError::Config(format!(
                    "gpu_attestation.nras_claims_version must be \"2.0\" or \"3.0\", got {:?}",
                    self.gpu_attestation.nras_claims_version
                )));
            }
            if self.gpu_attestation.nras_timeout_secs == 0 {
                return Err(PowerError::Config(
                    "gpu_attestation.nras_timeout_secs must be greater than zero".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply `A3S_POWER_*` environment variable overrides.
    pub(super) fn apply_env_overrides(&mut self) -> Result<()> {
        if let Ok(host) = std::env::var("A3S_POWER_HOST") {
            self.host = host;
        }

        if let Ok(port_str) = std::env::var("A3S_POWER_PORT") {
            self.port = parse_required_env_override::<u16>("A3S_POWER_PORT", &port_str)?;
        }

        if let Ok(data_dir) = std::env::var("A3S_POWER_DATA_DIR") {
            self.data_dir = PathBuf::from(data_dir);
        }

        if let Ok(max_str) = std::env::var("A3S_POWER_MAX_MODELS") {
            self.max_loaded_models =
                parse_required_env_override::<usize>("A3S_POWER_MAX_MODELS", &max_str)?;
        }

        if let Ok(value) = std::env::var("A3S_POWER_PROMPT_CACHE_MAX_ENTRIES") {
            self.prompt_cache_max_entries =
                parse_required_env_override::<usize>("A3S_POWER_PROMPT_CACHE_MAX_ENTRIES", &value)?;
        }

        if let Ok(value) = std::env::var("A3S_POWER_PROMPT_CACHE_TTL_SECONDS") {
            self.prompt_cache_ttl_seconds =
                parse_required_env_override::<u64>("A3S_POWER_PROMPT_CACHE_TTL_SECONDS", &value)?;
        }

        if let Ok(keep_alive) = std::env::var("A3S_POWER_KEEP_ALIVE") {
            self.keep_alive = keep_alive;
        }

        if let Ok(spec_mode) = std::env::var("A3S_POWER_SPEC_MODE") {
            self.spec_mode = spec_mode;
        }

        if let Ok(value) = std::env::var("A3S_POWER_SPEC_DRAFT_MAX") {
            self.spec_draft_max = Some(parse_required_env_override::<u32>(
                "A3S_POWER_SPEC_DRAFT_MAX",
                &value,
            )?);
        }

        if let Ok(value) = std::env::var("A3S_POWER_SPEC_MTP_RECURRENT_SNAPSHOTS") {
            self.spec_mtp_recurrent_snapshots = parse_required_env_override::<u32>(
                "A3S_POWER_SPEC_MTP_RECURRENT_SNAPSHOTS",
                &value,
            )?;
        }

        if let Ok(value) = std::env::var("A3S_POWER_SPEC_MTP_RECURRENT_CHAIN") {
            self.spec_mtp_recurrent_chain =
                parse_required_bool_env_override("A3S_POWER_SPEC_MTP_RECURRENT_CHAIN", &value)?;
        }

        if let Ok(value) = std::env::var("A3S_POWER_SPEC_MTP_ADAPTIVE") {
            self.spec_mtp_adaptive =
                parse_required_bool_env_override("A3S_POWER_SPEC_MTP_ADAPTIVE", &value)?;
        }

        if let Ok(value) = std::env::var("A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE") {
            self.spec_mtp_fr_vocab_size = Some(parse_required_env_override::<u32>(
                "A3S_POWER_SPEC_MTP_FR_VOCAB_SIZE",
                &value,
            )?);
        }

        if let Ok(value) = std::env::var("A3S_POWER_SPEC_DRAFT_MIN") {
            self.spec_draft_min =
                parse_required_env_override::<u32>("A3S_POWER_SPEC_DRAFT_MIN", &value)?;
        }

        if let Ok(value) = std::env::var("A3S_POWER_SPEC_DRAFT_P_MIN") {
            self.spec_draft_p_min =
                parse_required_env_override::<f32>("A3S_POWER_SPEC_DRAFT_P_MIN", &value)?;
        }

        if let Ok(gpu_str) = std::env::var("A3S_POWER_GPU_LAYERS") {
            self.gpu.gpu_layers =
                parse_required_env_override::<i32>("A3S_POWER_GPU_LAYERS", &gpu_str)?;
        }

        if let Ok(provider) = std::env::var("A3S_POWER_GPU_ATTESTATION_PROVIDER") {
            self.gpu_attestation.provider = provider;
        }

        if let Ok(source) = std::env::var("A3S_POWER_GPU_ATTESTATION_SOURCE") {
            self.gpu_attestation.source = parse_required_env_override::<GpuAttestationSource>(
                "A3S_POWER_GPU_ATTESTATION_SOURCE",
                &source,
            )?;
        }

        if let Ok(evidence_hex) = std::env::var("A3S_POWER_GPU_ATTESTATION_EVIDENCE_HEX") {
            self.gpu_attestation.evidence_hex = Some(evidence_hex);
            self.gpu_attestation.evidence_path = None;
        }

        if let Ok(evidence_path) = std::env::var("A3S_POWER_GPU_ATTESTATION_EVIDENCE_PATH") {
            self.gpu_attestation.evidence_path = Some(PathBuf::from(evidence_path));
            self.gpu_attestation.evidence_hex = None;
        }

        if let Ok(verdict_hex) = std::env::var("A3S_POWER_GPU_ATTESTATION_VERDICT_HEX") {
            self.gpu_attestation.verdict_hex = Some(verdict_hex);
            self.gpu_attestation.verdict_path = None;
        }

        if let Ok(verdict_path) = std::env::var("A3S_POWER_GPU_ATTESTATION_VERDICT_PATH") {
            self.gpu_attestation.verdict_path = Some(PathBuf::from(verdict_path));
            self.gpu_attestation.verdict_hex = None;
        }

        if let Ok(path) = std::env::var("A3S_POWER_GPU_ATTESTATION_NVATTEST_PATH") {
            self.gpu_attestation.nvattest_path = PathBuf::from(path);
        }

        if let Ok(verifier) = std::env::var("A3S_POWER_GPU_ATTESTATION_NVATTEST_VERIFIER") {
            self.gpu_attestation.nvattest_verifier = verifier;
        }

        if let Ok(source) = std::env::var("A3S_POWER_GPU_ATTESTATION_NVATTEST_GPU_EVIDENCE_SOURCE")
        {
            self.gpu_attestation.nvattest_gpu_evidence_source = source;
        }

        if let Ok(architecture) =
            std::env::var("A3S_POWER_GPU_ATTESTATION_NVATTEST_GPU_ARCHITECTURE")
        {
            self.gpu_attestation.nvattest_gpu_architecture = Some(architecture);
        }

        if let Ok(url) = std::env::var("A3S_POWER_GPU_ATTESTATION_NRAS_URL") {
            self.gpu_attestation.nras_url = Some(url);
        }

        if let Ok(architecture) = std::env::var("A3S_POWER_GPU_ATTESTATION_NRAS_GPU_ARCHITECTURE") {
            self.gpu_attestation.nras_gpu_architecture = Some(architecture);
        }

        if let Ok(claims_version) = std::env::var("A3S_POWER_GPU_ATTESTATION_NRAS_CLAIMS_VERSION") {
            self.gpu_attestation.nras_claims_version = claims_version;
        }

        if let Ok(token_env) = std::env::var("A3S_POWER_GPU_ATTESTATION_NRAS_BEARER_TOKEN_ENV") {
            self.gpu_attestation.nras_bearer_token_env = Some(token_env);
        }

        if let Ok(timeout) = std::env::var("A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS") {
            self.gpu_attestation.nras_timeout_secs = parse_required_env_override::<u64>(
                "A3S_POWER_GPU_ATTESTATION_NRAS_TIMEOUT_SECS",
                &timeout,
            )?;
        }

        if let Ok(url) = std::env::var("A3S_POWER_GPU_ATTESTATION_RIM_URL") {
            self.gpu_attestation.rim_url = Some(url);
        }

        if let Ok(url) = std::env::var("A3S_POWER_GPU_ATTESTATION_OCSP_URL") {
            self.gpu_attestation.ocsp_url = Some(url);
        }

        if let Ok(path) = std::env::var("A3S_POWER_GPU_ATTESTATION_RELYING_PARTY_POLICY_PATH") {
            self.gpu_attestation.relying_party_policy_path = Some(PathBuf::from(path));
        }

        if let Ok(timeout) = std::env::var("A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS") {
            self.gpu_attestation.nvattest_timeout_secs = parse_required_env_override::<u64>(
                "A3S_POWER_GPU_ATTESTATION_NVATTEST_TIMEOUT_SECS",
                &timeout,
            )?;
        }

        if let Ok(tee_str) = std::env::var("A3S_POWER_TEE_MODE") {
            self.tee_mode = parse_required_bool_env_override("A3S_POWER_TEE_MODE", &tee_str)?;
        }

        if let Ok(policy_mode) = std::env::var("A3S_POWER_TEE_POLICY_MODE") {
            self.tee_policy_mode = parse_required_env_override::<TeePolicyMode>(
                "A3S_POWER_TEE_POLICY_MODE",
                &policy_mode,
            )?;
        }

        if let Ok(redact_str) = std::env::var("A3S_POWER_REDACT_LOGS") {
            self.redact_logs =
                parse_required_bool_env_override("A3S_POWER_REDACT_LOGS", &redact_str)?;
        }

        // When TEE mode is enabled, default redact_logs to true unless explicitly disabled
        if self.tee_mode && std::env::var("A3S_POWER_REDACT_LOGS").is_err() && !self.redact_logs {
            self.redact_logs = true;
        }

        if let Ok(tls_port_str) = std::env::var("A3S_POWER_TLS_PORT") {
            self.tls_port = Some(parse_required_env_override::<u16>(
                "A3S_POWER_TLS_PORT",
                &tls_port_str,
            )?);
        }

        if let Ok(ra_tls_str) = std::env::var("A3S_POWER_RA_TLS") {
            self.ra_tls = parse_required_bool_env_override("A3S_POWER_RA_TLS", &ra_tls_str)?;
        }

        if let Ok(vsock_str) = std::env::var("A3S_POWER_VSOCK_PORT") {
            self.vsock_port = Some(parse_required_env_override::<u32>(
                "A3S_POWER_VSOCK_PORT",
                &vsock_str,
            )?);
        }

        if let Ok(keys_str) = std::env::var("A3S_POWER_API_KEYS") {
            let keys: Vec<String> = keys_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !keys.is_empty() {
                self.api_keys = keys;
            }
        }

        // A3S_POWER_TEE_STRICT=1 removes "simulated" from allowed TEE types
        if std::env::var("A3S_POWER_TEE_STRICT").as_deref() == Ok("1") {
            self.tee_policy_mode = TeePolicyMode::Strict;
            if self.allowed_tee_types.is_empty() {
                // Default to all hardware types when strict mode is enabled
                self.allowed_tee_types = vec!["sev-snp".to_string(), "tdx".to_string()];
            } else {
                self.allowed_tee_types.retain(|t| t != "simulated");
            }
        }

        if let Ok(audit_log) = std::env::var("A3S_POWER_AUDIT_LOG") {
            self.audit_log = parse_required_bool_env_override("A3S_POWER_AUDIT_LOG", &audit_log)?;
        }

        if let Ok(enabled) = std::env::var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST") {
            self.proxy_effective_prompt_digest = parse_required_bool_env_override(
                "A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST",
                &enabled,
            )?;
        }

        if let Ok(required) = std::env::var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED") {
            self.proxy_effective_prompt_digest_required = parse_required_bool_env_override(
                "A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_REQUIRED",
                &required,
            )?;
        }

        if let Ok(path) = std::env::var("A3S_POWER_PROXY_EFFECTIVE_PROMPT_DIGEST_PATH") {
            self.proxy_effective_prompt_digest_path = path;
        }

        Ok(())
    }

    /// Save the current configuration to the default config file path (A3S ACL format).
    pub fn save(&self) -> Result<()> {
        let path = dirs::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = self.to_acl()?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    /// Serialize the configuration to canonical A3S ACL syntax.
    pub fn to_acl(&self) -> Result<String> {
        acl::serialize(self)
    }
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Return true when production attestation checks must fail closed.
    pub fn strict_attestation(&self) -> bool {
        matches!(
            self.tee_policy_mode,
            TeePolicyMode::Strict | TeePolicyMode::GpuConfidential
        )
    }

    /// Return the effective TEE type allowlist for the configured policy.
    pub fn effective_allowed_tee_types(&self) -> Vec<String> {
        if !self.allowed_tee_types.is_empty() {
            if self.strict_attestation() {
                return self
                    .allowed_tee_types
                    .iter()
                    .filter(|tee_type| tee_type.as_str() != "simulated")
                    .cloned()
                    .collect();
            }
            return self.allowed_tee_types.clone();
        }

        if self.strict_attestation() {
            return vec!["sev-snp".to_string(), "tdx".to_string()];
        }

        Vec::new()
    }
}
