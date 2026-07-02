pub mod audit;
pub mod auth;
pub mod limiter;
pub(crate) mod lock;
pub mod log_stream;
pub mod metrics;
pub mod request_context;
pub mod router;
pub mod state;
#[cfg(any(
    all(feature = "vsock", target_os = "linux"),
    all(feature = "vsock", test)
))]
pub(crate) mod vsock;

use std::path::Path;
use std::sync::Arc;

use crate::backend;
use crate::config::{GpuAttestationSource, PowerConfig, TeePolicyMode};
use crate::dirs;
use crate::error::{PowerError, Result};
use crate::model::manifest::ModelFormat;
use crate::model::registry::ModelRegistry;
use crate::server::log_stream::LogBuffer;
use crate::tee;
use crate::tee::attestation::{TeeProvider, TeeType};
use crate::tee::gpu::{provider_from_config, GpuEvidenceProvider};
use crate::tee::key_provider;
use crate::tee::policy::{DefaultTeePolicy, TeePolicy};

const NVIDIA_NRAS_PROVIDER: &str = "nvidia-nras";

/// Start the HTTP server with the given configuration.
///
/// If `log_buffer` is provided the server will expose captured log entries via
/// the `GET /v1/logs` SSE endpoint.  Pass a `LogBuffer` whose `LogBufferLayer`
/// has already been installed in the global tracing subscriber so that startup
/// logs are also captured.
pub async fn start(config: PowerConfig) -> Result<()> {
    start_with_log_buffer(config, None).await
}

/// Start the HTTP server, injecting an existing `LogBuffer`.
pub async fn start_with_log_buffer(
    mut config: PowerConfig,
    log_buffer: Option<LogBuffer>,
) -> Result<()> {
    // Ensure storage directories exist
    dirs::ensure_dirs()?;

    // Auto-detect GPU and configure layers if not explicitly set
    backend::gpu::auto_configure(&mut config.gpu);

    // Detect GPU for metrics (recorded after AppState creation)
    let gpu_info = backend::gpu::detect();

    // Initialize model registry and scan for existing models
    let registry = Arc::new(ModelRegistry::new());
    registry.scan()?;
    tracing::info!(count = registry.count(), "Loaded model registry");

    // Initialize key provider early so startup integrity checks can verify
    // encrypted model plaintext hashes with the same semantics as autoload.
    let model_key_provider = key_provider::from_config(&config);

    // TEE initialization
    let mut gpu_evidence_provider: Option<Arc<dyn GpuEvidenceProvider>> = None;

    let tee_provider = if config.tee_mode {
        let provider = tee::attestation::DefaultTeeProvider::detect();
        let tee_type = provider.tee_type();
        tracing::info!(tee_type = %tee_type, "TEE mode enabled");

        // Validate TEE type against policy
        let policy = DefaultTeePolicy::new(
            config.effective_allowed_tee_types(),
            config.expected_measurements.clone(),
        )?;
        policy.validate_tee_type(tee_type)?;
        tracing::info!("TEE policy validation passed");

        validate_strict_tee_config(&config, &registry, tee_type)?;

        if matches!(config.tee_policy_mode, TeePolicyMode::GpuConfidential) {
            let provider = provider_from_config(&config.gpu_attestation)?.ok_or_else(|| {
                PowerError::Config(
                    "tee_policy_mode = \"gpu-confidential\" requires gpu_attestation.source = \"configured\", \"nvattest-cli\", or \"nras-rest\" with a usable evidence source".to_string(),
                )
            })?;
            let claim = provider.evidence_claim().await.map_err(|e| {
                PowerError::Config(format!(
                    "failed to load NVIDIA GPU confidential-computing evidence: {e}"
                ))
            })?;
            if claim.verdict_digest.is_none() {
                return Err(PowerError::Config(
                    "tee_policy_mode = \"gpu-confidential\" requires an NVIDIA NRAS verdict digest to be bound".to_string(),
                ));
            }
            tracing::info!(
                provider = %claim.provider,
                evidence_digest = %hex::encode(&claim.evidence_digest),
                verdict_digest = ?claim.verdict_digest.as_ref().map(hex::encode),
                "NVIDIA GPU confidential-computing evidence binding configured"
            );
            gpu_evidence_provider = Some(Arc::from(provider));
        }

        // Validate measurement against expected values if configured
        if !config.expected_measurements.is_empty() {
            match provider.attestation_report(None).await {
                Ok(report) => {
                    policy.validate_measurement(tee_type, &report.measurement)?;
                    tracing::info!("TEE measurement validation passed");
                }
                Err(e) => {
                    return Err(PowerError::Config(format!(
                        "Failed to get attestation report for measurement validation: {e}"
                    )));
                }
            }
        }

        // Verify model integrity against configured hashes
        if !config.model_hashes.is_empty() {
            let verified = tee::model_seal::verify_all_models_with_key_provider(
                &registry,
                &config.model_hashes,
                model_key_provider.as_deref(),
            )
            .await?;
            tracing::info!(count = verified, "All model integrity checks passed");
        }

        // Verify model signatures if signing key is configured
        if let Some(ref signing_key) = config.model_signing_key {
            let verified = tee::model_seal::verify_all_signatures(&registry, signing_key)?;
            tracing::info!(count = verified, "All model signature checks passed");
        }

        Some(Arc::new(provider) as Arc<dyn TeeProvider>)
    } else {
        None
    };

    let bind_addr = config.bind_address();
    let config = Arc::new(config);

    // Initialize backends
    let backends = Arc::new(backend::default_backends(config.clone()));
    tracing::info!(
        backends = ?backends.list_names(),
        "Initialized backends"
    );

    // Register proxied (remote) models from config so they appear in /v1/models
    // and route to the proxy backend. In-memory only (config is the source).
    for name in config.proxy_upstreams.keys() {
        match registry.register_transient(crate::model::manifest::ModelManifest::remote(name)) {
            Ok(()) => tracing::info!(model = %name, "Registered proxy (remote) model"),
            Err(e) => tracing::warn!(model = %name, error = %e, "Failed to register proxy model"),
        }
    }

    let mut app_state = match log_buffer {
        Some(buf) => state::AppState::with_log_buffer(registry, backends, config.clone(), buf),
        None => state::AppState::new(registry, backends, config.clone()),
    };
    if let Some(provider) = tee_provider {
        app_state = app_state.with_tee_provider(provider);
    }
    if let Some(provider) = gpu_evidence_provider {
        app_state = app_state.with_gpu_evidence_provider(provider);
    }
    if config.tee_mode || config.redact_logs || config.suppress_token_metrics {
        let redact = config.redact_logs || config.tee_mode;
        let privacy = tee::privacy::DefaultPrivacyProvider::new(redact)
            .with_suppress_token_metrics(config.suppress_token_metrics || redact);
        app_state = app_state.with_privacy(Arc::new(privacy));
    }

    if !config.api_keys.is_empty() {
        let auth_provider = auth::ApiKeyAuth::new(&config.api_keys);
        app_state = app_state.with_auth(Arc::new(auth_provider));
        tracing::info!(
            keys = config.api_keys.len(),
            "API key authentication enabled"
        );
    }

    if config.audit_log {
        let log_path = config
            .audit_log_path
            .clone()
            .unwrap_or_else(|| dirs::power_home().join("audit.jsonl"));

        if config.audit_log_encrypt {
            // Encrypted audit logging: AES-256-GCM per-line encryption
            let key_source = config.audit_key_source.as_ref().ok_or_else(|| {
                PowerError::Config(
                    "audit_log_encrypt = true but audit_key_source is not configured".to_string(),
                )
            })?;
            let key = crate::tee::encrypted_model::load_key(key_source)?;
            match audit::EncryptedAuditLogger::open(log_path.clone(), key) {
                Ok(logger) => {
                    app_state = app_state.with_audit(Arc::new(logger));
                    tracing::info!(path = %log_path.display(), "Audit logging enabled (encrypted)");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to open encrypted audit log, audit logging disabled");
                }
            }
        } else {
            match audit::AsyncJsonLinesAuditLogger::open(log_path.clone()) {
                Ok(logger) => {
                    app_state = app_state.with_audit(Arc::new(logger));
                    tracing::info!(path = %log_path.display(), "Audit logging enabled (async)");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to open audit log, audit logging disabled");
                }
            }
        }
    }

    // Attach key provider for encrypted model loading
    if let Some(provider) = model_key_provider {
        tracing::info!(
            provider = provider.provider_name(),
            "Key provider initialized"
        );
        app_state = app_state.with_key_provider(provider);
    }

    // Spawn background keep_alive reaper task
    spawn_keep_alive_reaper(app_state.clone());

    // Record detected GPU memory in Prometheus metrics
    if gpu_info.vram_bytes > 0 {
        app_state
            .metrics
            .set_gpu_memory(&gpu_info.name, gpu_info.vram_bytes);
        tracing::info!(
            gpu = %gpu_info.name,
            vram = %gpu_info.vram_display(),
            backend = %gpu_info.backend,
            "GPU detected"
        );
    }

    // Spawn SIGHUP handler for key rotation (Unix only)
    #[cfg(unix)]
    if app_state.key_provider.is_some() {
        spawn_key_rotation_handler(app_state.clone());
    }

    let app_state_for_shutdown = app_state.clone();
    let app = router::build(app_state.clone());

    // Start TLS server in a background task if configured.
    #[cfg(feature = "tls")]
    if config.tls_port.is_some() {
        spawn_tls_server(&config, app.clone(), app_state.tee_provider.as_ref()).await?;
    }

    // Start vsock server in a background task if configured (Linux only).
    #[cfg(all(feature = "vsock", target_os = "linux"))]
    if let Some(vsock_port) = config.vsock_port {
        vsock::spawn_vsock_server(vsock_port, app.clone()).await?;
    }

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| PowerError::Server(format!("Failed to bind to {bind_addr}: {e}")))?;

    tracing::info!("Server listening on {bind_addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(app_state_for_shutdown))
        .await
        .map_err(|e| PowerError::Server(format!("Server error: {e}")))?;

    Ok(())
}

/// Spawn a TLS (RA-TLS) server in a background task.
///
/// Generates a self-signed certificate at startup. When `ra_tls = true` and a
/// TEE provider is present, the attestation report is embedded in the cert as
/// a custom X.509 extension (OID 1.3.6.1.4.1.56560.1.1).
#[cfg(feature = "tls")]
async fn spawn_tls_server(
    config: &crate::config::PowerConfig,
    app: axum::Router,
    tee_provider: Option<&std::sync::Arc<dyn crate::tee::attestation::TeeProvider>>,
) -> Result<()> {
    let tls_port = config.tls_port.ok_or_else(|| {
        PowerError::Config("tls_port must be configured before starting TLS server".to_string())
    })?;

    // Optionally get an attestation report to embed in the certificate.
    let attestation = if config.ra_tls && config.tee_mode {
        match tee_provider {
            Some(provider) => match provider.attestation_report(None).await {
                Ok(report) => Some(report),
                Err(e) => {
                    tracing::warn!("RA-TLS: failed to get attestation report, proceeding without embedding: {e}");
                    None
                }
            },
            None => {
                tracing::warn!("RA-TLS requested but no TEE provider configured; proceeding without attestation");
                None
            }
        }
    } else {
        None
    };

    let cert = crate::tee::cert::CertManager::generate(attestation.as_ref(), &config.tls_sans)
        .map_err(|e| PowerError::Server(format!("TLS certificate generation failed: {e}")))?;

    let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem(
        cert.cert_pem().as_bytes().to_vec(),
        cert.key_pem().as_bytes().to_vec(),
    )
    .await
    .map_err(|e| PowerError::Server(format!("TLS rustls config error: {e}")))?;

    let tls_bind = format!("{}:{}", config.host, tls_port);
    let tls_addr: std::net::SocketAddr = tls_bind
        .parse()
        .map_err(|e| PowerError::Server(format!("Invalid TLS bind address {tls_bind}: {e}")))?;

    tracing::info!(
        addr = %tls_bind,
        ra_tls = config.ra_tls && attestation.is_some(),
        "TLS server listening"
    );

    tokio::spawn(async move {
        if let Err(e) = axum_server::bind_rustls(tls_addr, tls_config)
            .serve(app.into_make_service())
            .await
        {
            tracing::error!("TLS server error: {e}");
        }
    });

    Ok(())
}

/// Wait for SIGTERM or Ctrl-C, then perform graceful shutdown cleanup.
///
/// Cleanup order (TEE security requirements):
/// 1. Unload all loaded models — triggers RAII zeroize of decrypted weights
/// 2. Flush audit log — ensures no events are lost on shutdown
async fn shutdown_signal(state: state::AppState) {
    // Wait for either SIGTERM (systemd/Kubernetes) or Ctrl-C.
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                tokio::select! {
                    received = sigterm.recv() => {
                        if received.is_some() {
                            tracing::info!("SIGTERM received, starting graceful shutdown");
                        } else {
                            tracing::warn!("SIGTERM handler closed; starting graceful shutdown");
                        }
                    }
                    _ = wait_for_ctrl_c_signal() => {}
                }
            }
            Err(e) => {
                tracing::warn!("Failed to register SIGTERM handler; waiting for Ctrl-C only: {e}");
                wait_for_ctrl_c_signal().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        wait_for_ctrl_c_signal().await;
    }

    unload_models_for_shutdown(&state).await;

    // Flush audit log to ensure no events are lost
    if let Some(ref audit) = state.audit {
        if let Err(e) = audit.flush().await {
            tracing::warn!("Failed to flush audit log on shutdown: {e}");
        } else {
            tracing::info!("Audit log flushed");
        }
    }

    tracing::info!("Graceful shutdown complete");
}

// Unload all models to trigger RAII cleanup (zeroize decrypted weights).
async fn unload_models_for_shutdown(state: &state::AppState) {
    let loaded = state.loaded_model_names();
    if !loaded.is_empty() {
        tracing::info!(count = loaded.len(), "Unloading all models before shutdown");
        for model_name in &loaded {
            let format = state
                .registry
                .get(model_name)
                .map(|m| m.format.clone())
                .unwrap_or(crate::model::manifest::ModelFormat::Gguf);

            let backend = match state.backends.find_for_format(&format) {
                Ok(backend) => backend,
                Err(e) => {
                    tracing::warn!(
                        model = %model_name,
                        error = %e,
                        "Failed to find backend for shutdown unload"
                    );
                    continue;
                }
            };

            if let Err(e) = backend.unload(model_name).await {
                tracing::warn!(
                    model = %model_name,
                    error = %e,
                    "Failed to unload model on shutdown"
                );
                continue;
            }

            state.mark_unloaded(model_name);
            tracing::info!(model = %model_name, "Model unloaded on shutdown");
        }
    }
}

fn validate_strict_tee_config(
    config: &PowerConfig,
    registry: &ModelRegistry,
    tee_type: TeeType,
) -> Result<()> {
    if !config.strict_attestation() {
        return Ok(());
    }

    validate_strict_measurement_policy(config, tee_type)?;

    if matches!(config.tee_policy_mode, TeePolicyMode::GpuConfidential) {
        validate_gpu_confidential_provider(&config.gpu_attestation.provider)?;

        match config.gpu_attestation.source {
            GpuAttestationSource::Configured => {
                if !config.gpu_attestation.evidence_configured() {
                    return Err(PowerError::Config(
                        "tee_policy_mode = \"gpu-confidential\" with gpu_attestation.source = \"configured\" requires gpu_attestation.evidence_path or gpu_attestation.evidence_hex".to_string(),
                    ));
                }
                if !config.gpu_attestation.verdict_configured() {
                    return Err(PowerError::Config(
                        "tee_policy_mode = \"gpu-confidential\" with gpu_attestation.source = \"configured\" requires gpu_attestation.verdict_path or gpu_attestation.verdict_hex to bind an NVIDIA NRAS verdict".to_string(),
                    ));
                }
                if let Some(path) = &config.gpu_attestation.evidence_path {
                    validate_gpu_confidential_file_source_path(
                        "gpu_attestation.evidence_path",
                        path,
                    )?;
                }
                if let Some(path) = &config.gpu_attestation.verdict_path {
                    validate_gpu_confidential_file_source_path(
                        "gpu_attestation.verdict_path",
                        path,
                    )?;
                }
            }
            GpuAttestationSource::NvattestCli => {
                if config
                    .gpu_attestation
                    .nvattest_verifier
                    .trim()
                    .to_ascii_lowercase()
                    != "remote"
                {
                    return Err(PowerError::Config(
                        "tee_policy_mode = \"gpu-confidential\" with gpu_attestation.source = \"nvattest-cli\" requires gpu_attestation.nvattest_verifier = \"remote\" so NVIDIA NRAS verifies GPU evidence".to_string(),
                    ));
                }
                validate_gpu_confidential_nvattest_path(&config.gpu_attestation.nvattest_path)?;
                if let Some(path) = &config.gpu_attestation.relying_party_policy_path {
                    validate_gpu_confidential_policy_path(
                        "gpu_attestation.relying_party_policy_path",
                        path,
                    )?;
                }
                if let Some(url) = &config.gpu_attestation.nras_url {
                    validate_gpu_confidential_https_url("gpu_attestation.nras_url", url)?;
                }
                if let Some(url) = &config.gpu_attestation.rim_url {
                    validate_gpu_confidential_https_url("gpu_attestation.rim_url", url)?;
                }
                if let Some(url) = &config.gpu_attestation.ocsp_url {
                    validate_gpu_confidential_https_url("gpu_attestation.ocsp_url", url)?;
                }
                if config.gpu_attestation.nvattest_timeout_secs == 0 {
                    return Err(PowerError::Config(
                        "gpu_attestation.nvattest_timeout_secs must be greater than zero"
                            .to_string(),
                    ));
                }
            }
            GpuAttestationSource::NrasRest => {
                if !config.gpu_attestation.evidence_configured() {
                    return Err(PowerError::Config(
                        "tee_policy_mode = \"gpu-confidential\" with gpu_attestation.source = \"nras-rest\" requires gpu_attestation.evidence_path or gpu_attestation.evidence_hex".to_string(),
                    ));
                }
                if let Some(path) = &config.gpu_attestation.evidence_path {
                    validate_gpu_confidential_file_source_path(
                        "gpu_attestation.evidence_path",
                        path,
                    )?;
                }
                if config.gpu_attestation.verdict_configured() {
                    return Err(PowerError::Config(
                        "tee_policy_mode = \"gpu-confidential\" with gpu_attestation.source = \"nras-rest\" obtains the verdict from NRAS; gpu_attestation.verdict_path/verdict_hex must not be configured".to_string(),
                    ));
                }
                if config.gpu_attestation.nras_gpu_architecture.is_none() {
                    return Err(PowerError::Config(
                        "gpu_attestation.nras_gpu_architecture is required when source = \"nras-rest\"".to_string(),
                    ));
                }
                if !matches!(
                    config.gpu_attestation.nras_claims_version.trim(),
                    "2.0" | "3.0"
                ) {
                    return Err(PowerError::Config(
                        "gpu_attestation.nras_claims_version must be \"2.0\" or \"3.0\""
                            .to_string(),
                    ));
                }
                if config.gpu_attestation.nras_timeout_secs == 0 {
                    return Err(PowerError::Config(
                        "gpu_attestation.nras_timeout_secs must be greater than zero".to_string(),
                    ));
                }
                if let Some(url) = &config.gpu_attestation.nras_url {
                    validate_gpu_confidential_https_url("gpu_attestation.nras_url", url)?;
                }
            }
        }
    }

    if matches!(config.tee_policy_mode, TeePolicyMode::GpuConfidential)
        && config.gpu.gpu_layers == 0
    {
        return Err(PowerError::Config(
            "tee_policy_mode = \"gpu-confidential\" requires gpu.gpu_layers to enable GPU execution/offload; use tee_policy_mode = \"strict\" for CPU-only TEE deployments".to_string(),
        ));
    }

    if config.model_hashes.is_empty() && config.model_signing_key.is_none() {
        return Err(PowerError::Config(
            "strict TEE mode requires model_hashes or model_signing_key to pin local model integrity".to_string(),
        ));
    }

    if config.model_signing_key.is_none() {
        for manifest in registry.list()? {
            if manifest.format == ModelFormat::Remote {
                continue;
            }
            if !config.model_hashes.contains_key(&manifest.name) {
                return Err(PowerError::Config(format!(
                    "strict TEE mode requires a model_hashes entry for '{}'",
                    manifest.name
                )));
            }
        }
    }

    Ok(())
}

fn validate_gpu_confidential_provider(provider: &str) -> Result<()> {
    if provider.trim() != NVIDIA_NRAS_PROVIDER {
        return Err(PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires gpu_attestation.provider = \"{NVIDIA_NRAS_PROVIDER}\" for NVIDIA GPU confidential-computing evidence, got {provider:?}"
        )));
    }

    Ok(())
}

fn validate_gpu_confidential_nvattest_path(path: &Path) -> Result<()> {
    validate_gpu_confidential_regular_file_path(
        "gpu_attestation.nvattest_path",
        path,
        RegularFileRequirement::Executable,
    )
}

fn validate_gpu_confidential_policy_path(field: &str, path: &Path) -> Result<()> {
    validate_gpu_confidential_regular_file_path(field, path, RegularFileRequirement::NonEmpty)
}

fn validate_gpu_confidential_file_source_path(field: &str, path: &Path) -> Result<()> {
    validate_gpu_confidential_regular_file_path(field, path, RegularFileRequirement::NonEmpty)
}

#[derive(Clone, Copy)]
enum RegularFileRequirement {
    Executable,
    NonEmpty,
}

fn validate_gpu_confidential_regular_file_path(
    field: &str,
    path: &Path,
    requirement: RegularFileRequirement,
) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires {field} to be non-empty when configured"
        )));
    }
    if !path.is_absolute() {
        return Err(PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires {field} to be an absolute path, got {}",
            path.display()
        )));
    }

    let metadata = std::fs::metadata(path).map_err(|e| {
        PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires {field} to point to an existing regular file at {}: {e}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires {field} to point to a regular file, got {}",
            path.display()
        )));
    }

    match requirement {
        RegularFileRequirement::Executable => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                if metadata.permissions().mode() & 0o111 == 0 {
                    return Err(PowerError::Config(format!(
                        "tee_policy_mode = \"gpu-confidential\" requires {field} to be executable by at least one class, got {}",
                        path.display()
                    )));
                }
            }
        }
        RegularFileRequirement::NonEmpty => {
            if metadata.len() == 0 {
                return Err(PowerError::Config(format!(
                    "tee_policy_mode = \"gpu-confidential\" requires {field} to be non-empty, got {}",
                    path.display()
                )));
            }
        }
    }

    Ok(())
}

fn validate_gpu_confidential_https_url(field: &str, url: &str) -> Result<()> {
    let url = url.trim();
    if url.is_empty() {
        return Err(PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires {field} to be non-empty when configured"
        )));
    }

    let parsed = reqwest::Url::parse(url).map_err(|e| {
        PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires {field} to be a valid HTTPS URL: {e}"
        ))
    })?;
    if parsed.scheme() != "https" {
        return Err(PowerError::Config(format!(
            "tee_policy_mode = \"gpu-confidential\" requires {field} to use https, got {:?}",
            parsed.scheme()
        )));
    }

    Ok(())
}

fn validate_strict_measurement_policy(config: &PowerConfig, tee_type: TeeType) -> Result<()> {
    let tee_type_name = tee_type_config_name(tee_type);
    let Some(expected_measurement) = config.expected_measurements.get(tee_type_name) else {
        return Err(PowerError::Config(format!(
            "strict TEE mode requires expected_measurements entry for detected TEE type '{tee_type_name}'"
        )));
    };

    let expected_measurement = expected_measurement.trim();
    let measurement = hex::decode(expected_measurement).map_err(|e| {
        PowerError::Config(format!(
            "expected_measurements.{tee_type_name} must be a hex-encoded 48-byte launch measurement: {e}"
        ))
    })?;
    if measurement.len() != 48 {
        return Err(PowerError::Config(format!(
            "expected_measurements.{tee_type_name} must be 48 bytes (96 hex characters), got {} bytes",
            measurement.len()
        )));
    }

    Ok(())
}

fn tee_type_config_name(tee_type: TeeType) -> &'static str {
    match tee_type {
        TeeType::SevSnp => "sev-snp",
        TeeType::Tdx => "tdx",
        TeeType::Simulated => "simulated",
        TeeType::None => "none",
    }
}

async fn wait_for_ctrl_c_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => tracing::info!("Ctrl-C received, starting graceful shutdown"),
        Err(e) => {
            tracing::warn!("Failed to wait for Ctrl-C; graceful shutdown signal disabled: {e}");
            std::future::pending::<()>().await;
        }
    }
}

/// Spawn a background task that periodically checks for models whose keep_alive
/// has expired and unloads them to free memory.
fn spawn_keep_alive_reaper(state: state::AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            reap_expired_models_once(&state).await;
        }
    });
}

async fn reap_expired_models_once(state: &state::AppState) {
    let expired = state.expired_models();
    for model_name in expired {
        // Look up the model's actual format to find the right backend.
        let format = state
            .registry
            .get(&model_name)
            .map(|m| m.format.clone())
            .unwrap_or(crate::model::manifest::ModelFormat::Gguf);

        let backend = match state.backends.find_for_format(&format) {
            Ok(backend) => backend,
            Err(e) => {
                tracing::warn!(
                    model = %model_name,
                    error = %e,
                    "Failed to find backend for expired model"
                );
                continue;
            }
        };

        if let Err(e) = backend.unload(&model_name).await {
            tracing::warn!(
                model = %model_name,
                "Failed to unload expired model: {e}"
            );
            continue;
        }

        state.mark_unloaded(&model_name);
        state.metrics.increment_evictions();
        state.metrics.remove_model_memory(&model_name);
        tracing::info!(
            model = %model_name,
            "Unloaded model (keep_alive expired)"
        );
    }
}

/// Spawn a SIGHUP handler that triggers key rotation.
///
/// When a SIGHUP signal is received, calls `rotate_key()` on the key provider.
/// This enables zero-downtime key rotation: deploy a new key, send SIGHUP,
/// then remove the old key.
#[cfg(unix)]
fn spawn_key_rotation_handler(state: state::AppState) {
    tokio::spawn(async move {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sighup = match signal(SignalKind::hangup()) {
            Ok(sighup) => sighup,
            Err(e) => {
                tracing::warn!(
                    "Failed to register SIGHUP handler; key rotation signal disabled: {e}"
                );
                return;
            }
        };

        loop {
            if sighup.recv().await.is_none() {
                tracing::warn!("SIGHUP handler closed; key rotation signal disabled");
                return;
            }
            tracing::info!("SIGHUP received, attempting key rotation");

            if let Some(ref kp) = state.key_provider {
                match kp.rotate_key().await {
                    Ok(_) => {
                        tracing::info!(provider = kp.provider_name(), "Key rotation successful");
                    }
                    Err(e) => {
                        tracing::warn!(
                            provider = kp.provider_name(),
                            error = %e,
                            "Key rotation failed"
                        );
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{test_state_with_mock, MockBackend};
    use crate::backend::BackendRegistry;
    use crate::config::{GpuAttestationConfig, GpuAttestationSource, GpuConfig, TeePolicyMode};
    use crate::model::manifest::ModelManifest;
    use crate::model::registry::ModelRegistry;
    use serial_test::serial;
    use std::{collections::HashMap, fs, path::PathBuf};

    fn local_manifest(name: &str) -> ModelManifest {
        ModelManifest {
            name: name.to_string(),
            format: ModelFormat::Gguf,
            size: 8,
            sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: PathBuf::from(format!("/tmp/{name}.gguf")),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: Vec::new(),
            family: None,
            families: None,
        }
    }

    fn measurement_pins() -> HashMap<String, String> {
        HashMap::from([("sev-snp".to_string(), "00".repeat(48))])
    }

    fn test_nvattest_path() -> PathBuf {
        std::env::current_exe().unwrap()
    }

    fn gpu_confidential_nvattest_config(gpu_attestation: GpuAttestationConfig) -> PowerConfig {
        let mut gpu_attestation = gpu_attestation;
        if matches!(gpu_attestation.source, GpuAttestationSource::NvattestCli)
            && !gpu_attestation.nvattest_path.is_absolute()
        {
            gpu_attestation.nvattest_path = test_nvattest_path();
        }

        PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation,
            expected_measurements: measurement_pins(),
            ..Default::default()
        }
    }

    #[test]
    fn test_spawn_keep_alive_reaper_compiles() {
        // Test that the function signature is correct
        // We can't easily test the actual spawning without a full runtime setup
        // but we can verify the function exists and has the right signature
    }

    #[test]
    fn test_server_module_exports() {
        // Verify that the module exports the expected items
        let _ = start;
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_keep_alive_reaper_unloads_expired_models() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());

        // Load a model with very short keep_alive
        state.mark_loaded_with_keep_alive("test-model", std::time::Duration::from_millis(10));
        assert!(state.is_model_loaded("test-model"));

        // Spawn the reaper
        spawn_keep_alive_reaper(state.clone());

        // Wait for expiry and reaper tick (reaper runs every 5 seconds)
        tokio::time::sleep(std::time::Duration::from_secs(6)).await;

        // Model should be unloaded
        assert!(!state.is_model_loaded("test-model"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_keep_alive_reaper_preserves_non_expired() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());

        // Load a model with long keep_alive
        state.mark_loaded_with_keep_alive("long-lived", std::time::Duration::from_secs(300));
        assert!(state.is_model_loaded("long-lived"));

        // Spawn the reaper
        spawn_keep_alive_reaper(state.clone());

        // Wait for reaper tick
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Model should still be loaded
        assert!(state.is_model_loaded("long-lived"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_reap_expired_models_preserves_loaded_state_without_backend() {
        let state = state::AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded_with_keep_alive("orphaned-model", std::time::Duration::from_millis(1));
        std::thread::sleep(std::time::Duration::from_millis(5));

        reap_expired_models_once(&state).await;

        assert!(state.is_model_loaded("orphaned-model"));
    }

    #[tokio::test]
    async fn test_reap_expired_models_preserves_loaded_state_when_unload_fails() {
        let state = test_state_with_mock(MockBackend::unload_fails());
        state.mark_loaded_with_keep_alive("sticky-model", std::time::Duration::from_millis(1));
        std::thread::sleep(std::time::Duration::from_millis(5));

        reap_expired_models_once(&state).await;

        assert!(state.is_model_loaded("sticky-model"));
    }

    #[tokio::test]
    async fn test_shutdown_unload_marks_unloaded_on_success() {
        let state = test_state_with_mock(MockBackend::success());
        state.mark_loaded("shutdown-model");

        unload_models_for_shutdown(&state).await;

        assert!(!state.is_model_loaded("shutdown-model"));
    }

    #[tokio::test]
    async fn test_shutdown_unload_preserves_loaded_state_without_backend() {
        let state = state::AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("orphaned-shutdown-model");

        unload_models_for_shutdown(&state).await;

        assert!(state.is_model_loaded("orphaned-shutdown-model"));
    }

    #[tokio::test]
    async fn test_shutdown_unload_preserves_loaded_state_when_unload_fails() {
        let state = test_state_with_mock(MockBackend::unload_fails());
        state.mark_loaded("sticky-shutdown-model");

        unload_models_for_shutdown(&state).await;

        assert!(state.is_model_loaded("sticky-shutdown-model"));
    }

    #[test]
    fn test_config_bind_address() {
        let config = PowerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
            ..Default::default()
        };
        assert_eq!(config.bind_address(), "0.0.0.0:8080");
    }

    #[test]
    fn test_config_bind_address_default() {
        let config = PowerConfig::default();
        assert_eq!(config.bind_address(), "127.0.0.1:11434");
    }

    #[test]
    fn test_validate_strict_tee_config_development_allows_unpinned_models() {
        let registry = ModelRegistry::new();
        registry
            .register_transient(local_manifest("dev-model"))
            .unwrap();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::Development,
            ..Default::default()
        };

        validate_strict_tee_config(&config, &registry, TeeType::Simulated).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_requires_measurement_pin() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();
        assert!(err
            .to_string()
            .contains("requires expected_measurements entry"));
    }

    #[test]
    fn test_validate_strict_tee_config_rejects_short_measurement_pin() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            expected_measurements: HashMap::from([("sev-snp".to_string(), "deadbeef".to_string())]),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();
        assert!(err.to_string().contains("must be 48 bytes"));
    }

    #[test]
    fn test_validate_strict_tee_config_requires_integrity_policy() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();
        assert!(err
            .to_string()
            .contains("requires model_hashes or model_signing_key"));
    }

    #[test]
    fn test_validate_strict_tee_config_requires_each_local_model_hash() {
        let registry = ModelRegistry::new();
        registry
            .register_transient(local_manifest("unpinned-model"))
            .unwrap();
        let config = PowerConfig {
            model_hashes: HashMap::from([(
                "other-model".to_string(),
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            )]),
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();
        assert!(err.to_string().contains("unpinned-model"));
    }

    #[test]
    fn test_validate_strict_tee_config_accepts_model_signing_key_policy() {
        let registry = ModelRegistry::new();
        registry
            .register_transient(local_manifest("signed-model"))
            .unwrap();
        let config = PowerConfig {
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_requires_gpu_evidence() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();
        assert!(err.to_string().contains("gpu_attestation.evidence"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_requires_nras_verdict() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                evidence_hex: Some("0011".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();
        assert!(err.to_string().contains("gpu_attestation.verdict"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_accepts_bound_evidence() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                evidence_hex: Some("0011".to_string()),
                verdict_hex: Some("2233".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_accepts_configured_file_sources() {
        let registry = ModelRegistry::new();
        let evidence = tempfile::NamedTempFile::new().unwrap();
        let verdict = tempfile::NamedTempFile::new().unwrap();
        fs::write(evidence.path(), "evidence").unwrap();
        fs::write(verdict.path(), "verdict").unwrap();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                evidence_path: Some(evidence.path().to_path_buf()),
                verdict_path: Some(verdict.path().to_path_buf()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_relative_configured_evidence_path()
    {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                evidence_path: Some(PathBuf::from("gpu-evidence.json")),
                verdict_hex: Some("2233".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.evidence_path to be an absolute path"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_non_nvidia_provider() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                provider: "custom-provider".to_string(),
                evidence_hex: Some("0011".to_string()),
                verdict_hex: Some("2233".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("gpu_attestation.provider = \"nvidia-nras\""));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_requires_gpu_offload() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                evidence_hex: Some("0011".to_string()),
                verdict_hex: Some("2233".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();
        assert!(err.to_string().contains("gpu.gpu_layers"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_accepts_nvattest_cli() {
        let registry = ModelRegistry::new();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            ..Default::default()
        });

        validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_accepts_nvattest_policy_file() {
        let registry = ModelRegistry::new();
        let policy = tempfile::NamedTempFile::new().unwrap();
        fs::write(policy.path(), "{}").unwrap();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            relying_party_policy_path: Some(policy.path().to_path_buf()),
            ..Default::default()
        });

        validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_relative_nvattest_path() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                source: GpuAttestationSource::NvattestCli,
                nvattest_path: PathBuf::from("nvattest"),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.nvattest_path to be an absolute path"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_missing_nvattest_path() {
        let registry = ModelRegistry::new();
        let dir = tempfile::tempdir().unwrap();
        let missing_path = dir.path().join("missing-nvattest");
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_path: missing_path,
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err.to_string().contains(
            "requires gpu_attestation.nvattest_path to point to an existing regular file"
        ));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_relative_policy_path() {
        let registry = ModelRegistry::new();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            relying_party_policy_path: Some(PathBuf::from("nras-policy.json")),
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.relying_party_policy_path to be an absolute path"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_empty_policy_file() {
        let registry = ModelRegistry::new();
        let policy = tempfile::NamedTempFile::new().unwrap();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            relying_party_policy_path: Some(policy.path().to_path_buf()),
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.relying_party_policy_path to be non-empty"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_policy_directory() {
        let registry = ModelRegistry::new();
        let dir = tempfile::tempdir().unwrap();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            relying_party_policy_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err.to_string().contains(
            "requires gpu_attestation.relying_party_policy_path to point to a regular file"
        ));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_accepts_https_nvattest_service_urls() {
        let registry = ModelRegistry::new();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nras_url: Some("https://nras.attestation.nvidia.com".to_string()),
            rim_url: Some("https://rim.attestation.nvidia.com".to_string()),
            ocsp_url: Some("https://ocsp.attestation.nvidia.com".to_string()),
            ..Default::default()
        });

        validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_local_nvattest_verifier() {
        let registry = ModelRegistry::new();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nvattest_verifier: "local".to_string(),
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err.to_string().contains("nvattest_verifier = \"remote\""));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_http_nvattest_nras_url() {
        let registry = ModelRegistry::new();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            nras_url: Some("http://nras.attestation.nvidia.com".to_string()),
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.nras_url to use https"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_http_nvattest_rim_url() {
        let registry = ModelRegistry::new();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            rim_url: Some("http://rim.attestation.nvidia.com".to_string()),
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.rim_url to use https"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_http_nvattest_ocsp_url() {
        let registry = ModelRegistry::new();
        let config = gpu_confidential_nvattest_config(GpuAttestationConfig {
            source: GpuAttestationSource::NvattestCli,
            ocsp_url: Some("http://ocsp.attestation.nvidia.com".to_string()),
            ..Default::default()
        });

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.ocsp_url to use https"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_accepts_nras_rest() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                source: GpuAttestationSource::NrasRest,
                evidence_hex: Some(hex::encode(
                    br#"{"evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}"#,
                )),
                nras_gpu_architecture: Some("HOPPER".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap();
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_relative_nras_rest_evidence_path() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                source: GpuAttestationSource::NrasRest,
                evidence_path: Some(PathBuf::from("gpu-evidence.json")),
                nras_gpu_architecture: Some("HOPPER".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.evidence_path to be an absolute path"));
    }

    #[test]
    fn test_validate_strict_tee_config_gpu_confidential_rejects_http_nras_rest_url() {
        let registry = ModelRegistry::new();
        let config = PowerConfig {
            tee_policy_mode: TeePolicyMode::GpuConfidential,
            gpu: GpuConfig {
                gpu_layers: -1,
                ..Default::default()
            },
            model_signing_key: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
            gpu_attestation: GpuAttestationConfig {
                source: GpuAttestationSource::NrasRest,
                evidence_hex: Some(hex::encode(
                    br#"{"evidence":"ZXZpZGVuY2U","certificate":"Y2VydA"}"#,
                )),
                nras_url: Some("http://nras.attestation.nvidia.com".to_string()),
                nras_gpu_architecture: Some("HOPPER".to_string()),
                ..Default::default()
            },
            expected_measurements: measurement_pins(),
            ..Default::default()
        };

        let err = validate_strict_tee_config(&config, &registry, TeeType::SevSnp).unwrap_err();

        assert!(err
            .to_string()
            .contains("requires gpu_attestation.nras_url to use https"));
    }
}
