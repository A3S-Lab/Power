pub mod audit;
pub mod auth;
pub(crate) mod lock;
pub mod metrics;
pub mod request_context;
pub mod router;
pub mod state;
#[cfg(all(feature = "vsock", target_os = "linux"))]
pub(crate) mod vsock;

use std::sync::Arc;

use crate::backend;
use crate::config::PowerConfig;
use crate::dirs;
use crate::error::{PowerError, Result};
use crate::model::registry::ModelRegistry;
use crate::tee;
use crate::tee::attestation::TeeProvider;
use crate::tee::key_provider;
use crate::tee::policy::{DefaultTeePolicy, TeePolicy};

/// Start the HTTP server with the given configuration.
pub async fn start(mut config: PowerConfig) -> Result<()> {
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

    // TEE initialization
    let tee_provider = if config.tee_mode {
        let provider = tee::attestation::DefaultTeeProvider::detect();
        let tee_type = provider.tee_type();
        tracing::info!(tee_type = %tee_type, "TEE mode enabled");

        // Validate TEE type against policy
        let policy = DefaultTeePolicy::new(
            config.allowed_tee_types.clone(),
            config.expected_measurements.clone(),
        )?;
        policy.validate_tee_type(tee_type)?;
        tracing::info!("TEE policy validation passed");

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
            let verified = tee::model_seal::verify_all_models(&registry, &config.model_hashes)?;
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

    // Auto-enable in-memory decryption in TEE mode for maximum security
    if config.tee_mode && !config.in_memory_decrypt {
        tracing::info!("TEE mode: auto-enabling in_memory_decrypt");
        config.in_memory_decrypt = true;
    }

    let bind_addr = config.bind_address();
    let config = Arc::new(config);

    // Initialize backends
    let backends = Arc::new(backend::default_backends(config.clone()));
    tracing::info!(
        backends = ?backends.list_names(),
        "Initialized backends"
    );

    let mut app_state = state::AppState::new(registry, backends, config.clone());
    if let Some(provider) = tee_provider {
        app_state = app_state.with_tee_provider(provider);
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

    // Initialize key provider for encrypted model loading
    if let Some(provider) = key_provider::from_config(&config) {
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
    let tls_port = config.tls_port.expect("tls_port checked by caller");

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
    // Wait for either SIGTERM (systemd/Kubernetes) or Ctrl-C
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("SIGTERM received, starting graceful shutdown");
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Ctrl-C received, starting graceful shutdown");
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Ctrl-C received, starting graceful shutdown");
    }

    // Unload all models to trigger RAII cleanup (zeroize decrypted weights)
    let loaded = state.loaded_model_names();
    if !loaded.is_empty() {
        tracing::info!(count = loaded.len(), "Unloading all models before shutdown");
        for model_name in &loaded {
            let format = state
                .registry
                .get(model_name)
                .map(|m| m.format.clone())
                .unwrap_or(crate::model::manifest::ModelFormat::Gguf);
            if let Ok(backend) = state.backends.find_for_format(&format) {
                if let Err(e) = backend.unload(model_name).await {
                    tracing::warn!(model = %model_name, "Failed to unload model on shutdown: {e}");
                }
            }
            state.mark_unloaded(model_name);
            tracing::info!(model = %model_name, "Model unloaded on shutdown");
        }
    }

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

/// Spawn a background task that periodically checks for models whose keep_alive
/// has expired and unloads them to free memory.
fn spawn_keep_alive_reaper(state: state::AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            let expired = state.expired_models();
            for model_name in expired {
                // Look up the model's actual format to find the right backend.
                let format = state
                    .registry
                    .get(&model_name)
                    .map(|m| m.format.clone())
                    .unwrap_or(crate::model::manifest::ModelFormat::Gguf);

                if let Ok(backend) = state.backends.find_for_format(&format) {
                    if let Err(e) = backend.unload(&model_name).await {
                        tracing::warn!(
                            model = %model_name,
                            "Failed to unload expired model: {e}"
                        );
                        continue;
                    }
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
    });
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
        let mut sighup = signal(SignalKind::hangup()).expect("failed to register SIGHUP handler");

        loop {
            sighup.recv().await;
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
    use serial_test::serial;

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
}
