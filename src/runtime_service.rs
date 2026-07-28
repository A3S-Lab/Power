//! A3S Runtime Service profile for hosting Power through A3S Box.
//!
//! This module only compiles Power-specific process, configuration, endpoint,
//! health, and restart requirements into the shared Runtime contract. Placement,
//! accelerator leases, rollout, authorization, and routing remain Cloud concerns.

use std::collections::BTreeMap;

use a3s_runtime::contract::{
    ArtifactRef, HealthProbe, IsolationLevel, NetworkMode, ResourceLimits, RestartPolicy,
    RuntimeHealthCheck, RuntimeMount, RuntimeNetworkSpec, RuntimePort, RuntimeProcessSpec,
    RuntimeUnitClass, RuntimeUnitSpec, SecretReference, SecretTarget, TransportProtocol,
};
use serde::{Deserialize, Serialize};

/// Power configuration file injected by the Runtime provider.
pub const POWER_CONFIG_PATH: &str = "/run/a3s-power/config.acl";

/// Named Runtime endpoint exposed by a Power Service.
pub const POWER_HTTP_PORT_NAME: &str = "http";

/// Immutable inputs needed to compile one Power Service generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerRuntimeServiceProfile {
    pub unit_id: String,
    pub generation: u64,
    pub artifact: ArtifactRef,
    /// Opaque provider-resolved reference to a complete A3S ACL configuration.
    pub config_secret_reference: String,
    pub port: u16,
    pub resources: ResourceLimits,
    pub isolation: IsolationLevel,
    #[serde(default)]
    pub mounts: Vec<RuntimeMount>,
    /// Additional secret references, such as model-decryption keys. Secret
    /// values must never be placed in `environment`.
    #[serde(default)]
    pub secrets: Vec<SecretReference>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// Digest of the Cloud-owned immutable Power deployment semantics.
    pub semantics_profile_digest: String,
}

#[derive(Debug, thiserror::Error)]
pub enum PowerRuntimeServiceError {
    #[error("Power requires sandbox or confidential isolation when hosted by A3S Box")]
    UnsupportedIsolation,
    #[error("invalid Power Runtime Service: {0}")]
    InvalidRuntimeSpec(String),
}

impl PowerRuntimeServiceProfile {
    /// Compile this Power profile into the provider-neutral Runtime Service
    /// contract consumed by A3S Box.
    pub fn compile(&self) -> Result<RuntimeUnitSpec, PowerRuntimeServiceError> {
        if !matches!(
            self.isolation,
            IsolationLevel::Sandbox | IsolationLevel::Confidential
        ) {
            return Err(PowerRuntimeServiceError::UnsupportedIsolation);
        }

        let mut secrets = Vec::with_capacity(self.secrets.len() + 1);
        secrets.push(SecretReference {
            name: "power-config".into(),
            reference: self.config_secret_reference.clone(),
            target: SecretTarget::File {
                path: POWER_CONFIG_PATH.into(),
                mode: 0o400,
            },
        });
        secrets.extend(self.secrets.clone());

        let spec = RuntimeUnitSpec {
            schema: RuntimeUnitSpec::SCHEMA.into(),
            unit_id: self.unit_id.clone(),
            generation: self.generation,
            class: RuntimeUnitClass::Service,
            artifact: self.artifact.clone(),
            process: RuntimeProcessSpec {
                command: Vec::new(),
                args: vec![
                    "serve".into(),
                    "--config".into(),
                    POWER_CONFIG_PATH.into(),
                    "--host".into(),
                    "0.0.0.0".into(),
                    "--port".into(),
                    self.port.to_string(),
                ],
                working_directory: None,
                environment: self.environment.clone(),
            },
            mounts: self.mounts.clone(),
            secrets,
            network: RuntimeNetworkSpec {
                mode: NetworkMode::Service,
                ports: vec![RuntimePort {
                    name: POWER_HTTP_PORT_NAME.into(),
                    container_port: self.port,
                    protocol: TransportProtocol::Tcp,
                }],
            },
            resources: self.resources.clone(),
            isolation: self.isolation,
            health: Some(RuntimeHealthCheck {
                probe: HealthProbe::Http {
                    port: POWER_HTTP_PORT_NAME.into(),
                    path: "/health".into(),
                    expected_statuses: vec![200],
                },
                interval_ms: 10_000,
                timeout_ms: 2_000,
                start_period_ms: 60_000,
                success_threshold: 1,
                failure_threshold: 3,
            }),
            restart: RestartPolicy::Always,
            outputs: Vec::new(),
            semantics_profile_digest: Some(self.semantics_profile_digest.clone()),
        };
        spec.validate()
            .map_err(PowerRuntimeServiceError::InvalidRuntimeSpec)?;
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> PowerRuntimeServiceProfile {
        PowerRuntimeServiceProfile {
            unit_id: "environment/inference/power/revision-1".into(),
            generation: 1,
            artifact: ArtifactRef {
                uri: format!("oci://registry.example/a3s-power@sha256:{}", "a".repeat(64)),
                digest: format!("sha256:{}", "a".repeat(64)),
                media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            },
            config_secret_reference: "secret://environment/power-config/revision-1".into(),
            port: 11_434,
            resources: ResourceLimits {
                cpu_millis: 2_000,
                memory_bytes: 4 * 1024 * 1024 * 1024,
                pids: 256,
                ephemeral_storage_bytes: Some(16 * 1024 * 1024 * 1024),
                execution_timeout_ms: None,
            },
            isolation: IsolationLevel::Sandbox,
            mounts: Vec::new(),
            secrets: Vec::new(),
            environment: BTreeMap::new(),
            semantics_profile_digest: format!("sha256:{}", "b".repeat(64)),
        }
    }

    #[test]
    fn compiles_standard_runtime_service_for_box() {
        let spec = profile().compile().unwrap();
        assert_eq!(spec.class, RuntimeUnitClass::Service);
        assert_eq!(spec.network.mode, NetworkMode::Service);
        assert_eq!(spec.network.ports[0].name, POWER_HTTP_PORT_NAME);
        assert!(matches!(spec.restart, RestartPolicy::Always));
        assert!(matches!(
            spec.health.as_ref().map(|health| &health.probe),
            Some(HealthProbe::Http { path, .. }) if path == "/health"
        ));
        assert!(matches!(
            &spec.secrets[0].target,
            SecretTarget::File { path, mode } if path == POWER_CONFIG_PATH && *mode == 0o400
        ));
    }

    #[test]
    fn rejects_non_box_isolation_profile() {
        let mut value = profile();
        value.isolation = IsolationLevel::Container;
        assert!(matches!(
            value.compile(),
            Err(PowerRuntimeServiceError::UnsupportedIsolation)
        ));
    }

    #[test]
    fn rejects_task_only_execution_timeout() {
        let mut value = profile();
        value.resources.execution_timeout_ms = Some(1_000);
        let error = value.compile().unwrap_err();
        assert!(error.to_string().contains("execution timeout"));
    }
}
