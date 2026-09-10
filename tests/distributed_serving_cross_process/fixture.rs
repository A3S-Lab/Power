//! Cross-process worker that installs the product BufferedHostLoopback pair
//! via ACL `serving_execution.transport = "buffered-host-loopback"`.
//!
//! This is loopback conformance evidence only — not HSN or model-semantic P/D.

use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::sync::Arc;

use a3s_power::backend::BackendRegistry;
use a3s_power::config::PowerConfig;
use a3s_power::model::registry::ModelRegistry;
use a3s_power::server::auth::ApiKeyAuth;
use a3s_power::server::router;
use a3s_power::server::state::AppState;
use a3s_power::serving::{
    BoundedStateTransferService, BufferedHostLoopbackPhaseExecutor, DisaggregatedServingRole,
    ServingCompositionTransport, StateTransferService,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

pub const SERVICE_KEY: &str = "cross-process-service-key";
pub const MODEL: &str = "internal/cross-process-model-v1";
pub const CONFORMANCE_TOKEN: &str = "loopback-conformance-token";

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn product_pair_acl(role: DisaggregatedServingRole) -> String {
    let role = match role {
        DisaggregatedServingRole::Prefill => "prefill",
        DisaggregatedServingRole::Decode => "decode",
    };
    format!(
        r#"
host = "127.0.0.1"
port = 0
api_keys = ["{SERVICE_KEY}"]

serving_execution {{
  profile = "prefill-decode"
  role = "{role}"
  model = "{MODEL}"
  model_sha256 = "{model}"
  backend = "buffered-host-loopback"
  backend_sha256 = "{backend}"
  execution_sha256 = "{execution}"
  device_sha256 = "{device}"
  layout_sha256 = "{layout}"
  peer_set_sha256 = "{peer_set}"
  generation = 7
  protocol = "buffered-host-memory-pull-v1"
  state_kind = "kv-cache"
  max_state_bytes = 64
  max_inflight_transfers = 2
  transfer_timeout_ms = 10000
  cancellation_timeout_ms = 1000
  privacy = "authenticated-encrypted-transport"
  privacy_policy_sha256 = "{privacy}"
  weight_cache = "shared-weight-hierarchy"
  session_pool = "shared-session-pool"
  transport = "buffered-host-loopback"
}}
"#,
        model = digest('1'),
        backend = digest('2'),
        execution = digest('3'),
        device = digest('4'),
        layout = digest('5'),
        peer_set = digest('6'),
        privacy = digest('7'),
    )
}

/// Install the product pair the same way ACL composition resolve does.
fn install_product_pair_from_acl_transport(
    config: &PowerConfig,
) -> a3s_power::error::Result<(
    Arc<dyn StateTransferService>,
    Arc<BufferedHostLoopbackPhaseExecutor>,
)> {
    match config.serving_execution.composition_transport() {
        Some(ServingCompositionTransport::BufferedHostLoopback) => {
            let (transfer, executor) =
                BufferedHostLoopbackPhaseExecutor::paired_for_profile(&config.serving_execution)?;
            Ok((transfer as Arc<dyn StateTransferService>, executor))
        }
        None => Err(a3s_power::error::PowerError::Config(
            "cross-process product-pair worker requires serving_execution.transport = buffered-host-loopback"
                .to_string(),
        )),
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyWorker {
    pub address: SocketAddr,
    pub process_id: u32,
    pub worker_epoch: uuid::Uuid,
    pub execution_profile_sha256: String,
}

pub async fn serve_worker(
    role: DisaggregatedServingRole,
    ready_path: &Path,
    shutdown_path: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = ready_path
        .parent()
        .ok_or("ready path must have a parent directory")?;
    let acl_path = directory.join(format!(
        "worker-{}-{}.acl",
        match role {
            DisaggregatedServingRole::Prefill => "prefill",
            DisaggregatedServingRole::Decode => "decode",
        },
        std::process::id()
    ));
    std::fs::write(&acl_path, product_pair_acl(role))?;
    let config = PowerConfig::load_from(acl_path.to_str().ok_or("ACL path is not UTF-8")?)?;
    assert_eq!(
        config.serving_execution.composition_transport(),
        Some(ServingCompositionTransport::BufferedHostLoopback),
        "ACL transport opt-in must survive load_from"
    );

    let profile = config.serving_execution.clone();
    let profile_sha256 = profile.sha256()?;
    let (transfer, executor) = install_product_pair_from_acl_transport(&config)?;
    let state = AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(BackendRegistry::new()),
        Arc::new(config),
    );
    let worker_epoch = state.worker_epoch();
    let transfer = Arc::new(BoundedStateTransferService::new(
        profile.clone(),
        worker_epoch,
        transfer,
    )?);
    let runtime = a3s_power::serving::DistributedServingRuntime::new(profile, transfer, executor)?;
    let state = state
        .with_distributed_serving(Arc::new(runtime))
        .with_auth(Arc::new(ApiKeyAuth::new(&[SERVICE_KEY.to_string()])));
    let listener = TcpListener::bind((IpAddr::from([127, 0, 0, 1]), 0)).await?;
    let ready = ReadyWorker {
        address: listener.local_addr()?,
        process_id: std::process::id(),
        worker_epoch,
        execution_profile_sha256: profile_sha256,
    };
    write_ready(ready_path, &ready)?;
    let shutdown_path = shutdown_path.to_path_buf();
    axum::serve(listener, router::build(state))
        .with_graceful_shutdown(async move { wait_for_shutdown(&shutdown_path).await })
        .await?;
    Ok(())
}

fn write_ready(
    ready_path: &Path,
    ready: &ReadyWorker,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let temporary = ready_path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(ready)?)?;
    std::fs::rename(temporary, ready_path)?;
    Ok(())
}

async fn wait_for_shutdown(shutdown_path: &Path) {
    while !shutdown_path.exists() {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}
