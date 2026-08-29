use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use a3s_power::backend::BackendRegistry;
use a3s_power::config::PowerConfig;
use a3s_power::error::{PowerError, Result};
use a3s_power::model::registry::ModelRegistry;
use a3s_power::server::auth::ApiKeyAuth;
use a3s_power::server::router;
use a3s_power::server::state::AppState;
use a3s_power::serving::{
    AbortStateTransfer, BoundedStateTransferService, ConsumeStateTransfer,
    DisaggregatedServingRole, ModelStateHandle, PhaseExecutorCapabilities,
    PrefillDecodeExecutionProfile, PrepareStateTransfer, PublishStateTransfer,
    ServingExecutionProfile, ServingPrivacyMode, StateKind, StateTransferBinding,
    StateTransferCapabilities, StateTransferIntegrity, StateTransferProtocol, StateTransferReceipt,
    StateTransferService, StateTransferSource, StateTransferTarget, TransferHealth,
    STATE_TRANSFER_RECEIPT_SCHEMA, STATE_TRANSFER_SOURCE_SCHEMA, STATE_TRANSFER_TARGET_SCHEMA,
};
use aes_gcm::aead::{Aead, Payload};
use aes_gcm::{Aes256Gcm, KeyInit};
use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::Utc;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use uuid::Uuid;

#[path = "phase_fixture.rs"]
mod phase_fixture;

use phase_fixture::FixturePhaseExecutor;

pub const SERVICE_KEY: &str = "cross-process-service-key";
pub const MODEL: &str = "internal/cross-process-model-v1";
const STATE_BYTES: usize = 512;
const SOURCE_TICKET_SCHEMA: &str = "a3s.power.test.encrypted-loopback-source.v1";
const DATA_PATH_DOMAIN: &[u8] = b"a3s.power.test.encrypted-loopback.v1\0";

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

pub fn profile(role: DisaggregatedServingRole) -> ServingExecutionProfile {
    ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
        role,
        model: MODEL.to_string(),
        model_sha256: digest('1'),
        backend: "cross-process-fixture".to_string(),
        backend_sha256: digest('2'),
        execution_sha256: digest('3'),
        device_sha256: digest('4'),
        layout_sha256: digest('5'),
        peer_set_sha256: digest('6'),
        generation: 7,
        protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
        state_kind: StateKind::KvCache,
        max_state_bytes: STATE_BYTES as u64,
        max_inflight_transfers: 2,
        transfer_timeout_ms: 10_000,
        cancellation_timeout_ms: 1_000,
        privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
        privacy_policy_sha256: digest('7'),
        attestation_policy_sha256: None,
    })
    .expect("the cross-process fixture profile is valid")
}

fn binding() -> StateTransferBinding {
    StateTransferBinding {
        model_sha256: digest('1'),
        execution_sha256: digest('3'),
        layout_sha256: digest('5'),
        state_kind: StateKind::KvCache,
        token_count: 16,
        state_bytes: STATE_BYTES as u64,
    }
}

fn fixture_state() -> Vec<u8> {
    (0..STATE_BYTES)
        .map(|index| ((index * 31 + 17) % 251) as u8)
        .collect()
}

#[derive(Default)]
struct FixtureStateStore {
    states: Mutex<HashMap<String, Vec<u8>>>,
}

impl FixtureStateStore {
    fn states(&self) -> Result<MutexGuard<'_, HashMap<String, Vec<u8>>>> {
        self.states.lock().map_err(|_| {
            PowerError::BackendNotAvailable("fixture state store lock is unavailable".to_string())
        })
    }

    fn insert(&self, handle: &ModelStateHandle, state: Vec<u8>) -> Result<()> {
        self.states()?.insert(handle.as_str().to_string(), state);
        Ok(())
    }

    fn take(&self, handle: &ModelStateHandle) -> Result<Vec<u8>> {
        self.states()?.remove(handle.as_str()).ok_or_else(|| {
            PowerError::BackendNotAvailable("fixture model state is unavailable".to_string())
        })
    }

    fn remove_execution(&self, execution_id: Uuid) -> Result<()> {
        let mut states = self.states()?;
        states.remove(&format!("source:{execution_id}"));
        states.remove(&format!("destination:{execution_id}"));
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceTicket {
    schema: String,
    address: SocketAddr,
    key: String,
    nonce: String,
    state_sha256: String,
}

struct FixtureTransferService {
    capabilities: StateTransferCapabilities,
    store: Arc<FixtureStateStore>,
    sources: AsyncMutex<HashMap<Uuid, JoinHandle<()>>>,
}

impl FixtureTransferService {
    fn new(profile: &ServingExecutionProfile, store: Arc<FixtureStateStore>) -> Self {
        Self {
            capabilities: StateTransferCapabilities {
                execution_profile_sha256: profile
                    .sha256()
                    .expect("the fixture profile digest is valid"),
                phases: vec![profile.phase()],
                protocols: vec![StateTransferProtocol::BufferedHostMemoryPullV1],
                max_transfer_bytes: STATE_BYTES as u64,
                max_inflight_transfers: 2,
            },
            store,
            sources: AsyncMutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl StateTransferService for FixtureTransferService {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        TransferHealth::Ready
    }

    async fn prepare_destination(
        &self,
        command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget> {
        Ok(StateTransferTarget {
            schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
            transfer_id: command.transfer_id,
            destination_worker_epoch: command.local_worker_epoch,
            binding: command.binding,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            prepared_at: Utc::now(),
            expires_at: command.expires_at,
            ticket: format!("encrypted-loopback-target:{}", command.transfer_id),
        })
    }

    async fn publish_source(&self, command: PublishStateTransfer) -> Result<StateTransferSource> {
        let state = self.store.take(&command.source)?;
        if state.len() as u64 != command.target.binding.state_bytes {
            return Err(PowerError::InvalidRequest(
                "fixture source state size does not match its binding".to_string(),
            ));
        }

        let listener = TcpListener::bind((IpAddr::from([127, 0, 0, 1]), 0))
            .await
            .map_err(peer_unavailable)?;
        let address = listener.local_addr().map_err(peer_unavailable)?;
        let published_at = Utc::now();
        let aad = transfer_aad(
            command.target.transfer_id,
            command.local_worker_epoch,
            command.target.destination_worker_epoch,
            &command.target.binding,
        )?;
        let mut key = [0_u8; 32];
        let mut nonce = [0_u8; 12];
        OsRng.fill_bytes(&mut key);
        OsRng.fill_bytes(&mut nonce);
        let state_sha256 = hex::encode(Sha256::digest(&state));
        let ticket = SourceTicket {
            schema: SOURCE_TICKET_SCHEMA.to_string(),
            address,
            key: URL_SAFE_NO_PAD.encode(key),
            nonce: URL_SAFE_NO_PAD.encode(nonce),
            state_sha256,
        };
        let encoded_ticket = serde_json::to_string(&ticket)?;
        let transfer_id = command.target.transfer_id;
        let task = tokio::spawn(serve_state_once(
            listener,
            transfer_id,
            key,
            nonce,
            aad,
            state,
        ));
        self.sources.lock().await.insert(transfer_id, task);

        Ok(StateTransferSource {
            schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
            transfer_id,
            source_worker_epoch: command.local_worker_epoch,
            destination_worker_epoch: command.target.destination_worker_epoch,
            binding: command.target.binding,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            published_at,
            expires_at: command.target.expires_at,
            ticket: encoded_ticket,
        })
    }

    async fn consume_source(&self, command: ConsumeStateTransfer) -> Result<StateTransferReceipt> {
        let ticket: SourceTicket = serde_json::from_str(&command.source.ticket)?;
        if ticket.schema != SOURCE_TICKET_SCHEMA || !ticket.address.ip().is_loopback() {
            return Err(PowerError::InvalidRequest(
                "fixture source ticket is invalid".to_string(),
            ));
        }
        let key = decode_array::<32>(&ticket.key, "fixture transfer key")?;
        let nonce = decode_array::<12>(&ticket.nonce, "fixture transfer nonce")?;
        let aad = transfer_aad(
            command.source.transfer_id,
            command.source.source_worker_epoch,
            command.source.destination_worker_epoch,
            &command.source.binding,
        )?;
        let auth = authentication_token(&key, &aad);
        let mut stream = TcpStream::connect(ticket.address)
            .await
            .map_err(peer_unavailable)?;
        stream.write_all(&auth).await.map_err(peer_unavailable)?;

        let ciphertext_len = stream.read_u64().await.map_err(peer_unavailable)?;
        let expected_ciphertext_len = command
            .source
            .binding
            .state_bytes
            .checked_add(16)
            .ok_or_else(|| PowerError::InvalidRequest("fixture state size overflow".to_string()))?;
        if ciphertext_len != expected_ciphertext_len {
            return Err(PowerError::InvalidRequest(
                "fixture encrypted state size is invalid".to_string(),
            ));
        }
        let ciphertext_len = usize::try_from(ciphertext_len).map_err(|_| {
            PowerError::InvalidRequest("fixture encrypted state is too large".to_string())
        })?;
        let mut ciphertext = vec![0_u8; ciphertext_len];
        stream
            .read_exact(&mut ciphertext)
            .await
            .map_err(peer_unavailable)?;
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| {
            PowerError::InvalidRequest("fixture transfer key is invalid".to_string())
        })?;
        let cipher_nonce = nonce.into();
        let state = cipher
            .decrypt(
                &cipher_nonce,
                Payload {
                    msg: &ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| {
                PowerError::InvalidRequest(
                    "fixture encrypted state failed authentication".to_string(),
                )
            })?;
        let state_sha256 = hex::encode(Sha256::digest(&state));
        if state.len() as u64 != command.source.binding.state_bytes
            || state_sha256 != ticket.state_sha256
        {
            return Err(PowerError::InvalidRequest(
                "fixture transferred state failed integrity validation".to_string(),
            ));
        }
        self.store.insert(&command.destination, state)?;

        Ok(StateTransferReceipt {
            schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
            transfer_id: command.source.transfer_id,
            source_worker_epoch: command.source.source_worker_epoch,
            destination_worker_epoch: command.local_worker_epoch,
            binding: command.source.binding,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            bytes_transferred: STATE_BYTES as u64,
            integrity: StateTransferIntegrity::Sha256 {
                digest: state_sha256,
            },
            completed_at: Utc::now(),
        })
    }

    async fn abort(&self, command: AbortStateTransfer) -> Result<()> {
        if let Some(task) = self.sources.lock().await.remove(&command.transfer_id) {
            task.abort();
        }
        self.store.remove_execution(command.transfer_id)
    }
}

async fn serve_state_once(
    listener: TcpListener,
    _transfer_id: Uuid,
    key: [u8; 32],
    nonce: [u8; 12],
    aad: Vec<u8>,
    state: Vec<u8>,
) {
    let result: std::io::Result<()> = async {
        let (mut stream, peer) = listener.accept().await?;
        if !peer.ip().is_loopback() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fixture data path accepts loopback peers only",
            ));
        }
        let mut supplied_auth = [0_u8; 32];
        stream.read_exact(&mut supplied_auth).await?;
        if !constant_time_eq(&supplied_auth, &authentication_token(&key, &aad)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fixture data-path authentication failed",
            ));
        }
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| std::io::Error::other("fixture transfer key is invalid"))?;
        let cipher_nonce = nonce.into();
        let ciphertext = cipher
            .encrypt(
                &cipher_nonce,
                Payload {
                    msg: &state,
                    aad: &aad,
                },
            )
            .map_err(|_| std::io::Error::other("fixture state encryption failed"))?;
        stream.write_u64(ciphertext.len() as u64).await?;
        stream.write_all(&ciphertext).await?;
        stream.shutdown().await
    }
    .await;
    let _ = result;
}

fn transfer_aad(
    transfer_id: Uuid,
    source_worker_epoch: Uuid,
    destination_worker_epoch: Uuid,
    binding: &StateTransferBinding,
) -> Result<Vec<u8>> {
    serde_json::to_vec(&(
        "a3s.power.test.encrypted-loopback.aad.v1",
        transfer_id,
        source_worker_epoch,
        destination_worker_epoch,
        binding,
    ))
    .map_err(PowerError::from)
}

fn authentication_token(key: &[u8; 32], aad: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(DATA_PATH_DOMAIN);
    digest.update(key);
    digest.update(aad);
    digest.finalize().into()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn decode_array<const N: usize>(value: &str, label: &str) -> Result<[u8; N]> {
    let decoded = URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| PowerError::InvalidRequest(format!("{label} is not canonical base64url")))?;
    decoded.try_into().map_err(|_| {
        PowerError::InvalidRequest(format!("{label} does not have the required length"))
    })
}

fn peer_unavailable(error: std::io::Error) -> PowerError {
    PowerError::BackendNotAvailable(format!("fixture transfer peer is unavailable: {error}"))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyWorker {
    pub address: SocketAddr,
    pub process_id: u32,
    pub worker_epoch: Uuid,
    pub execution_profile_sha256: String,
}

pub async fn serve_worker(
    role: DisaggregatedServingRole,
    ready_path: &Path,
    shutdown_path: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let profile = profile(role);
    let profile_sha256 = profile.sha256()?;
    let config = PowerConfig {
        serving_execution: profile.clone(),
        api_keys: vec![SERVICE_KEY.to_string()],
        ..PowerConfig::default()
    };
    config.validate()?;
    let state = AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(BackendRegistry::new()),
        Arc::new(config),
    );
    let worker_epoch = state.worker_epoch();
    let store = Arc::new(FixtureStateStore::default());
    let transfer = Arc::new(BoundedStateTransferService::new(
        profile.clone(),
        worker_epoch,
        Arc::new(FixtureTransferService::new(&profile, Arc::clone(&store))),
    )?);
    let executor = Arc::new(FixturePhaseExecutor {
        capabilities: PhaseExecutorCapabilities {
            execution_profile_sha256: profile_sha256.clone(),
            phase: profile.phase(),
        },
        profile_sha256: profile_sha256.clone(),
        store,
    });
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
