//! Product-surface `BufferedHostMemoryPullV1` state-transfer adapter.
//!
//! Moves opaque, adapter-owned host buffers over an authenticated AES-GCM
//! loopback TCP data path. Satisfies [`ProductionAdapterContract::REQUIRED`]
//! (adapter-owned registration, adapter-owned transport integrity, confirmed
//! reclaim) so prefill/decode composition can inject a real path instead of an
//! Empty placeholder.
//!
//! Pair with [`crate::serving::BufferedHostLoopbackPhaseExecutor`] for a
//! complete product-surface injection. This is **not** high-speed-network
//! evidence, model-semantic phase execution, or production readiness. Wire
//! tickets stay connection metadata; KV layout and decode success remain owned
//! by the separately injected phase executor / backend. Host-buffer sealing
//! through [`SealedStateEnvelope`] remains available via `transfer_host_buffer`
//! when `embedded-inference` is enabled; this adapter does not embed sealed
//! envelopes in tickets. Transfer AAD v2 binds the profile privacy mode,
//! `privacy_policy_sha256`, and optional `attestation_policy_sha256` so peers
//! with matching model/layout bindings but mismatched privacy fail closed.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::Aes256Gcm;
use async_trait::async_trait;
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

use crate::error::{PowerError, Result};

use super::{
    AbortStateTransfer, AdapterProvisionState, ConsumeStateTransfer, ModelStateHandle,
    PrepareStateTransfer, ProductionAdapterContract, PublishStateTransfer, ServingExecutionProfile,
    ServingPrivacyMode, StateTransferCapabilities, StateTransferIntegrity, StateTransferProtocol,
    StateTransferReceipt, StateTransferService, StateTransferSource, StateTransferTarget,
    TransferHealth, STATE_TRANSFER_RECEIPT_SCHEMA, STATE_TRANSFER_SOURCE_SCHEMA,
    STATE_TRANSFER_TARGET_SCHEMA,
};

const SOURCE_TICKET_SCHEMA: &str = "a3s.power.buffered-host-loopback-source.v1";
const DATA_PATH_DOMAIN: &[u8] = b"a3s.power.buffered-host-loopback.v1\0";
/// AAD v2 binds privacy/attestation policy digests into the data path so peers
/// with matching model/layout bindings but mismatched privacy fail closed.
const AAD_SCHEMA: &str = "a3s.power.buffered-host-loopback.aad.v2";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceTicket {
    schema: String,
    address: SocketAddr,
    key_hex: String,
    nonce_hex: String,
    state_sha256: String,
}

/// Profile privacy/attestation facts cryptographically bound into every
/// loopback transfer AAD. Model/execution/layout bindings alone are not enough:
/// a peer with a different privacy or attestation policy must fail closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrivacyAttestationBinding {
    privacy: ServingPrivacyMode,
    privacy_policy_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    attestation_policy_sha256: Option<String>,
}

#[derive(Default)]
struct OwnedStateStore {
    states: Mutex<HashMap<String, Vec<u8>>>,
    /// Destination handles reserved by prepare for confirmed abort reclaim.
    reservations: Mutex<HashMap<Uuid, String>>,
}

impl OwnedStateStore {
    fn states(&self) -> Result<MutexGuard<'_, HashMap<String, Vec<u8>>>> {
        self.states.lock().map_err(|_| {
            PowerError::BackendNotAvailable(
                "buffered-host loopback state store lock is unavailable".to_string(),
            )
        })
    }

    fn reservations(&self) -> Result<MutexGuard<'_, HashMap<Uuid, String>>> {
        self.reservations.lock().map_err(|_| {
            PowerError::BackendNotAvailable(
                "buffered-host loopback reservation lock is unavailable".to_string(),
            )
        })
    }

    fn insert(&self, handle: &ModelStateHandle, state: Vec<u8>) -> Result<()> {
        let key = handle.as_str().to_string();
        let mut states = self.states()?;
        if states.contains_key(&key) {
            return Err(PowerError::InvalidRequest(format!(
                "buffered-host loopback handle is already registered: {}",
                handle.as_str()
            )));
        }
        states.insert(key, state);
        Ok(())
    }

    fn take(&self, handle: &ModelStateHandle) -> Result<Vec<u8>> {
        self.states()?.remove(handle.as_str()).ok_or_else(|| {
            PowerError::BackendNotAvailable(
                "buffered-host loopback model state is unavailable".to_string(),
            )
        })
    }

    fn remove_handle(&self, handle: &str) -> Result<()> {
        self.states()?.remove(handle);
        Ok(())
    }

    fn reserve_destination(&self, transfer_id: Uuid, destination: &ModelStateHandle) -> Result<()> {
        let key = destination.as_str().to_string();
        let mut reservations = self.reservations()?;
        if reservations.contains_key(&transfer_id) {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback destination is already prepared for this transfer"
                    .to_string(),
            ));
        }
        let mut states = self.states()?;
        if states.contains_key(&key) {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback destination handle is already registered".to_string(),
            ));
        }
        // Empty reservation marks adapter-owned ownership until consume fills it
        // or abort reclaims it. Power never copies these bytes into the wrapper.
        states.insert(key.clone(), Vec::new());
        reservations.insert(transfer_id, key);
        Ok(())
    }

    fn reclaim_transfer(&self, transfer_id: Uuid) -> Result<()> {
        if let Some(destination) = self.reservations()?.remove(&transfer_id) {
            self.remove_handle(&destination)?;
        }
        Ok(())
    }
}

/// Injectable buffered-host / loopback [`StateTransferService`].
///
/// Construct with [`Self::for_profile`] against a prefill/decode profile that
/// pins `BufferedHostMemoryPullV1` and `AuthenticatedEncryptedTransport`.
#[derive(Clone)]
pub struct BufferedHostLoopbackStateTransfer {
    capabilities: StateTransferCapabilities,
    deployment: super::ServingDeploymentIdentity,
    privacy: PrivacyAttestationBinding,
    store: Arc<OwnedStateStore>,
    sources: Arc<AsyncMutex<HashMap<Uuid, JoinHandle<()>>>>,
}

impl std::fmt::Debug for BufferedHostLoopbackStateTransfer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BufferedHostLoopbackStateTransfer")
            .field("capabilities", &self.capabilities)
            .field("deployment", &self.deployment)
            .field("privacy", &self.privacy)
            .finish_non_exhaustive()
    }
}

impl BufferedHostLoopbackStateTransfer {
    /// Bind one process profile to the product buffered-host loopback path.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        let ServingExecutionProfile::PrefillDecode { execution } = profile else {
            return Err(PowerError::Config(
                "buffered-host loopback adapter requires a prefill-decode serving profile"
                    .to_string(),
            ));
        };
        if !matches!(
            execution.protocol,
            StateTransferProtocol::BufferedHostMemoryPullV1
        ) {
            return Err(PowerError::Config(
                "buffered-host loopback adapter requires BufferedHostMemoryPullV1".to_string(),
            ));
        }
        if !matches!(
            execution.privacy,
            ServingPrivacyMode::AuthenticatedEncryptedTransport
        ) {
            return Err(PowerError::Config(
                "buffered-host loopback adapter requires AuthenticatedEncryptedTransport privacy"
                    .to_string(),
            ));
        }
        let capabilities = StateTransferCapabilities {
            execution_profile_sha256: profile.sha256()?,
            phases: vec![super::ServingPhase::from(execution.role)],
            protocols: vec![StateTransferProtocol::BufferedHostMemoryPullV1],
            max_transfer_bytes: execution.max_state_bytes,
            max_inflight_transfers: execution.max_inflight_transfers,
        };
        profile.validate_state_transfer_capabilities(&capabilities)?;
        Ok(Self {
            capabilities,
            deployment: profile.deployment_identity()?,
            privacy: PrivacyAttestationBinding {
                privacy: execution.privacy,
                privacy_policy_sha256: execution.privacy_policy_sha256.clone(),
                attestation_policy_sha256: execution.attestation_policy_sha256.clone(),
            },
            store: Arc::new(OwnedStateStore::default()),
            sources: Arc::new(AsyncMutex::new(HashMap::new())),
        })
    }

    /// Deposit opaque adapter-owned state under a local handle (prefill side).
    ///
    /// Callers (phase executors / backends) own layout semantics; Power only
    /// moves these bytes as opaque host buffers.
    pub fn register_owned_state(&self, handle: &ModelStateHandle, state: Vec<u8>) -> Result<()> {
        if state.is_empty() {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback owned state must be non-empty".to_string(),
            ));
        }
        if state.len() as u64 > self.capabilities.max_transfer_bytes {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback owned state exceeds the adapter byte limit".to_string(),
            ));
        }
        self.store.insert(handle, state)
    }

    /// Remove and return opaque adapter-owned state (decode side after consume).
    pub fn take_owned_state(&self, handle: &ModelStateHandle) -> Result<Vec<u8>> {
        let state = self.store.take(handle)?;
        if state.is_empty() {
            return Err(PowerError::BackendNotAvailable(
                "buffered-host loopback destination has not received transferred state".to_string(),
            ));
        }
        Ok(state)
    }
}

#[async_trait]
impl StateTransferService for BufferedHostLoopbackStateTransfer {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        TransferHealth::Ready
    }

    fn provision(&self) -> AdapterProvisionState {
        AdapterProvisionState::Injected
    }

    fn production_contract(&self) -> ProductionAdapterContract {
        ProductionAdapterContract::REQUIRED
    }

    async fn prepare_destination(
        &self,
        command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget> {
        self.store
            .reserve_destination(command.transfer_id, &command.destination)?;
        Ok(StateTransferTarget {
            schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
            transfer_id: command.transfer_id,
            destination_worker_epoch: command.local_worker_epoch,
            deployment: self.deployment.clone(),
            binding: command.binding,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            prepared_at: Utc::now(),
            expires_at: command.expires_at,
            ticket: format!("buffered-host-loopback-target:{}", command.transfer_id),
        })
    }

    async fn publish_source(&self, command: PublishStateTransfer) -> Result<StateTransferSource> {
        let state = self.store.take(&command.source)?;
        if state.len() as u64 != command.target.binding.state_bytes {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback source state size does not match its binding".to_string(),
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
            &self.privacy,
        )?;
        let mut key = [0_u8; 32];
        let mut nonce = [0_u8; 12];
        OsRng.fill_bytes(&mut key);
        OsRng.fill_bytes(&mut nonce);
        let state_sha256 = hex::encode(Sha256::digest(&state));
        let ticket = SourceTicket {
            schema: SOURCE_TICKET_SCHEMA.to_string(),
            address,
            key_hex: hex::encode(key),
            nonce_hex: hex::encode(nonce),
            state_sha256,
        };
        let encoded_ticket = serde_json::to_string(&ticket)?;
        let transfer_id = command.target.transfer_id;
        let task = tokio::spawn(serve_state_once(listener, key, nonce, aad, state));
        self.sources.lock().await.insert(transfer_id, task);

        Ok(StateTransferSource {
            schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
            transfer_id,
            source_worker_epoch: command.local_worker_epoch,
            destination_worker_epoch: command.target.destination_worker_epoch,
            deployment: command.target.deployment.clone(),
            binding: command.target.binding,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            published_at,
            expires_at: command.target.expires_at,
            ticket: encoded_ticket,
        })
    }

    async fn consume_source(&self, command: ConsumeStateTransfer) -> Result<StateTransferReceipt> {
        let ticket: SourceTicket = serde_json::from_str(&command.source.ticket).map_err(|_| {
            PowerError::InvalidRequest(
                "buffered-host loopback source ticket is not valid JSON".to_string(),
            )
        })?;
        if ticket.schema != SOURCE_TICKET_SCHEMA || !ticket.address.ip().is_loopback() {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback source ticket is invalid".to_string(),
            ));
        }
        let key = decode_hex_array::<32>(&ticket.key_hex, "buffered-host loopback transfer key")?;
        let nonce =
            decode_hex_array::<12>(&ticket.nonce_hex, "buffered-host loopback transfer nonce")?;
        let aad = transfer_aad(
            command.source.transfer_id,
            command.source.source_worker_epoch,
            command.source.destination_worker_epoch,
            &command.source.binding,
            &self.privacy,
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
            .ok_or_else(|| {
                PowerError::InvalidRequest("buffered-host loopback state size overflow".to_string())
            })?;
        if ciphertext_len != expected_ciphertext_len {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback encrypted state size is invalid".to_string(),
            ));
        }
        let ciphertext_len = usize::try_from(ciphertext_len).map_err(|_| {
            PowerError::InvalidRequest(
                "buffered-host loopback encrypted state is too large".to_string(),
            )
        })?;
        let mut ciphertext = vec![0_u8; ciphertext_len];
        stream
            .read_exact(&mut ciphertext)
            .await
            .map_err(peer_unavailable)?;
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| {
            PowerError::InvalidRequest("buffered-host loopback transfer key is invalid".to_string())
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
                    "buffered-host loopback encrypted state failed authentication".to_string(),
                )
            })?;
        let state_sha256 = hex::encode(Sha256::digest(&state));
        if state.len() as u64 != command.source.binding.state_bytes
            || state_sha256 != ticket.state_sha256
        {
            return Err(PowerError::InvalidRequest(
                "buffered-host loopback transferred state failed integrity validation".to_string(),
            ));
        }

        // Replace the empty prepare reservation with received opaque bytes and
        // drop the reservation so a later abort does not wipe consumed state.
        {
            let mut reservations = self.store.reservations()?;
            let reserved = reservations.remove(&command.source.transfer_id);
            let key = command.destination.as_str();
            if reserved.as_deref() != Some(key) {
                return Err(PowerError::InvalidRequest(
                    "buffered-host loopback destination was not prepared".to_string(),
                ));
            }
            let mut states = self.store.states()?;
            if !states.contains_key(key) {
                return Err(PowerError::InvalidRequest(
                    "buffered-host loopback destination reservation is missing".to_string(),
                ));
            }
            states.insert(key.to_string(), state);
        }
        let _ = self
            .sources
            .lock()
            .await
            .remove(&command.source.transfer_id);

        Ok(StateTransferReceipt {
            schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
            transfer_id: command.source.transfer_id,
            source_worker_epoch: command.source.source_worker_epoch,
            destination_worker_epoch: command.local_worker_epoch,
            deployment: command.source.deployment.clone(),
            binding: command.source.binding.clone(),
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            bytes_transferred: command.source.binding.state_bytes,
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
        self.store.reclaim_transfer(command.transfer_id)
    }
}

async fn serve_state_once(
    listener: TcpListener,
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
                "buffered-host loopback data path accepts loopback peers only",
            ));
        }
        let mut supplied_auth = [0_u8; 32];
        stream.read_exact(&mut supplied_auth).await?;
        if !constant_time_eq(&supplied_auth, &authentication_token(&key, &aad)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "buffered-host loopback data-path authentication failed",
            ));
        }
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| std::io::Error::other("buffered-host loopback transfer key is invalid"))?;
        let cipher_nonce = nonce.into();
        let ciphertext = cipher
            .encrypt(
                &cipher_nonce,
                Payload {
                    msg: &state,
                    aad: &aad,
                },
            )
            .map_err(|_| std::io::Error::other("buffered-host loopback state encryption failed"))?;
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
    binding: &super::StateTransferBinding,
    privacy: &PrivacyAttestationBinding,
) -> Result<Vec<u8>> {
    serde_json::to_vec(&(
        AAD_SCHEMA,
        transfer_id,
        source_worker_epoch,
        destination_worker_epoch,
        binding,
        privacy,
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

fn decode_hex_array<const N: usize>(value: &str, label: &str) -> Result<[u8; N]> {
    let decoded = hex::decode(value)
        .map_err(|_| PowerError::InvalidRequest(format!("{label} is not canonical hexadecimal")))?;
    decoded.try_into().map_err(|_| {
        PowerError::InvalidRequest(format!("{label} does not have the required length"))
    })
}

fn peer_unavailable(error: std::io::Error) -> PowerError {
    PowerError::BackendNotAvailable(format!(
        "buffered-host loopback transfer peer is unavailable: {error}"
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    use crate::error::{PowerError, Result};
    use crate::serving::{
        validate_injected_production_adapters, AbortPhaseExecution, AbortStateTransfer,
        AdapterProvisionState, BoundedStateTransferService, BufferedHostLoopbackStateTransfer,
        ConsumeStateTransfer, DisaggregatedServingRole, EmptyServingPhaseExecutor,
        ExecutePhaseExecution, ModelStateHandle, PhaseDecision, PhaseExecutionOutput,
        PhaseExecutorCapabilities, PhaseExecutorHealth, PhaseSessionPoolMode, PhaseWeightCacheMode,
        PrefillDecodeExecutionProfile, PreparePhaseExecution, PrepareStateTransfer,
        PreparedPhaseExecution, ProductionAdapterContract, PublishStateTransfer,
        ServingExecutionProfile, ServingPhaseExecutor, ServingPrivacyMode, StateKind,
        StateTransferBinding, StateTransferProtocol, StateTransferService, TransferHealth,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile(role: DisaggregatedServingRole) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role,
            model: "fixture".to_string(),
            backend: "buffered-host-loopback".to_string(),
            model_sha256: digest('1'),
            backend_sha256: digest('2'),
            execution_sha256: digest('3'),
            device_sha256: digest('4'),
            layout_sha256: digest('5'),
            peer_set_sha256: digest('6'),
            generation: 7,
            state_kind: StateKind::KvCache,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            max_state_bytes: 64,
            max_inflight_transfers: 2,
            transfer_timeout_ms: 30_000,
            cancellation_timeout_ms: 5_000,
            privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
            privacy_policy_sha256: digest('7'),
            attestation_policy_sha256: None,
            weight_cache: PhaseWeightCacheMode::SharedWeightHierarchy,
            residency_policy_sha256: None,
            session_pool: PhaseSessionPoolMode::SharedSessionPool,
            session_pool_policy_sha256: None,
            transport: None,
            phase_executor: None,
            state_ownership: None,
        })
        .unwrap()
    }

    fn binding() -> StateTransferBinding {
        StateTransferBinding {
            model_sha256: digest('1'),
            execution_sha256: digest('3'),
            layout_sha256: digest('5'),
            state_kind: StateKind::KvCache,
            token_count: 8,
            state_bytes: 64,
        }
    }

    fn opaque_state() -> Vec<u8> {
        (0..64)
            .map(|index| ((index * 17 + 3) % 251) as u8)
            .collect()
    }

    struct ReadyPhaseStub {
        capabilities: PhaseExecutorCapabilities,
    }

    #[async_trait]
    impl ServingPhaseExecutor for ReadyPhaseStub {
        fn capabilities(&self) -> PhaseExecutorCapabilities {
            self.capabilities.clone()
        }

        fn health(&self) -> PhaseExecutorHealth {
            PhaseExecutorHealth::Ready
        }

        async fn prepare(
            &self,
            _command: PreparePhaseExecution,
        ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
            Err(PowerError::BackendNotAvailable(
                "buffered-host loopback tests do not execute phases".to_string(),
            ))
        }

        async fn execute(
            &self,
            _command: ExecutePhaseExecution,
        ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
            Err(PowerError::BackendNotAvailable(
                "buffered-host loopback tests do not execute phases".to_string(),
            ))
        }

        async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn product_adapter_is_injected_with_required_contract() {
        let adapter = BufferedHostLoopbackStateTransfer::for_profile(&profile(
            DisaggregatedServingRole::Decode,
        ))
        .unwrap();
        assert!(adapter.provision().is_injected());
        assert_eq!(
            adapter.production_contract(),
            ProductionAdapterContract::REQUIRED
        );
        assert_eq!(adapter.health(), TransferHealth::Ready);
        assert!(!matches!(adapter.provision(), AdapterProvisionState::Empty));
    }

    #[test]
    fn refuses_direct_device_and_aggregated_profiles() {
        let mut execution = match profile(DisaggregatedServingRole::Prefill) {
            ServingExecutionProfile::PrefillDecode { execution } => *execution,
            ServingExecutionProfile::Aggregated {} => panic!("expected prefill-decode"),
        };
        execution.protocol = StateTransferProtocol::DirectDeviceMemoryPullV1;
        let direct = ServingExecutionProfile::prefill_decode(execution).unwrap();
        let err = BufferedHostLoopbackStateTransfer::for_profile(&direct).unwrap_err();
        assert!(err.to_string().contains("BufferedHostMemoryPullV1"));

        let err =
            BufferedHostLoopbackStateTransfer::for_profile(&ServingExecutionProfile::Aggregated {})
                .unwrap_err();
        assert!(err.to_string().contains("prefill-decode"));
    }

    #[test]
    fn injected_production_validation_accepts_product_transfer_with_phase_stub() {
        let profile = profile(DisaggregatedServingRole::Decode);
        let transfer = BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap();
        let executor = ReadyPhaseStub {
            capabilities: PhaseExecutorCapabilities::for_profile(&profile).unwrap(),
        };
        validate_injected_production_adapters(&transfer, &executor).unwrap();
    }

    #[test]
    fn empty_phase_still_fails_composition_even_with_product_transfer() {
        let profile = profile(DisaggregatedServingRole::Decode);
        let transfer = BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap();
        let empty = EmptyServingPhaseExecutor::for_profile(&profile).unwrap();
        let err = validate_injected_production_adapters(&transfer, &empty).unwrap_err();
        assert!(err.to_string().contains("Empty placeholders"));
    }

    #[tokio::test]
    async fn loopback_prepare_publish_consume_round_trip_and_abort_reclaim() {
        let prefill_profile = profile(DisaggregatedServingRole::Prefill);
        let decode_profile = profile(DisaggregatedServingRole::Decode);
        let prefill = BufferedHostLoopbackStateTransfer::for_profile(&prefill_profile).unwrap();
        let decode = BufferedHostLoopbackStateTransfer::for_profile(&decode_profile).unwrap();

        let transfer_id = Uuid::new_v4();
        let source_epoch = Uuid::new_v4();
        let destination_epoch = Uuid::new_v4();
        let source = ModelStateHandle::new(format!("source:{transfer_id}")).unwrap();
        let destination = ModelStateHandle::new(format!("destination:{transfer_id}")).unwrap();
        let state = opaque_state();
        prefill
            .register_owned_state(&source, state.clone())
            .unwrap();

        let expires_at = Utc::now() + Duration::seconds(30);
        let target = decode
            .prepare_destination(PrepareStateTransfer {
                transfer_id,
                local_worker_epoch: destination_epoch,
                binding: binding(),
                destination: destination.clone(),
                expires_at,
            })
            .await
            .unwrap();

        let published = prefill
            .publish_source(PublishStateTransfer {
                local_worker_epoch: source_epoch,
                source: source.clone(),
                target: target.clone(),
            })
            .await
            .unwrap();

        let receipt = decode
            .consume_source(ConsumeStateTransfer {
                local_worker_epoch: destination_epoch,
                destination: destination.clone(),
                source: published,
            })
            .await
            .unwrap();
        assert_eq!(receipt.bytes_transferred, 64);
        assert_eq!(decode.take_owned_state(&destination).unwrap(), state);

        // Fresh transfer aborted after prepare reclaims the reservation.
        let abort_id = Uuid::new_v4();
        let abort_destination = ModelStateHandle::new(format!("destination:{abort_id}")).unwrap();
        decode
            .prepare_destination(PrepareStateTransfer {
                transfer_id: abort_id,
                local_worker_epoch: destination_epoch,
                binding: binding(),
                destination: abort_destination.clone(),
                expires_at,
            })
            .await
            .unwrap();
        decode
            .abort(AbortStateTransfer {
                transfer_id: abort_id,
                local_worker_epoch: destination_epoch,
            })
            .await
            .unwrap();
        assert!(decode.take_owned_state(&abort_destination).is_err());
    }

    fn profile_with_privacy(
        role: DisaggregatedServingRole,
        privacy_policy_sha256: String,
        attestation_policy_sha256: Option<String>,
    ) -> ServingExecutionProfile {
        let mut execution = match profile(role) {
            ServingExecutionProfile::PrefillDecode { execution } => *execution,
            ServingExecutionProfile::Aggregated {} => panic!("expected prefill-decode"),
        };
        execution.privacy_policy_sha256 = privacy_policy_sha256;
        execution.attestation_policy_sha256 = attestation_policy_sha256;
        ServingExecutionProfile::prefill_decode(execution).unwrap()
    }

    async fn publish_across(
        prefill: &BufferedHostLoopbackStateTransfer,
        decode: &BufferedHostLoopbackStateTransfer,
    ) -> Result<(
        crate::serving::StateTransferSource,
        ModelStateHandle,
        Uuid,
        Uuid,
    )> {
        let transfer_id = Uuid::new_v4();
        let source_epoch = Uuid::new_v4();
        let destination_epoch = Uuid::new_v4();
        let source = ModelStateHandle::new(format!("source:{transfer_id}")).unwrap();
        let destination = ModelStateHandle::new(format!("destination:{transfer_id}")).unwrap();
        prefill
            .register_owned_state(&source, opaque_state())
            .unwrap();
        let expires_at = Utc::now() + Duration::seconds(30);
        let target = decode
            .prepare_destination(PrepareStateTransfer {
                transfer_id,
                local_worker_epoch: destination_epoch,
                binding: binding(),
                destination: destination.clone(),
                expires_at,
            })
            .await
            .unwrap();
        let published = prefill
            .publish_source(PublishStateTransfer {
                local_worker_epoch: source_epoch,
                source,
                target,
            })
            .await
            .unwrap();
        Ok((published, destination, destination_epoch, transfer_id))
    }

    #[tokio::test]
    async fn mismatched_privacy_policy_fail_closed_on_consume() {
        let prefill = BufferedHostLoopbackStateTransfer::for_profile(&profile_with_privacy(
            DisaggregatedServingRole::Prefill,
            digest('7'),
            None,
        ))
        .unwrap();
        let decode = BufferedHostLoopbackStateTransfer::for_profile(&profile_with_privacy(
            DisaggregatedServingRole::Decode,
            digest('a'),
            None,
        ))
        .unwrap();
        let (published, destination, destination_epoch, _) =
            publish_across(&prefill, &decode).await.unwrap();
        let err = decode
            .consume_source(ConsumeStateTransfer {
                local_worker_epoch: destination_epoch,
                destination,
                source: published,
            })
            .await
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("authentication")
                || message.contains("unavailable")
                || message.contains("integrity"),
            "privacy policy mismatch must fail closed: {message}"
        );
    }

    #[tokio::test]
    async fn mismatched_attestation_policy_fail_closed_on_consume() {
        let prefill = BufferedHostLoopbackStateTransfer::for_profile(&profile_with_privacy(
            DisaggregatedServingRole::Prefill,
            digest('7'),
            Some(digest('8')),
        ))
        .unwrap();
        let decode = BufferedHostLoopbackStateTransfer::for_profile(&profile_with_privacy(
            DisaggregatedServingRole::Decode,
            digest('7'),
            None,
        ))
        .unwrap();
        let (published, destination, destination_epoch, _) =
            publish_across(&prefill, &decode).await.unwrap();
        let err = decode
            .consume_source(ConsumeStateTransfer {
                local_worker_epoch: destination_epoch,
                destination,
                source: published,
            })
            .await
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("authentication")
                || message.contains("unavailable")
                || message.contains("integrity"),
            "attestation policy mismatch must fail closed: {message}"
        );
    }

    #[tokio::test]
    async fn matching_privacy_and_attestation_policies_still_round_trip() {
        let prefill = BufferedHostLoopbackStateTransfer::for_profile(&profile_with_privacy(
            DisaggregatedServingRole::Prefill,
            digest('7'),
            Some(digest('8')),
        ))
        .unwrap();
        let decode = BufferedHostLoopbackStateTransfer::for_profile(&profile_with_privacy(
            DisaggregatedServingRole::Decode,
            digest('7'),
            Some(digest('8')),
        ))
        .unwrap();
        let (published, destination, destination_epoch, _) =
            publish_across(&prefill, &decode).await.unwrap();
        let receipt = decode
            .consume_source(ConsumeStateTransfer {
                local_worker_epoch: destination_epoch,
                destination: destination.clone(),
                source: published,
            })
            .await
            .unwrap();
        assert_eq!(receipt.bytes_transferred, 64);
        assert_eq!(
            decode.take_owned_state(&destination).unwrap(),
            opaque_state()
        );
    }

    #[tokio::test]
    async fn bounded_wrapper_accepts_product_adapter() {
        let profile = profile(DisaggregatedServingRole::Decode);
        let adapter = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        let bounded = BoundedStateTransferService::new(profile, Uuid::new_v4(), adapter).unwrap();
        assert_eq!(bounded.health(), TransferHealth::Ready);
        assert!(bounded.provision().is_injected());
        assert_eq!(
            bounded.production_contract(),
            ProductionAdapterContract::REQUIRED
        );
    }
}
