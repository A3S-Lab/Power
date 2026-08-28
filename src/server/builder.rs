use std::sync::Arc;

use crate::backend::{Backend, BackendRegistry};
use crate::config::PowerConfig;
use crate::error::Result;
use crate::model::manifest::ModelManifest;
use crate::serving::StateTransferService;

use super::log_stream::LogBuffer;

/// Typed composition root for the Power HTTP server.
///
/// Injected backends are registered before Power's built-in backends and
/// therefore have higher priority. Call [`Self::without_default_backends`] when
/// the caller supplies the complete backend set.
pub struct PowerServerBuilder {
    options: PowerServerOptions,
}

pub(super) struct PowerServerOptions {
    pub(super) config: PowerConfig,
    pub(super) log_buffer: Option<LogBuffer>,
    pub(super) backends: BackendRegistry,
    pub(super) model_manifests: Vec<ModelManifest>,
    pub(super) state_transfer_service: Option<Arc<dyn StateTransferService>>,
    pub(super) include_default_backends: bool,
}

impl PowerServerBuilder {
    pub fn new(config: PowerConfig) -> Self {
        Self {
            options: PowerServerOptions {
                config,
                log_buffer: None,
                backends: BackendRegistry::new(),
                model_manifests: Vec::new(),
                state_transfer_service: None,
                include_default_backends: true,
            },
        }
    }

    pub fn with_log_buffer(mut self, log_buffer: LogBuffer) -> Self {
        self.options.log_buffer = Some(log_buffer);
        self
    }

    /// Register a typed backend ahead of the built-in backends.
    pub fn with_backend(mut self, backend: Arc<dyn Backend>) -> Self {
        self.options.backends.register(backend);
        self
    }

    /// Replace the currently injected backend set.
    ///
    /// Built-in backends are still appended unless
    /// [`Self::without_default_backends`] is also selected.
    pub fn with_backend_registry(mut self, backends: BackendRegistry) -> Self {
        self.options.backends = backends;
        self
    }

    /// Register an in-memory model manifest when the server starts.
    ///
    /// This is intended for typed downstream composition roots whose model is
    /// already available locally. The manifest is not persisted into Power's
    /// global model directory and replaces a scanned manifest with the same
    /// name for this server process only.
    pub fn with_model_manifest(mut self, manifest: ModelManifest) -> Self {
        self.options.model_manifests.push(manifest);
        self
    }

    /// Install the model-state transfer data path used by disaggregated
    /// prefill/decode adapters.
    ///
    /// The service owns registered memory and transport-specific metadata.
    /// Power exposes only its typed, bounded capabilities and never forwards
    /// model-state bytes through the control or request-routing planes.
    pub fn with_state_transfer_service(mut self, service: Arc<dyn StateTransferService>) -> Self {
        self.options.state_transfer_service = Some(service);
        self
    }

    /// Start with only caller-injected backends.
    pub fn without_default_backends(mut self) -> Self {
        self.options.include_default_backends = false;
        self
    }

    pub async fn start(self) -> Result<()> {
        super::start_with_options(self.into_options()).await
    }

    pub(super) fn into_options(self) -> PowerServerOptions {
        self.options
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::backend::test_utils::MockBackend;
    use crate::config::PowerConfig;
    use crate::model::manifest::{ModelFormat, ModelManifest};
    use crate::serving::{
        AbortStateTransfer, ConsumeStateTransfer, PrepareStateTransfer, PublishStateTransfer,
        ServingPhase, StateTransferCapabilities, StateTransferProtocol, StateTransferReceipt,
        StateTransferService, StateTransferSource, StateTransferTarget, TransferHealth,
    };

    use super::PowerServerBuilder;

    struct TestStateTransferService;

    #[async_trait]
    impl StateTransferService for TestStateTransferService {
        fn capabilities(&self) -> StateTransferCapabilities {
            StateTransferCapabilities {
                execution_profile_sha256: "9".repeat(64),
                phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
                protocols: vec![StateTransferProtocol::BufferedHostMemoryPullV1],
                max_transfer_bytes: 1024,
                max_inflight_transfers: 1,
            }
        }

        fn health(&self) -> TransferHealth {
            TransferHealth::Ready
        }

        async fn prepare_destination(
            &self,
            _command: PrepareStateTransfer,
        ) -> crate::error::Result<StateTransferTarget> {
            Err(crate::error::PowerError::BackendNotAvailable(
                "test adapter".to_string(),
            ))
        }

        async fn publish_source(
            &self,
            _command: PublishStateTransfer,
        ) -> crate::error::Result<StateTransferSource> {
            Err(crate::error::PowerError::BackendNotAvailable(
                "test adapter".to_string(),
            ))
        }

        async fn consume_source(
            &self,
            _command: ConsumeStateTransfer,
        ) -> crate::error::Result<StateTransferReceipt> {
            Err(crate::error::PowerError::BackendNotAvailable(
                "test adapter".to_string(),
            ))
        }

        async fn abort(&self, _command: AbortStateTransfer) -> crate::error::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn injected_backends_are_typed_and_keep_registration_order() {
        let options = PowerServerBuilder::new(PowerConfig::default())
            .with_backend(Arc::new(
                MockBackend::success()
                    .with_name("olmoe")
                    .with_family("olmoe"),
            ))
            .with_backend(Arc::new(MockBackend::success().with_name("fallback")))
            .into_options();

        assert_eq!(options.backends.list_names(), ["olmoe", "fallback"]);
        assert!(options.include_default_backends);
    }

    #[test]
    fn default_backends_can_be_disabled_explicitly() {
        let options = PowerServerBuilder::new(PowerConfig::default())
            .without_default_backends()
            .into_options();

        assert!(!options.include_default_backends);
        assert!(options.backends.list_names().is_empty());
    }

    #[test]
    fn state_transfer_service_is_a_typed_optional_extension() {
        let options = PowerServerBuilder::new(PowerConfig::default())
            .with_state_transfer_service(Arc::new(TestStateTransferService))
            .into_options();

        let service = options
            .state_transfer_service
            .expect("state-transfer service");
        assert_eq!(service.health(), TransferHealth::Ready);
        service.capabilities().validate().unwrap();
    }

    #[test]
    fn downstream_models_are_process_local_and_keep_registration_order() {
        let first = ModelManifest {
            name: "olmoe-a".to_string(),
            format: ModelFormat::SafeTensors,
            size: 10,
            sha256: "11".repeat(32),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: PathBuf::from("/models/olmoe-a"),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            adapter_artifact: None,
            external_draft: None,
            projector_path: None,
            projector_artifact: None,
            messages: Vec::new(),
            family: Some("olmoe".to_string()),
            families: None,
        };
        let mut second = first.clone();
        second.name = "olmoe-b".to_string();

        let options = PowerServerBuilder::new(PowerConfig::default())
            .with_model_manifest(first)
            .with_model_manifest(second)
            .into_options();

        assert_eq!(
            options
                .model_manifests
                .iter()
                .map(|manifest| manifest.name.as_str())
                .collect::<Vec<_>>(),
            ["olmoe-a", "olmoe-b"]
        );
    }

    #[test]
    fn builder_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PowerServerBuilder>();
    }
}
