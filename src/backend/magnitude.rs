mod request;
mod stream;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use magnitude_engine::composition::{start_in_process, EngineConfiguration};
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::options::{
    standard_service_limits, ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection,
};
use magnitude_engine::worker::transport::stdio_worker_transport;
use magnitude_engine::worker::{serve_worker, EngineClient, WorkerExit};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_executor::ExecutionPath;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;

use super::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
};
use super::{Backend, LoadedSpeculativeArtifact};
use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::external_draft::ExternalDraftArtifact;
use crate::model::manifest::{ModelFormat, ModelManifest};
use crate::speculative::SpeculativeStrategy;

pub const MAGNITUDE_WORKER_ARG: &str = "--magnitude-worker";

pub fn worker_main() -> i32 {
    match serve_worker(stdio_worker_transport()) {
        WorkerExit::Prepared | WorkerExit::Shutdown => 0,
        _ => 1,
    }
}

pub struct MagnitudeBackend {
    config: Arc<PowerConfig>,
    models: RwLock<HashMap<String, LoadedModel>>,
}

struct LoadedModel {
    host: Arc<HostArtifacts>,
    engine: EngineClient,
    draft: Option<ExternalDraftArtifact>,
}

impl MagnitudeBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        Self {
            config,
            models: RwLock::new(HashMap::new()),
        }
    }

    async fn loaded(&self, model_name: &str) -> Result<(Arc<HostArtifacts>, EngineClient)> {
        let loaded = self.models.read().await;
        let model = loaded.get(model_name).ok_or_else(|| {
            PowerError::ModelNotFound(format!("magnitude model '{model_name}' is not loaded"))
        })?;
        model.engine.check().map_err(|error| {
            PowerError::InferenceFailed(format!("magnitude worker is unavailable: {error}"))
        })?;
        Ok((model.host.clone(), model.engine.clone()))
    }
}

#[async_trait]
impl Backend for MagnitudeBackend {
    fn name(&self) -> &str {
        "magnitude"
    }

    fn supports_openai_tool_selection(&self) -> bool {
        true
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }

    fn supports_manifest(&self, manifest: &ModelManifest) -> bool {
        if self.config.tee_mode
            || !self.supports(&manifest.format)
            || super::is_reserved_gguf_manifest(manifest)
        {
            return false;
        }
        let package = PackageOptions {
            target: manifest.path.clone(),
            projector: ProjectorSelection::Disabled,
            draft: None,
        };
        package.open().is_ok_and(|package| {
            magnitude_engine::families::recognize(package.target().directory()).is_ok()
        })
    }

    fn reports_exact_chat_usage(&self) -> bool {
        true
    }

    async fn loaded_speculative_artifacts(&self) -> Vec<LoadedSpeculativeArtifact> {
        self.models
            .read()
            .await
            .iter()
            .filter_map(|(model_name, loaded)| {
                loaded
                    .draft
                    .as_ref()
                    .map(|draft| LoadedSpeculativeArtifact {
                        backend: self.name().into(),
                        model: model_name.clone(),
                        strategy: draft.kind.as_str().into(),
                        artifact_size: draft.size,
                        artifact_sha256: draft.sha256.clone(),
                        target_sha256: draft.target_sha256.clone(),
                    })
            })
            .collect()
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        validate_load(self, manifest)?;
        if self.models.read().await.contains_key(&manifest.name) {
            return Ok(());
        }
        let method = model_method(&self.config.spec_mode)?;
        let mtp_proposals = self
            .config
            .spec_draft_max
            .map(|value| {
                u8::try_from(value)
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| {
                        PowerError::Config(
                            "magnitude spec_draft_max must be between 1 and 255".into(),
                        )
                    })
            })
            .transpose()?;
        let device = device_request(&self.config)?;
        let manifest = manifest.clone();
        let data_dir = self.config.data_dir.clone();
        let model_name = manifest.name.clone();
        let loaded = tokio::task::spawn_blocking(move || {
            if let Some(projector) = &manifest.projector_artifact {
                projector.verify("Multimodal projector")?;
            }
            if let Some(draft) = &manifest.external_draft {
                draft.verify_for_target_file(&manifest.path, manifest.size, &manifest.sha256)?;
            }
            let projector = match manifest.resolved_projector_path()? {
                Some(path) => ProjectorSelection::Explicit(path.to_path_buf()),
                None => ProjectorSelection::Discover,
            };
            let resolved = EngineConfiguration {
                package: PackageOptions {
                    target: manifest.path.clone(),
                    projector,
                    draft: manifest
                        .external_draft
                        .as_ref()
                        .map(|draft| draft.path.clone()),
                },
                model: ModelPolicy {
                    method,
                    mtp_proposals,
                    ..ModelPolicy::default()
                },
                context_tokens: manifest
                    .parameters
                    .as_ref()
                    .and_then(|parameters| parameters.context_length)
                    .map(|value| value as usize),
                service: standard_service_limits(),
                path: ExecutionPath::Native,
                device,
                kernel_cache: Some(data_dir.join("kernels").join("magnitude")),
                reserves: MemoryReserves::standard(),
            }
            .resolve()
            .map_err(|error| {
                PowerError::InferenceFailed(format!("magnitude model resolution failed: {error}"))
            })?;
            let mut worker =
                std::process::Command::new(std::env::current_exe().map_err(|error| {
                    PowerError::InferenceFailed(format!(
                        "cannot locate a3s-power for Magnitude Prepare: {error}"
                    ))
                })?);
            worker.arg(MAGNITUDE_WORKER_ARG);
            let ready = start_in_process(resolved, worker, |progress| {
                tracing::debug!(?progress, "Magnitude model load progress");
            })
            .map_err(|error| {
                PowerError::InferenceFailed(format!("magnitude model load failed: {error}"))
            })?;
            let draft = if matches!(
                ready.ready_info().model.method,
                magnitude_engine::options::ResolvedMethod::DFlash { .. }
            ) {
                manifest.external_draft.clone()
            } else {
                None
            };
            let (host, engine, _) = ready.into_parts();
            Ok::<_, PowerError>(LoadedModel {
                host,
                engine,
                draft,
            })
        })
        .await
        .map_err(|error| {
            PowerError::InferenceFailed(format!("Magnitude load task failed: {error}"))
        })??;
        self.models.write().await.insert(model_name, loaded);
        Ok(())
    }

    async fn unload(&self, model_name: &str) -> Result<()> {
        if let Some(loaded) = self.models.write().await.remove(model_name) {
            loaded.engine.shutdown().map_err(|error| {
                PowerError::InferenceFailed(format!("magnitude unload failed: {error}"))
            })?;
        }
        Ok(())
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        let (host, engine) = self.loaded(model_name).await?;
        let request = request::generation_request(&request)?;
        let (session_sender, mut session_events) = mpsc::channel(32);
        tokio::spawn(magnitude_engine::chat::generate(
            host,
            engine,
            request,
            magnitude_engine::chat::SessionLimits {
                output_capacity: 32,
                max_output_bytes: 8 << 20,
            },
            session_sender,
        ));
        let (sender, receiver) = mpsc::channel(32);
        tokio::spawn(async move {
            let mut assembler = stream::StreamAssembler::default();
            while let Some(event) = session_events.recv().await {
                match assembler.apply(event) {
                    Ok(chunks) => {
                        for chunk in chunks {
                            if sender.send(Ok(chunk)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                }
            }
            if let Err(error) = assembler.finish() {
                let _ = sender.send(Err(error)).await;
            }
        });
        Ok(Box::pin(ReceiverStream::new(receiver)))
    }

    async fn complete(
        &self,
        _model_name: &str,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        Err(PowerError::BackendNotAvailable(
            "magnitude text completion is not implemented; use chat or another backend".into(),
        ))
    }

    async fn embed(
        &self,
        _model_name: &str,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        Err(PowerError::BackendNotAvailable(
            "magnitude embeddings are not implemented; use another backend".into(),
        ))
    }
}

fn validate_load(backend: &MagnitudeBackend, manifest: &ModelManifest) -> Result<()> {
    if backend.config.tee_mode {
        return Err(PowerError::BackendNotAvailable(
            "magnitude does not implement Power's confidential in-memory loader".into(),
        ));
    }
    if !backend.supports(&manifest.format) {
        return Err(PowerError::InvalidFormat(format!(
            "magnitude requires GGUF, got {}",
            manifest.format
        )));
    }
    if super::is_reserved_gguf_manifest(manifest) {
        return Err(PowerError::InvalidFormat(
            "magnitude cannot load a reserved Prism or Xolotl GGUF pack".into(),
        ));
    }
    if manifest.resolved_adapter_path()?.is_some() {
        return Err(PowerError::BackendNotAvailable(
            "magnitude LoRA loading is not implemented".into(),
        ));
    }
    if manifest.template_override.is_some() {
        return Err(PowerError::BackendNotAvailable(
            "magnitude template_override is not implemented".into(),
        ));
    }
    if manifest.system_prompt.is_some() || !manifest.messages.is_empty() {
        return Err(PowerError::BackendNotAvailable(
            "magnitude manifest-provided prompts are not implemented".into(),
        ));
    }
    if backend.config.use_mlock {
        return Err(PowerError::BackendNotAvailable(
            "magnitude use_mlock is controlled by Seismic residency, not Power".into(),
        ));
    }
    if backend.config.num_thread.is_some() {
        return Err(PowerError::BackendNotAvailable(
            "magnitude num_thread is controlled by Seismic workers".into(),
        ));
    }
    Ok(())
}

fn model_method(configured: &str) -> Result<ModelMethod> {
    let strategy = SpeculativeStrategy::parse(configured)
        .ok_or_else(|| PowerError::Config(format!("unsupported spec_mode '{configured}'")))?;
    match strategy {
        SpeculativeStrategy::Auto => Ok(ModelMethod::Auto),
        SpeculativeStrategy::Off => Ok(ModelMethod::Plain),
        SpeculativeStrategy::Mtp => Ok(ModelMethod::Mtp),
        SpeculativeStrategy::Dflash => Ok(ModelMethod::DFlash),
        SpeculativeStrategy::Dflash2 => Ok(ModelMethod::DFlash2),
        SpeculativeStrategy::Dspark => Ok(ModelMethod::DSpark),
        SpeculativeStrategy::PromptLookup
        | SpeculativeStrategy::NgramContext
        | SpeculativeStrategy::DraftModel => Err(PowerError::BackendNotAvailable(format!(
            "magnitude does not support spec_mode '{}'",
            strategy.as_str()
        ))),
    }
}

fn device_request(config: &PowerConfig) -> Result<DeviceRequest> {
    if config.gpu.main_gpu != 0
        || !config.gpu.tensor_split.is_empty()
        || !config.gpu.cpu_tensors.is_empty()
        || !config.gpu.gpu_tensors.is_empty()
    {
        return Err(PowerError::BackendNotAvailable(
            "magnitude does not support llama.cpp GPU placement overrides".into(),
        ));
    }
    config
        .magnitude_device
        .parse()
        .map_err(|error| PowerError::Config(format!("invalid magnitude_device: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertises_validated_openai_tool_selection() {
        let backend = MagnitudeBackend::new(Arc::new(PowerConfig::default()));
        assert!(backend.supports_openai_tool_selection());
    }

    #[test]
    fn maps_only_supported_speculative_modes() {
        assert_eq!(model_method("auto").unwrap(), ModelMethod::Auto);
        assert_eq!(model_method("off").unwrap(), ModelMethod::Plain);
        assert_eq!(model_method("dflash2").unwrap(), ModelMethod::DFlash2);
        assert!(model_method("draft-model").is_err());
        assert!(model_method("unknown").is_err());
    }

    #[test]
    fn maps_explicit_device_without_gpu_layer_interference() {
        assert_eq!(
            device_request(&PowerConfig {
                magnitude_device: "cpu".into(),
                gpu: crate::config::GpuConfig {
                    gpu_layers: -1,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap()
            .to_string(),
            "cpu"
        );
        assert_eq!(
            device_request(&PowerConfig::default()).unwrap(),
            DeviceRequest::Automatic
        );
        assert!(device_request(&PowerConfig {
            magnitude_device: "invalid".into(),
            ..Default::default()
        })
        .is_err());
        assert!(device_request(&PowerConfig {
            gpu: crate::config::GpuConfig {
                main_gpu: 1,
                ..Default::default()
            },
            ..Default::default()
        })
        .is_err());
    }
}
