//! Native PP-OCRv6-small graph execution.
//!
//! The reviewed graph plans are compiled into this crate. SafeTensors files
//! provide weights only: loading a session does not parse ONNX, download a
//! model, start another process, or bind a network listener.

mod executor;
mod plan;
mod value;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::{
    DevicePreference, ExecutionReceipt, InferenceLimits, ModelIdentity, RuntimeDevice,
    RuntimeIdentity, TensorInput, TensorOutput, WeightStore,
};
use executor::GraphExecutor;
use plan::{GraphPlan, GraphRole, DETECTION_PLAN, RECOGNITION_PLAN};

const MODEL_FAMILY: &str = "pp-ocr-v6-small";
const MODEL_REVISION: &str = "paddlex-paddle3.0.0";
const DETECTION_WEIGHTS_SHA256: &str =
    "0439824a102e0b365ca905355553985a885773ca0ea9f6a526e5f7317fc15592";
const RECOGNITION_WEIGHTS_SHA256: &str =
    "e8bf34a6900addc8cd9ec1d1ea73ea56e97cb0d668c8c45508a885924078761f";

/// Typed location of the reviewed PP-OCRv6-small SafeTensors weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpOcrV6Model {
    detection: PathBuf,
    recognition: PathBuf,
}

impl PpOcrV6Model {
    /// Uses `<root>/det` and `<root>/rec`, the layout produced by
    /// `tools/pack_ppocr_v6.py`.
    pub fn from_directory(root: impl AsRef<Path>) -> Self {
        Self {
            detection: root.as_ref().join("det"),
            recognition: root.as_ref().join("rec"),
        }
    }

    pub fn from_weight_directories(
        detection: impl Into<PathBuf>,
        recognition: impl Into<PathBuf>,
    ) -> Self {
        Self {
            detection: detection.into(),
            recognition: recognition.into(),
        }
    }

    pub fn detection(&self) -> &Path {
        &self.detection
    }

    pub fn recognition(&self) -> &Path {
        &self.recognition
    }
}

/// Output and reproducibility evidence for one native graph execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PpOcrV6Execution {
    pub output: TensorOutput,
    pub receipt: ExecutionReceipt,
}

/// In-process PP-OCRv6-small inference session.
///
/// Construction validates the complete tensor inventory and exact reviewed
/// weight digests before materializing constants on the selected device.
pub struct PpOcrV6Session {
    detection: GraphExecutor,
    recognition: GraphExecutor,
    detection_store: Arc<WeightStore>,
    recognition_store: Arc<WeightStore>,
    detection_identity: ModelIdentity,
    recognition_identity: ModelIdentity,
    device: RuntimeDevice,
    limits: InferenceLimits,
    active: AtomicUsize,
}

impl PpOcrV6Session {
    pub fn load(
        model: PpOcrV6Model,
        device: DevicePreference,
        limits: InferenceLimits,
    ) -> Result<Self> {
        limits.validate()?;
        let device = RuntimeDevice::resolve(device)?;
        let detection_store = Arc::new(WeightStore::open(model.detection(), &limits)?);
        verify_weights(
            GraphRole::Detection,
            &detection_store,
            DETECTION_WEIGHTS_SHA256,
        )?;
        let recognition_store = Arc::new(WeightStore::open(model.recognition(), &limits)?);
        verify_weights(
            GraphRole::Recognition,
            &recognition_store,
            RECOGNITION_WEIGHTS_SHA256,
        )?;

        let detection_plan =
            GraphPlan::parse(DETECTION_PLAN, GraphRole::Detection, &detection_store)?;
        let recognition_plan =
            GraphPlan::parse(RECOGNITION_PLAN, GraphRole::Recognition, &recognition_store)?;
        let detection = GraphExecutor::new(
            detection_plan,
            Arc::clone(&detection_store),
            &device.candle,
            limits.clone(),
        )?;
        let recognition = GraphExecutor::new(
            recognition_plan,
            Arc::clone(&recognition_store),
            &device.candle,
            limits.clone(),
        )?;

        Ok(Self {
            detection,
            recognition,
            detection_store,
            recognition_store,
            detection_identity: model_identity(GraphRole::Detection, DETECTION_WEIGHTS_SHA256),
            recognition_identity: model_identity(
                GraphRole::Recognition,
                RECOGNITION_WEIGHTS_SHA256,
            ),
            device,
            limits,
            active: AtomicUsize::new(0),
        })
    }

    pub fn device(&self) -> &RuntimeDevice {
        &self.device
    }

    pub fn detection_weights(&self) -> &WeightStore {
        &self.detection_store
    }

    pub fn recognition_weights(&self) -> &WeightStore {
        &self.recognition_store
    }

    /// Executes the native text-detection graph on an NCHW F32 tensor.
    pub fn detect(
        &self,
        input: TensorInput,
        cancellation: &CancellationToken,
    ) -> Result<PpOcrV6Execution> {
        validate_nchw(&input, GraphRole::Detection)?;
        self.execute(
            input,
            cancellation,
            &self.detection,
            &self.detection_identity,
        )
    }

    /// Executes the native text-recognition graph on an NCHW F32 tensor.
    /// The reviewed recognition model requires an input height of 48 pixels.
    pub fn recognize(
        &self,
        input: TensorInput,
        cancellation: &CancellationToken,
    ) -> Result<PpOcrV6Execution> {
        validate_nchw(&input, GraphRole::Recognition)?;
        self.execute(
            input,
            cancellation,
            &self.recognition,
            &self.recognition_identity,
        )
    }

    fn execute(
        &self,
        input: TensorInput,
        cancellation: &CancellationToken,
        executor: &GraphExecutor,
        identity: &ModelIdentity,
    ) -> Result<PpOcrV6Execution> {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "PP-OCRv6 inference was cancelled".to_string(),
            ));
        }
        let _permit = ActivePermit::acquire(&self.active, self.limits.max_concurrent_requests)?;
        let input_elements = self
            .limits
            .checked_elements(&input.shape, "PP-OCRv6 input tensor")?;
        let input_sha256 = tensor_sha256(&input.shape, &input.values);
        let output = executor.run(input.into_candle(&self.device.candle)?, cancellation)?;
        let output = TensorOutput::from_candle(&output, &self.limits)?;
        let output_elements = self
            .limits
            .checked_elements(&output.shape, "PP-OCRv6 output tensor")?;
        let output_sha256 = tensor_sha256(&output.shape, &output.values);
        Ok(PpOcrV6Execution {
            output,
            receipt: ExecutionReceipt {
                model: identity.clone(),
                runtime: RuntimeIdentity::current(&self.device),
                input_sha256,
                output_sha256,
                input_elements,
                output_elements,
            },
        })
    }
}

impl std::fmt::Debug for PpOcrV6Session {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PpOcrV6Session")
            .field("device", &self.device)
            .field("detection_weights", &self.detection_store)
            .field("recognition_weights", &self.recognition_store)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

fn validate_nchw(input: &TensorInput, role: GraphRole) -> Result<()> {
    if input.shape.len() != 4 || input.shape[1] != 3 {
        return Err(PowerError::InvalidRequest(format!(
            "PP-OCRv6 {} input must have NCHW shape [batch, 3, height, width]",
            role.name()
        )));
    }
    if role == GraphRole::Recognition && input.shape[2] != 48 {
        return Err(PowerError::InvalidRequest(
            "PP-OCRv6 recognition input height must be 48 pixels".to_string(),
        ));
    }
    Ok(())
}

fn verify_weights(role: GraphRole, store: &WeightStore, expected: &str) -> Result<()> {
    if store.sha256() != expected {
        return Err(PowerError::IntegrityCheckFailed {
            model: format!("{MODEL_FAMILY}-{}", role.name()),
            expected: expected.to_string(),
            actual: store.sha256().to_string(),
        });
    }
    Ok(())
}

fn model_identity(role: GraphRole, weights_sha256: &str) -> ModelIdentity {
    ModelIdentity::new(
        format!("{MODEL_FAMILY}-{}", role.name()),
        MODEL_REVISION,
        weights_sha256,
    )
}

fn tensor_sha256(shape: &[usize], values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"a3s-power-f32-tensor-v1\0");
    hasher.update((shape.len() as u64).to_le_bytes());
    for dimension in shape {
        hasher.update((*dimension as u64).to_le_bytes());
    }
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

struct ActivePermit<'a> {
    active: &'a AtomicUsize,
}

impl<'a> ActivePermit<'a> {
    fn acquire(active: &'a AtomicUsize, maximum: usize) -> Result<Self> {
        let mut observed = active.load(Ordering::Acquire);
        loop {
            if observed >= maximum {
                return Err(PowerError::InferenceFailed(format!(
                    "PP-OCRv6 session already has {maximum} active request(s)"
                )));
            }
            match active.compare_exchange_weak(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(Self { active }),
                Err(updated) => observed = updated,
            }
        }
    }
}

impl Drop for ActivePermit<'_> {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_layout_is_explicit() {
        let model = PpOcrV6Model::from_directory("/models/ppocr-v6");
        assert_eq!(model.detection(), Path::new("/models/ppocr-v6/det"));
        assert_eq!(model.recognition(), Path::new("/models/ppocr-v6/rec"));
    }

    #[test]
    fn recognition_shape_requires_reviewed_height() {
        let limits = InferenceLimits::default();
        let input = TensorInput::new(vec![1, 3, 32, 100], vec![0.0; 9_600], &limits).unwrap();
        assert!(validate_nchw(&input, GraphRole::Recognition).is_err());
    }

    #[test]
    fn tensor_digest_includes_shape() {
        assert_ne!(
            tensor_sha256(&[1, 2], &[1.0, 2.0]),
            tensor_sha256(&[2, 1], &[1.0, 2.0])
        );
    }

    #[test]
    fn public_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PpOcrV6Session>();
    }

    #[test]
    fn real_reviewed_weights_execute_when_configured() {
        let Some(root) = std::env::var_os("A3S_PPOCR_V6_MODEL") else {
            return;
        };
        let limits = InferenceLimits::default();
        let session = PpOcrV6Session::load(
            PpOcrV6Model::from_directory(root),
            DevicePreference::Cpu,
            limits.clone(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();

        let detection = TensorInput::new(vec![1, 3, 64, 64], vec![0.0; 12_288], &limits)
            .and_then(|input| session.detect(input, &cancellation))
            .unwrap();
        assert_eq!(detection.output.shape[0..2], [1, 1]);

        let recognition = TensorInput::new(vec![1, 3, 48, 320], vec![0.0; 46_080], &limits)
            .and_then(|input| session.recognize(input, &cancellation))
            .unwrap();
        assert_eq!(recognition.output.shape[0], 1);
        assert_eq!(recognition.output.shape[2], 18_710);
    }
}
