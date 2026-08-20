use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::binding::{DynamicShapeFallback, ShapeProfileBinding};
use super::digest::{canonical_sha256, domain_sha256, validate_canonical_sha256};
use super::evidence::{
    ShapeProfileExecutionEvidence, ShapeProfileExecutionPath, ShapeProfileFallbackReason,
    ShapeProfileSelection,
};

const MAX_SHAPE_PROFILES: usize = 256;

/// One opaque model-owned shape class and its aggregate execution envelope.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShapeProfile {
    implementation_sha256: String,
    shape_class_sha256: String,
    max_batch_size: usize,
    max_tensor_elements: usize,
    host_scratch_bytes: u64,
    device_scratch_bytes: u64,
    profile_sha256: String,
}

impl ShapeProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        implementation_sha256: impl Into<String>,
        shape_class_sha256: impl Into<String>,
        max_batch_size: usize,
        max_tensor_elements: usize,
        host_scratch_bytes: u64,
        device_scratch_bytes: u64,
    ) -> Result<Self> {
        let mut profile = Self {
            implementation_sha256: canonical_sha256(
                implementation_sha256,
                "shape-profile implementation",
            )?,
            shape_class_sha256: canonical_sha256(shape_class_sha256, "shape-profile class")?,
            max_batch_size,
            max_tensor_elements,
            host_scratch_bytes,
            device_scratch_bytes,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 = profile.recompute_sha256()?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn implementation_sha256(&self) -> &str {
        &self.implementation_sha256
    }

    pub fn shape_class_sha256(&self) -> &str {
        &self.shape_class_sha256
    }

    pub fn profile_sha256(&self) -> &str {
        &self.profile_sha256
    }

    fn validate(&self) -> Result<()> {
        validate_canonical_sha256(&self.implementation_sha256, "shape-profile implementation")?;
        validate_canonical_sha256(&self.shape_class_sha256, "shape-profile class")?;
        validate_canonical_sha256(&self.profile_sha256, "shape-profile identity")?;
        if self.max_batch_size == 0 || self.max_tensor_elements == 0 {
            return Err(PowerError::InvalidRequest(
                "shape profiles require positive batch and tensor-element bounds".to_string(),
            ));
        }
        if self.profile_sha256 != self.recompute_sha256()? {
            return Err(PowerError::InvalidFormat(
                "shape-profile identity does not match its canonical payload".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_scratch(&self, binding: &ShapeProfileBinding) -> Result<()> {
        if self.host_scratch_bytes > binding.runtime_reservations.host_scratch_bytes
            || self.device_scratch_bytes > binding.runtime_reservations.device_scratch_bytes
        {
            return Err(PowerError::InvalidRequest(
                "shape-profile scratch exceeds the bound runtime reservation".to_string(),
            ));
        }
        Ok(())
    }

    fn recompute_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Payload<'a> {
            implementation_sha256: &'a str,
            shape_class_sha256: &'a str,
            max_batch_size: usize,
            max_tensor_elements: usize,
            host_scratch_bytes: u64,
            device_scratch_bytes: u64,
        }
        domain_sha256(
            b"a3s-power-shape-profile-v1\0",
            &serde_json::to_vec(&Payload {
                implementation_sha256: &self.implementation_sha256,
                shape_class_sha256: &self.shape_class_sha256,
                max_batch_size: self.max_batch_size,
                max_tensor_elements: self.max_tensor_elements,
                host_scratch_bytes: self.host_scratch_bytes,
                device_scratch_bytes: self.device_scratch_bytes,
            })?,
        )
    }
}

impl std::fmt::Debug for ShapeProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ShapeProfile")
            .field("implementation", &"sha256")
            .field("shape_class", &"sha256")
            .field("profile", &"sha256")
            .field("execution_envelope", &"bounded")
            .finish()
    }
}

/// Canonical finite profile set published by a model integration.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShapeProfileDeclaration {
    schema: String,
    binding: ShapeProfileBinding,
    profiles: Vec<ShapeProfile>,
    dynamic_fallback: DynamicShapeFallback,
    declaration_sha256: String,
}

impl ShapeProfileDeclaration {
    pub const SCHEMA: &'static str = "a3s.power.shape-profile-declaration.v1";

    pub fn new(
        binding: ShapeProfileBinding,
        mut profiles: Vec<ShapeProfile>,
        dynamic_fallback: DynamicShapeFallback,
    ) -> Result<Self> {
        profiles.sort_by(|left, right| left.shape_class_sha256.cmp(&right.shape_class_sha256));
        let mut declaration = Self {
            schema: Self::SCHEMA.to_string(),
            binding,
            profiles,
            dynamic_fallback,
            declaration_sha256: String::new(),
        };
        declaration.declaration_sha256 = declaration.recompute_sha256()?;
        declaration.validate()?;
        Ok(declaration)
    }

    pub fn binding(&self) -> &ShapeProfileBinding {
        &self.binding
    }

    pub fn profiles(&self) -> &[ShapeProfile] {
        &self.profiles
    }

    pub fn dynamic_fallback(&self) -> &DynamicShapeFallback {
        &self.dynamic_fallback
    }

    pub fn declaration_sha256(&self) -> &str {
        &self.declaration_sha256
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA
            || self.profiles.is_empty()
            || self.profiles.len() > MAX_SHAPE_PROFILES
        {
            return Err(PowerError::InvalidFormat(format!(
                "shape-profile declaration must use the supported schema and contain between 1 and {MAX_SHAPE_PROFILES} profiles"
            )));
        }
        self.binding.validate()?;
        self.dynamic_fallback.validate()?;
        validate_canonical_sha256(&self.declaration_sha256, "shape-profile declaration")?;

        let mut previous_class = None;
        for profile in &self.profiles {
            profile.validate()?;
            profile.validate_scratch(&self.binding)?;
            if previous_class.is_some_and(|previous| previous >= profile.shape_class_sha256()) {
                return Err(PowerError::InvalidFormat(
                    "shape-profile classes must be unique and canonically ordered".to_string(),
                ));
            }
            previous_class = Some(profile.shape_class_sha256());
        }
        if self.declaration_sha256 != self.recompute_sha256()? {
            return Err(PowerError::InvalidFormat(
                "shape-profile declaration digest does not match its canonical payload".to_string(),
            ));
        }
        Ok(())
    }

    pub fn select(
        &self,
        current_binding: &ShapeProfileBinding,
        request: &ShapeProfileRequest,
    ) -> Result<ShapeProfileSelection> {
        self.validate()?;
        current_binding.validate()?;
        request.validate()?;
        if current_binding != &self.binding {
            return Err(PowerError::InvalidRequest(
                "shape-profile declaration is stale for the current model, graph, device topology, scratch bounds, or TEE policy"
                    .to_string(),
            ));
        }

        let profile = self
            .profiles
            .binary_search_by(|profile| {
                profile
                    .shape_class_sha256
                    .as_str()
                    .cmp(&request.shape_class_sha256)
            })
            .ok()
            .map(|index| &self.profiles[index]);
        let path = match profile {
            None => self.fallback(ShapeProfileFallbackReason::ShapeClassUnavailable)?,
            Some(profile) if request.batch_size > profile.max_batch_size => {
                self.fallback(ShapeProfileFallbackReason::BatchBoundExceeded)?
            }
            Some(profile) if request.tensor_elements > profile.max_tensor_elements => {
                self.fallback(ShapeProfileFallbackReason::TensorElementBoundExceeded)?
            }
            Some(profile) => ShapeProfileExecutionPath::Profile {
                profile_sha256: profile.profile_sha256.clone(),
                implementation_sha256: profile.implementation_sha256.clone(),
            },
        };
        let evidence = ShapeProfileExecutionEvidence {
            schema: ShapeProfileExecutionEvidence::SCHEMA.to_string(),
            declaration_sha256: self.declaration_sha256.clone(),
            binding_sha256: self.binding.binding_sha256()?,
            request_sha256: request.request_sha256()?,
            weights_sha256: self.binding.weights_sha256.clone(),
            runtime_device: self.binding.runtime_device,
            input_sha256: request.input_sha256.clone(),
            path,
        };
        evidence.validate()?;
        Ok(ShapeProfileSelection::new(evidence))
    }

    fn fallback(&self, reason: ShapeProfileFallbackReason) -> Result<ShapeProfileExecutionPath> {
        let Some(implementation_sha256) = self.dynamic_fallback.implementation_sha256() else {
            return Err(PowerError::InvalidRequest(format!(
                "shape-profile request requires {reason:?}, but dynamic fallback is denied"
            )));
        };
        Ok(ShapeProfileExecutionPath::DynamicFallback {
            reason,
            implementation_sha256: implementation_sha256.to_string(),
        })
    }

    fn recompute_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Payload<'a> {
            schema: &'a str,
            binding: &'a ShapeProfileBinding,
            profiles: &'a [ShapeProfile],
            dynamic_fallback: &'a DynamicShapeFallback,
        }
        domain_sha256(
            b"a3s-power-shape-profile-declaration-v1\0",
            &serde_json::to_vec(&Payload {
                schema: &self.schema,
                binding: &self.binding,
                profiles: &self.profiles,
                dynamic_fallback: &self.dynamic_fallback,
            })?,
        )
    }
}

impl std::fmt::Debug for ShapeProfileDeclaration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ShapeProfileDeclaration")
            .field("schema", &self.schema)
            .field("binding", &"sha256")
            .field("profile_count", &self.profiles.len())
            .field("dynamic_fallback", &self.dynamic_fallback)
            .field("declaration", &"sha256")
            .finish()
    }
}

/// Opaque model-class request plus aggregate bounds checked by Power.
#[derive(Clone, PartialEq, Eq)]
pub struct ShapeProfileRequest {
    input_sha256: String,
    shape_class_sha256: String,
    batch_size: usize,
    tensor_elements: usize,
}

impl ShapeProfileRequest {
    pub fn new(
        input_sha256: impl Into<String>,
        shape_class_sha256: impl Into<String>,
        batch_size: usize,
        tensor_elements: usize,
    ) -> Result<Self> {
        let request = Self {
            input_sha256: canonical_sha256(input_sha256, "shape-profile input")?,
            shape_class_sha256: canonical_sha256(
                shape_class_sha256,
                "shape-profile request class",
            )?,
            batch_size,
            tensor_elements,
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<()> {
        validate_canonical_sha256(&self.input_sha256, "shape-profile input")?;
        validate_canonical_sha256(&self.shape_class_sha256, "shape-profile request class")?;
        if self.batch_size == 0 || self.tensor_elements == 0 {
            return Err(PowerError::InvalidRequest(
                "shape-profile requests require positive batch and tensor-element counts"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn request_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Payload<'a> {
            input_sha256: &'a str,
            shape_class_sha256: &'a str,
            batch_size: usize,
            tensor_elements: usize,
        }
        domain_sha256(
            b"a3s-power-shape-profile-request-v1\0",
            &serde_json::to_vec(&Payload {
                input_sha256: &self.input_sha256,
                shape_class_sha256: &self.shape_class_sha256,
                batch_size: self.batch_size,
                tensor_elements: self.tensor_elements,
            })?,
        )
    }
}

impl std::fmt::Debug for ShapeProfileRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ShapeProfileRequest")
            .field("input", &"sha256")
            .field("shape_class", &"sha256")
            .field("execution_envelope", &"bounded")
            .finish()
    }
}
