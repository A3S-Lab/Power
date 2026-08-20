use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::super::{InferenceLimits, TensorDescriptor, WeightStore};

const GRAPH_SCHEMA_VERSION: u32 = 1;
const MAX_NODE_ATTRIBUTES: usize = 32;

/// Model-owned identity expected for a reviewed static graph.
///
/// Power validates this identity but does not define model families, graph
/// roles, revisions, or source hashes. Those remain the responsibility of the
/// model crate that embeds the reviewed plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GraphIdentity {
    pub family: String,
    pub role: String,
    pub source_format: String,
    pub source_sha256: String,
    pub opset: u32,
}

impl GraphIdentity {
    pub fn new(
        family: impl Into<String>,
        role: impl Into<String>,
        source_format: impl Into<String>,
        source_sha256: impl Into<String>,
        opset: u32,
    ) -> Self {
        Self {
            family: family.into(),
            role: role.into(),
            source_format: source_format.into(),
            source_sha256: source_sha256.into(),
            opset,
        }
    }

    /// Canonical digest of every semantic field in the reviewed graph
    /// identity. The source digest is normalized because hexadecimal case does
    /// not change the identified bytes.
    pub(super) fn binding_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Binding<'a> {
            family: &'a str,
            role: &'a str,
            source_format: &'a str,
            source_sha256: String,
            opset: u32,
        }

        let payload = serde_json::to_vec(&Binding {
            family: &self.family,
            role: &self.role,
            source_format: &self.source_format,
            source_sha256: self.source_sha256.to_ascii_lowercase(),
            opset: self.opset,
        })?;
        let payload_len = u64::try_from(payload.len()).map_err(|_| {
            PowerError::InvalidRequest(
                "reviewed graph identity payload length overflowed".to_string(),
            )
        })?;
        let mut digest = Sha256::new();
        digest.update(b"a3s-power-reviewed-graph-identity-v1\0");
        digest.update(payload_len.to_le_bytes());
        digest.update(payload);
        Ok(format!("{:x}", digest.finalize()))
    }

    fn validate(&self, limits: &InferenceLimits) -> Result<()> {
        for (label, value) in [
            ("family", self.family.as_str()),
            ("role", self.role.as_str()),
            ("source format", self.source_format.as_str()),
        ] {
            validate_name(value, limits).map_err(|_| {
                PowerError::InvalidFormat(format!("static graph {label} is invalid"))
            })?;
        }
        if self.source_sha256.len() != 64
            || !self
                .source_sha256
                .bytes()
                .all(|value| value.is_ascii_hexdigit())
        {
            return Err(PowerError::InvalidFormat(
                "static graph source SHA-256 must contain 64 hexadecimal characters".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GraphPlan {
    schema_version: u32,
    family: String,
    role: String,
    source: GraphSource,
    pub(super) inputs: Vec<GraphTensor>,
    pub(super) outputs: Vec<GraphTensor>,
    pub(super) initializers: Vec<Initializer>,
    pub(super) nodes: Vec<GraphNode>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphSource {
    format: String,
    sha256: String,
    opset: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GraphTensor {
    pub(super) name: String,
    #[allow(dead_code)]
    shape: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Initializer {
    pub(super) name: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GraphNode {
    pub(super) name: String,
    pub(super) op: GraphOp,
    pub(super) inputs: Vec<String>,
    pub(super) outputs: Vec<String>,
    pub(super) attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub(super) enum GraphOp {
    Add,
    AveragePool,
    BatchNormalization,
    Concat,
    Conv,
    ConvTranspose,
    Div,
    Erf,
    GlobalAveragePool,
    HardSigmoid,
    Identity,
    MatMul,
    MaxPool,
    Mul,
    Pow,
    ReduceMean,
    Relu,
    Reshape,
    Resize,
    Shape,
    Sigmoid,
    Slice,
    Softmax,
    Sqrt,
    Squeeze,
    Sub,
    Transpose,
    Unsqueeze,
}

impl GraphPlan {
    pub fn parse(
        source: &str,
        expected: &GraphIdentity,
        weights: &WeightStore,
        limits: &InferenceLimits,
    ) -> Result<Self> {
        expected.validate(limits)?;
        if source.len() > limits.max_graph_plan_bytes {
            return Err(PowerError::InvalidFormat(format!(
                "static graph plan contains {} bytes, exceeding the {} byte limit",
                source.len(),
                limits.max_graph_plan_bytes
            )));
        }
        let plan: Self = serde_json::from_str(source).map_err(|error| {
            PowerError::InvalidFormat(format!("failed to parse static graph plan: {error}"))
        })?;
        plan.validate(expected, weights, limits)?;
        Ok(plan)
    }

    pub fn identity(&self) -> GraphIdentity {
        GraphIdentity {
            family: self.family.clone(),
            role: self.role.clone(),
            source_format: self.source.format.clone(),
            source_sha256: self.source.sha256.clone(),
            opset: self.source.opset,
        }
    }

    fn validate(
        &self,
        expected: &GraphIdentity,
        weights: &WeightStore,
        limits: &InferenceLimits,
    ) -> Result<()> {
        if self.schema_version != GRAPH_SCHEMA_VERSION || self.identity() != *expected {
            return Err(PowerError::InvalidFormat(
                "static graph identity does not match the model-owned reviewed identity"
                    .to_string(),
            ));
        }
        if self.inputs.len() != 1 || self.outputs.len() != 1 {
            return Err(PowerError::InvalidFormat(
                "static graph must expose exactly one input and one output".to_string(),
            ));
        }
        if self.nodes.is_empty() || self.nodes.len() > limits.max_graph_nodes {
            return Err(PowerError::InvalidFormat(
                "static graph node count is outside the configured bound".to_string(),
            ));
        }
        if self.initializers.len() > limits.max_graph_initializers {
            return Err(PowerError::InvalidFormat(
                "static graph initializer count is outside the configured bound".to_string(),
            ));
        }

        let inventory = weights
            .inventory()
            .map(|descriptor| (descriptor.name.as_str(), descriptor))
            .collect::<BTreeMap<_, _>>();
        if inventory.len() != self.initializers.len() {
            return Err(PowerError::InvalidFormat(format!(
                "static graph weight inventory contains {} tensors; expected {}",
                inventory.len(),
                self.initializers.len()
            )));
        }
        let mut available = BTreeSet::new();
        validate_name(&self.inputs[0].name, limits)?;
        validate_name(&self.outputs[0].name, limits)?;
        available.insert(self.inputs[0].name.as_str());
        for initializer in &self.initializers {
            validate_name(&initializer.name, limits)?;
            let descriptor = inventory.get(initializer.name.as_str()).ok_or_else(|| {
                PowerError::InvalidFormat(format!(
                    "static graph is missing initializer '{}'",
                    initializer.name
                ))
            })?;
            validate_initializer(initializer, descriptor)?;
            if !available.insert(initializer.name.as_str()) {
                return Err(PowerError::InvalidFormat(format!(
                    "static graph declares duplicate value '{}'",
                    initializer.name
                )));
            }
        }
        for node in &self.nodes {
            validate_name(&node.name, limits)?;
            if node.outputs.len() != 1 {
                return Err(PowerError::InvalidFormat(format!(
                    "static graph node '{}' must have exactly one output",
                    node.name
                )));
            }
            if node
                .inputs
                .iter()
                .any(|name| !available.contains(name.as_str()))
            {
                return Err(PowerError::InvalidFormat(format!(
                    "static graph node '{}' consumes an undeclared value",
                    node.name
                )));
            }
            let output = &node.outputs[0];
            validate_name(output, limits)?;
            if !available.insert(output) {
                return Err(PowerError::InvalidFormat(format!(
                    "static graph writes value '{output}' more than once"
                )));
            }
            if node.attributes.len() > MAX_NODE_ATTRIBUTES {
                return Err(PowerError::InvalidFormat(format!(
                    "static graph node '{}' has too many attributes",
                    node.name
                )));
            }
        }
        if !available.contains(self.outputs[0].name.as_str()) {
            return Err(PowerError::InvalidFormat(
                "static graph output is not produced by its graph".to_string(),
            ));
        }
        Ok(())
    }
}

impl GraphNode {
    pub(super) fn int(&self, name: &str, default: i64) -> Result<i64> {
        match self.attributes.get(name) {
            None => Ok(default),
            Some(value) => value.as_i64().ok_or_else(|| self.attribute_error(name)),
        }
    }

    pub(super) fn float(&self, name: &str, default: f64) -> Result<f64> {
        match self.attributes.get(name) {
            None => Ok(default),
            Some(value) => value.as_f64().ok_or_else(|| self.attribute_error(name)),
        }
    }

    pub(super) fn string<'a>(&'a self, name: &str, default: &'a str) -> Result<&'a str> {
        match self.attributes.get(name) {
            None => Ok(default),
            Some(value) => value.as_str().ok_or_else(|| self.attribute_error(name)),
        }
    }

    pub(super) fn ints(&self, name: &str, default: &[i64]) -> Result<Vec<i64>> {
        match self.attributes.get(name) {
            None => Ok(default.to_vec()),
            Some(serde_json::Value::Array(values)) => values
                .iter()
                .map(|value| value.as_i64().ok_or_else(|| self.attribute_error(name)))
                .collect(),
            Some(_) => Err(self.attribute_error(name)),
        }
    }

    fn attribute_error(&self, name: &str) -> PowerError {
        PowerError::InvalidFormat(format!(
            "static graph node '{}' has an invalid '{name}' attribute",
            self.name
        ))
    }
}

fn validate_initializer(initializer: &Initializer, descriptor: &TensorDescriptor) -> Result<()> {
    let dtype = match initializer.dtype.as_str() {
        "float32" => "f32",
        "float16" => "f16",
        "int64" => "i64",
        "int32" => "i32",
        other => other,
    };
    if descriptor.dtype != dtype || descriptor.shape != initializer.shape {
        return Err(PowerError::InvalidFormat(format!(
            "static graph initializer '{}' expected {dtype} {:?}, found {} {:?}",
            initializer.name, initializer.shape, descriptor.dtype, descriptor.shape
        )));
    }
    Ok(())
}

fn validate_name(value: &str, limits: &InferenceLimits) -> Result<()> {
    if value.is_empty()
        || value.len() > limits.max_graph_name_bytes
        || value.chars().any(char::is_control)
    {
        return Err(PowerError::InvalidFormat(
            "static graph contains an invalid value name".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewed_identity_requires_a_sha256() {
        let limits = InferenceLimits::default();
        let valid = GraphIdentity::new("model", "encoder", "onnx", "a".repeat(64), 17);
        assert!(valid.validate(&limits).is_ok());

        let invalid = GraphIdentity::new("model", "encoder", "onnx", "not-a-hash", 17);
        assert!(invalid.validate(&limits).is_err());
    }

    #[test]
    fn reviewed_identity_digest_binds_every_semantic_field() {
        let identity = GraphIdentity::new("model", "encoder", "onnx", "a".repeat(64), 17);
        let expected = identity.binding_sha256().unwrap();
        let variants = [
            GraphIdentity::new("other", "encoder", "onnx", "a".repeat(64), 17),
            GraphIdentity::new("model", "decoder", "onnx", "a".repeat(64), 17),
            GraphIdentity::new("model", "encoder", "gguf", "a".repeat(64), 17),
            GraphIdentity::new("model", "encoder", "onnx", "b".repeat(64), 17),
            GraphIdentity::new("model", "encoder", "onnx", "a".repeat(64), 18),
        ];
        for variant in variants {
            assert_ne!(variant.binding_sha256().unwrap(), expected);
        }
        let uppercase_source = GraphIdentity::new("model", "encoder", "onnx", "A".repeat(64), 17);
        assert_eq!(uppercase_source.binding_sha256().unwrap(), expected);
    }
}
