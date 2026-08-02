use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::error::{PowerError, Result};

use super::super::{TensorDescriptor, WeightStore};

pub(super) const DETECTION_PLAN: &str = include_str!("graphs/small_detection.json");
pub(super) const RECOGNITION_PLAN: &str = include_str!("graphs/small_recognition.json");

const FAMILY: &str = "pp-ocr-v6-small";
const DETECTION_SOURCE_SHA256: &str =
    "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e";
const RECOGNITION_SOURCE_SHA256: &str =
    "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634";
const MAX_NODES: usize = 2_048;
const MAX_INITIALIZERS: usize = 4_096;
const MAX_NAME_BYTES: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GraphRole {
    Detection,
    Recognition,
}

impl GraphRole {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Detection => "detection",
            Self::Recognition => "recognition",
        }
    }

    fn source_sha256(self) -> &'static str {
        match self {
            Self::Detection => DETECTION_SOURCE_SHA256,
            Self::Recognition => RECOGNITION_SOURCE_SHA256,
        }
    }

    fn opset(self) -> u32 {
        match self {
            Self::Detection => 14,
            Self::Recognition => 11,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct GraphPlan {
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
    pub(super) fn parse(source: &str, role: GraphRole, weights: &WeightStore) -> Result<Self> {
        let plan: Self = serde_json::from_str(source).map_err(|error| {
            PowerError::InvalidFormat(format!(
                "failed to parse embedded PP-OCRv6 {} graph: {error}",
                role.name()
            ))
        })?;
        plan.validate(role, weights)?;
        Ok(plan)
    }

    fn validate(&self, role: GraphRole, weights: &WeightStore) -> Result<()> {
        if self.schema_version != 1
            || self.family != FAMILY
            || self.role != role.name()
            || self.source.format != "onnx"
            || self.source.sha256 != role.source_sha256()
            || self.source.opset != role.opset()
        {
            return Err(PowerError::InvalidFormat(format!(
                "embedded PP-OCRv6 {} graph identity does not match the reviewed model",
                role.name()
            )));
        }
        if self.inputs.len() != 1 || self.outputs.len() != 1 {
            return Err(PowerError::InvalidFormat(format!(
                "PP-OCRv6 {} graph must expose exactly one input and one output",
                role.name()
            )));
        }
        if self.nodes.is_empty() || self.nodes.len() > MAX_NODES {
            return Err(PowerError::InvalidFormat(format!(
                "PP-OCRv6 {} graph node count is outside the supported bound",
                role.name()
            )));
        }
        if self.initializers.is_empty() || self.initializers.len() > MAX_INITIALIZERS {
            return Err(PowerError::InvalidFormat(format!(
                "PP-OCRv6 {} initializer count is outside the supported bound",
                role.name()
            )));
        }

        let inventory = weights
            .inventory()
            .map(|descriptor| (descriptor.name.as_str(), descriptor))
            .collect::<BTreeMap<_, _>>();
        if inventory.len() != self.initializers.len() {
            return Err(PowerError::InvalidFormat(format!(
                "PP-OCRv6 {} weight inventory contains {} tensors; expected {}",
                role.name(),
                inventory.len(),
                self.initializers.len()
            )));
        }
        let mut available = BTreeSet::new();
        available.insert(self.inputs[0].name.as_str());
        for initializer in &self.initializers {
            validate_name(&initializer.name)?;
            let descriptor = inventory.get(initializer.name.as_str()).ok_or_else(|| {
                PowerError::InvalidFormat(format!(
                    "PP-OCRv6 {} is missing initializer '{}'",
                    role.name(),
                    initializer.name
                ))
            })?;
            validate_initializer(initializer, descriptor)?;
            if !available.insert(initializer.name.as_str()) {
                return Err(PowerError::InvalidFormat(format!(
                    "PP-OCRv6 {} declares duplicate value '{}'",
                    role.name(),
                    initializer.name
                )));
            }
        }
        for node in &self.nodes {
            validate_name(&node.name)?;
            if node.outputs.len() != 1 {
                return Err(PowerError::InvalidFormat(format!(
                    "PP-OCRv6 node '{}' must have exactly one output",
                    node.name
                )));
            }
            if node
                .inputs
                .iter()
                .any(|name| !available.contains(name.as_str()))
            {
                return Err(PowerError::InvalidFormat(format!(
                    "PP-OCRv6 node '{}' consumes an undeclared value",
                    node.name
                )));
            }
            let output = &node.outputs[0];
            validate_name(output)?;
            if !available.insert(output) {
                return Err(PowerError::InvalidFormat(format!(
                    "PP-OCRv6 graph writes value '{output}' more than once"
                )));
            }
            if node.attributes.len() > 32 {
                return Err(PowerError::InvalidFormat(format!(
                    "PP-OCRv6 node '{}' has too many attributes",
                    node.name
                )));
            }
        }
        if !available.contains(self.outputs[0].name.as_str()) {
            return Err(PowerError::InvalidFormat(format!(
                "PP-OCRv6 {} output is not produced by its graph",
                role.name()
            )));
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
            "PP-OCRv6 node '{}' has an invalid '{name}' attribute",
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
            "PP-OCRv6 initializer '{}' expected {dtype} {:?}, found {} {:?}",
            initializer.name, initializer.shape, descriptor.dtype, descriptor.shape
        )));
    }
    Ok(())
}

fn validate_name(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > MAX_NAME_BYTES || value.chars().any(char::is_control) {
        return Err(PowerError::InvalidFormat(
            "PP-OCRv6 graph contains an invalid value name".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_graphs_have_reviewed_identity() {
        let detection: GraphPlan = serde_json::from_str(DETECTION_PLAN).unwrap();
        let recognition: GraphPlan = serde_json::from_str(RECOGNITION_PLAN).unwrap();
        assert_eq!(detection.source.sha256, DETECTION_SOURCE_SHA256);
        assert_eq!(recognition.source.sha256, RECOGNITION_SOURCE_SHA256);
        assert_eq!(detection.nodes.len(), 242);
        assert_eq!(recognition.nodes.len(), 481);
    }
}
