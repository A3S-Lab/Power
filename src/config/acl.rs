use std::collections::{HashMap, HashSet};

use a3s_acl::{
    generate, parse, validate_document, AttributeSchema, Block, BlockSchema, Cardinality, Document,
    Schema, Value, ValueSchema,
};
use serde_json::{Map, Number};

use super::PowerConfig;
use crate::error::{PowerError, Result};

const NORMAL_FIELDS: &[&str] = &[
    "host",
    "port",
    "data_dir",
    "max_loaded_models",
    "spec_mode",
    "spec_draft_max",
    "spec_draft_min",
    "spec_draft_p_min",
    "keep_alive",
    "use_mlock",
    "num_thread",
    "flash_attention",
    "num_parallel",
    "tee_mode",
    "tee_policy_mode",
    "redact_logs",
    "tls_port",
    "tls_sans",
    "ra_tls",
    "vsock_port",
    "api_keys",
    "allowed_tee_types",
    "audit_log",
    "audit_log_path",
    "audit_log_encrypt",
    "model_signing_key",
    "key_provider",
    "in_memory_decrypt",
    "streaming_decrypt",
    "suppress_token_metrics",
    "rate_limit_rps",
    "max_concurrent_requests",
    "proxy_effective_prompt_digest",
    "proxy_effective_prompt_digest_required",
    "proxy_effective_prompt_digest_path",
    "timing_padding_ms",
];

const ALWAYS_GENERATED_FIELDS: &[&str] = &[
    "host",
    "port",
    "data_dir",
    "max_loaded_models",
    "spec_mode",
    "keep_alive",
    "use_mlock",
    "flash_attention",
    "num_parallel",
    "tee_mode",
    "tee_policy_mode",
    "redact_logs",
];

pub(super) fn deserialize(input: &str) -> Result<PowerConfig> {
    let document =
        parse(input).map_err(|error| PowerError::Acl(format!("failed to parse ACL: {error}")))?;
    let report = validate_document(&document, &power_schema());
    if !report.is_empty() {
        let mut diagnostics = report
            .diagnostics
            .iter()
            .map(|diagnostic| {
                format!(
                    "{} at {}: {}",
                    diagnostic.code.as_str(),
                    diagnostic.path,
                    diagnostic.message
                )
            })
            .collect::<Vec<_>>();
        if report.truncated {
            diagnostics.push("additional ACL schema diagnostics were truncated".to_string());
        }
        return Err(PowerError::Acl(diagnostics.join("; ")));
    }

    let value = document_to_json(&document)?;
    serde_json::from_value(value)
        .map_err(|error| PowerError::Acl(format!("failed to decode ACL configuration: {error}")))
}

pub(super) fn serialize(config: &PowerConfig) -> Result<String> {
    let current = serde_json::to_value(config)?;
    let defaults = serde_json::to_value(PowerConfig::default())?;
    let mut current = into_json_object(current, "Power configuration")?;
    let defaults = into_json_object(defaults, "default Power configuration")?;
    let mut blocks = Vec::new();

    let gpu = current
        .remove("gpu")
        .ok_or_else(|| PowerError::Config("serialized Power configuration omitted gpu".into()))?;
    blocks.push(json_object_block("gpu", gpu)?);

    let gpu_attestation = current.remove("gpu_attestation").ok_or_else(|| {
        PowerError::Config("serialized Power configuration omitted gpu_attestation".into())
    })?;
    if defaults.get("gpu_attestation") != Some(&gpu_attestation) {
        blocks.push(json_object_block("gpu_attestation", gpu_attestation)?);
    }

    append_optional_object_block(
        &mut blocks,
        "model_key_source",
        current.remove("model_key_source"),
    )?;
    append_optional_object_block(
        &mut blocks,
        "audit_key_source",
        current.remove("audit_key_source"),
    )?;
    append_ordered_object_blocks(
        &mut blocks,
        "key_rotation_source",
        current.remove("key_rotation_sources"),
    )?;
    append_labeled_map_blocks(
        &mut blocks,
        "model_hash",
        "digest",
        current.remove("model_hashes"),
    )?;
    append_labeled_map_blocks(
        &mut blocks,
        "expected_measurement",
        "digest",
        current.remove("expected_measurements"),
    )?;
    append_labeled_map_blocks(
        &mut blocks,
        "proxy_upstream",
        "url",
        current.remove("proxy_upstreams"),
    )?;

    for field in NORMAL_FIELDS {
        let value = current.remove(*field).unwrap_or(serde_json::Value::Null);
        let default = defaults.get(*field).unwrap_or(&serde_json::Value::Null);
        let always_generate = ALWAYS_GENERATED_FIELDS.contains(field);
        if !value.is_null() && (always_generate || default != &value) {
            blocks.push(assignment(field, json_to_acl(value)?));
        }
    }

    if !current.is_empty() {
        let fields = current.keys().cloned().collect::<Vec<_>>().join(", ");
        return Err(PowerError::Config(format!(
            "ACL serializer does not define fields: {fields}"
        )));
    }

    Ok(generate(&Document { blocks }))
}

fn power_schema() -> Schema {
    let mut schema = Schema::new();
    for name in [
        "host",
        "data_dir",
        "spec_mode",
        "keep_alive",
        "tee_policy_mode",
        "audit_log_path",
        "model_signing_key",
        "key_provider",
        "proxy_effective_prompt_digest_path",
    ] {
        schema = schema.attribute(name, AttributeSchema::optional(ValueSchema::string()));
    }
    for name in [
        "port",
        "max_loaded_models",
        "spec_draft_max",
        "spec_draft_min",
        "spec_draft_p_min",
        "num_thread",
        "num_parallel",
        "tls_port",
        "vsock_port",
        "rate_limit_rps",
        "max_concurrent_requests",
        "timing_padding_ms",
    ] {
        schema = schema.attribute(name, AttributeSchema::optional(ValueSchema::number()));
    }
    for name in [
        "use_mlock",
        "flash_attention",
        "tee_mode",
        "redact_logs",
        "ra_tls",
        "audit_log",
        "audit_log_encrypt",
        "in_memory_decrypt",
        "streaming_decrypt",
        "suppress_token_metrics",
        "proxy_effective_prompt_digest",
        "proxy_effective_prompt_digest_required",
    ] {
        schema = schema.attribute(name, AttributeSchema::optional(ValueSchema::bool()));
    }
    for name in ["tls_sans", "api_keys", "allowed_tee_types"] {
        schema = schema.attribute(
            name,
            AttributeSchema::optional(ValueSchema::list(ValueSchema::string())),
        );
    }

    schema
        .block("gpu", optional_singleton_block(gpu_schema()))
        .block(
            "gpu_attestation",
            optional_singleton_block(gpu_attestation_schema()),
        )
        .block(
            "model_key_source",
            optional_singleton_block(key_source_schema()),
        )
        .block(
            "audit_key_source",
            optional_singleton_block(key_source_schema()),
        )
        .block(
            "key_rotation_source",
            BlockSchema::new(key_source_schema())
                .occurrences(Cardinality::at_least(0))
                .labels(Cardinality::exactly(0)),
        )
        .block("model_hash", labeled_string_map_block("digest"))
        .block("expected_measurement", labeled_string_map_block("digest"))
        .block("proxy_upstream", labeled_string_map_block("url"))
}

fn gpu_schema() -> Schema {
    Schema::new()
        .attribute(
            "gpu_layers",
            AttributeSchema::optional(ValueSchema::number()),
        )
        .attribute("main_gpu", AttributeSchema::optional(ValueSchema::number()))
        .attribute(
            "tensor_split",
            AttributeSchema::optional(ValueSchema::list(ValueSchema::number())),
        )
}

fn gpu_attestation_schema() -> Schema {
    let mut schema = Schema::new();
    for name in [
        "source",
        "provider",
        "evidence_hex",
        "evidence_path",
        "verdict_hex",
        "verdict_path",
        "nvattest_path",
        "nvattest_verifier",
        "nvattest_gpu_evidence_source",
        "nvattest_gpu_architecture",
        "nras_url",
        "nras_gpu_architecture",
        "nras_claims_version",
        "nras_bearer_token_env",
        "rim_url",
        "ocsp_url",
        "relying_party_policy_path",
    ] {
        schema = schema.attribute(name, AttributeSchema::optional(ValueSchema::string()));
    }
    for name in ["nras_timeout_secs", "nvattest_timeout_secs"] {
        schema = schema.attribute(name, AttributeSchema::optional(ValueSchema::number()));
    }
    schema
}

fn key_source_schema() -> Schema {
    Schema::new()
        .attribute("type", AttributeSchema::required(ValueSchema::string()))
        .attribute("value", AttributeSchema::required(ValueSchema::string()))
}

fn optional_singleton_block(body: Schema) -> BlockSchema {
    BlockSchema::new(body)
        .occurrences(Cardinality::new(0, Some(1)).expect("valid singleton cardinality"))
        .labels(Cardinality::exactly(0))
}

fn labeled_string_map_block(attribute: &str) -> BlockSchema {
    BlockSchema::new(
        Schema::new().attribute(attribute, AttributeSchema::required(ValueSchema::string())),
    )
    .occurrences(Cardinality::at_least(0))
    .labels(Cardinality::exactly(1))
    .unordered(true)
}

fn document_to_json(document: &Document) -> Result<serde_json::Value> {
    let mut root = Map::new();
    let mut key_rotation_sources = Vec::new();
    let mut model_hashes = Map::new();
    let mut expected_measurements = Map::new();
    let mut proxy_upstreams = Map::new();

    for block in &document.blocks {
        if let Some((name, value)) = bare_attribute(block) {
            root.insert(name.to_string(), acl_to_json(value)?);
            continue;
        }

        match block.name.as_str() {
            "gpu" | "gpu_attestation" | "model_key_source" | "audit_key_source" => {
                root.insert(block.name.clone(), block_to_json(block)?);
            }
            "key_rotation_source" => key_rotation_sources.push(block_to_json(block)?),
            "model_hash" => insert_labeled_value(block, "digest", &mut model_hashes)?,
            "expected_measurement" => {
                insert_labeled_value(block, "digest", &mut expected_measurements)?
            }
            "proxy_upstream" => insert_labeled_value(block, "url", &mut proxy_upstreams)?,
            other => {
                return Err(PowerError::Acl(format!(
                    "unsupported ACL block {other:?} passed schema validation"
                )))
            }
        }
    }

    if !key_rotation_sources.is_empty() {
        root.insert(
            "key_rotation_sources".to_string(),
            serde_json::Value::Array(key_rotation_sources),
        );
    }
    if !model_hashes.is_empty() {
        root.insert(
            "model_hashes".to_string(),
            serde_json::Value::Object(model_hashes),
        );
    }
    if !expected_measurements.is_empty() {
        root.insert(
            "expected_measurements".to_string(),
            serde_json::Value::Object(expected_measurements),
        );
    }
    if !proxy_upstreams.is_empty() {
        root.insert(
            "proxy_upstreams".to_string(),
            serde_json::Value::Object(proxy_upstreams),
        );
    }

    Ok(serde_json::Value::Object(root))
}

fn bare_attribute(block: &Block) -> Option<(&str, &Value)> {
    if block.labels.is_empty() && block.blocks.is_empty() && block.attributes.len() == 1 {
        block
            .attributes
            .get(&block.name)
            .map(|value| (block.name.as_str(), value))
    } else {
        None
    }
}

fn block_to_json(block: &Block) -> Result<serde_json::Value> {
    let mut object = Map::new();
    for (name, value) in &block.attributes {
        object.insert(name.clone(), acl_to_json(value)?);
    }
    Ok(serde_json::Value::Object(object))
}

fn insert_labeled_value(
    block: &Block,
    attribute: &str,
    destination: &mut Map<String, serde_json::Value>,
) -> Result<()> {
    let label = block
        .labels
        .first()
        .ok_or_else(|| PowerError::Acl(format!("ACL block {:?} requires one label", block.name)))?;
    let value = block.attributes.get(attribute).ok_or_else(|| {
        PowerError::Acl(format!(
            "ACL block {:?} requires attribute {attribute:?}",
            block.name
        ))
    })?;
    if destination.contains_key(label) {
        return Err(PowerError::Acl(format!(
            "ACL block {:?} repeats label {label:?}",
            block.name
        )));
    }
    destination.insert(label.clone(), acl_to_json(value)?);
    Ok(())
}

fn acl_to_json(value: &Value) -> Result<serde_json::Value> {
    match value {
        Value::String(value) => Ok(serde_json::Value::String(value.clone())),
        Value::Bool(value) => Ok(serde_json::Value::Bool(*value)),
        Value::Null => Ok(serde_json::Value::Null),
        Value::Number(value) => acl_number_to_json(*value),
        Value::List(values) => values
            .iter()
            .map(acl_to_json)
            .collect::<Result<Vec<_>>>()
            .map(serde_json::Value::Array),
        Value::Object(fields) => {
            let mut object = Map::new();
            for (name, value) in fields {
                if object.contains_key(name) {
                    return Err(PowerError::Acl(format!(
                        "ACL object repeats field {name:?}"
                    )));
                }
                object.insert(name.clone(), acl_to_json(value)?);
            }
            Ok(serde_json::Value::Object(object))
        }
        Value::Call(name, _) => Err(PowerError::Acl(format!(
            "ACL function call {name:?} is not allowed in Power configuration"
        ))),
    }
}

fn acl_number_to_json(value: f64) -> Result<serde_json::Value> {
    if !value.is_finite() {
        return Err(PowerError::Acl("ACL number must be finite".into()));
    }
    if value.fract() == 0.0 {
        if value < 0.0 && value >= i64::MIN as f64 {
            return Ok(serde_json::Value::Number(Number::from(value as i64)));
        }
        if value >= 0.0 && value <= u64::MAX as f64 {
            return Ok(serde_json::Value::Number(Number::from(value as u64)));
        }
    }
    Number::from_f64(value)
        .map(serde_json::Value::Number)
        .ok_or_else(|| PowerError::Acl("ACL number cannot be represented as JSON".into()))
}

fn assignment(name: &str, value: Value) -> Block {
    Block {
        name: name.to_string(),
        labels: Vec::new(),
        blocks: Vec::new(),
        attributes: HashMap::from([(name.to_string(), value)]),
    }
}

fn json_object_block(name: &str, value: serde_json::Value) -> Result<Block> {
    let object = into_json_object(value, name)?;
    let attributes = object
        .into_iter()
        .map(|(field, value)| Ok((field, json_to_acl(value)?)))
        .collect::<Result<HashMap<_, _>>>()?;
    Ok(Block {
        name: name.to_string(),
        labels: Vec::new(),
        blocks: Vec::new(),
        attributes,
    })
}

fn append_optional_object_block(
    blocks: &mut Vec<Block>,
    name: &str,
    value: Option<serde_json::Value>,
) -> Result<()> {
    if let Some(value) = value.filter(|value| !value.is_null()) {
        blocks.push(json_object_block(name, value)?);
    }
    Ok(())
}

fn append_ordered_object_blocks(
    blocks: &mut Vec<Block>,
    name: &str,
    value: Option<serde_json::Value>,
) -> Result<()> {
    let Some(serde_json::Value::Array(values)) = value else {
        return Ok(());
    };
    for value in values {
        blocks.push(json_object_block(name, value)?);
    }
    Ok(())
}

fn append_labeled_map_blocks(
    blocks: &mut Vec<Block>,
    name: &str,
    attribute: &str,
    value: Option<serde_json::Value>,
) -> Result<()> {
    let Some(serde_json::Value::Object(values)) = value else {
        return Ok(());
    };
    for (label, value) in values {
        blocks.push(Block {
            name: name.to_string(),
            labels: vec![label],
            blocks: Vec::new(),
            attributes: HashMap::from([(attribute.to_string(), json_to_acl(value)?)]),
        });
    }
    Ok(())
}

fn json_to_acl(value: serde_json::Value) -> Result<Value> {
    match value {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(value) => Ok(Value::Bool(value)),
        serde_json::Value::String(value) => Ok(Value::String(value)),
        serde_json::Value::Number(value) => {
            let number = value.as_f64().ok_or_else(|| {
                PowerError::Config(format!("JSON number {value} cannot be represented by ACL"))
            })?;
            if value.is_u64() && number as u64 != value.as_u64().unwrap_or_default() {
                return Err(PowerError::Config(format!(
                    "JSON integer {value} exceeds exact ACL number precision"
                )));
            }
            Ok(Value::Number(number))
        }
        serde_json::Value::Array(values) => values
            .into_iter()
            .map(json_to_acl)
            .collect::<Result<Vec<_>>>()
            .map(Value::List),
        serde_json::Value::Object(fields) => {
            let mut seen = HashSet::new();
            let mut values = Vec::with_capacity(fields.len());
            for (name, value) in fields {
                if !seen.insert(name.clone()) {
                    return Err(PowerError::Config(format!(
                        "JSON object repeats field {name:?}"
                    )));
                }
                values.push((name, json_to_acl(value)?));
            }
            Ok(Value::Object(values))
        }
    }
}

fn into_json_object(
    value: serde_json::Value,
    label: &str,
) -> Result<Map<String, serde_json::Value>> {
    match value {
        serde_json::Value::Object(object) => Ok(object),
        _ => Err(PowerError::Config(format!("{label} must be an object"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_attributes() {
        let error = deserialize("host = \"127.0.0.1\"\nunknown = true\n").unwrap_err();
        assert!(error.to_string().contains("acl.schema.unknown_attribute"));
    }

    #[test]
    fn rejects_duplicate_labeled_map_entries() {
        let error = deserialize(
            "model_hash \"model-a\" { digest = \"one\" }\nmodel_hash \"model-a\" { digest = \"two\" }\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("repeats label"));
    }

    #[test]
    fn round_trips_closed_acl() {
        let config = PowerConfig {
            host: "0.0.0.0".into(),
            port: 11_435,
            proxy_upstreams: HashMap::from([(
                "model/with-punctuation".into(),
                "http://power-member:8000".into(),
            )]),
            ..PowerConfig::default()
        };
        let encoded = serialize(&config).unwrap();
        assert!(encoded.contains("proxy_upstream \"model/with-punctuation\""));
        let decoded = deserialize(&encoded).unwrap();
        assert_eq!(decoded.host, config.host);
        assert_eq!(decoded.port, config.port);
        assert_eq!(decoded.proxy_upstreams, config.proxy_upstreams);
    }
}
