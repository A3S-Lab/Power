use super::*;
use crate::config::XolotlManagedConfig;
fn configured() -> PowerConfig {
    let mut config = PowerConfig::default();
    config.xolotl.enabled = true;
    config.xolotl.managed = Some(XolotlManagedConfig {
        model_name: "exact-model".into(),
        model_sha256: "1".repeat(64),
        model_parts: vec![crate::config::XolotlModelPart {
            path: "/models/main/part-01".into(),
            size_bytes: 32,
            sha256: "2".repeat(64),
        }],
        gpu_uuids: vec!["GPU-511088e2-3e4c-2a27-42c4-fd6659c7fbfc".into()],
        python: "/usr/bin/python3.12".into(),
        bridge: "/opt/a3s/bin/power_xolotl_bridge.py".into(),
        server: "/opt/xolotl/tools/v41/serve/serve_ds41.py".into(),
        tokenizer: "/models/main/tokenizer".into(),
        startup_timeout_seconds: 1800,
        shutdown_timeout_seconds: 15,
    });
    config
}
#[test]
fn xolotl_managed_acl_round_trip_preserves_the_exact_runtime_binding() {
    let config = configured();
    let encoded = serialize(&config).unwrap();
    let decoded = deserialize(&encoded).unwrap();
    assert_eq!(decoded.xolotl.managed, config.xolotl.managed);
    assert!(encoded.contains("managed ="));
}
#[test]
fn xolotl_managed_acl_schema_refuses_unknown_or_missing_authority_fields() {
    let encoded = serialize(&configured()).unwrap();
    for unknown in [true, false] {
        let mut document = parse(&encoded).unwrap();
        let xolotl = document
            .blocks
            .iter_mut()
            .find(|b| b.name == "xolotl")
            .unwrap();
        let Value::Object(managed) = xolotl.attributes.get_mut("managed").unwrap() else {
            panic!("managed must be an ACL object")
        };
        if unknown {
            managed.push(("unknown".into(), Value::Bool(true)));
        } else {
            managed.retain(|(key, _)| key != "gpu_uuids");
        }
        assert!(!validate_document(&document, &power_schema()).is_empty());
    }
}

#[test]
fn xolotl_tuning_native_acl_round_trip_and_closed_schema() {
    let mut cfg = configured();
    cfg.xolotl.tuning = Some(crate::config::XolotlTuning {
        context_tokens: 1_048_576,
        cpu_threads: 16,
        arena_gib: 120,
        prompt_batch: 8192,
        prefill_chunk: 512,
        pcie_bp: 7500,
        tier_batch_pcie_bp: 10_000,
    });
    let encoded = serialize(&cfg).unwrap();
    assert_eq!(
        crate::config::PowerConfig::from_acl(&encoded)
            .unwrap()
            .xolotl
            .tuning,
        cfg.xolotl.tuning
    );
    let mut doc = parse(&encoded).unwrap();
    let x = doc.blocks.iter_mut().find(|b| b.name == "xolotl").unwrap();
    let Value::Object(t) = x.attributes.get_mut("tuning").unwrap() else {
        panic!("tuning")
    };
    t.push(("extra_args".into(), Value::String("--mock".into())));
    assert!(!validate_document(&doc, &power_schema()).is_empty());
    let invalid = encoded.replace("1048576", "2097152");
    assert!(crate::config::PowerConfig::from_acl(&invalid).is_err());
}
