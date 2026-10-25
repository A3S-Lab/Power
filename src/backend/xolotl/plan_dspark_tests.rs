use super::*;
use crate::backend::test_utils::sample_manifest;
use crate::config::{XolotlDSparkConfig, XolotlManagedConfig, XolotlTuning};
fn fixture() -> (PowerConfig, ModelManifest, Vec<CudaDevice>) {
    let mut c = PowerConfig::default();
    c.xolotl.enabled = true;
    c.xolotl.bin = Some("/runtime/ds41_generate".into());
    c.xolotl.model = Some("/models/DeepSeek-V4.1-Flash-1017-XQ.gguf".into());
    c.xolotl.engram = Some("/models/engram".into());
    c.xolotl.rank = Some("/models/rank-1017-v3.bin".into());
    c.xolotl.gpus = vec![6];
    c.xolotl.layer_split = "none".into();
    c.xolotl.cuda_architecture = "8.9".into();
    c.xolotl.kv_fp4 = true;
    c.xolotl.tuning = Some(XolotlTuning {
        context_tokens: 1048576,
        cpu_threads: 24,
        arena_gib: 144,
        prompt_batch: 8192,
        prefill_chunk: 512,
        pcie_bp: 0,
        tier_batch_pcie_bp: 10000,
    });
    c.xolotl.managed = Some(XolotlManagedConfig {
        model_name: "xq".into(),
        model_sha256: "a".repeat(64),
        model_parts: vec![],
        gpu_uuids: vec![],
        python: "/python".into(),
        bridge: "/bridge".into(),
        server: "/server".into(),
        tokenizer: "/tokenizer".into(),
        startup_timeout_seconds: 1800,
        shutdown_timeout_seconds: 15,
    });
    c.xolotl.dspark = Some(XolotlDSparkConfig {
        sidecar: "/models/draft.gguf".into(),
        sidecar_sha256: "b".repeat(64),
        depth: 4,
        confidence_bp: 6000,
        evict_prompt_tokens: 4096,
    });
    let mut m = sample_manifest("xq");
    m.path = c.xolotl.model.clone().unwrap().into();
    let gpu = parse_nvidia_smi_csv("6, NVIDIA GeForce RTX 4090 D, 8.9, 24564\n").unwrap();
    (c, m, gpu)
}
#[test]
fn dspark_plan_preserves_exact_routing_one_gpu_and_one_million_context() {
    let (c, m, g) = fixture();
    let p = plan(&c, &m, &g).unwrap();
    assert_eq!(p.nvidia_indices, vec![6]);
    assert_eq!(arg_value(&p.argv, "--ctx").unwrap(), "1048576");
    for key in ["--route-bias", "--route-bias-file", "--route-bias-remote"] {
        assert_eq!(arg_value(&p.argv, key).unwrap(), "0");
    }
    assert_eq!(
        p.env
            .iter()
            .find(|(k, _)| k == "STRATA_DSPARK_DEPTH")
            .unwrap()
            .1,
        "4"
    );
    assert_eq!(p.env.len(), 5);
}
#[test]
fn dspark_refuses_unowned_or_unproven_shapes() {
    for mode in ["managed", "tuning", "kv", "placement", "depth"] {
        let (mut c, m, g) = fixture();
        match mode {
            "managed" => c.xolotl.managed = None,
            "tuning" => c.xolotl.tuning = None,
            "kv" => c.xolotl.kv_fp4 = false,
            "placement" => c.xolotl.layer_split = REQUIRED_LAYER_SPLIT.into(),
            "depth" => c.xolotl.dspark.as_mut().unwrap().depth = 6,
            _ => unreachable!(),
        }
        assert!(plan(&c, &m, &g).is_err(), "{mode}");
    }
}

#[test]
fn indexer_tile_env_preserves_every_native_argument_with_or_without_drafts() {
    for on in [false, true] {
        let (mut c, m, g) = fixture();
        if !on {
            c.xolotl.dspark = None;
        }
        let original = plan(&c, &m, &g).unwrap();
        c.xolotl.indexer_tile_elements = Some(1 << 26);
        let tiled = plan(&c, &m, &g).unwrap();
        assert_eq!(tiled.argv, original.argv);
        assert_eq!(
            tiled
                .env
                .iter()
                .filter(|(k, _)| k == "STRATA_INDEXER_TILE")
                .map(|(_, v)| v.as_str())
                .collect::<Vec<_>>(),
            vec!["67108864"]
        );
        c.xolotl.indexer_tile_elements = Some((1 << 26) + 1);
        assert!(plan(&c, &m, &g).is_err());
    }
}
