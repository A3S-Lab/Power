use super::*;
use crate::backend::test_utils::sample_manifest;

#[test]
fn xolotl_fp4_kv_preserves_the_admitted_placement_and_context() {
    let mut config = PowerConfig::default();
    config.xolotl.enabled = true;
    config.xolotl.bin = Some("/runtime/ds41_generate".into());
    config.xolotl.model = Some("/models/DeepSeek-V4.1-Flash-1017-XQ.gguf".into());
    config.xolotl.engram = Some("/models/engram".into());
    config.xolotl.rank = Some("/models/rank-1017.bin".into());
    config.xolotl.tuning = Some(crate::config::XolotlTuning {
        context_tokens: 1_048_576,
        cpu_threads: 16,
        arena_gib: 120,
        prompt_batch: 8192,
        prefill_chunk: 512,
        pcie_bp: 7500,
        tier_batch_pcie_bp: 10000,
    });
    let mut manifest = sample_manifest("xq-1017");
    manifest.path = config.xolotl.model.clone().unwrap().into();
    for single in [false, true] {
        config.xolotl.gpus = if single { vec![6] } else { (0..8).collect() };
        config.xolotl.cuda_architecture = if single { "8.9" } else { "9.0" }.into();
        config.xolotl.layer_split = if single { "none" } else { REQUIRED_LAYER_SPLIT }.into();
        let inventory = if single {
            parse_nvidia_smi_csv("6, NVIDIA GeForce RTX 4090 D, 8.9, 24564\n").unwrap()
        } else {
            (0..8)
                .map(|index| CudaDevice {
                    index,
                    name: "NVIDIA H200".into(),
                    compute_major: 9,
                    compute_minor: 0,
                    memory_bytes: 143_771 * 1024 * 1024,
                })
                .collect()
        };
        config.xolotl.kv_fp4 = false;
        let baseline = plan(&config, &manifest, &inventory).unwrap();
        config.xolotl.kv_fp4 = true;
        let packed = plan(&config, &manifest, &inventory).unwrap();
        let mut expected = baseline;
        expected.argv.push("--kv-fp4".into());
        assert_eq!(packed, expected);
        assert_eq!(arg_value(&packed.argv, "--ctx").unwrap(), "1048576");
        assert_eq!(packed.nvidia_indices, config.xolotl.gpus);
    }
}
