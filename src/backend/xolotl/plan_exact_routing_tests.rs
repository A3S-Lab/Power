#![cfg(test)]
use super::*;
use crate::backend::test_utils::sample_manifest;
#[test]
fn every_owned_placement_keeps_exact_routing_with_or_without_kv_packing() {
    let mut config = PowerConfig::default();
    config.xolotl.enabled = true;
    config.xolotl.bin = Some("/runtime/ds41_generate".into());
    config.xolotl.model = Some("/models/DeepSeek-V4.1-Flash-1017-XQ.gguf".into());
    config.xolotl.engram = Some("/models/engram".into());
    config.xolotl.rank = Some("/models/rank-1017.bin".into());
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
        for packed in [false, true] {
            config.xolotl.kv_fp4 = packed;
            let launch = plan(&config, &manifest, &inventory).unwrap();
            for flag in ["--route-bias", "--route-bias-file", "--route-bias-remote"] {
                assert_eq!(
                    launch
                        .argv
                        .iter()
                        .filter(|value| value.as_str() == flag)
                        .count(),
                    1
                );
                assert_eq!(arg_value(&launch.argv, flag).unwrap(), "0");
            }
            assert_eq!(
                launch.argv.last().is_some_and(|value| value == "--kv-fp4"),
                packed
            );
        }
    }
}
