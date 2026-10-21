//! Drive the shipped nvidia-smi parser and the shipped Xolotl planner.
//!
//! `cargo run --example xolotl_local_plan --no-default-features --features server -- <smi.csv> <bin> <model.gguf> <engram> <rank.bin>`

use std::path::PathBuf;
use std::process::ExitCode;

use a3s_power::backend::test_utils::sample_manifest;
use a3s_power::backend::xolotl::{parse_nvidia_smi_csv, plan_from_nvidia_smi_csv};
use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::ModelFormat;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(smi_path) = args.next() else {
        eprintln!("usage: xolotl_local_plan <smi.csv> <bin> <model.gguf> <engram> <rank.bin>");
        return ExitCode::from(2);
    };
    let Some(bin) = args.next() else {
        eprintln!("missing ds41_generate path");
        return ExitCode::from(2);
    };
    let Some(model) = args.next() else {
        eprintln!("missing XQ GGUF path");
        return ExitCode::from(2);
    };
    let Some(engram) = args.next() else {
        eprintln!("missing engram directory list");
        return ExitCode::from(2);
    };
    let Some(rank) = args.next() else {
        eprintln!("missing rank-1017 file");
        return ExitCode::from(2);
    };

    let text = match std::fs::read_to_string(&smi_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("could not read {smi_path}: {error}");
            return ExitCode::from(2);
        }
    };
    let devices = match parse_nvidia_smi_csv(&text) {
        Ok(devices) => devices,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };
    if devices.len() != 1 {
        eprintln!("expected one GPU from nvidia-smi, parsed {}", devices.len());
        return ExitCode::from(2);
    }

    let mut config = PowerConfig::default();
    config.xolotl.enabled = true;
    config.xolotl.bin = Some(bin);
    config.xolotl.model = Some(model.clone());
    config.xolotl.engram = Some(engram);
    config.xolotl.rank = Some(rank);
    config.xolotl.gpus = vec![devices[0].index];
    config.xolotl.cuda_architecture = "8.9".into();
    config.xolotl.layer_split = "none".into();

    let mut manifest = sample_manifest("xq-1017");
    manifest.format = ModelFormat::Gguf;
    manifest.path = PathBuf::from(&model);

    let plan = match plan_from_nvidia_smi_csv(&config, &manifest, &text) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };

    let device = &devices[0];
    println!("NAME={}", device.name);
    println!("INDEX={}", device.index);
    println!("COMPUTE={}.{}", device.compute_major, device.compute_minor);
    println!("MEMORY_MIB={}", device.memory_bytes / (1024 * 1024));
    println!("COMMAND={}", plan.command_line());
    ExitCode::SUCCESS
}
