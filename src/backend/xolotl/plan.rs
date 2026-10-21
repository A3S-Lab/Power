//! Fail-closed 8× H200 launch plan for one `ds41_generate` process.
//!
//! `ds41_generate` rejects an unknown `--gpus` flag. With `--layer-split` set
//! and `--split-device` omitted it keeps only the first two visible devices,
//! and `--layer-split auto` is a two-way split at half the layer count
//! (`ds41_pipeline.cpp`). One H200 cannot hold the XQ expert pack, so
//! `--split-skip-if-fits` is not emitted. The 40-layer 1017 pack on eight
//! cards is the explicit split below. The engine may move a boundary that is
//! not indexer-safe; that process is not ready.

use std::path::{Path, PathBuf};

use crate::config::{PowerConfig, XolotlConfig};
use crate::error::{PowerError, Result};
use crate::model::manifest::ModelManifest;

use super::detect::is_xolotl_xq_path;

/// DeepSeek-V4.1-Flash-1017 layer count.
pub const REQUIRED_LAYER_COUNT: u32 = 40;

/// Cards required for this placement. Fewer cards are a different plan.
pub const REQUIRED_CARD_COUNT: usize = 8;

/// First layer of each later card for an equal 5-layer split of 40 layers.
pub const REQUIRED_LAYER_SPLIT: &str = "5,10,15,20,25,30,35";

/// Compute capability nvidia-smi prints for sm_90. The binary itself must be
/// built with `-DCMAKE_CUDA_ARCHITECTURES=90`.
pub const REQUIRED_COMPUTE_CAPABILITY: &str = "9.0";

/// Bytes below this are not an H200. 120 GiB rejects H100 80/94 GB and H20 96 GB.
/// An H200 reports about 140 GiB through nvidia-smi.
pub const H200_MIN_MEMORY_BYTES: u64 = 120 * 1024 * 1024 * 1024;

/// One CUDA device from `nvidia-smi --query-gpu=index,name,compute_cap,memory.total`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDevice {
    pub index: u32,
    pub name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub memory_bytes: u64,
}

/// Argv and environment for one 8-card `ds41_generate --serve` process.
///
/// Power does not start that process. A supervisor may start it only after
/// [`split_log_is_ready`] accepts the engine log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XolotlLaunchPlan {
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    /// nvidia-smi indices, in stage order. `CUDA_VISIBLE_DEVICES` uses this order.
    pub nvidia_indices: Vec<u32>,
    /// CUDA ordinals after `CUDA_VISIBLE_DEVICES` remaps `nvidia_indices`.
    pub visible_ordinals: Vec<u32>,
    pub ready_log_line: String,
}

impl XolotlLaunchPlan {
    /// One shell-ready command. Paths with spaces are double-quoted.
    pub fn command_line(&self) -> String {
        let env = self
            .env
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(" ");
        let args = self
            .argv
            .iter()
            .map(|arg| quote_arg(arg))
            .collect::<Vec<_>>()
            .join(" ");
        if env.is_empty() {
            args
        } else {
            format!("{env} {args}")
        }
    }
}

/// Build the only 8× H200 plan this backend will admit.
///
/// `devices` is the full nvidia-smi inventory. The plan keeps the eight
/// configured indices and ignores extra cards.
pub fn plan(
    config: &PowerConfig,
    manifest: &ModelManifest,
    devices: &[CudaDevice],
) -> Result<XolotlLaunchPlan> {
    let xolotl = &config.xolotl;
    if !xolotl.enabled {
        return Err(PowerError::Config(
            "xolotl backend refuses to load until xolotl.enabled = true".into(),
        ));
    }
    if config.max_concurrent_requests != 1 {
        return Err(PowerError::Config(format!(
            "xolotl 8× H200 serves one request at a time; max_concurrent_requests must be 1, got {}",
            config.max_concurrent_requests
        )));
    }
    if config.num_parallel != 1 {
        return Err(PowerError::Config(format!(
            "xolotl layer split has no batch slots; num_parallel must be 1, got {}",
            config.num_parallel
        )));
    }
    if xolotl.cuda_architecture != REQUIRED_COMPUTE_CAPABILITY {
        return Err(PowerError::Config(format!(
            "xolotl.cuda_architecture must be {REQUIRED_COMPUTE_CAPABILITY} (H200 / sm_90), got {}",
            xolotl.cuda_architecture
        )));
    }
    if xolotl.layer_count != REQUIRED_LAYER_COUNT {
        return Err(PowerError::Config(format!(
            "xolotl.layer_count must be {REQUIRED_LAYER_COUNT} for DeepSeek-V4.1-Flash-1017, got {}",
            xolotl.layer_count
        )));
    }
    if xolotl.layer_split == "auto" || xolotl.layer_split.eq_ignore_ascii_case("auto") {
        return Err(PowerError::Config(format!(
            "xolotl.layer_split=auto is a 2-way split; the 8× H200 plan requires {REQUIRED_LAYER_SPLIT}"
        )));
    }
    if xolotl.layer_split != REQUIRED_LAYER_SPLIT {
        return Err(PowerError::Config(format!(
            "xolotl.layer_split must be {REQUIRED_LAYER_SPLIT}, got {}",
            xolotl.layer_split
        )));
    }
    let nvidia_indices = require_eight_indices(&xolotl.gpus)?;
    require_h200_devices(&nvidia_indices, devices)?;

    let bin = require_ds41_binary(xolotl)?;
    let model = require_model_path(xolotl, manifest)?;
    let engram = require_engram(xolotl)?;
    let rank = require_rank(xolotl)?;

    let visible_ordinals = (0..REQUIRED_CARD_COUNT as u32).collect::<Vec<_>>();
    let split_device = visible_ordinals
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let visible_devices = nvidia_indices
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let ready_log_line = format!(
        "ds41: {}-way layer split at {} across GPUs {}",
        REQUIRED_CARD_COUNT - 1,
        REQUIRED_LAYER_SPLIT.replace(',', " "),
        split_device.replace(',', " ")
    );
    let argv = vec![
        bin.display().to_string(),
        "--serve".into(),
        "-m".into(),
        model.display().to_string(),
        "--engram".into(),
        engram,
        "--rank".into(),
        rank.display().to_string(),
        "--backend".into(),
        "cuda".into(),
        "--experts".into(),
        "gpu".into(),
        "--layer-split".into(),
        REQUIRED_LAYER_SPLIT.into(),
        "--split-device".into(),
        split_device,
    ];
    reject_forbidden_launch(&argv)?;
    Ok(XolotlLaunchPlan {
        argv,
        env: vec![("CUDA_VISIBLE_DEVICES".into(), visible_devices)],
        nvidia_indices,
        visible_ordinals,
        ready_log_line,
    })
}

/// Query the live inventory. Unit tests inject [`CudaDevice`] values and do not call this.
pub fn query_nvidia_smi() -> Result<Vec<CudaDevice>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| PowerError::Config(format!("nvidia-smi failed to start: {error}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PowerError::Config(format!(
            "nvidia-smi exited {}: {}",
            output.status,
            stderr.trim()
        )));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| PowerError::Config(format!("nvidia-smi output is not UTF-8: {error}")))?;
    parse_nvidia_smi_csv(&stdout)
}

/// Parse `nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader,nounits`.
///
/// `memory.total` in that mode is MiB.
pub fn parse_nvidia_smi_csv(text: &str) -> Result<Vec<CudaDevice>> {
    let mut devices = Vec::new();
    for (line_number, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let fields = split_smi_fields(line).map_err(|error| {
            PowerError::Config(format!("nvidia-smi line {}: {error}", line_number + 1))
        })?;
        let index: u32 = fields[0].trim().parse().map_err(|_| {
            PowerError::Config(format!(
                "nvidia-smi line {}: gpu index {:?} is not an integer",
                line_number + 1,
                fields[0]
            ))
        })?;
        let (compute_major, compute_minor) =
            parse_compute_capability(fields[2].trim()).map_err(|error| {
                PowerError::Config(format!("nvidia-smi line {}: {error}", line_number + 1))
            })?;
        let memory_mib: u64 = fields[3].trim().parse().map_err(|_| {
            PowerError::Config(format!(
                "nvidia-smi line {}: memory.total {:?} is not MiB",
                line_number + 1,
                fields[3]
            ))
        })?;
        let memory_bytes = memory_mib
            .checked_mul(1024)
            .and_then(|value| value.checked_mul(1024))
            .ok_or_else(|| {
                PowerError::Config(format!(
                    "nvidia-smi line {}: memory.total {memory_mib} MiB overflows",
                    line_number + 1
                ))
            })?;
        devices.push(CudaDevice {
            index,
            name: fields[1].trim().to_string(),
            compute_major,
            compute_minor,
            memory_bytes,
        });
    }
    if devices.is_empty() {
        return Err(PowerError::Config("nvidia-smi returned no GPUs".into()));
    }
    Ok(devices)
}

/// The engine is ready only when the log shows this plan's 7 boundaries on
/// the 8 visible ordinals, and it did not move a boundary.
pub fn split_log_is_ready(log: &str, plan: &XolotlLaunchPlan) -> Result<()> {
    if log.contains("not indexer-safe") {
        return Err(PowerError::Config(
            "ds41_generate moved a layer-split boundary (not indexer-safe); refusing the 8× H200 plan".into(),
        ));
    }
    if !log.lines().any(|line| line.trim() == plan.ready_log_line) {
        return Err(PowerError::Config(format!(
            "ds41_generate log is missing the 8-card split line: {}",
            plan.ready_log_line
        )));
    }
    Ok(())
}

/// Check the binary, joined GGUF, Engram directories, and rank file exist.
pub fn ensure_artifacts_exist(plan: &XolotlLaunchPlan) -> Result<()> {
    let bin = Path::new(&plan.argv[0]);
    if !bin.is_file() {
        return Err(PowerError::Config(format!(
            "xolotl.bin is not a file: {}",
            bin.display()
        )));
    }
    let model = arg_value(&plan.argv, "-m")?;
    if !Path::new(model).is_file() {
        return Err(PowerError::Config(format!(
            "xolotl model is not a file: {model}"
        )));
    }
    let engram = arg_value(&plan.argv, "--engram")?;
    for segment in engram_segments(engram)? {
        if !Path::new(segment).is_dir() {
            return Err(PowerError::Config(format!(
                "xolotl.engram directory does not exist: {segment}"
            )));
        }
    }
    let rank = arg_value(&plan.argv, "--rank")?;
    if !Path::new(rank).is_file() {
        return Err(PowerError::Config(format!(
            "xolotl.rank is not a file: {rank}"
        )));
    }
    Ok(())
}

fn require_eight_indices(gpus: &[u32]) -> Result<Vec<u32>> {
    if gpus.len() != REQUIRED_CARD_COUNT {
        return Err(PowerError::Config(format!(
            "xolotl.gpus must list exactly {REQUIRED_CARD_COUNT} indices, got {}",
            gpus.len()
        )));
    }
    let mut seen = Vec::with_capacity(gpus.len());
    for index in gpus {
        if seen.contains(index) {
            return Err(PowerError::Config(format!(
                "xolotl.gpus repeats index {index}"
            )));
        }
        seen.push(*index);
    }
    Ok(seen)
}

fn require_h200_devices(indices: &[u32], devices: &[CudaDevice]) -> Result<()> {
    for index in indices {
        let Some(device) = devices.iter().find(|device| device.index == *index) else {
            return Err(PowerError::Config(format!(
                "xolotl.gpus index {index} is not in the nvidia-smi inventory"
            )));
        };
        if !is_h200_product_name(&device.name) {
            return Err(PowerError::Config(format!(
                "GPU {index} ({}) is not an H200; compute capability 9.0 and capacity are not enough",
                device.name
            )));
        }
        if device.compute_major != 9 || device.compute_minor != 0 {
            return Err(PowerError::Config(format!(
                "GPU {index} ({}) is compute {}.{}, not {REQUIRED_COMPUTE_CAPABILITY}",
                device.name, device.compute_major, device.compute_minor
            )));
        }
        if device.memory_bytes < H200_MIN_MEMORY_BYTES {
            return Err(PowerError::Config(format!(
                "GPU {index} ({}) has {} bytes, below the {H200_MIN_MEMORY_BYTES}-byte H200 floor",
                device.name, device.memory_bytes
            )));
        }
    }
    Ok(())
}

/// `H200` as a whole product token. `H20`, `H100`, and `H2000` do not match.
fn is_h200_product_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    let needle = b"H200";
    let mut start = 0;
    while let Some(relative) = bytes[start..]
        .windows(needle.len())
        .position(|window| window == needle)
    {
        let at = start + relative;
        let before_ok = at == 0 || !bytes[at - 1].is_ascii_alphanumeric();
        let after = at + needle.len();
        let after_ok = after == bytes.len() || !bytes[after].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        start = at + 1;
    }
    false
}

fn require_ds41_binary(config: &XolotlConfig) -> Result<PathBuf> {
    let Some(path) = non_empty(config.bin.as_deref()) else {
        return Err(PowerError::Config(
            "xolotl.bin must point at ds41_generate".into(),
        ));
    };
    let path = PathBuf::from(path);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if name != "ds41_generate" && name != "ds41_generate.exe" {
        return Err(PowerError::Config(format!(
            "xolotl.bin must be named ds41_generate, got {name}"
        )));
    }
    Ok(path)
}

fn require_model_path(config: &XolotlConfig, manifest: &ModelManifest) -> Result<PathBuf> {
    let manifest_path = manifest.path.clone();
    let path = if let Some(configured) = non_empty(config.model.as_deref()) {
        let configured = PathBuf::from(configured);
        if configured != manifest_path {
            return Err(PowerError::Config(format!(
                "xolotl.model {} does not match the manifest path {}",
                configured.display(),
                manifest_path.display()
            )));
        }
        configured
    } else {
        manifest_path
    };
    if !is_xolotl_xq_path(&path) {
        return Err(PowerError::Config(format!(
            "xolotl model path is not an XQ GGUF: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn require_engram(config: &XolotlConfig) -> Result<String> {
    let Some(engram) = non_empty(config.engram.as_deref()) else {
        return Err(PowerError::Config(
            "xolotl.engram must list the 1017 Engram directories".into(),
        ));
    };
    let _segments = engram_segments(engram)?;
    Ok(engram.to_string())
}

fn require_rank(config: &XolotlConfig) -> Result<PathBuf> {
    let Some(path) = non_empty(config.rank.as_deref()) else {
        return Err(PowerError::Config(
            "xolotl.rank must point at rank-1017-*.bin".into(),
        ));
    };
    let path = PathBuf::from(path);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if !name.contains("rank-1017") || !name.to_ascii_lowercase().ends_with(".bin") {
        return Err(PowerError::Config(format!(
            "xolotl.rank filename must contain rank-1017 and end in .bin, got {name}"
        )));
    }
    Ok(path)
}

fn engram_segments(value: &str) -> Result<Vec<&str>> {
    let separator = if value.contains(';') { ';' } else { ':' };
    let segments = value
        .split(separator)
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.len() < 2 {
        return Err(PowerError::Config(
            "xolotl.engram needs at least two directories separated by ':' on Linux or ';' on Windows".into(),
        ));
    }
    Ok(segments)
}

fn reject_forbidden_launch(argv: &[String]) -> Result<()> {
    for arg in argv {
        if arg == "--gpus"
            || arg == "--split-skip-if-fits"
            || arg == "-ngl"
            || arg == "--ngl"
            || arg == "auto"
        {
            return Err(PowerError::Config(format!(
                "xolotl launch must not include {arg}"
            )));
        }
    }
    Ok(())
}

fn arg_value<'a>(argv: &'a [String], flag: &str) -> Result<&'a str> {
    argv.iter()
        .position(|arg| arg == flag)
        .and_then(|index| argv.get(index + 1))
        .map(String::as_str)
        .ok_or_else(|| PowerError::Config(format!("xolotl argv is missing {flag}")))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn parse_compute_capability(value: &str) -> std::result::Result<(u32, u32), String> {
    let Some((major, minor)) = value.split_once('.') else {
        return Err(format!("compute capability {value:?} must look like 9.0"));
    };
    if minor.contains('.') {
        return Err(format!("compute capability {value:?} must look like 9.0"));
    }
    let major = major
        .parse::<u32>()
        .map_err(|_| format!("compute capability {value:?} must look like 9.0"))?;
    let minor = minor
        .parse::<u32>()
        .map_err(|_| format!("compute capability {value:?} must look like 9.0"))?;
    Ok((major, minor))
}

/// `index, name, compute_cap, memory`. The name may contain commas.
fn split_smi_fields(line: &str) -> std::result::Result<Vec<String>, String> {
    let parts = line.split(',').map(|part| part.trim()).collect::<Vec<_>>();
    if parts.len() < 4 {
        return Err(format!(
            "expected index,name,compute_cap,memory.total, got {line:?}"
        ));
    }
    let index = parts[0].to_string();
    let memory = parts[parts.len() - 1].to_string();
    let compute = parts[parts.len() - 2].to_string();
    let name = parts[1..parts.len() - 2].join(", ");
    if index.is_empty() || name.is_empty() || compute.is_empty() || memory.is_empty() {
        return Err(format!("nvidia-smi row has an empty field: {line:?}"));
    }
    Ok(vec![index, name, compute, memory])
}

fn quote_arg(arg: &str) -> String {
    if arg.is_empty() || arg.contains(' ') || arg.contains('\t') {
        format!("\"{arg}\"")
    } else {
        arg.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::backend::test_utils::sample_manifest;
    use crate::config::PowerConfig;

    fn h200(index: u32) -> CudaDevice {
        CudaDevice {
            index,
            name: "NVIDIA H200".into(),
            compute_major: 9,
            compute_minor: 0,
            memory_bytes: 143_771 * 1024 * 1024,
        }
    }

    fn eight_h200() -> Vec<CudaDevice> {
        (0..8).map(h200).collect()
    }

    fn enabled_config(root: &Path) -> PowerConfig {
        let mut config = PowerConfig::default();
        config.xolotl.enabled = true;
        config.xolotl.bin = Some(root.join("ds41_generate").display().to_string());
        config.xolotl.model = Some(
            root.join("DeepSeek-V4.1-Flash-1017-XQ.gguf")
                .display()
                .to_string(),
        );
        config.xolotl.engram = Some(format!(
            "{}:{}",
            root.join("engram-a").display(),
            root.join("engram-b").display()
        ));
        config.xolotl.rank = Some(root.join("rank-1017-085.bin").display().to_string());
        config
    }

    fn manifest_for(config: &PowerConfig) -> ModelManifest {
        let mut manifest = sample_manifest("xq-1017");
        manifest.path = PathBuf::from(config.xolotl.model.clone().unwrap());
        manifest.family = Some("xolotl".into());
        manifest
    }

    #[test]
    fn eight_h200_plan_uses_split_device_not_gpus() {
        let root = Path::new("/models/xq");
        let config = enabled_config(root);
        let plan = plan(&config, &manifest_for(&config), &eight_h200()).unwrap();
        let command = plan.command_line();
        assert!(command.starts_with("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "));
        assert!(command.contains("--serve"));
        assert!(command.contains("--layer-split 5,10,15,20,25,30,35"));
        assert!(command.contains("--split-device 0,1,2,3,4,5,6,7"));
        assert!(command.contains("--backend cuda"));
        assert!(command.contains("--experts gpu"));
        assert!(command.contains("--engram"));
        assert!(command.contains("rank-1017-085.bin"));
        assert!(!command.contains("--gpus"));
        assert!(!command.contains("--split-skip-if-fits"));
        assert!(!command.contains("-ngl"));
        assert!(!command.contains("STRATA_DSPARK"));
        assert!(!command.contains("auto"));
        assert_eq!(
            plan.ready_log_line,
            "ds41: 7-way layer split at 5 10 15 20 25 30 35 across GPUs 0 1 2 3 4 5 6 7"
        );
        assert_eq!(
            plan.env,
            vec![("CUDA_VISIBLE_DEVICES".into(), "0,1,2,3,4,5,6,7".into())]
        );
    }

    #[test]
    fn non_h200_hopper_cards_are_rejected() {
        let root = Path::new("/models/xq");
        let config = enabled_config(root);
        let manifest = manifest_for(&config);
        let cases = [
            (
                "NVIDIA H100 80GB HBM3",
                9,
                0,
                81_559 * 1024 * 1024,
                "not an H200",
            ),
            ("NVIDIA H20", 9, 0, 143_771 * 1024 * 1024, "not an H200"),
            ("NVIDIA H200", 8, 9, 143_771 * 1024 * 1024, "compute 8.9"),
            ("NVIDIA H200", 9, 0, 96 * 1024 * 1024 * 1024, "H200 floor"),
            ("NVIDIA H2000", 9, 0, 143_771 * 1024 * 1024, "not an H200"),
        ];
        for (name, major, minor, memory, needle) in cases {
            let mut devices = eight_h200();
            devices[3] = CudaDevice {
                index: 3,
                name: name.into(),
                compute_major: major,
                compute_minor: minor,
                memory_bytes: memory,
            };
            let error = plan(&config, &manifest, &devices).unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains(needle),
                "name={name} expected {needle}, got {message}"
            );
        }
    }

    #[test]
    fn auto_split_short_inventory_and_disabled_fail() {
        let root = Path::new("/models/xq");
        let manifest_config = enabled_config(root);
        let manifest = manifest_for(&manifest_config);

        let mut auto = manifest_config.clone();
        auto.xolotl.layer_split = "auto".into();
        let message = plan(&auto, &manifest, &eight_h200())
            .unwrap_err()
            .to_string();
        assert!(message.contains("auto"));
        assert!(message.contains(REQUIRED_LAYER_SPLIT));

        let mut disabled = manifest_config.clone();
        disabled.xolotl.enabled = false;
        assert!(plan(&disabled, &manifest, &eight_h200())
            .unwrap_err()
            .to_string()
            .contains("xolotl.enabled"));

        let seven = (0..7).map(h200).collect::<Vec<_>>();
        let message = plan(&manifest_config, &manifest, &seven)
            .unwrap_err()
            .to_string();
        assert!(message.contains("not in the nvidia-smi inventory"));

        let mut concurrent = manifest_config.clone();
        concurrent.max_concurrent_requests = 8;
        assert!(plan(&concurrent, &manifest, &eight_h200())
            .unwrap_err()
            .to_string()
            .contains("max_concurrent_requests"));
    }

    #[test]
    fn smi_parser_converts_mib_and_keeps_commas_in_the_name() {
        let text = "\
0, NVIDIA H200, 9.0, 143771
1, NVIDIA H200 NVL, 9.0, 143771
2, NVIDIA H200, SXM, 9.0, 143771
";
        let devices = parse_nvidia_smi_csv(text).unwrap();
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0].memory_bytes, 143_771 * 1024 * 1024);
        assert_eq!(devices[0].compute_major, 9);
        assert_eq!(devices[0].compute_minor, 0);
        assert_eq!(devices[2].name, "NVIDIA H200, SXM");
        assert!(is_h200_product_name(&devices[1].name));
        assert!(is_h200_product_name(&devices[2].name));
    }

    #[test]
    fn ready_log_requires_the_unmodified_eight_card_line() {
        let root = Path::new("/models/xq");
        let config = enabled_config(root);
        let plan = plan(&config, &manifest_for(&config), &eight_h200()).unwrap();
        let ready = format!(
            "ds41: indexer-safe layer-split boundaries: 5 10 15\n{}\n",
            plan.ready_log_line
        );
        split_log_is_ready(&ready, &plan).unwrap();

        let rewritten = format!(
            "{}\nds41: layer-split 5 is not indexer-safe; using 6 instead\n",
            plan.ready_log_line
        );
        assert!(split_log_is_ready(&rewritten, &plan).is_err());

        let two_way = "ds41: 1-way layer split at 20 across GPUs 0 1\n";
        assert!(split_log_is_ready(two_way, &plan).is_err());
    }

    #[test]
    fn wrong_rank_binary_and_unsloth_model_are_refused() {
        let root = Path::new("/models/xq");
        let base = enabled_config(root);
        let manifest = manifest_for(&base);

        let mut rank = base.clone();
        rank.xolotl.rank = Some(root.join("routes.bin").display().to_string());
        assert!(plan(&rank, &manifest, &eight_h200())
            .unwrap_err()
            .to_string()
            .contains("rank-1017"));

        let mut bin = base.clone();
        bin.xolotl.bin = Some(root.join("llama-server").display().to_string());
        assert!(plan(&bin, &manifest, &eight_h200())
            .unwrap_err()
            .to_string()
            .contains("ds41_generate"));

        let mut model = base.clone();
        let ud = root.join("DeepSeek-V4.1-Flash-1017-UD-IQ1_M.gguf");
        model.xolotl.model = Some(ud.display().to_string());
        let mut ud_manifest = manifest.clone();
        ud_manifest.path = ud;
        assert!(plan(&model, &ud_manifest, &eight_h200())
            .unwrap_err()
            .to_string()
            .contains("XQ"));
    }
}
