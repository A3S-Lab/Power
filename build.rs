use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use cudaforge::{detect_compute_cap, CudaToolkit, GpuArch};

fn main() -> Result<(), Box<dyn Error>> {
    const DEPTHWISE_SOURCE: &str = "src/inference/graph/executor/depthwise/cuda/depthwise.cu";
    const GATED_HARD_SIGMOID_SOURCE: &str =
        "src/inference/graph/executor/gated_hard_sigmoid/cuda/gated_hard_sigmoid.cu";

    println!("cargo::rerun-if-changed={DEPTHWISE_SOURCE}");
    println!("cargo::rerun-if-changed={GATED_HARD_SIGMOID_SOURCE}");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed=NVCC");
    println!("cargo::rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo::rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    if env::var_os("CARGO_FEATURE_EMBEDDED_CUDA").is_none() {
        return Ok(());
    }

    let output_directory = PathBuf::from(env::var("OUT_DIR")?);
    let toolkit = CudaToolkit::detect()?;
    let gpu_arch = detect_compute_cap()?;
    println!(
        "cargo::rustc-env=CUDA_INCLUDE_DIR={}",
        toolkit.include_dir.display()
    );

    build_kernel(
        &toolkit,
        &gpu_arch,
        DEPTHWISE_SOURCE,
        &output_directory,
        "DEPTHWISE",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        GATED_HARD_SIGMOID_SOURCE,
        &output_directory,
        "GATED_HARD_SIGMOID",
    )?;
    Ok(())
}

fn build_kernel(
    toolkit: &CudaToolkit,
    gpu_arch: &GpuArch,
    source: &str,
    output_directory: &Path,
    constant_name: &str,
) -> Result<(), Box<dyn Error>> {
    let source_path = Path::new(source);
    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| io::Error::other(format!("invalid CUDA source path: {source}")))?;
    let ptx_path = output_directory.join(format!("{stem}.ptx"));
    let binding_path = output_directory.join(format!("{stem}_ptx.rs"));

    let mut command = Command::new(&toolkit.nvcc_path);
    command
        .arg(gpu_arch.to_gencode_arg())
        .arg("--ptx")
        .args(["--default-stream", "per-thread"])
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-o")
        .arg(&ptx_path);
    if let Some(ccbin) = env::var_os("NVCC_CCBIN") {
        command
            .arg("-allow-unsupported-compiler")
            .arg("-ccbin")
            .arg(ccbin);
    }
    command.arg(source_path);

    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "failed to compile {source} for {gpu_arch}:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }

    let ptx_size = fs::metadata(&ptx_path)?.len();
    if ptx_size == 0 {
        return Err(io::Error::other(format!(
            "nvcc produced an empty PTX file for {source}: {}",
            ptx_path.display()
        ))
        .into());
    }

    fs::write(
        &binding_path,
        format!(
            "pub const {constant_name}: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{stem}.ptx\"));\n"
        ),
    )?;
    if !binding_path.is_file() {
        return Err(io::Error::other(format!(
            "failed to generate CUDA binding: {}",
            binding_path.display()
        ))
        .into());
    }

    Ok(())
}
