use std::env;
use std::error::Error;
use std::path::Path;
use std::path::PathBuf;

use cudaforge::KernelBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    const DEPTHWISE_SOURCE: &str = "src/inference/graph/executor/depthwise/cuda/depthwise.cu";
    const GATED_HARD_SIGMOID_SOURCE: &str =
        "src/inference/graph/executor/gated_hard_sigmoid/cuda/gated_hard_sigmoid.cu";

    println!("cargo::rerun-if-changed={DEPTHWISE_SOURCE}");
    println!("cargo::rerun-if-changed={GATED_HARD_SIGMOID_SOURCE}");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    if env::var_os("CARGO_FEATURE_EMBEDDED_CUDA").is_none() {
        return Ok(());
    }

    let output_directory = PathBuf::from(env::var("OUT_DIR")?);
    build_kernel(DEPTHWISE_SOURCE, &output_directory.join("depthwise_ptx.rs"))?;
    build_kernel(
        GATED_HARD_SIGMOID_SOURCE,
        &output_directory.join("gated_hard_sigmoid_ptx.rs"),
    )?;
    Ok(())
}

fn build_kernel(source: &str, binding: &Path) -> Result<(), Box<dyn Error>> {
    KernelBuilder::new()
        .source_files(vec![source])
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?
        .write(binding)?;
    Ok(())
}
