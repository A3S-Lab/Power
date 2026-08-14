use std::env;
use std::error::Error;
use std::path::PathBuf;

use cudaforge::KernelBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    const DEPTHWISE_SOURCE: &str = "src/inference/graph/executor/depthwise/cuda/depthwise.cu";

    println!("cargo::rerun-if-changed={DEPTHWISE_SOURCE}");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    if env::var_os("CARGO_FEATURE_EMBEDDED_CUDA").is_none() {
        return Ok(());
    }

    let output = KernelBuilder::new()
        .source_files(vec![DEPTHWISE_SOURCE])
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;
    output.write(PathBuf::from(env::var("OUT_DIR")?).join("depthwise_ptx.rs"))?;
    Ok(())
}
