use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use cudaforge::{detect_compute_cap, CudaToolkit, GpuArch};

fn main() -> Result<(), Box<dyn Error>> {
    const CPU_GRAPH_SEGMENT_SOURCE: &str =
        "src/inference/graph/executor/cpu_graph_segment/native.cpp";
    const CPU_GRAPH_SEGMENT_ARENA: &str =
        "src/inference/graph/executor/cpu_graph_segment/activation_arena.hpp";
    const BIASED_ACTIVATION_SOURCE: &str =
        "src/inference/graph/executor/biased_activation/cuda/biased_activation.cu";
    const BIASED_SWISH_SOURCE: &str =
        "src/inference/graph/executor/biased_swish/cuda/biased_swish.cu";
    const BATCH_NORM_SOURCE: &str = "src/inference/graph/executor/batch_norm/cuda/batch_norm.cu";
    const CONTIGUOUS_TRANSPOSE_SOURCE: &str =
        "src/inference/graph/executor/contiguous_transpose/cuda/contiguous_transpose.cu";
    const CONTIGUOUS_MEAN_SOURCE: &str =
        "src/inference/graph/executor/contiguous_mean/cuda/contiguous_mean.cu";
    const CUDA_FAST_DIVIDE_HEADER: &str = "src/inference/graph/executor/cuda_fast_divide.cuh";
    const DEPTHWISE_SOURCE: &str = "src/inference/graph/executor/depthwise/cuda/depthwise.cu";
    const DEPTHWISE_BATCH_NORM_SOURCE: &str =
        "src/inference/graph/executor/depthwise/cuda/depthwise_batch_norm.cu";
    const DEPTHWISE_CONTIGUOUS_HEADER: &str =
        "src/inference/graph/executor/depthwise/cuda/contiguous.cuh";
    const GATED_HARD_SIGMOID_SOURCE: &str =
        "src/inference/graph/executor/gated_hard_sigmoid/cuda/gated_hard_sigmoid.cu";
    const GELU_ERF_SOURCE: &str = "src/inference/graph/executor/gelu_erf/cuda/gelu_erf.cu";
    const LAYER_NORM_AFFINE_SOURCE: &str =
        "src/inference/graph/executor/layer_norm_affine/cuda/layer_norm_affine.cu";
    const MATMUL_BIAS_SOURCE: &str = "src/inference/graph/executor/matmul_bias/cuda/matmul_bias.cu";
    const SPATIAL_IM2COL_SOURCE: &str =
        "src/inference/graph/executor/spatial_convolution/cuda/im2col.cu";
    const ROW_TOP1_SOURCE: &str = "src/inference/graph/row_top1/cuda/row_top1.cu";
    const ROW_SOFTMAX_TOP1_SOURCE: &str =
        "src/inference/graph/row_softmax_top1/cuda/row_softmax_top1.cu";
    const SIGMOID_PRODUCT_SOURCE: &str =
        "src/inference/graph/executor/sigmoid_product/cuda/sigmoid_product.cu";

    println!("cargo::rerun-if-changed={BIASED_ACTIVATION_SOURCE}");
    println!("cargo::rerun-if-changed={BIASED_SWISH_SOURCE}");
    println!("cargo::rerun-if-changed={BATCH_NORM_SOURCE}");
    println!("cargo::rerun-if-changed={CONTIGUOUS_TRANSPOSE_SOURCE}");
    println!("cargo::rerun-if-changed={CONTIGUOUS_MEAN_SOURCE}");
    println!("cargo::rerun-if-changed={CUDA_FAST_DIVIDE_HEADER}");
    println!("cargo::rerun-if-changed={DEPTHWISE_SOURCE}");
    println!("cargo::rerun-if-changed={DEPTHWISE_BATCH_NORM_SOURCE}");
    println!("cargo::rerun-if-changed={DEPTHWISE_CONTIGUOUS_HEADER}");
    println!("cargo::rerun-if-changed={GATED_HARD_SIGMOID_SOURCE}");
    println!("cargo::rerun-if-changed={GELU_ERF_SOURCE}");
    println!("cargo::rerun-if-changed={LAYER_NORM_AFFINE_SOURCE}");
    println!("cargo::rerun-if-changed={MATMUL_BIAS_SOURCE}");
    println!("cargo::rerun-if-changed={SPATIAL_IM2COL_SOURCE}");
    println!("cargo::rerun-if-changed={ROW_TOP1_SOURCE}");
    println!("cargo::rerun-if-changed={ROW_SOFTMAX_TOP1_SOURCE}");
    println!("cargo::rerun-if-changed={SIGMOID_PRODUCT_SOURCE}");
    println!("cargo::rerun-if-changed={CPU_GRAPH_SEGMENT_SOURCE}");
    println!("cargo::rerun-if-changed={CPU_GRAPH_SEGMENT_ARENA}");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed=NVCC");
    println!("cargo::rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo::rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    if env::var_os("CARGO_FEATURE_EMBEDDED_CPU_OPTIMIZED").is_some() {
        build_cpu_graph_segment(CPU_GRAPH_SEGMENT_SOURCE)?;
    }
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
        BIASED_ACTIVATION_SOURCE,
        &output_directory,
        "BIASED_ACTIVATION",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        BIASED_SWISH_SOURCE,
        &output_directory,
        "BIASED_SWISH",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        BATCH_NORM_SOURCE,
        &output_directory,
        "BATCH_NORM",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        CONTIGUOUS_MEAN_SOURCE,
        &output_directory,
        "CONTIGUOUS_MEAN",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        CONTIGUOUS_TRANSPOSE_SOURCE,
        &output_directory,
        "CONTIGUOUS_TRANSPOSE",
    )?;
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
        DEPTHWISE_BATCH_NORM_SOURCE,
        &output_directory,
        "DEPTHWISE_BATCH_NORM",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        GATED_HARD_SIGMOID_SOURCE,
        &output_directory,
        "GATED_HARD_SIGMOID",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        GELU_ERF_SOURCE,
        &output_directory,
        "GELU_ERF",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        LAYER_NORM_AFFINE_SOURCE,
        &output_directory,
        "LAYER_NORM_AFFINE",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        MATMUL_BIAS_SOURCE,
        &output_directory,
        "MATMUL_BIAS",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        SPATIAL_IM2COL_SOURCE,
        &output_directory,
        "SPATIAL_IM2COL",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        ROW_TOP1_SOURCE,
        &output_directory,
        "ROW_TOP1",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        ROW_SOFTMAX_TOP1_SOURCE,
        &output_directory,
        "ROW_SOFTMAX_TOP1",
    )?;
    build_kernel(
        &toolkit,
        &gpu_arch,
        SIGMOID_PRODUCT_SOURCE,
        &output_directory,
        "SIGMOID_PRODUCT",
    )?;
    Ok(())
}

fn build_cpu_graph_segment(source: &str) -> Result<(), Box<dyn Error>> {
    let include = env::var_os("DEP_DNNL_INCLUDE_PATH")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("oneDNN did not publish its include directory"))?;
    let library = env::var_os("DEP_DNNL_LIBRARY_PATH")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("oneDNN did not publish its library directory"))?;
    let mut build = cc::Build::new();
    build.cpp(true).file(source).include(include);
    if cfg!(target_env = "msvc") {
        build.flag("/std:c++17");
    } else {
        build.flag("-std=c++17");
    }
    build.compile("a3s_power_cpu_graph_segment");
    println!("cargo::rustc-link-search=native={}", library.display());
    println!("cargo::rustc-link-lib=static=dnnl");
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
