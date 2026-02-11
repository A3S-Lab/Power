// GPU detection and auto-configuration.
//
// Detects available GPU hardware and automatically configures the number of
// layers to offload. Supports Metal (macOS), CUDA (Linux/Windows), and
// falls back to CPU-only when no GPU is detected.

use tracing;

/// Detected GPU information.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU device name (e.g. "Apple M2 Pro", "NVIDIA RTX 4090").
    pub name: String,
    /// Available VRAM in bytes (0 if unknown).
    pub vram_bytes: u64,
    /// GPU backend type.
    pub backend: GpuBackend,
}

/// GPU compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Apple Metal (macOS Apple Silicon).
    Metal,
    /// NVIDIA CUDA.
    Cuda,
    /// No GPU detected — CPU only.
    Cpu,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Cpu => write!(f, "CPU"),
        }
    }
}

impl GpuInfo {
    /// Format VRAM as a human-readable string.
    pub fn vram_display(&self) -> String {
        if self.vram_bytes == 0 {
            "unknown".to_string()
        } else {
            format_bytes(self.vram_bytes)
        }
    }
}

/// Detect available GPU hardware.
///
/// Returns information about the best available GPU. Detection order:
/// 1. Apple Metal (macOS) — uses `system_profiler` to read GPU info
/// 2. NVIDIA CUDA (Linux/Windows) — uses `nvidia-smi` to read GPU info
/// 3. Falls back to CPU if no GPU is detected
pub fn detect() -> GpuInfo {
    // Try Metal first (macOS)
    if cfg!(target_os = "macos") {
        if let Some(info) = detect_metal() {
            return info;
        }
    }

    // Try CUDA (Linux/Windows)
    if let Some(info) = detect_cuda() {
        return info;
    }

    // Fallback: CPU only
    GpuInfo {
        name: "CPU".to_string(),
        vram_bytes: 0,
        backend: GpuBackend::Cpu,
    }
}

/// Recommend the number of GPU layers to offload based on detected hardware.
///
/// Returns:
/// - `-1` (all layers) if a GPU is detected
/// - `0` (CPU only) if no GPU is detected
///
/// This is only used when the user hasn't explicitly set `gpu_layers` in config.
pub fn recommended_gpu_layers(gpu: &GpuInfo) -> i32 {
    match gpu.backend {
        GpuBackend::Metal | GpuBackend::Cuda => -1, // Offload all layers
        GpuBackend::Cpu => 0,
    }
}

/// Auto-configure GPU layers if not explicitly set by the user.
///
/// When `gpu_layers` is 0 (the default), detects GPU hardware and sets
/// `gpu_layers = -1` (all layers) if a GPU is available.
/// Logs the detection result.
pub fn auto_configure(config: &mut crate::config::GpuConfig) {
    let gpu = detect();

    match gpu.backend {
        GpuBackend::Metal => {
            tracing::info!(
                gpu = %gpu.name,
                vram = %gpu.vram_display(),
                backend = "Metal",
                "GPU detected"
            );
        }
        GpuBackend::Cuda => {
            tracing::info!(
                gpu = %gpu.name,
                vram = %gpu.vram_display(),
                backend = "CUDA",
                "GPU detected"
            );
        }
        GpuBackend::Cpu => {
            tracing::info!("No GPU detected, using CPU only");
        }
    }

    // Only auto-configure if user hasn't explicitly set gpu_layers
    if config.gpu_layers == 0 {
        let recommended = recommended_gpu_layers(&gpu);
        if recommended != 0 {
            tracing::info!(
                gpu_layers = recommended,
                "Auto-configuring GPU layers (set gpu_layers in config or OLLAMA_NUM_GPU to override)"
            );
            config.gpu_layers = recommended;
        }
    } else {
        tracing::info!(gpu_layers = config.gpu_layers, "Using configured GPU layers");
    }
}

/// Detect Apple Metal GPU on macOS.
///
/// Uses `system_profiler SPDisplaysDataType` to read GPU name and memory.
fn detect_metal() -> Option<GpuInfo> {
    let output = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;

    // Navigate: SPDisplaysDataType[0].sppci_model
    let displays = json.get("SPDisplaysDataType")?.as_array()?;
    let gpu = displays.first()?;

    let name = gpu
        .get("sppci_model")
        .and_then(|v| v.as_str())
        .unwrap_or("Apple GPU")
        .to_string();

    // Try to get VRAM from spdisplays_vram or _spdisplays_vram
    let vram_str = gpu
        .get("spdisplays_vram")
        .or_else(|| gpu.get("_spdisplays_vram"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let vram_bytes = parse_vram_string(vram_str);

    // For Apple Silicon unified memory, try to get total system memory
    // since VRAM is shared with system RAM
    let vram_bytes = if vram_bytes == 0 {
        get_system_memory_bytes().unwrap_or(0)
    } else {
        vram_bytes
    };

    Some(GpuInfo {
        name,
        vram_bytes,
        backend: GpuBackend::Metal,
    })
}

/// Detect NVIDIA CUDA GPU.
///
/// Uses `nvidia-smi` to read GPU name and memory.
fn detect_cuda() -> Option<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() < 2 {
        return None;
    }

    let name = parts[0].to_string();
    // nvidia-smi reports memory in MiB
    let vram_mib: u64 = parts[1].parse().ok()?;
    let vram_bytes = vram_mib * 1_048_576;

    Some(GpuInfo {
        name,
        vram_bytes,
        backend: GpuBackend::Cuda,
    })
}

/// Parse a VRAM string like "16 GB" or "8192 MB" into bytes.
fn parse_vram_string(s: &str) -> u64 {
    let s = s.trim().to_uppercase();
    if let Some(num_str) = s.strip_suffix("GB") {
        if let Ok(n) = num_str.trim().parse::<f64>() {
            return (n * 1_073_741_824.0) as u64;
        }
    }
    if let Some(num_str) = s.strip_suffix("MB") {
        if let Ok(n) = num_str.trim().parse::<f64>() {
            return (n * 1_048_576.0) as u64;
        }
    }
    // Try plain number (assume bytes)
    s.parse::<u64>().unwrap_or(0)
}

/// Get total system memory in bytes (for Apple Silicon unified memory).
fn get_system_memory_bytes() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<u64>().ok()
}

/// Format bytes as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;

    if bytes >= GB {
        format!("{:.1} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MiB", bytes as f64 / MB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(GpuBackend::Metal.to_string(), "Metal");
        assert_eq!(GpuBackend::Cuda.to_string(), "CUDA");
        assert_eq!(GpuBackend::Cpu.to_string(), "CPU");
    }

    #[test]
    fn test_recommended_gpu_layers_metal() {
        let gpu = GpuInfo {
            name: "Apple M2 Pro".to_string(),
            vram_bytes: 16_000_000_000,
            backend: GpuBackend::Metal,
        };
        assert_eq!(recommended_gpu_layers(&gpu), -1);
    }

    #[test]
    fn test_recommended_gpu_layers_cuda() {
        let gpu = GpuInfo {
            name: "NVIDIA RTX 4090".to_string(),
            vram_bytes: 24_000_000_000,
            backend: GpuBackend::Cuda,
        };
        assert_eq!(recommended_gpu_layers(&gpu), -1);
    }

    #[test]
    fn test_recommended_gpu_layers_cpu() {
        let gpu = GpuInfo {
            name: "CPU".to_string(),
            vram_bytes: 0,
            backend: GpuBackend::Cpu,
        };
        assert_eq!(recommended_gpu_layers(&gpu), 0);
    }

    #[test]
    fn test_parse_vram_string_gb() {
        assert_eq!(parse_vram_string("16 GB"), 16 * 1_073_741_824);
        assert_eq!(parse_vram_string("8GB"), 8 * 1_073_741_824);
    }

    #[test]
    fn test_parse_vram_string_mb() {
        assert_eq!(parse_vram_string("8192 MB"), 8192 * 1_048_576);
    }

    #[test]
    fn test_parse_vram_string_empty() {
        assert_eq!(parse_vram_string(""), 0);
        assert_eq!(parse_vram_string("unknown"), 0);
    }

    #[test]
    fn test_gpu_info_vram_display() {
        let gpu = GpuInfo {
            name: "Test".to_string(),
            vram_bytes: 16 * 1_073_741_824,
            backend: GpuBackend::Metal,
        };
        assert_eq!(gpu.vram_display(), "16.0 GiB");
    }

    #[test]
    fn test_gpu_info_vram_display_unknown() {
        let gpu = GpuInfo {
            name: "CPU".to_string(),
            vram_bytes: 0,
            backend: GpuBackend::Cpu,
        };
        assert_eq!(gpu.vram_display(), "unknown");
    }

    #[test]
    fn test_detect_returns_valid_info() {
        // detect() should always return something (at minimum CPU fallback)
        let gpu = detect();
        assert!(!gpu.name.is_empty());
    }

    #[test]
    fn test_auto_configure_respects_explicit_setting() {
        let mut config = crate::config::GpuConfig {
            gpu_layers: 10,
            main_gpu: 0,
            tensor_split: vec![],
        };
        auto_configure(&mut config);
        // Should NOT change explicitly set value
        assert_eq!(config.gpu_layers, 10);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
        assert_eq!(format_bytes(1_048_576), "1 MiB");
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(1), "1 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_boundary_mb() {
        // Just below 1 GiB
        assert_eq!(format_bytes(1_073_741_823), "1024 MiB");
    }

    #[test]
    fn test_format_bytes_large_gib() {
        assert_eq!(format_bytes(32 * 1_073_741_824), "32.0 GiB");
    }

    #[test]
    fn test_parse_vram_string_lowercase() {
        // parse_vram_string uppercases internally
        assert_eq!(parse_vram_string("16 gb"), 16 * 1_073_741_824);
        assert_eq!(parse_vram_string("8192 mb"), 8192 * 1_048_576);
    }

    #[test]
    fn test_parse_vram_string_plain_number() {
        assert_eq!(parse_vram_string("1024"), 1024);
    }

    #[test]
    fn test_parse_vram_string_fractional_gb() {
        let result = parse_vram_string("1.5 GB");
        assert_eq!(result, (1.5 * 1_073_741_824.0) as u64);
    }

    #[test]
    fn test_gpu_info_clone() {
        let gpu = GpuInfo {
            name: "Test GPU".to_string(),
            vram_bytes: 8_000_000_000,
            backend: GpuBackend::Metal,
        };
        let cloned = gpu.clone();
        assert_eq!(cloned.name, "Test GPU");
        assert_eq!(cloned.vram_bytes, 8_000_000_000);
        assert_eq!(cloned.backend, GpuBackend::Metal);
    }

    #[test]
    fn test_gpu_backend_eq() {
        assert_eq!(GpuBackend::Metal, GpuBackend::Metal);
        assert_ne!(GpuBackend::Metal, GpuBackend::Cuda);
        assert_ne!(GpuBackend::Cuda, GpuBackend::Cpu);
    }

    #[test]
    fn test_gpu_info_vram_display_mib() {
        let gpu = GpuInfo {
            name: "Test".to_string(),
            vram_bytes: 512 * 1_048_576, // 512 MiB
            backend: GpuBackend::Cuda,
        };
        assert_eq!(gpu.vram_display(), "512 MiB");
    }

    #[test]
    fn test_auto_configure_with_zero_gpu_layers() {
        let mut config = crate::config::GpuConfig {
            gpu_layers: 0,
            main_gpu: 0,
            tensor_split: vec![],
        };
        auto_configure(&mut config);
        // On macOS with Metal, should auto-set to -1
        // On CI without GPU, stays at 0
        // Either way, the function should not panic
        assert!(config.gpu_layers == -1 || config.gpu_layers == 0);
    }

    #[test]
    fn test_gpu_info_debug() {
        let gpu = GpuInfo {
            name: "Test".to_string(),
            vram_bytes: 0,
            backend: GpuBackend::Cpu,
        };
        let debug = format!("{:?}", gpu);
        assert!(debug.contains("Test"));
        assert!(debug.contains("Cpu"));
    }

    #[test]
    fn test_gpu_backend_copy() {
        let backend = GpuBackend::Metal;
        let copied = backend;
        assert_eq!(backend, copied);
    }
}
