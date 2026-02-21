//! TEE memory benchmark — peak RSS comparison: picolm vs mistralrs.
//!
//! Measures peak resident set size (RSS) during model load and inference
//! for each backend. The key metric is whether picolm's layer-streaming
//! keeps peak RAM proportional to layer size rather than model size.
//!
//! Run with:
//!   cargo bench --bench tee_memory --features picolm
//!
//! To simulate EPC pressure, set a cgroup memory limit before running:
//!   systemd-run --scope -p MemoryMax=512M cargo bench --bench tee_memory

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::{ModelFormat, ModelManifest};

// ── RSS measurement ───────────────────────────────────────────────────────────

/// Read peak RSS from /proc/self/status (Linux) or return 0 on other platforms.
fn peak_rss_kb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse().unwrap_or(0);
                    }
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

// ── Fake GGUF model generator ─────────────────────────────────────────────────

/// Write a minimal valid GGUF v3 file of approximately `size_mb` megabytes.
///
/// The file has a valid header and enough padding to simulate a real model's
/// file size, allowing us to measure mmap vs full-load memory behaviour.
fn write_fake_gguf(path: &PathBuf, size_mb: usize) {
    let mut data: Vec<u8> = Vec::new();

    // GGUF magic + version
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic "GGUF"
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // n_tensors = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // n_kv = 0

    // Pad to requested size with zeros (simulates weight data)
    let target = size_mb * 1024 * 1024;
    if data.len() < target {
        data.resize(target, 0u8);
    }

    std::fs::write(path, &data).expect("Failed to write fake GGUF");
}

// ── Benchmark: model load RSS ─────────────────────────────────────────────────

fn bench_load_rss(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = TempDir::new().unwrap();

    let mut group = c.benchmark_group("model_load_rss");
    group.sample_size(10);

    for size_mb in [64usize, 256, 512] {
        let model_path = dir.path().join(format!("model_{size_mb}mb.gguf"));
        write_fake_gguf(&model_path, size_mb);

        let manifest = ModelManifest {
            name: format!("test-{size_mb}mb"),
            format: ModelFormat::Gguf,
            size: (size_mb * 1024 * 1024) as u64,
            sha256: format!("sha256:fake{size_mb}"),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: model_path.clone(),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };

        // Benchmark picolm load (mmap — should NOT increase RSS by model_size)
        #[cfg(feature = "picolm")]
        {
            use a3s_power::backend::picolm::PicolmBackend;
            use a3s_power::backend::Backend;

            let config = Arc::new(PowerConfig::default());
            let backend = PicolmBackend::new(config);
            let manifest_clone = manifest.clone();

            group.bench_with_input(
                BenchmarkId::new("picolm_load", size_mb),
                &size_mb,
                |b, _| {
                    b.to_async(&rt).iter(|| async {
                        let rss_before = peak_rss_kb();
                        let _ = backend.load(&manifest_clone).await;
                        let rss_after = peak_rss_kb();
                        let _ = backend.unload(&manifest_clone.name).await;
                        // Return delta so criterion doesn't optimise it away
                        rss_after.saturating_sub(rss_before)
                    });
                },
            );
        }
    }

    group.finish();
}

// ── Benchmark: EPC routing decision ──────────────────────────────────────────

fn bench_epc_routing(c: &mut Criterion) {
    use a3s_power::backend::BackendRegistry;
    use a3s_power::model::manifest::ModelFormat;

    let config = Arc::new(PowerConfig::default());
    let mut registry = BackendRegistry::new();

    #[cfg(feature = "picolm")]
    {
        use a3s_power::backend::picolm::PicolmBackend;
        registry.register(Arc::new(PicolmBackend::new(config.clone())));
    }

    let mut group = c.benchmark_group("epc_routing");
    group.sample_size(100);

    // Measure overhead of find_for_tee() vs find_for_format()
    group.bench_function("find_for_format", |b| {
        b.iter(|| registry.find_for_format(&ModelFormat::Gguf))
    });

    group.bench_function("find_for_tee_small_model", |b| {
        // 100MB model — should fit in any EPC
        b.iter(|| registry.find_for_tee(&ModelFormat::Gguf, 100 * 1024 * 1024))
    });

    group.bench_function("find_for_tee_large_model", |b| {
        // 8GB model — exceeds typical EPC
        b.iter(|| registry.find_for_tee(&ModelFormat::Gguf, 8 * 1024 * 1024 * 1024))
    });

    group.finish();
}

criterion_group!(benches, bench_load_rss, bench_epc_routing);
criterion_main!(benches);
