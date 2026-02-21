//! Layer-streaming throughput benchmark.
//!
//! Measures tokens/second for the picolm backend at various simulated
//! model sizes, verifying that throughput degrades gracefully as model
//! size increases (layer-streaming amortises the cost across layers).
//!
//! Run with:
//!   cargo bench --bench layer_stream --features picolm

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::{ModelFormat, ModelManifest};

/// Write a minimal valid GGUF v3 file of approximately `size_mb` megabytes.
fn write_fake_gguf(path: &PathBuf, size_mb: usize) {
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let target = size_mb * 1024 * 1024;
    if data.len() < target {
        data.resize(target, 0u8);
    }
    std::fs::write(path, &data).expect("Failed to write fake GGUF");
}

fn bench_layer_stream_throughput(c: &mut Criterion) {
    #[cfg(not(feature = "picolm"))]
    {
        let _ = c;
        eprintln!("layer_stream bench requires --features picolm");
        return;
    }

    #[cfg(feature = "picolm")]
    {
        use a3s_power::backend::picolm::PicolmBackend;
        use a3s_power::backend::types::{ChatMessage, ChatRequest, MessageContent};
        use a3s_power::backend::Backend;
        use futures::StreamExt;

        let rt = tokio::runtime::Runtime::new().unwrap();
        let dir = TempDir::new().unwrap();
        let config = Arc::new(PowerConfig::default());

        let mut group = c.benchmark_group("layer_stream_throughput");
        group.sample_size(10);

        for size_mb in [64usize, 128, 256] {
            let model_path = dir.path().join(format!("model_{size_mb}mb.gguf"));
            write_fake_gguf(&model_path, size_mb);

            let manifest = ModelManifest {
                name: format!("bench-{size_mb}mb"),
                format: ModelFormat::Gguf,
                size: (size_mb * 1024 * 1024) as u64,
                sha256: format!("sha256:bench{size_mb}"),
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

            let backend = Arc::new(PicolmBackend::new(config.clone()));
            let backend_clone = Arc::clone(&backend);
            let manifest_clone = manifest.clone();

            // Pre-load the model outside the benchmark loop
            rt.block_on(async {
                backend_clone.load(&manifest_clone).await.ok();
            });

            let model_name = manifest.name.clone();

            group.bench_with_input(
                BenchmarkId::new("tokens_per_sec", size_mb),
                &size_mb,
                |b, _| {
                    let backend_ref = Arc::clone(&backend);
                    let name = model_name.clone();
                    b.to_async(&rt).iter(|| async {
                        let req = ChatRequest {
                            messages: vec![ChatMessage {
                                role: "user".to_string(),
                                content: MessageContent::Text("Hello".to_string()),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                                images: None,
                            }],
                            session_id: None,
                            temperature: Some(0.0), // greedy for determinism
                            top_p: None,
                            max_tokens: Some(16), // small for bench speed
                            stop: None,
                            stream: true,
                            top_k: None,
                            min_p: None,
                            repeat_penalty: None,
                            frequency_penalty: None,
                            presence_penalty: None,
                            seed: Some(42),
                            num_ctx: None,
                            mirostat: None,
                            mirostat_tau: None,
                            mirostat_eta: None,
                            tfs_z: None,
                            typical_p: None,
                            response_format: None,
                            tools: None,
                            tool_choice: None,
                            repeat_last_n: None,
                            penalize_newline: None,
                            num_batch: None,
                            num_thread: None,
                            num_thread_batch: None,
                            flash_attention: None,
                            num_gpu: None,
                            main_gpu: None,
                            use_mmap: None,
                            use_mlock: None,
                            num_parallel: None,
                            images: None,
                        };

                        let mut stream = backend_ref.chat(&name, req).await.unwrap();
                        let mut token_count = 0u32;
                        while let Some(chunk) = stream.next().await {
                            if let Ok(c) = chunk {
                                if !c.content.is_empty() {
                                    token_count += 1;
                                }
                                if c.done {
                                    break;
                                }
                            }
                        }
                        token_count
                    });
                },
            );

            // Clean up
            rt.block_on(async {
                backend.unload(&model_name).await.ok();
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_layer_stream_throughput);
criterion_main!(benches);
