# Q6_K prompt-prefix cache capture

This capture measures repeated-prefix prefill through Power's real streaming
`POST /v1/completions` API. Five fresh caller keys each produce one cold miss
followed by one strict prompt extension and cache hit. Model loading is warmed
outside the sample window, and suffix order alternates between pairs.

## Result

| Metric, five paired requests | Cold | Warm | Change |
| --- | ---: | ---: | ---: |
| Median backend prompt evaluation | 786.1375 ms | 33.4102 ms | **23.5299x faster** |
| Median time to first token | 950.0142 ms | 72.1932 ms | **13.1593x faster** |
| Tokens evaluated across all pairs | 9,745 | 60 | **99.3843% fewer** |
| Tokens reused across all pairs | 0 | 9,740 | 1,948 per hit |
| Cache outcome | 5 misses | 5 hits | Exact for every pair |

The configured one-entry bound intentionally caused four evictions as each new
pair replaced the previous key. That is expected lifecycle evidence, not a
capacity miss within a pair.

The machine-readable [accepted report](final-5x.json) has SHA-256
`3de3ea4a7d98633f98ee7fd4c6889b24dc831d87e0f0292a4cc0252142a59fd0`.
It contains health, model, request, metric-delta, receipt-digest, output-digest,
and per-sample timing evidence, but no raw prompt, caller key, or generated
text.

## Pinned environment

| Boundary | Captured value |
| --- | --- |
| Power source | `84e1eec70784e686208971f09343b95aa607c58a`, clean before build |
| CUDA binary | SHA-256 `00b86e1243613d1c98c76e8c04c0f5a03b745e1ada387e4522f4c83496e8987b` |
| Backend | `llama.cpp`, full GPU offload, CUDA architecture 8.9 |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, driver 610.74, WDDM |
| CPU | Intel Xeon w5-2445, 10 physical / 20 logical cores |
| OS | Windows 11 build 22631 |
| Model | 22,884,408,288-byte Qwen3.8-27B Q6_K GGUF |
| Model SHA-256 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| Shared-prefix source | [`docs/prompt-prefix-cache.md`](../../../prompt-prefix-cache.md), 8,288 bytes, SHA-256 `e257a710b8c3dd21dec2c01dd31449c3155c1b6e271f3fa1ce8523771f46b886` |
| Runtime shape | context 8,192; batch 512; one parallel request; ten CPU threads |
| Cache policy | one entry per model; 900-second idle TTL |
| Sampling | greedy, seed 0, eight generated tokens |

Flash Attention was enabled. Speculation, MTP, DFlash v1, DFlash2, and DSpark
were disabled
so the experiment isolates prefix reuse. The unchanged target model performs
all suffix evaluation and generation.

## Reproduce

Apply the reviewed llama.cpp patches and build Power for the RTX 4090. Run the
build inside a Visual Studio developer environment with Ninja available:

```powershell
.\tools\apply-llamacpp-power-patches.ps1
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_CUDA_ARCHITECTURES = "89"
cargo build --locked --release --bin a3s-power `
  --target-dir target-native-sm89-ninja `
  --no-default-features --features llamacpp-cuda
```

Register the exact model digest in the normal Power manifest directory. If the
local model root differs, change only `data_dir` in the checked-in
[`prompt-cache-q6k-rtx4090.acl`](../prompt-cache-q6k-rtx4090.acl), then start an
isolated server. Omitting `--host` and `--port` deliberately verifies that the
ACL owns those values:

```powershell
.\target-native-sm89-ninja\release\a3s-power.exe serve `
  --config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt-cache-q6k-rtx4090.acl
```

From a second terminal at the same clean revision:

```powershell
$revision = git rev-parse HEAD
py -3 tools\prompt_cache_benchmark.py `
  --base-url http://127.0.0.1:11537 `
  --model qwen3.8-27b-q6-k `
  --expected-model-sha256 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  --server-revision $revision `
  --prefix-file docs\prompt-prefix-cache.md `
  --run-id pcache-rtx4090-20260822-final `
  --pairs 5 `
  --minimum-prompt-eval-speedup 10 `
  --output target-prompt-cache\final-5x.json
```

The client rejects an unexpected model hash, backend, revision, speculative
mode, privacy policy, concurrency shape, cache transition, receipt, or minimum
speedup. Its deterministic contract tests run with:

```powershell
py -3 -m unittest tools\test_prompt_cache_benchmark.py
```

## Scope

This result applies to a long shared prefix on one request lane. It establishes
prefill and TTFT reuse, not steady decode token/s, concurrent service
throughput, external-draft support, or a general quality score. Prefix caching
does not change model weights or replace target evaluation of the new suffix,
but this performance capture is not a substitute for the separate quality
matrix.
