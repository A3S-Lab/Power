---
title: Performance evidence
description: A3S Power's model-neutral performance-evidence method and a reproducible Qwen3.8-27B Q6_K case study measured through the real API.
---

# Performance evidence

Power accepts performance work only when the measured system still preserves
its execution contract. The published Qwen3.8 captures therefore use the real
streaming API, pin artifact and binary identities, verify deterministic output,
and report quality and workload throughput beside steady decode.

This page documents one backend/model/hardware integration, not the scope of
the engine. Power's shared execution, admission, memory, cancellation, and
evidence contracts are model-neutral and also serve vision, OCR, embedding,
audio, multimodal, scientific, and caller-owned reviewed graphs.

To verify the numbers directly, open [Reproduction](./reproduction): run the
model-free one-command verifier first, then replay the full RTX 4090 protocol
against the same input identities.

## Quality and speed by mode

| Qwen3.8-27B artifact and mode | Fixed-task quality proxy | Request-wide throughput | Median steady decode |
| --- | --- | ---: | ---: |
| Untouched Q6_K, autoregressive | 67/100 lenient; 60/100 strict (100 tasks, 3x) | 30.883 token/s | 35.5793 token/s (earlier capture) |
| **Untouched Q6_K + prefix-FR8192, fixed K6/S6/B8** | 9/12 lenient and strict in both paired modes (1x; 3 truncated) | **46.923 token/s** | - |
| **Untouched Q6_K + prefix-FR8192, fixed K7/S6/B11, high-priority CUDA** | Fixed peak prompt retained the same output digest | - | **172.835 token/s** on a shared WDDM desktop |
| Untouched Q6_K, full-vocabulary MTP, K7/S7 | Exact greedy parity on the fixed peak prompt | - | 147.0207 token/s |
| Untouched Q6_K, full-vocabulary MTP, K7/S6 | 5/12 lenient; 3/12 strict (1x; 11 truncated) | **47.032 token/s** | - |
| **Untouched Q6_K + prefix-FR8192 MTP, K7/S6** | 4/12 lenient; 3/12 strict (1x; 11 truncated) | 37.290 token/s | **176.6109 token/s** |
| TBQ4 mixed, autoregressive | 70/100 lenient; 64/100 strict (100 tasks, 3x) | 38.724 token/s | - |
| **TBQ4 mixed + full-vocabulary fixed MTP, K7/S7** | **76/100 lenient; 66/100 strict** (100 tasks, 3x) | **83.228 token/s** | **175.2089 token/s** |
| TBQ4 mixed + full-vocabulary guarded MTP, K7/S6 | 5/12 lenient; 3/12 strict (12 tasks, 3x) | 54.060 token/s | **177.7165 token/s** |
| TBQ4 mixed + MTP + prefix FR (historical) | 72/100 lenient; 60/100 strict (100 tasks, 3x) | 27.951 token/s | 184.3665 token/s |
| UD-Q8_K_XL, heterogeneous MTP K4/S4 | Cross-mode output hashes differ; matrix not run | - | 9.7577 token/s |

`Request-wide` includes prompt processing, generation, HTTP, and request
overhead. `Steady decode` is a warmed-up, repetitive 1,024-token shape. A dash
means there is no defensible apples-to-apples capture for that cell.

## The untouched-Q6_K execution-path case study

The 22,884,408,288-byte Q6_K artifact was not requantized. The current clean
capture uses fixed K7/S6/B11, an 8,192-row draft-only token-ID prefix,
short-batch Flash Attention off, normal CUDA Graphs, high-priority CUDA
streams, full CUDA offload, physical-core affinity, and one loaded model with
one concurrent request. Across nine 1,024-token samples it reached 172.835
token/s median, 171.298 minimum, and 175.533 maximum while the shared Windows
display GPU already showed 5–8% utilization.

The earlier quiet-host high-water mark for the same artifact remains 176.6109
token/s. Its 20.13% comparison used the full-vocabulary K7/S7 control at
147.0207 token/s. Both captures retained the same deterministic output digest;
the earlier result is not a service floor for the current shared desktop.

The general short-task profile uses fixed K6/S6/B8. The current paired
12-task, 256-token capture reached 46.923 token/s versus 28.713 token/s for the
target-only control, a 63.42% gain. Proposal acceptance was 26.81%, yet each
target pass committed 2.591 verified tokens and replay remained zero.

From first principles, throughput is approximately emitted tokens per target
pass divided by draft, target verification, synchronization, and sampling
cost. The peak prompt already reaches 96.64% acceptance and 7.75 emitted
tokens per target pass, leaving only about 3.2% perfect-acceptance headroom.
Shared WDDM contention is now the main stable-175 boundary.

Negative results remain part of the evidence. Disabling CUDA Graphs reached
133.876 token/s; `GGML_CUDA_GRAPH_OPT=1` reached 160.613; and
`CUDA_DEVICE_MAX_CONNECTIONS=32` reached 168.900. Adaptive K raised
representative-workload acceptance to 50.07% but fell to 35.178 token/s because
shape variation reduced graph reuse. Flash Attention is therefore
profile-specific, not a global switch.

This is one llama.cpp/CUDA integration, **not** a service floor or universal
default. Power supplies finite-shape, scheduling, exact-fallback, and evidence
contracts. Other backends and models must select their own shapes and kernels.

The previous mixed-artifact boundary remains separate. Its rollback-complete
K7/S7 profile reached 175.2089 token/s steady decode and 83.228 token/s on the
repeated 100-task workload. That 19,187,686,464-byte artifact uses Q4_0 main
FFN tensors, a Q6_K MTP block, and a Q4_K draft head.

The current acceptance host is Windows 11 with an RTX 4090 and a 10-core,
20-thread Intel Xeon w5-2445. Its `0x55555` affinity mask is topology-specific,
not a portable product default.

## Did quality fall?

In the current pure-Q6_K K6/S6/B8 pair, all 12 final answers matched, both modes
scored 9/12 lenient and strict, eight complete content digests matched, and
three tasks per mode reached the 256-token cap. Batched and serial target
kernels can follow different floating-point trajectories, so exact target
verification does not promise byte-identical prose.

This sample is still too small for a general-intelligence claim. Exact target
verification proves that every committed token remains target-authoritative;
it does not replace a representative quality evaluation.

For the previous mixed-artifact K7/S7 profile, no regression was observed on
the fixed repeated sample:

- TBQ4 autoregressive scored 70/100 lenient and 64/100 strict.
- Full-vocabulary K7/S7 scored 76/100 lenient and 66/100 strict.
- All 900 requests completed without errors.
- Every prediction was stable across three repetitions.
- Proposal acceptance was 51.33%; replay and guard activation were both zero.

These are fixed-task accuracy proxies, not general intelligence or IQ scores.
The paired differences did not reach conventional statistical significance, so
the result does not prove that MTP improves general model intelligence.

The archived prefix-FR mode is useful negative evidence. Its steady peak was
high, but mixed-workload acceptance fell to 25.55%, request-wide throughput fell
to 27.951 token/s, and C-Eval acceptance was only 14.21%. Full-vocabulary K7/S7
removes that draft-coverage bottleneck.

## Reproduce the boundary

The dedicated [Reproduction page](./reproduction) provides copyable commands,
the fixed environment, input SHA-256 values, complete runner arguments, output
files, and pass criteria.

The full replay has two levels:

1. **Offline verification** recomputes archived report statistics, hashes,
   thresholds, and output identities without loading the model.
2. **Performance replay** rebuilds the pinned CUDA profile, verifies the exact
   artifact, prompt, ACL, and binaries, then runs Power's streaming API workload.

The current clean capture fixes one warm-up, nine measured requests, 1,024
generated tokens per request, batch 11, greedy sampling, high-priority CUDA
streams, and a deterministic output SHA-256. The runner fails on a changed
model identity, wrong backend, short output, non-deterministic digest, missing
host control, or a missed explicit gate. The offline verifier pins 23 evidence
hashes and recomputes both current and historical statistics.

Use the repository's checked-in guide and raw evidence:

- [Complete Windows/CUDA reproduction procedure](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md)
- [Benchmark record and all mode interpretations](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md)
- [Untouched-Q6_K boundary and dynamic-quantization analysis](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/PURE-Q6.md)
- [Repeated 100-task quality protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md)
- [Current compact machine-readable evidence](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/full-vocabulary-s7-current-rtx4090-3x.json)
- [UD-Q8_K_XL heterogeneous-placement boundary](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-ud-q8-k-xl-rtx4090/README.md)

Treat a replay on different silicon, driver, display load, clock policy, model
bytes, or prompt as a new result rather than silently combining it with this
acceptance capture.
