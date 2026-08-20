---
title: Performance evidence
description: Reproducible Qwen3.8-27B speed, quality, rollback, and artifact evidence measured through the A3S Power API.
---

# Performance evidence

Power accepts performance work only when the measured system still preserves
its execution contract. The published Qwen3.8 captures therefore use the real
streaming API, pin artifact and binary identities, verify deterministic output,
and report quality and workload throughput beside steady decode.

To verify the numbers directly, open [Reproduction](./reproduction): run the
model-free one-command verifier first, then replay the full RTX 4090 protocol
against the same input identities.

## Quality and speed by mode

| Qwen3.8-27B artifact and mode | Fixed-task quality proxy | Request-wide throughput | Median steady decode |
| --- | --- | ---: | ---: |
| Untouched Q6_K, autoregressive | 67/100 lenient; 60/100 strict (100 tasks, 3x) | 30.883 token/s | 35.5793 token/s (earlier capture) |
| Untouched Q6_K, native MTP | Exact greedy parity on the fixed peak prompt; matrix not run | - | 140.1600 token/s |
| TBQ4 mixed, autoregressive | 70/100 lenient; 64/100 strict (100 tasks, 3x) | 38.724 token/s | - |
| **TBQ4 mixed + full-vocabulary fixed MTP, K7/S7** | **76/100 lenient; 66/100 strict** (100 tasks, 3x) | **83.228 token/s** | **175.2089 token/s** |
| TBQ4 mixed + full-vocabulary guarded MTP, K7/S6 | 5/12 lenient; 3/12 strict (12 tasks, 3x) | 54.060 token/s | **177.7165 token/s** |
| TBQ4 mixed + MTP + prefix FR (historical) | 72/100 lenient; 60/100 strict (100 tasks, 3x) | 27.951 token/s | 184.3665 token/s |
| UD-Q8_K_XL, heterogeneous MTP K4/S4 | Cross-mode output hashes differ; matrix not run | - | 9.7577 token/s |

`Request-wide` includes prompt processing, generation, HTTP, and request
overhead. `Steady decode` is a warmed-up, repetitive 1,024-token shape. A dash
means there is no defensible apples-to-apples capture for that cell.

## What the 175+ boundary means

The balanced K7/S7 profile reached a 175.2089 token/s median across nine
1,024-token samples, with a 174.2211 minimum and five samples at or above 175.
The guarded K7/S6 peak profile reached 177.7165 token/s with all nine samples
above 175, but K7/S7 is the mixed-workload default because its seven resident
snapshots make every proposal rollback-complete without replay.

This result is **not** an untouched 6-bit model and **not** a service floor. The
19,187,686,464-byte artifact is derived from Q6_K: main FFN tensors are Q4_0,
the MTP block remains Q6_K, and the full-vocabulary draft head is Q4_K. The
measurement also depends on Flash Attention, full CUDA offload, batched target
and draft greedy sampling, an idle-enough WDDM display GPU, and host tuning.

The current acceptance host is Windows 11 with an RTX 4090 and a 10-core,
20-thread Intel Xeon w5-2445. Its `0x55555` affinity mask is topology-specific,
not a portable product default.

## Did quality fall?

No regression was observed on the fixed current sample:

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

The current gate fixes one warm-up, nine measured requests, 1,024 generated
tokens per request, batch 14, greedy sampling, and a deterministic output
SHA-256. The runner fails on a changed model identity, wrong backend, short
output, non-deterministic digest, missing host control, or missed median gate.

Use the repository's checked-in guide and raw evidence:

- [Complete Windows/CUDA reproduction procedure](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md)
- [Benchmark record and all mode interpretations](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md)
- [Repeated 100-task quality protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md)
- [Current compact machine-readable evidence](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/full-vocabulary-s7-current-rtx4090-3x.json)
- [UD-Q8_K_XL heterogeneous-placement boundary](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-ud-q8-k-xl-rtx4090/README.md)

Treat a replay on different silicon, driver, display load, clock policy, model
bytes, or prompt as a new result rather than silently combining it with this
acceptance capture.
