---
title: Performance evidence
description: Reproducible Qwen3.8-27B speed, quality, rollback, and artifact evidence measured through the A3S Power API.
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

| Active Qwen3.8-27B Q6_K mode | Fixed 100-task score, mean of 3 runs | Request-wide throughput | Three-run range |
| --- | --- | ---: | ---: |
| Untouched Q6_K, autoregressive control | 67/100 lenient; 60/100 strict | 23.642 token/s | 23.543--23.832 |
| **Same Q6_K, full-vocabulary MTP** | 67/100 lenient; 58/100 strict | **41.035 token/s (1.736x)** | 40.197--41.696 |

All six runs completed 100/100 requests without errors and passed continuous
per-process GPU exclusivity checks. Each mode was internally deterministic.
Across modes, MTP retained 89/100 extracted answers and 50/100 complete output
hashes. Strict formatting moved from 60/100 to 58/100, so this faster mode is
not a lossless production default.

`Request-wide` includes prompt processing, generation, HTTP, and request
overhead. It is not the same metric as warmed-up 1,024-token steady decode.
The [Q6_K-only evidence package](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/pure-q6-rtx4090-3x.evidence.json)
can be checked without a model or GPU.

## Archived auxiliary-proposer research

The DSpark rows use a shorter context-512, 256-token acceptance shape rather
than the 1,024-token MTP peak shape. They are paired with each other, not with
the historical autoregressive row.

## Native external-DSpark boundary

Power's verified external-draft path keeps the target Q6_K bytes unchanged and
loads a separate 1.10 GB DSpark Q4 proposal model. Fixed K10/S6 produced
169.561, 167.102, and 169.324 token/s, a 169.324 median and 5.250x gain over
its 32.249 token/s paired target-only control. Proposal acceptance was 90.873%,
verified tokens per target pass were 9.8077, and replay was zero.

Every paired request, 256-token output, and execution receipt had the same
digest. That is exact evidence for this deterministic request, not a
cross-prompt intelligence result. The profile peaked at 23,847 MiB and left
only 717 MiB on the recorded RTX 4090, so low utilization alone does not prove
that model admission will be stable.

The separate 600-request quality capture used 100 fixed MMLU/GSM8K/C-Eval
tasks, three repetitions per mode, context 1,024, and batch 12. Target-only
scored 67/100 lenient and 58/100 strict at 22.618 token/s request-wide. DSpark
scored 73/100 and 59/100 at 32.678 token/s, a 1.445x gain. Both modes were
deterministic, but only 54/100 complete outputs matched across modes and every
DSpark request entered exact fallback replay. The observed score did not fall;
nevertheless K10/S6 fails the lossless production-default gate and remains an
explicit benchmark profile.

DFlash is not part of this number. It uses a different artifact contract and
cannot be layered onto DSpark. No genuine DFlash GGUF has completed the
acceptance gate on this host.

## What the pure-Q6_K peak boundary means

The latest exact-build replay of the untouched 22,884,408,288-byte Q6_K
artifact reached 174.413 token/s median across nine 1,024-token samples with
prefix-FR8192 K7/S6. Its minimum was 172.723 and every output digest matched,
so it did not establish a stable 175 token/s floor. An earlier quiet-host
capture reached 176.6109 token/s median with a 173.263 minimum; neither is a
service guarantee.

No model weight was requantized. The speedup comes from native MTP, exact target
verification, an 8,192-row draft-only token-ID prefix, six resident recurrent
snapshots, batched GPU greedy sampling, normal CUDA Graphs, short-batch Flash
Attention off, full CUDA offload, batch 11, and host controls.

This is a high-coverage peak, **not** a service floor or universal default. On
the one-pass 12-task calibration, full-vocabulary K7/S6 reached 47.032 token/s
request-wide with 52.30% acceptance, while prefix FR reached 37.290 token/s
with 24.82% acceptance. Eleven tasks per mode hit the output cap.

The previous mixed-artifact boundary remains separate. Its rollback-complete
K7/S7 profile reached 175.2089 token/s steady decode and 83.228 token/s on the
repeated 100-task workload. That 19,187,686,464-byte artifact uses Q4_0 main
FFN tensors, a Q6_K MTP block, and a Q4_K draft head.

The current acceptance host is Windows 11 with an RTX 4090 and a 10-core,
20-thread Intel Xeon w5-2445. Its `0x55555` affinity mask is topology-specific,
not a portable product default.

## Did quality fall?

The active Q6_K-only matrix kept the lenient score at 67/100 while strict
format scoring moved from 60/100 to 58/100. MTP had two lenient gains and two
losses, 89/100 extracted-answer parity, and 50/100 complete-output parity. All
59 tasks untruncated in both modes retained the same extracted answer. This
does not prove unchanged intelligence or lossless output, so the default gate
remains closed.

The external-DSpark matrix observed no score decrease: 73/100 lenient and
59/100 strict versus its target-only control at 67/100 and 58/100. All 58 tasks
that were untruncated in both modes retained the same extracted answer. The
54/100 complete-output parity still blocks a lossless-default claim; fixed-task
scores are not a measure of general intelligence.

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

The current peak gate fixes one warm-up, nine measured requests, 1,024 generated
tokens per request, batch 11, greedy sampling, and a deterministic output
SHA-256. The runner fails on a changed model identity, wrong backend, short
output, non-deterministic digest, missing host control, or missed median gate.

Use the repository's checked-in guide and raw evidence:

- [Complete Windows/CUDA reproduction procedure](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md)
- [Benchmark record and all mode interpretations](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md)
- [Untouched-Q6_K boundary and dynamic-quantization analysis](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/PURE-Q6.md)
- [Repeated 100-task quality protocol](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md)
- [Current Q6_K-only machine-readable evidence](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/pure-q6-rtx4090-3x.evidence.json)
- [Native DSpark Q4 reports and exact reproduction](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark)
- [UD-Q8_K_XL heterogeneous-placement boundary](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-ud-q8-k-xl-rtx4090/README.md)

Treat a replay on different silicon, driver, display load, clock policy, model
bytes, or prompt as a new result rather than silently combining it with this
acceptance capture.
