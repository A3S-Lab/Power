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

The active gate tests only the unchanged Q6_K target. Q4 files are present only
as auxiliary speculative proposers; they are not target-model quality results.
Mixed-quantization and Q8 captures are historical research outside this
acceptance table.

| Active Qwen3.8-27B Q6_K mode | Fixed 100-task score, mean of 3 runs | Request-wide throughput | Three-run range |
| --- | --- | ---: | ---: |
| Untouched Q6_K, autoregressive control | 67/100 lenient; 60/100 strict | 23.642 token/s | 23.543--23.832 |
| **Same Q6_K, full-vocabulary MTP** | 67/100 lenient; 58/100 strict | **41.035 token/s (1.736x)** | 40.197--41.696 |

All six runs completed 100/100 requests without errors and passed continuous
per-process GPU exclusivity checks. Each mode was internally deterministic.
Across modes, MTP retained 89/100 extracted answers and 50/100 complete output
hashes, with two lenient gains and two losses. Strict formatting had zero gains
and two losses, so the candidate is faster but not a lossless default.

`Request-wide` includes prompt processing, generation, HTTP, and request
overhead. It is not the same metric as warmed-up 1,024-token steady decode.
The compact [Q6_K-only evidence package](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/pure-q6-rtx4090-3x.evidence.json)
can be verified without a model or GPU.

## Archived auxiliary-proposer research

The archived DSpark captures use a shorter context-512, 256-token acceptance
shape rather than the 1,024-token MTP peak shape. Their controls are local to
those captures and must not be compared with the active table above.

## Q6_K-only DFlash2 boundary

DFlash2 was measured with the target fixed to the same 22,884,408,288-byte
Q6_K GGUF in both modes. The 1.14 GB Q4 file is an auxiliary proposer only;
there is no Q4 target result in this capture.

Native Power produced the following controlled five-sample pair:

| Same Q6_K target | Median decode | Minimum decode | Median end-to-end |
| --- | ---: | ---: | ---: |
| Target-only | 33.075 token/s | 32.938 token/s | 25.744 token/s |
| **DFlash2 K7/S6** | **144.453 token/s** | **141.267 token/s** | **63.182 token/s** |

That is a 4.367x decode and 2.454x end-to-end speedup. All five paired outputs
matched, proposal acceptance was 98.230%, and replay was zero. The integer-
sequence prompt is a high-coverage synthetic boundary, not a 175 token/s
result or a general-workload claim.

The separate three-order 12-task calibration retained 9/12 lenient and strict
answers in both modes. All 12 extracted answers matched, while only 7/12
complete response hashes matched. Mean request-wide throughput increased from
29.702 to 45.143 token/s (1.520x), but the candidate ranged from 33.689 to
54.586 token/s. DFlash2 is therefore native and useful as an explicit profile,
not a lossless default. The complete
[report, offline verifier, and replay commands](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)
are published with the repository.

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

The current request-local controller replaces the unconditional wide start
with one rollback-safe K6 probe. A fully accepted first probe opens K10; a
partial first probe closes that path for the request, and sustained low-yield
rounds open a one-way target-only circuit. On clean commit
`cbdb3f673446b3532c9683dabc816a149ae27b1f`, the controlled peak produced
166.988, 160.881, and 164.756 token/s. Its 164.756 median and 160.881 minimum
passed both 160 token/s gates with identical output and receipt hashes,
92.713% acceptance, 9.8077 verified tokens per target pass, and zero replay.

The adaptive 100-task run reached 31.052 token/s against 22.872 target-only,
a 1.358x request-wide gain. It recorded 62.878% acceptance, 3.373 verified
tokens per target pass, 24 target-only requests, and zero replay or guard
activation. Scores moved from 67/58 to 69/56 lenient/strict, with five lenient
gains and three losses, one strict gain and three losses, 89/100 extracted-
answer parity, and 55/100 complete-output parity. All 57 tasks untruncated in
both modes retained the same extracted answer.

A clean follow-up selected every observed lenient or strict loss plus one
positive control. Three alternating 512-token repetitions produced 5/5 paired
answers, zero gains, zero losses, and 30.521 versus 24.967 token/s. Raising the
budget to 1,024 tokens let all five tasks finish; target-only and DSpark both
scored 4/5 and retained 5/5 untruncated answer parity. The apparent 256-token
losses were therefore cutoff-sensitive diagnostics, not reproduced answer
regressions. Complete outputs still differed 0/5, so this establishes a
controlled 160-plus opt-in peak and selected-answer non-regression—not a 175
token/s floor, a general-intelligence result, or an exact-output default.

DFlash v1 is not part of this number. It uses a different artifact contract
and cannot be layered onto DSpark. No genuine DFlash v1 GGUF has completed the
acceptance gate on this host; the native DFlash2 capture above is a separate
contract.

## The untouched-Q6_K execution-path case study

The 22,884,408,288-byte Q6_K artifact was not requantized. The latest
exact-build replay uses fixed K7/S6/B11, an 8,192-row draft-only token-ID prefix,
short-batch Flash Attention off, normal CUDA Graphs, high-priority CUDA
streams, full CUDA offload, physical-core affinity, and one loaded model with
one concurrent request. Across nine 1,024-token samples it reached 174.413
token/s median and 172.723 minimum with one deterministic output digest. This
fresh run did not establish a stable 175 token/s floor.

The earlier quiet-host high-water mark for the same artifact remains 176.6109
token/s with a 173.263 minimum. Its 20.13% comparison used the full-vocabulary
K7/S7 control at 147.0207 token/s. These are peak-shape boundaries, not service
guarantees for the shared desktop.

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
`CUDA_DEVICE_MAX_CONNECTIONS=32` reached 168.900. An earlier unrestricted
adaptive-K experiment raised representative-workload acceptance to 50.07% but
fell to 35.178 token/s because shape variation reduced graph reuse. Flash Attention is therefore
profile-specific, not a global switch.

This is one llama.cpp/CUDA integration, **not** a service floor or universal
default. Power supplies finite-shape, scheduling, exact-fallback, and evidence
contracts. Other backends and models must select their own shapes and kernels.

The current acceptance host is Windows 11 with an RTX 4090 and a 10-core,
20-thread Intel Xeon w5-2445. Its `0x55555` affinity mask is topology-specific,
not a portable product default.

## Did quality fall?

The active Q6_K-only matrix kept the lenient score at 67/100 while strict
format scoring moved from 60/100 to 58/100. MTP produced two lenient gains and
two losses, 89/100 extracted-answer parity, and 50/100 complete-output parity.
All 59 tasks untruncated in both modes kept the same extracted answer. The
fixed sample therefore shows no aggregate lenient-score decrease, but it does
not prove unchanged intelligence or lossless output; the production-default
gate remains closed.

The fixed external-DSpark matrix observed no score decrease: 73/100 lenient
and 59/100 strict versus its target-only control at 67/100 and 58/100. Its
54/100 complete-output parity still blocks a lossless-default claim. The newer
adaptive run moved lenient score up by two and strict score down by two, with
three paired lenient losses. All 57 tasks untruncated in both adaptive modes
retained the same extracted answer. The focused 512/1,024-token follow-up
reproduced none of those answer losses and reached 5/5 untruncated parity at
1,024 tokens, but only on the selected five-task diagnostic. Fixed-task scores
are not a measure of general intelligence, and 0/5 complete-output parity keeps
the exact-output production gate closed.

In the current pure-Q6_K K6/S6/B8 pair, all 12 final answers matched, both modes
scored 9/12 lenient and strict, eight complete content digests matched, and
three tasks per mode reached the 256-token cap. Batched and serial target
kernels can follow different floating-point trajectories, so exact target
verification does not promise byte-identical prose.

The DFlash2 calibration reached the same 9/12 score in both modes and retained
12/12 extracted answers, but only 7/12 complete response hashes. It therefore
shows no observed score regression on this small sample, not general
intelligence equivalence or lossless output identity.

This sample is still too small for a general-intelligence claim. Exact target
verification proves that every committed token remains target-authoritative;
it does not replace a representative quality evaluation.

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
- [Native DSpark Q4 reports and exact reproduction](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark)
- [Q6_K-only DFlash2 evidence and exact reproduction](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)
- [Adaptive DSpark peak and paired-quality evidence](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/adaptive/evidence.json)

Treat a replay on different silicon, driver, display load, clock policy, model
bytes, or prompt as a new result rather than silently combining it with this
acceptance capture.
