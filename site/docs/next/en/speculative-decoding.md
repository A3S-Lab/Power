---
title: Speculative decoding
description: How A3S Power separates advisory proposal generation from exact target verification, rollback, and quality gates.
---

# Model-neutral Speculative Decoding

Speculative decoding is a runtime capability, not a Qwen-specific branch.
Proposal generation is advisory; the target model remains authoritative for
every emitted token.

Power provides the common transaction, scheduling, rollback, and evidence
protocol. A backend adapter provides the architecture-specific draft graph,
state snapshots, logits access, and compatibility metadata.

## The exactness transaction

Each model-backed round follows one order:

1. Retain a checkpoint for target, draft, sampler, and decoder state.
2. Ask the adapter for at most the scheduled number of proposal tokens.
3. Evaluate the anchor and proposal block in one target pass.
4. Sample target rows only through the first mismatch.
5. Commit the accepted prefix and discard every rejected state row.
6. Emit exactly one correction or bonus token when generation continues.
7. Restore the last committed transaction on cancellation or failure.

For greedy decoding, speculative and autoregressive token IDs must match
exactly. For stochastic decoding, target sampler state advances once per
emitted target sample and never for an unobserved rejected suffix.

## Capabilities fail closed

| Backend/model capability | Available strategies |
| --- | --- |
| mistral.rs or proxy | `off`; `auto` resolves to `off` |
| picolm | `off`, `prompt-lookup`, `ngram-context` |
| llama.cpp without native prediction tensors | `off` |
| llama.cpp with `*.nextn_predict_layers > 0` | `off`, `mtp` |
| llama.cpp with a verified external DFlash GGUF | `off`, `dflash` |
| llama.cpp with a verified external DSpark GGUF | `off`, `dspark` |

`draft-model` remains a reserved shared strategy without a production
llama.cpp adapter. DFlash and DSpark use different external-artifact contracts;
they are not interchangeable and do not stack. Power parses and hashes both
GGUF files, binds the draft to the target digest, validates artifact-specific
metadata and tensors, then compares the complete target/draft vocabularies when
their contexts bind. An explicit unsupported or mismatched mode fails closed.

`auto` selects a verified external artifact when the model manifest contains
one; otherwise it considers native MTP. Power does not load both draft
mechanisms because they would compete for device memory and mutable state.

## Draft width and rollback width are different

```text
spec_mode = "mtp"
spec_draft_max = 7
spec_mtp_recurrent_snapshots = 7

# Experimental compact draft projection. Omit for full vocabulary.
# spec_mtp_fr_vocab_size = 8192
```

`spec_draft_max` bounds proposal width. `spec_mtp_recurrent_snapshots` bounds
resident target rollback state. K7/S7 keeps a rollback point for every proposed
token. K7/S6 can be faster on high-acceptance prompts, but a rejected suffix may
exceed the resident window.

Power's guarded K7/S6 path permits one exact replay, then clamps all later
rounds in that request to six proposals. The guard bounds replay without
changing requests that remain on the high-acceptance path. K7/S7 avoids the
replay condition entirely and is the balanced default.

## TBQ4 and FR solve different problems

TBQ4 is an artifact-construction choice. It reduces selected tensor bandwidth;
it is not a generic runtime switch. The current mixed artifact keeps the MTP
block at Q6_K, uses Q4_0 for main FFN tensors, and uses Q4_K for the separate
draft head.

FR reduces only the rows projected by the MTP draft head. That can raise a peak
on a narrow vocabulary distribution, but its acceptance is language- and
domain-sensitive. The historical prefix-FR matrix reached a high steady rate
while becoming slower than autoregressive TBQ4 on the representative workload.
The mixed-artifact release profile therefore retains all 248,320 draft-head
rows. The current pure-Q6_K peak deliberately enables an 8,192-token-ID prefix,
but keeps full-vocabulary MTP as its balanced workload profile.

## Current measured profiles

The untouched Q6_K peak combines:

- the original 22,884,408,288-byte Q6_K artifact;
- native MTP with an 8,192-row draft-only token-ID prefix;
- seven proposals and six recurrent snapshots;
- fixed B11 target-verification capacity and normal CUDA Graphs;
- batched target and draft greedy CUDA sampling;
- short-batch Flash Attention off and full CUDA layer offload;
- high-priority CUDA streams, physical-core affinity, and single-model,
  single-request scheduling;
- exact target verification and deterministic output digests.

The current clean nine-run capture reached 172.835 token/s median steady
decode, 171.298 minimum, and 175.533 maximum while the shared Windows display
GPU already showed 5–8% utilization. The earlier quiet-host high-water mark is
176.6109 token/s; the same-artifact full-vocabulary K7/S7 control reached
147.0207 token/s.

The general short-task profile uses fixed K6/S6/B8. In the current paired
12-task, 256-token calibration it reached 46.923 token/s versus 28.713 token/s
with speculation off, a 63.42% gain. Both modes retained all 12 final answers
and the 9/12 score. Acceptance was 26.81%, verified tokens per target pass were
2.591, and replay was zero.

Stable shapes mattered more than nominal acceptance. Adaptive K raised
acceptance on the same representative workload to 50.07% but fell to 35.178
token/s because changing verification shapes reduced CUDA Graph reuse.
Disabling CUDA Graphs also reduced the peak workload to 133.876 token/s. K, S,
target batch, and graph shape must be tuned as one system.

The previous mixed-artifact K7/S7 profile remains the representative quality
capture: 175.2089 token/s median steady decode, 83.228 token/s request-wide,
and no observed regression against its TBQ4 autoregressive control. That sample
does not establish a general intelligence improvement.

The native external-DSpark acceptance capture is a separate context-512,
batch-12 boundary. The untouched Q6_K target-only control reached 32.249
token/s median; DSpark Q4 K10/S6 reached 169.324 token/s median and 167.102
minimum, a 5.250x decode gain. All three 256-token outputs and execution
receipts matched the control exactly, proposal acceptance was 90.873%, and
fallback replay was zero. Peak VRAM was 23,847 MiB, so this profile requires a
quiet 24 GB device. A genuine DFlash artifact has not yet been benchmarked; a
DSpark artifact mislabeled as DFlash is rejected and is not a DFlash result.

See the [Optimization playbook](./optimization) for the complete execution
path, [Performance evidence](./performance) for the measurement interpretation, and
the [canonical speculative-decoding design](https://github.com/A3S-Lab/Power/blob/main/docs/speculative-decoding.md)
for adapter APIs and acceptance rules, and the
[DSpark evidence package](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark)
for raw paired reports and exact replay commands.
