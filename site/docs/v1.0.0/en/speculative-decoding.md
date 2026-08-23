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
| llama.cpp with a verified external DFlash2 GGUF | `off`, `dflash2` |
| llama.cpp with a verified external DSpark GGUF | `off`, `dspark` |

`draft-model` remains a reserved shared strategy without a production
llama.cpp adapter. DFlash v1, DFlash2, and DSpark use different external-
artifact contracts; they are not interchangeable and do not stack. Power
parses and hashes both GGUF files, binds the draft to the target digest,
validates artifact-specific metadata and tensors, then compares the complete
target/draft vocabularies when their contexts bind. An explicit unsupported or
mismatched mode fails closed.

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
it is not a generic runtime switch. The archived mixed artifact keeps the MTP
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
- exact target verification and deterministic output digests.

The latest exact-build capture reached 174.413 token/s median steady decode,
with a 172.723 minimum and 177.150 maximum. The earlier quiet-host high-water
mark was 176.6109 token/s; the same-artifact full-vocabulary K7/S7 control was
147.0207 token/s. On the one-pass 12-task calibration, however,
full-vocabulary K7/S6 reached 47.032 token/s request-wide while prefix FR
reached 37.290 token/s. The prefix-FR peak profile has not completed the
repeated 100-task matrix; the active repeated matrix uses full-vocabulary MTP.

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

Native DFlash2 is another separate contract. It keeps the Q6_K target unchanged
and uses a 1.14 GB Q4 artifact only as an auxiliary proposer. Five paired
256-token samples reached 144.453 token/s median decode versus 33.075
target-only (4.367x), with 98.230% acceptance, exact outputs, and zero replay.
Median end-to-end throughput was 63.182 versus 25.744 token/s. The separate
12-task quality calibration retained every extracted answer but only 7/12
complete outputs, so DFlash2 remains opt-in and the native peak is not a stable
175 token/s floor.

The context-1024 quality diagnostic tells a different, workload-representative
story. Across 600 successful requests, DSpark K10/S6 delivered 32.678 token/s
request-wide versus 22.618 for target-only (1.445x), with no observed score
decrease. Exact complete-output parity was only 54/100 and every DSpark request
entered fallback replay. The profile is therefore deterministic and useful for
experimentation, but it is not a lossless production default.

See [Performance evidence](/performance) for the complete interpretation and
the [canonical speculative-decoding design](https://github.com/A3S-Lab/Power/blob/main/docs/speculative-decoding.md)
for adapter APIs and acceptance rules, and the
[DSpark evidence package](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark)
for raw paired reports and exact replay commands, and the
[DFlash2 evidence package](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)
for native performance, representative quality, and reproduction.
