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

The shared strategy vocabulary also includes `draft-model`, `dflash`, and
`dspark`. They remain unavailable until a compatible adapter artifact and graph
exist. An explicit unsupported mode returns an error; Power never silently
substitutes a cheaper algorithm or relabels n-gram lookup.

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
The current profile therefore retains all 248,320 draft-head rows.

## Current accepted profile

The balanced Qwen3.8-27B K7/S7 capture combines:

- the Q6_K-derived mixed TBQ4 artifact;
- native full-vocabulary MTP;
- seven proposals and seven recurrent snapshots;
- batched target and draft greedy CUDA sampling;
- Flash Attention and full CUDA layer offload;
- exact target verification and deterministic output digests.

It reached 175.2089 token/s median steady decode and 83.228 token/s mean
request-wide throughput on the fixed 100-task workload. The quality matrix
recorded no regression against its TBQ4 autoregressive control, but that sample
does not establish a general intelligence improvement.

See [Performance evidence](/performance) for the complete interpretation and
the [canonical speculative-decoding design](https://github.com/A3S-Lab/Power/blob/main/docs/speculative-decoding.md)
for adapter APIs, benchmark commands, and acceptance rules.
