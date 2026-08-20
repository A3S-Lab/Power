---
title: Architecture
description: The model-neutral ownership, resource, integrity, and evidence contracts behind A3S Power.
---

# Embedded Inference Architecture

Power treats inference as a bounded operation whose identity and effects can be
verified. It does not treat a model name as an execution contract.

## Derive the boundary from first principles

| Constraint | Required contract | Runtime mechanism |
| --- | --- | --- |
| Memory, compute, and queue capacity are finite. | Every request needs explicit limits and cancellation. | Bounded admission, deterministic microbatching, placement plans, and session pools. |
| A filename or model alias does not identify executed bytes. | Artifact identity must survive storage, mirrors, and device placement. | SHA-256 descriptors, signed identities, verified mirrors, and residency evidence. |
| A response alone does not reveal the path that produced it. | Runtime policy, device path, input, and output must be committed together. | Canonical execution receipts and accelerator evidence. |
| The service cannot be trusted to approve its own claims. | Acceptance policy belongs outside the execution boundary. | Nonce-bound TEE evidence and an independent verifier. |

## One core, three surfaces

```text
revision-locked bundle       model-owned graph       API client
          |                         |                     |
    provisioner                 embedded               service
          \                         |                    /
           +----------- shared runtime core -----------+
                                  |
              admission / placement / cancellation
                                  |
                    devices / weights / state
                                  |
                   evidence / canonical receipt
                                  |
                      independent verification
```

The embedded library and hosted service are entry points into the same
contracts, not two model implementations. Artifact provisioning is independent
because downloading and installing a reviewed bundle is a different authority
from executing it.

## Ownership is explicit

| Power owns | Model-owning crates own |
| --- | --- |
| Typed CPU, CUDA, and Metal devices; bounded graph execution | Architecture, topology, layers, kernels, and arithmetic |
| Admission, session pools, microbatching, cancellation, and limits | Tokenizer, preprocessing, postprocessing, and generation policy |
| Artifact identity, replicas, mirrors, placement, and residency | Revision pins, conversion, tensor contracts, and quality gates |
| TEE privacy, attestation binding, sealed state, and receipts | KV/recurrent layout and semantic state |

This boundary keeps Power model-neutral. Language, vision, OCR, embedding, and
future model crates can share resource and evidence machinery without moving
their semantics into a central model switch statement.

## The runtime contracts

### Bounded execution

Admission limits active and queued work. Device admission prevents independent
models from overcommitting the same accelerator. Cancellation is checked before
admission and remains safe while a request waits or executes.

### Model-owned finite profiles

Model crates may declare finite optimized shape classes as opaque SHA-256
identities. Power checks only aggregate batch, tensor-element, scratch, device,
artifact, and TEE-policy bounds; it never interprets sequence lengths, image
geometry, tokenization, or a model family. Unsupported classes either fail
closed or select an explicitly identified dynamic implementation. Receipt v5
records that decision without exposing private geometry.

### Model-neutral mutable replicas

Stateful model crates can request a finite set of lazy, independently
initialized session replicas for one exact model and execution identity. Each
non-cloneable lease owns one anonymous slot; all slots share the same resolved
runtime and physical-device admission gate. Power reserves the worst-case
resident bytes before any loader runs and reports only aggregate replica
counts. Language, vision, OCR, embedding, and multimodal contexts use the same
path: the model family is opaque identity, never a dispatch branch.

### Verified weights

Weight descriptors bind tensor ranges to storage identities. Complete and
partial mirrors retain that identity across storage tiers. Placement and
residency evidence record the actual selected path rather than the preferred
path alone.

### Accelerator evidence

Execution receipts can include the concrete device, fallback, fused-batch, or
multi-device mesh selected for a declaration-bound execution. Evidence must
match the model, runtime device, input digest, and output digest before it can
be attached to a receipt.

### Recoverable state

Authenticated sealed-state envelopes bind model and runtime identity to warm
state. Recovery policy distinguishes primary and backup sources, authorizes
export scopes, detects rollback, and zeroizes sensitive material.

### Private observability

Digest-only receipts and telemetry make execution inspectable without requiring
prompt or response content. Opaque renderer paths omit claims they cannot
derive instead of fabricating deterministic evidence.

## Execution lifecycle

```text
declare limits and identity
          |
          v
admit model + device capacity
          |
          v
resolve storage + placement
          |
          v
execute reviewed model plan
          |
          v
commit input/output/device evidence
          |
          v
release permits and verify receipt
```

The detailed design includes tensor batches, residency budgets, partial mirrors,
prefetch hints, heterogeneous meshes, sealed state, and tuning evidence. Read
the [canonical architecture document](https://github.com/A3S-Lab/Power/blob/main/docs/embedded-inference-architecture.md)
for the complete APIs, invariants, and validation gates. The
[shape-profile contract](https://github.com/A3S-Lab/Power/blob/main/docs/shape-profiles.md)
and [session-replica contract](https://github.com/A3S-Lab/Power/blob/main/docs/session-replicas.md)
document their model-neutral ownership boundaries and reproduction commands.
