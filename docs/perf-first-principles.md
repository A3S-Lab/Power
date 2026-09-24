# Performance First-Principles Gate (non-TEE)

This gate governs Power performance work when confidential / TEE constraints are
out of scope. It exists to reject overfitted micro-optimizations and require
end-to-end evidence before a change is called a win.

Related: [optimization-playbook.md](optimization-playbook.md),
[prism-acceleration-plan.md](prism-acceleration-plan.md),
[device-resident-graphs.md](device-resident-graphs.md).

## Mission

Power is a model-neutral **execution boundary**. It may:

- select backend acceleration profiles;
- fuse **reviewed** Candle graph patterns under `embedded-cuda`;
- admit, batch, cache, and speculate when the owning backend can honor the claim.

It must not become a second LLM kernel engine, and it must not claim wins that
only appear on synthetic microbenchmarks.

## Accepted surfaces

| Surface | What may be optimized | Evidence required |
| --- | --- | --- |
| `llamacpp-cuda` | Finite shapes, FA, CUDA Graphs, exact speculation, prefix cache | Request-wide tok/s and/or TTFT on a fixed ACL + artifact pin |
| `PrismBackend` + Prism upstream | Packing (`PTQ1_0` / `PQ2_0`), FA/ngl, `prism_profile` (baseline/dspark/kv4) | Power chat API timings under `docs/benchmarks/bonsai2-…/` |
| `embedded-cuda` | Reviewed fusions, device-resident graph chains | GraphExecutor path: **byte parity** vs unfused + e2e latency on representative shapes |

## Rejected (overfit / misaligned)

| Change | Why rejected |
| --- | --- |
| Power-owned ternary / Hadamard / `PQ2_0` kernels | Prism owns those kernels; merging them into Power core contaminates pins |
| Mapping Power `spec_mode=mtp\|dflash\|dspark` onto Bonsai packs | Different draft contracts; fail-closed instead |
| Adaptive draft widths that explode CUDA Graph shapes | Acceptance-rate-up / tok/s-down is not a win (see playbook Q6_K case) |
| Fusion that only wins on `[1,4]` unit tensors | Must show GraphExecutor latency or H↔D reduction on shapes that match real graphs |
| Claiming “faster” from unit `CustomOp` tests alone | Unit parity is necessary; e2e (GraphExecutor or Power API) is sufficient |
| Relabeling Qwen speculative benches as Bonsai evidence | Wrong weight lineage |
| Rewriting cuBLAS / FlashAttention inside Power | Backend-owned; absorb engines, do not reimplement |

## Definition of done for a lever

A performance change is accepted only when **all** of the following hold:

1. **First-principles fit** — the lever belongs to an accepted surface above.
2. **Parity** — temp=0 or byte-exact output agreement with the previous path for at least one fixed workload (or an explicit documented quality gate).
3. **End-to-end metric** — request-wide decode/prefill/TTFT (language) or GraphExecutor wall time / copy counts (embedded), not only kernel µs.
4. **Negative evidence preserved** — regressions and “no win” results stay in the capture README.
5. **Repro command** — a script or cargo invocation under `tools/` / `docs/benchmarks/` regenerates the capture.

## Measurement contracts

### Ternary / Prism

Use `tools/run-prism-baseline-bench.ps1` and `tools/run-prism-packing-compare.ps1`.

Every packing or profile claim publishes:

- GPU SKU + Prism binary identity;
- target GGUF packing (`PTQ1_0` or `PQ2_0`) and SHA-256 when available;
- `prism_profile`;
- workload class (chat / code / math);
- `predicted_per_second` / `prompt_per_second` (and draft counters for dspark);
- store under `docs/benchmarks/bonsai2-…/`, never under Qwen trees.

Tier-1 packing advisory (Ada → PTQ1 decode) is a **hypothesis** until the packing
compare script produces a same-host capture.

### Native Rust CUDA

Use GraphExecutor plans that match real fusion windows (for example the checked-in
GELU-erf Div/Erf/Add/Mul/Mul plan), on CUDA, with:

- output parity vs the CPU (or vs an intentionally unfused CUDA plan); and
- an e2e timing capture on shapes large enough that launch fusion can dominate
  (≥1M elements for elementwise chains unless a model-owned shape is smaller).

`a3s-power-cuda-fusion-bench` is the reproducible harness for that contract.

## Optimization order (non-TEE)

1. Backend profile / packing / speculation / prefix (largest tok/s and TTFT).
2. Power boundary: warm residency, streaming, admission aligned to `-np`.
3. Embedded CUDA: resident graph chains, then reviewed fusions with the contract above.
4. Artifact quantization only after 1–3 have workload-wide evidence.

## Current status snapshot

| Lever | Status |
| --- | --- |
| Prism `baseline` PTQ1_0 on RTX 4090 | Thinking-on ≈80; `--reasoning off` **85.3**; raw `llama-bench` **89.1** |
| Prism packing PTQ1 vs PQ2 same-host compare | **Done** — decode PTQ1, prefill PQ2 (`docs/benchmarks/bonsai2-27b-packing-rtx4090/`) |
| Prism `dspark` | Fail-closed for Bonsai-2 (no official `*dspark-dflash*`); unit-tested |
| Prism `mtp` (Bonsai-2) | Stock Prism peak **89.9 tok/s** @ n-max=1 + reasoning off (97.8% accept) |
| Prism `dflash` (Bonsai-2) | **Done** — patched SM89 Prism+DFlash2: **137.7 tok/s** mean (max 151.5, 98.6% accept); **1.86×** vs same-binary PQ2 baseline 74.2; stock Prism rejects draft GGUF |
| Embedded CUDA GELU-erf fusion | E2E parity on RTX 4090; wall speedup ≈1.02× at ≥16M elems — **not a claimed win** (`docs/benchmarks/cuda-fusion-rtx4090/`) |
| Embedded CUDA resident chains | Digest-fair e2e: **1.25×** (2g) → **2.12×** (4g) → **3.38×** (8g) at 4M elems; H2D/D2H 1/1 vs N/N |
| Prism upstream streaming + health TTL | **Done** — tool-free SSE; TTFT median 2318 ms (PTQ1) / **2107 ms (PQ2)** vs buffered walls; `/health` TTL 5s |
| Prism `kv4` profile | Short-chat decode ≈baseline on PQ2 (~76 tok/s); claim VRAM relief, not tok/s |
