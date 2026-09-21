# Prism Backend for A3S Power — First-Principles Plan

Status: **Phase 1 MVP implemented** (`PrismBackend` + `prism_upstream` config; verified
chat against local Bonsai 2 `PTQ1_0` via external Prism `llama-server`)  
Audience: Power maintainers deciding how (and whether) to host PrismML / Bonsai inference  
Related: `src/backend/prism/`, `src/backend/mod.rs` (`Backend` trait), `src/backend/proxy.rs`, `src/backend/llamacpp.rs`, PrismML `Bonsai-demo` / `PrismML-Eng/llama.cpp`

---

## Implementation notes (2026-09)

Shipped MVP (proxy-style dedicated backend, not a kernel merge):

- `src/backend/prism/` — detects Prism-required packs (`PTQ1_0` / `PQ2_0` / Bonsai-2
  filename markers), admits them via `supports_manifest`, forwards OpenAI chat /
  completions to a Prism `llama-server`.
- Generic `mistralrs` / `llamacpp` backends refuse Prism packs (`supports_manifest`).
- Config: `prism_upstream = "http://127.0.0.1:8080"` (or env `A3S_PRISM_UPSTREAM`).
- Verified: `POST /v1/chat/completions` through `a3s-power` → Prism upstream →
  `Ternary-Bonsai-2-27B-PTQ1_0.gguf` returned `Paris` for a capital-of-France prompt.

Not done (later phases): in-process Prism link, process supervision, mmproj wiring
through Power, TEE claims for proxied Prism.

**Acceleration planning:** see [`prism-acceleration-plan.md`](prism-acceleration-plan.md)
(profiled upstream DSpark / packing / fail-closed speculative control — separate from
this admission plan).

---

## 0. Objective

Define an executable architecture for adding a **Prism** capability to Power so that Ternary / Bonsai GGUF packs (`PTQ1_0`, `PQ2_0`, and future Prism-required bands) can be served through Power’s OpenAI-compatible boundary **without** contaminating the existing pinned `llama.cpp` / speculative stack.

This document is the decision record. Implementation starts only after Phase 0 gates pass.

---

## 1. First-principles gate (Rule 0–2)

### 1.1 Mission fit

| Question | Answer |
| --- | --- |
| What is Power’s mission? | Model-neutral, bounded, verifiable **inference execution boundary**: admission, placement, API, receipts, optional TEE — not ownership of every kernel family. |
| Does Prism serve that mission? | **Yes, as an accelerator backend.** Bonsai 2 delivers 27B-class quality at ~6 GB; Power’s job is to admit/serve that capability behind the same contract as `mistralrs` / `llamacpp` / `proxy`. |
| Is the problem real? | **Yes.** Stock / Power-pinned llama.cpp cannot correctly run Bonsai 2 (`PTQ1_0`/`PQ2_0` need Prism kernels; lookalike `Q2_0` can silently gibberish). Users already run Prism out-of-band (`Bonsai-demo`). |
| Does it strengthen architecture? | **Only if Prism stays a backend capability**, not a fork-merge into the existing `llamacpp` feature. |
| Simpler alternative? | Point clients at Prism `llama-server` alone. That skips Power’s boundary (auth, admission, receipts, registry). Acceptable for demos; insufficient if the product goal is “Power hosts low-bit local models.” |

**Gate verdict:** Proceed. Refuse only the designs that merge Prism into the pinned `llamacpp` revision or claim TEE confidentiality for proxied Prism.

### 1.2 Core vs extension

| Piece | Classification | Rule |
| --- | --- | --- |
| `Backend` trait, registry, admission, receipts, OpenAI surface | **Core** | Unchanged by Prism |
| Prism Hadamard / ternary kernels, GGUF type IDs, mmproj path | **Extension (backend-owned)** | Never in Power core |
| Existing MTP / DFlash / DSpark patches on pinned `llama-cpp-rs` | **Existing extension** | Must remain isolated from Prism |
| Routing that selects Prism for Bonsai manifests | **Core plumbing + backend policy** | Use `supports_manifest`, not format-only `Gguf` |

### 1.3 What Power must *not* own

- Prism kernel source or binary distribution licensing beyond pinning a reviewed artifact identity
- Claiming “Power speculative decoding accelerates Bonsai” unless Prism’s own draft path is explicitly wired and measured
- Treating all `ModelFormat::Gguf` as interchangeable

---

## 2. Constraint inventory (evidence)

### 2.1 Power side (current tree)

- Backends implement `Backend` (`load` / `unload` / `chat` / `complete` / `embed` + optional System One, prompt cache, speculative telemetry).
- Registry is priority-ordered; `supports_manifest` exists so **architecture-specific GGUF backends can coexist** with generic GGUF backends (`docs/embedded-inference-architecture.md`).
- `ModelFormat::Remote` + `ProxyBackend` already front OpenAI-compatible upstreams (documented as the “absorb vLLM without reimplementing kernels” path).
- `llamacpp` is pinned to a specific `llama-cpp-rs` revision with Power-owned speculative patches. That stack is **not** Prism.

### 2.2 Prism side (product reality)

- Bonsai 2 language packs require **PrismML-Eng/llama.cpp** (or Bonsai-demo’s pinned release binaries).
- Safe failure: `PTQ1_0` / `PQ2_0` rejected by mainline as unknown types.
- Dangerous failure: some `Q2_0` Prism-dev packs load on mainline and emit garbage — Power must **refuse** those on non-Prism backends.
- Capabilities already in Prism `llama-server`: chat, vision (mmproj), tools, thinking / reasoning budget, optional Prism DSpark speculative path.
- Sampling defaults differ by mode (thinking vs instruct); GGUF metadata carries recommended sampling.

### 2.3 Hard incompatibility

| Concern | Implication |
| --- | --- |
| Two llama.cpp lineages | Cannot share one `llama-cpp-2` link unit without ABI / patch collision |
| Speculative contracts | Power MTP/DFlash ≠ Prism DSpark packaging; do not auto-compose |
| Receipt honesty | Proxy cannot claim effective prompt digests or in-process KV metrics unless upstream exposes them |

---

## 3. Options (ranked)

### Option A — Proxy Prism `llama-server` (Phase 1 product)

**Mechanism:** Register Bonsai as `ModelFormat::Remote` (or a dedicated remote alias) with `proxy_upstreams[model] = http://127.0.0.1:8080`. Optionally add a thin lifecycle helper that starts/stops a pinned Prism binary.

| Pros | Cons |
| --- | --- |
| Zero kernel merge; reuses `ProxyBackend` | Out-of-process; weaker receipt / no TEE |
| Ships value in days | Process supervision, port, health checks are extra |
| Matches Power’s stated “front accelerated engines” pattern | Prefill/decode metrics depend on upstream |

**Fits first principles:** Yes — kernels stay outside; Power owns the door.

### Option B — Dedicated `prism` / `PrismBackend` feature (Phase 2)

**Mechanism:** New Cargo feature `prism` (name TBD) that links **only** Prism’s llama.cpp (sys crate or process-isolated FFI). Implements `Backend`. `supports_manifest` returns true **only** for Prism-required GGUF (detect via quant type metadata / filename contract / explicit manifest marker). Generic `llamacpp` / `mistralrs` must return false for those manifests.

| Pros | Cons |
| --- | --- |
| In-process load, stronger digests, native mmproj wiring | Build matrix (CUDA/Metal), binary size, dual llama.cpp CI |
| Clean fail-closed routing | Ongoing pin of Prism release tags |
| Can later expose Prism DSpark as backend-local speculative | Must not reuse Power’s `llamacpp-external-draft` types without a Prism adapter |

**Fits first principles:** Yes — extension backend, core unchanged.

### Option C — Replace / merge Prism into existing `llamacpp` (rejected)

| Why rejected |
| --- |
| Destroys Power’s reviewed speculative patch set |
| Couples release cadence to Prism |
| Violates “backends are capabilities” and Rule 2 (core pollution) |

### Option D — Wait for upstream Hadamard (watch only)

Track Prism → mainline landing. When mainline can run Bonsai 2 safely, reconsider collapsing into `llamacpp`. Until then, Option D is **not** a delivery plan.

---

## 4. Recommended architecture

```text
Client
  │
  ▼
Power core (admission, auth, OpenAI API, receipts)
  │
  ├─ mistralrs / llamacpp / picolm     ← ordinary GGUF / tensors
  ├─ ProxyBackend (Phase 1)            ← Prism llama-server upstream
  └─ PrismBackend (Phase 2, optional)  ← Prism kernels in-process
         │
         ▼
   PrismML llama.cpp (pinned release)
         │
         ▼
   Bonsai / Ternary GGUF (+ optional mmproj)
```

### 4.1 Manifest identity (mandatory)

Do **not** route by `ModelFormat::Gguf` alone.

Minimum detection contract (any one may be used; prefer layered fail-closed):

1. **Explicit marker** in manifest / ACL: `runtime = "prism"` or `family = "bonsai2"` (caller-owned, clearest).
2. **GGUF probe:** reject load on `llamacpp` when tensor types include Prism-only enums (`PTQ1_0`, `PQ2_0`) or declared Hadamard rotation metadata.
3. **Filename allowlist** is insufficient alone (rename attacks); use as UX hint only.

`LlamaCppBackend::supports_manifest` and `MistralRsBackend::supports_manifest` must return **false** for Prism-required packs so registry order cannot steal the model.

### 4.2 Capability matrix (honest defaults)

| Capability | Phase 1 Proxy | Phase 2 In-process |
| --- | --- | --- |
| Chat / complete | Yes (upstream) | Yes |
| Vision (mmproj) | Yes if server started with mmproj | Yes if wired |
| Tools / thinking | Passthrough | Map to Prism flags |
| Power MTP/DFlash | No | No (unless separate Prism draft contract) |
| Prism DSpark | Upstream flag only | Optional backend feature |
| Effective prompt digest | Absent unless upstream API | Implement when local render exists |
| TEE / memory load | No | Default no; fail closed |
| System One logits | No unless upstream exposes | Only if Prism API allows |

### 4.3 Artifact provisioning

- Prefer content-addressed blobs under Power’s model store with SHA-256 in the manifest (same as other GGUF).
- Pin Prism **runtime** separately: release tag + platform digest (e.g. `prism-b10709-…-win-cuda-13.3`) recorded in config / lockfile — same spirit as Power’s llama.cpp pin.
- Never download F16 reference weights by default.

### 4.4 Config sketch (non-normative)

```acl
# Phase 1 — proxy
proxy_upstreams {
  "bonsai2-27b" = "http://127.0.0.1:8080"
}

# Phase 2 — example shape only
# prism {
#   enabled = true
#   runtime_dir = "~/.a3s/power/runtimes/prism-b10709"
#   ngl = 99
# }
```

Exact ACL schema is an implementation task; the plan only requires: **upstream map** (P1), **runtime pin + model marker** (P2).

---

## 5. Phased delivery

### Phase 0 — Decision & safety (no merge)

**Deliverables**

- [ ] Written acceptance of Option A → B sequence; Option C rejected
- [ ] Threat note: silent gibberish if Bonsai hits non-Prism backend
- [ ] Choose pin policy: Bonsai-demo release tag vs PrismML-Eng/llama.cpp tag

**Exit evidence**

- This plan reviewed; owners recorded; non-goals listed in §6 unchanged

### Phase 1 — Proxy integration (shippable)

**Work**

1. Document how to register a Bonsai remote model + `proxy_upstreams`.
2. Optional: `a3s-power prism serve` / sidecar supervisor that launches pinned `llama-server` with reviewed flags (`-ngl`, `-c`, mmproj, reasoning budget).
3. Smoke: OpenAI chat through Power → Prism returns correct short answer; timings visible.
4. Fail-closed test: registering the same GGUF as local `Gguf` on `llamacpp` must error **before** generate.

**Exit evidence**

- Automated or scripted smoke against local Prism
- Receipt does not claim in-process digest / TEE
- README / docs page: “Prism via proxy”

### Phase 2 — In-process `PrismBackend` (optional productization)

**Work**

1. Feature-gated crate path `backend/prism/` implementing `Backend`.
2. Isolated build: do not link Power’s `llama-cpp-2` and Prism sys in one default feature set if symbols clash; prefer mutually exclusive features or dynamic load.
3. `supports_manifest` detection + registry registration **after** mistralrs but selected by manifest match (registry must use manifest-aware find everywhere — already required by architecture docs).
4. Map chat streaming, unload, cleanup; vision via mmproj artifact fields already on `ModelManifest`.
5. Speculative: either `ensure_non_speculative_mode` or a Prism-specific draft contract — **no** reuse of Power DFlash validators without a Prism adapter.
6. CI matrix: at least one CUDA Linux job with a tiny Prism-compatible fixture or skip-gated live job with pinned model hash.

**Exit evidence**

- Load PTQ1_0 or PQ2_0 succeeds only on `prism` backend
- Load attempt on `llamacpp` fails closed with actionable error
- Bench note on reference GPU (pp512 / tg128) recorded under `docs/benchmarks/`
- No change to default `llamacpp` speculative CI

### Phase 3 — Convergence watch

- When Prism Hadamard lands upstream and Power’s pin can absorb it without breaking speculative patches, revisit collapsing backends.
- Until then, keep dual stacks.

---

## 6. Non-goals

- Merging PrismML/llama.cpp into the pinned `bogdanr/llama-cpp-rs` revision used by `llamacpp`
- Promising Power’s MTP/DFlash speedups on Bonsai weights
- TEE / sealed-memory claims for proxied Prism
- Bundling Open WebUI / Bonsai-demo UI into Power
- Supporting Prism-dev `Q2_0-prism-fork-required` on mainline backends “for convenience”
- Auto-pulling multi-tens-of-GB F16 reference GGUFs

---

## 7. Acceptance gates (definition of done)

| Gate | Phase | Evidence |
| --- | --- | --- |
| G0 Architecture decision recorded | 0 | This doc + owner sign-off |
| G1 Proxy path works | 1 | Chat via Power → Prism correct; process documented |
| G2 Fail-closed routing | 1–2 | Test: Prism pack rejected by `llamacpp`/`mistralrs` |
| G3 In-process optional | 2 | Feature `prism` loads PTQ1_0/PQ2_0; unload clean |
| G4 No regress | 2 | Existing `llamacpp` speculative / System One tests still green without `prism` |
| G5 Honest receipts | 1–2 | Proxy receipts omit false digests; in-process digests only when local render exists |

---

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Dual llama.cpp CI cost | Phase 1 only until demand justifies Phase 2; Phase 2 feature-gated offline by default |
| Prism release churn | Pin tag + sha256; Bonsai-demo as compatibility reference |
| Registry steals model | Manifest-aware `supports_manifest`; add negative tests |
| Users confuse Power DSpark with Prism DSpark | Separate config keys and docs; fail closed on mismatched draft artifacts |
| Windows CUDA toolkit variance | Follow Bonsai-demo’s cudart co-install pattern for sidecar |

---

## 9. Immediate next engineering tickets (when implementation starts)

1. **P1-docs:** “Serving Bonsai via Power proxy” + ACL example  
2. **P1-guard:** GGUF quant-type probe rejecting Prism packs on `llamacpp` load  
3. **P1-smoke:** Scripted Power↔Prism chat smoke (reuse local `D:\Bonsai-demo` or CI service)  
4. **P2-spike:** Build Prism as isolated sys crate / dynamic library; prove no symbol clash with `llamacpp`  
5. **P2-backend:** `PrismBackend` MVP (load + chat stream + unload)  
6. **P2-routing:** Manifest marker + registry tests  

---

## 10. Summary decision

| Decision | Choice |
| --- | --- |
| Integrate Prism? | **Yes**, as backend capability |
| First ship | **Option A (Proxy)** |
| Product hardening | **Option B (dedicated PrismBackend)** |
| Merge into `llamacpp`? | **No** |
| Core changes | Registry/manifest routing + docs/config only |

Power remains the bounded door. Prism remains the kernel owner for Bonsai ternary packs.
