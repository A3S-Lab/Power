# Prism Backend Inference Acceleration — First-Principles Plan

Status: **Phase A0–A3 implemented** (profiled upstream + telemetry + fail-closed;
DSpark measured compare deferred until Bonsai-2 drafter pin). Phase A4 optional.  
Audience: Power maintainers deciding *how* (and whether) to accelerate Bonsai / Prism packs behind `PrismBackend`  
Related: [`prism-acceleration-runbook.md`](prism-acceleration-runbook.md), [`prism-backend-plan.md`](prism-backend-plan.md), `src/backend/prism/`, `src/speculative.rs`, PrismML `Bonsai-demo` (`SPECULATIVE.md`, `start_llama_server.ps1`), Power Qwen speculative benches under `docs/benchmarks/`

Baseline today (evidence): Phase-1 `PrismBackend` proxies OpenAI chat to an external Prism `llama-server`, refuses Power `spec_mode` ≠ `auto`/`off` via `ensure_non_speculative_mode`, and does not own kernels, drafts, or KV. Local smoke on RTX 4090 used `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~84 tg128 tok/s without Power-side speculation).

---

## 0. Objective

Define an executable acceleration architecture for the **Prism capability** in Power so that Ternary / Bonsai packs (`PTQ1_0`, `PQ2_0`, …) can get *honest, measured* decode/prefill gains **without**:

1. Contaminating Power’s pinned `llamacpp` MTP / DFlash / DSpark patch stack  
2. Claiming Power speculative speedups on weights that only Prism kernels can run  
3. Lying in receipts about prompt digests, draft acceptance, or TEE residency  

This document answers: *what accelerates Bonsai under Power, who owns each lever, and in what order.*

Implementation starts only after Phase A0 gates pass.

---

## 1. First-principles gate (Rule 0–2)

### 1.1 Mission fit

| Question | Answer |
| --- | --- |
| What is Power’s mission? | Model-neutral, bounded, verifiable **inference execution boundary** — admission, placement, API, receipts, optional TEE — not ownership of every kernel or draft graph. |
| What does “accelerate Prism” mean for that mission? | Raise useful tokens/s (and/or cut TTFT / VRAM) for Prism-required packs **behind the same Power contract**, with fail-closed capability advertising. |
| Is the problem real? | **Yes.** Baseline Bonsai decode is already useful (~80–90 tg on 4090 for PTQ1_0), but users will ask for Power’s speculative wins. Those wins today are on **ordinary GGUF + pinned llamacpp**, not Prism packs. Separately, Prism itself ships DSpark / packing / FA / KV4 levers that Power currently neither configures nor reports. |
| Does acceleration strengthen architecture? | **Only if acceleration stays backend-owned** (Prism runtime + Prism-compatible drafter), and Power core only gains: profile selection, admission, telemetry honesty, and fail-closed strategy resolution. |
| Simpler alternative? | Run `BONSAI_SPECULATIVE=1` on Bonsai-demo’s `llama-server` and point clients there. That accelerates decode; it skips Power’s boundary. Acceptable for raw benches; insufficient if the product claim is “Power serves accelerated Bonsai.” |

**Gate verdict:** Proceed with a **Prism-owned acceleration ladder**. Refuse designs that (a) feed Bonsai into Power MTP/DFlash validators, (b) merge Prism into pinned `llama-cpp-2`, or (c) advertise `spec_mode = dspark` globally when only the Prism upstream actually drafts.

### 1.2 Core vs extension

| Piece | Classification | Rule |
| --- | --- | --- |
| Strategy enum, capability resolve, receipts, OpenAI surface | **Core** | May learn a *Prism-scoped* capability bit; must not hard-code Prism tensors |
| Packing choice (`PTQ1_0` vs `PQ2_0`), FA, ngl, ctx, mmproj | **Extension (runtime profile)** | Owned by Prism process / future in-process backend |
| Prism DSpark drafter (`*dspark-dflash*`, `--spec-type draft-dspark`) | **Extension (Prism speculative)** | Never validated by Power’s `llamacpp-external-draft` adapters without a Prism adapter |
| Power MTP / DFlash / DSpark on pinned llamacpp | **Existing extension** | Stay isolated; do not compose with Prism packs |
| Prompt-cache / keep-alive / concurrency admission | **Core plumbing + backend policy** | Only claim what the Prism path actually guarantees |

### 1.3 What “acceleration” is *not*

- Not “make Power’s speculative bench numbers apply to Bonsai.”  
- Not “one `spec_mode` string that silently means different engines.”  
- Not TEE confidentiality for out-of-process Prism.  
- Not free lunch: Prism DSpark **disables cross-request prompt-cache reuse** and forces `-np 1` (single slot) — multi-turn agentic chat may get *worse* TTFT.

---

## 2. Constraint inventory (evidence)

### 2.1 Power side (current tree)

- `PrismBackend` (`src/backend/prism/`): HTTP forwarder; `supports_manifest` only for Prism-required GGUF; `ensure_non_speculative_mode` on load/chat/complete.  
- `SpeculativeStrategy` (`src/speculative.rs`): model-neutral control plane (`auto` / `off` / `prompt-lookup` / `ngram-context` / `draft-model` / `mtp` / `dflash` / `dflash2` / `dspark`). Backends advertise `SpeculativeCapabilities` and fail closed.  
- Proven Power speculative wins (ordinary Qwen GGUF on RTX 4090) live under `docs/benchmarks/qwen3.8-27b-q6k-rtx4090/` — **different weight lineage**, different backend.  
- Prompt-cache contract: backends that cannot guarantee keyed prefix reuse must report `PromptCacheSupport::Unsupported` (current Prism path).  
- Receipt honesty: proxy/upstream paths must not claim in-process effective prompt digests or draft digests they did not verify.

### 2.2 Prism side (product reality)

From PrismML Bonsai-demo (`SPECULATIVE.md`, server scripts, published packing notes):

| Lever | Effect | Cost / constraint |
| --- | --- | --- |
| **Packing: PTQ1_0 vs PQ2_0** | Ada / L4 often prefer **PTQ1_0** decode; Hopper / Blackwell / Ampere often prefer **PQ2_0**; prefill usually favors PQ2_0 | Wrong packing is a silent latency tax, not a quality gate |
| **Flash Attention (`-fa on`)** | Standard decode/prefill win on CUDA | Already default in demo server scripts |
| **Full offload (`-ngl 99`)** | Keeps layers on GPU | VRAM bound; mmproj adds ~0.6–0.9 GiB unless CPU-projected |
| **Prism DSpark** (`-md …dspark-dflash… --spec-type draft-dspark --spec-draft-n-max 4`) | ~**1.4–2.4×** decode on CUDA for code/math (workload-dependent); temp=0 lossless vs non-spec | Disables cross-request prompt cache; `-np 1`; needs roomy `-c` (16k+); drafter is **target-specific**; can *slow* some GPUs (e.g. Spark) |
| **KV4 (`BONSAI_KV4`)** | Shrinks KV for long context | Slight decode tax; optional mean-centering bias |
| **Continuous batching (`-np > 1`)** | Multi-request throughput | **Incompatible with DSpark profile** as shipped |
| **Vision mmproj** | Enables multimodal | Prefill cost; optional `--no-mmproj-offload` |

Local host evidence (this repo’s prior smoke): RTX 4090 + PTQ1_0 ≈ **84 tg128 / ~1520 pp512** without DSpark. DSpark drafter for Bonsai **2** was **not** present in the local `27B` directory at planning time — acceleration planning must treat drafter provisioning as a hard dependency, not an assumed file.

### 2.3 Hard incompatibilities

| Concern | Implication |
| --- | --- |
| Two llama.cpp lineages | Power MTP/DFlash patches ≠ Prism DSpark packaging; do not auto-compose |
| Same word “DSpark” | Power `spec_mode = dspark` means *pinned llamacpp* adapter; Prism `--spec-type draft-dspark` is upstream-owned. Names must not collide in config/docs |
| Prompt cache vs speculation | Choosing DSpark **trades** multi-turn TTFT for single-request decode |
| Receipt claims | Upstream `timings.draft_n` is evidence of Prism speculation; Power must not invent `LoadedSpeculativeArtifact` digests unless it verified the drafter bytes |

---

## 3. Acceleration taxonomy (what can actually get faster)

```text
                    ┌─────────────────────────────────────┐
                    │  Useful tokens / s  (or lower TTFT) │
                    └─────────────────────────────────────┘
                      ▲                ▲                ▲
         packing/FA/ngl│    Prism DSpark│   Power boundary│
                      │                │                │
              ┌───────┴──────┐  ┌──────┴───────┐  ┌─────┴──────────┐
              │ Runtime profile│  │ Speculative  │  │ Admission / IO │
              │ (always on path)│  │ (opt-in)     │  │ (Power-owned)  │
              └───────────────┘  └──────────────┘  └────────────────┘
```

### Tier 0 — Already “acceleration” relative to stock llama.cpp

Running Bonsai **at all** on Prism kernels is the primary win vs. gibberish/reject on mainline. Do not sell Tier 0 as Power speculative work.

### Tier 1 — Runtime profile (no Power speculative semantics)

Pick packing for the GPU class; FA on; sensible `-c` / `-ngl`; optional KV4 / mmproj placement.  
**Owner:** Prism process flags (today) or future `PrismBackend` supervisor.  
**Power role:** document + optionally select a named **profile**; do not invent kernels.

### Tier 2 — Prism DSpark (true speculative acceleration for Bonsai)

Opt-in upstream (or in-process later) with content-addressed drafter identity.  
**Owner:** Prism runtime.  
**Power role:** profile switch, fail-closed strategy resolve, optional telemetry passthrough, honest receipts.

### Tier 3 — Power-boundary latency (not token/s kernels)

Warm upstream, keep-alive of admitted models, streaming passthrough, concurrency limits that match `-np`, avoiding double JSON buffering.  
**Owner:** Power.  
**Honest claim:** “lower overhead / better utilization,” not “2× decode.”

### Explicitly out of scope for Prism packs

Power `mtp` / `dflash` / `dflash2` / pinned-llamacpp `dspark` artifacts and benches.

---

## 4. Options (ranked)

### Option A — Profiled upstream acceleration (recommended Phase A1)

**Mechanism:** Keep `PrismBackend` as HTTP boundary. Introduce named **Prism runtime profiles** that describe how the *external* `llama-server` was started (or how a supervisor must start it):

| Profile | Intent | Typical flags (non-normative) |
| --- | --- | --- |
| `baseline` | Multi-turn / agentic; prompt-cache friendly | `-fa on -ngl 99 -np N` ; no `-md` |
| `dspark` | Single-user decode peak (code/math) | `-md <drafter> --spec-type draft-dspark --spec-draft-n-max 4 -np 1` ; larger `-c` |
| `kv4` | Long-context VRAM relief | `--cache-type-k/v q4_0` (± bias) on top of baseline |

Power config points at one upstream URL **per profile** *or* a supervisor launches the matching argv. `spec_mode` stays `auto`/`off` for the proxy backend unless/until a Prism-scoped capability exists.

| Pros | Cons |
| --- | --- |
| Matches shipped MVP; zero kernel merge | Acceleration lives outside the Power process |
| Matches Bonsai-demo’s own trade-offs | Dual ports / process supervision complexity |
| Clear docs: “Power fronts accelerated Prism” | Receipts remain proxy-honest |

**Fits first principles:** Yes.

### Option B — Capability-honest Prism speculative control plane (Phase A2–A3)

**Mechanism:** Extend `PrismBackend` (still out-of-process) to:

1. Advertise a **backend-local** capability (e.g. resolve only `auto`→`off` or a dedicated `prism-dspark` profile key — **not** global `spec_mode = dspark`).  
2. Pass through / record upstream `timings` (`predicted_per_second`, `draft_n`, `draft_n_accepted`) into observation / optional receipt fields marked **upstream-reported**.  
3. Optionally register a `LoadedSpeculativeArtifact` **only when** Power hashed the drafter file and the supervisor confirmed `-md` identity.

| Pros | Cons |
| --- | --- |
| Stops lying by omission; enables dashboards | Requires careful receipt schema so “upstream-reported” ≠ “Power-verified” |
| Enables fail-closed if user asks for Power DFlash on a Prism model | More API surface |

**Fits first principles:** Yes — capability boundary stays honest.

### Option C — In-process Prism + native Prism DSpark (Phase A4, optional)

**Mechanism:** Future in-process `PrismBackend` (see `prism-backend-plan.md` Phase 2) implements `SpeculativeCapabilities` for **Prism DSpark only**, with its own artifact validators (filename/digest/contract), never reusing Power’s llamacpp external-draft parsers unchanged.

| Pros | Cons |
| --- | --- |
| Stronger digests; single process | Dual llama.cpp build matrix; pin cadence |
| Can expose health speculative metrics like llamacpp | Large engineering cost |

**Fits first principles:** Yes, as extension.

### Option D — Map Power `spec_mode = dspark|mtp|dflash` onto Prism packs (rejected)

| Why rejected |
| --- |
| Different draft graphs, different pins, different acceptance contracts |
| Would either no-op silently or crash — both violate fail-closed |
| Contaminates core strategy meaning across backends |

### Option E — “Just enable prompt-lookup / ngram in Power for PrismProxy” (rejected for now)

Zero-weight strategies require **Power-owned** draft/verify loops over model logits. The HTTP Prism path does not expose next-token logits. Implementing prompt-lookup at the proxy would mean a second decoder — nonsense. Revisit only with in-process Prism logits.

---

## 5. Recommended architecture

```text
Client
  │
  ▼
Power core (admission, auth, OpenAI, receipts, strategy resolve)
  │
  ├─ llamacpp / mistralrs     ← ordinary GGUF + Power MTP/DFlash/DSpark
  │
  └─ PrismBackend
         │  profile = baseline | dspark | kv4
         │  spec_mode = auto|off  (until Prism-scoped capability ships)
         ▼
   Prism llama-server (pinned release)
         │  packing PTQ1_0|PQ2_0, FA, optional -md dspark-dflash
         ▼
   Bonsai / Ternary GGUF (+ optional mmproj / drafter)
```

### 5.1 Config sketch (non-normative)

```acl
prism_upstream = "http://127.0.0.1:8080"
# Future:
# prism_profile = "baseline"   # or "dspark" | "kv4"
# prism_supervisor {
#   runtime_dir = "~/.a3s/power/runtimes/prism-b10709"
#   model = "…/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
#   drafter = "…/…-dspark-dflash-….gguf"   # required iff profile = dspark
# }
```

Do **not** overload `proxy_upstreams` or global `spec_mode = dspark` for this.

### 5.2 Capability matrix (honest defaults)

| Claim | baseline profile | dspark profile | in-process Prism DSpark (future) |
| --- | --- | --- | --- |
| Chat/complete | Yes | Yes | Yes |
| Cross-request prompt cache | Upstream yes (typical) | **No** (Prism constraint) | Backend policy |
| Multi-request `-np>1` | Yes | **No** | Policy |
| Power MTP/DFlash | No | No | No |
| Prism DSpark decode gain | No | Yes (measured) | Yes (measured) |
| `LoadedSpeculativeArtifact` Power-verified | No | Only if supervisor hashed drafter | Yes |
| Upstream timings in observation | Optional | Recommended | Native |
| TEE / sealed KV | No | No | Default no |

### 5.3 Measurement contract (definition of “faster”)

Every acceleration claim for Prism under Power must publish:

1. **Host:** GPU SKU, driver, Prism binary digest / release tag  
2. **Weights:** target SHA-256 + packing; drafter SHA-256 if used  
3. **Profile:** baseline vs dspark vs kv4  
4. **Workload class:** chat vs code vs math (DSpark is workload-sensitive)  
5. **Metrics:** tg / pp (or request-wide tok/s), and for DSpark: `draft_n`, `draft_n_accepted`, accept rate  
6. **Parity:** temp=0 exact match vs baseline for at least one fixed prompt when claiming lossless speculation  

Store under `docs/benchmarks/bonsai2-…/` (new), **not** under Qwen speculative trees.

---

## 6. Phased delivery

### Phase A0 — Decision & honesty (no code required beyond docs)

**Deliverables**

- [x] This plan: Tier taxonomy; Option D/E rejected; Option A→B→C sequence  
- [x] Explicit product copy: “Power does not apply MTP/DFlash to Bonsai; use Prism profiles” (`docs/prism-acceleration-runbook.md`)  
- [x] Confirm local/CI drafter availability for Bonsai **2**: **unsupported** until an official `*dspark-dflash*` pin exists — `prism_profile=dspark` fails closed without `prism_drafter`

**Exit evidence**

- Plan reviewed; non-goals in §7 unchanged

### Phase A1 — Profiled upstream (shippable acceleration)

**Work**

1. [x] Document operator runbooks: baseline vs DSpark behind `prism_upstream` (`docs/prism-acceleration-runbook.md`).  
2. [x] Supervisor script: `tools/start-prism-upstream.ps1` launches pinned Prism argv for a chosen profile; health-gates remain Power’s upstream `/health` check.  
3. [x] Packing advisory: default PTQ1_0 on Ada (4090); allow PQ2_0 override for prefill-heavy hosts (runbook table).  
4. [x] Smoke: same Power chat API baseline timings measured (3-sample scripted bench ≈81 tok/s mean); DSpark before/after **deferred** — no Bonsai-2 official drafter pin (`tools/run-prism-baseline-bench.ps1`, `docs/benchmarks/bonsai2-27b-ptq1-rtx4090/`).

**Exit evidence**

- Scripted before/after on reference GPU  
- README pointer: Prism acceleration profiles  
- No change to default `llamacpp` speculative CI

### Phase A2 — Telemetry & fail-closed control

**Work**

1. [x] Surface upstream `timings` into OpenAI response as `prism_upstream` (`source: upstream-reported`) and accumulate `speculative_metrics` when `draft_n` is present.  
2. [x] If `spec_mode` requests Power MTP/DFlash while using prism backend → **fail closed** with actionable error pointing at `prism_profile`.  
3. [x] `prism_profile` + `prism_drafter` ACL keys.

**Exit evidence**

- Unit/integration tests for fail-closed strategy on Prism manifests  
- One captured response JSON showing `draft_n` passthrough when profile=dspark

### Phase A3 — Prompt-cache honesty for baseline profile

**Work**

1. [x] Prism HTTP path keeps `PromptCacheSupport::Unsupported` (keyed `prompt_cache_key` rejected by API).  
2. [x] Document that enabling DSpark profile **withdraws** any cache claim (runbook + `PrismProfile::allows_cross_request_prompt_cache`).

**Exit evidence**

- Contract tests: cache key rejected on dspark profile; behavior documented for baseline

### Phase A4 — In-process Prism DSpark (optional)

**Work**

1. Only after `prism-backend-plan.md` Phase 2 in-process backend exists.  
2. Prism-specific draft artifact identity + `SpeculativeCapabilities` bit distinct from llamacpp DSpark.  
3. Bench note under `docs/benchmarks/` with parity + accept rate.

**Exit evidence**

- Load PTQ1_0 + drafter only on prism backend  
- Health endpoint lists Prism speculative artifact digests  
- Zero regressions in llamacpp speculative suites

---

## 7. Non-goals

- Applying Power MTP / DFlash / DFlash2 / pinned-llamacpp DSpark to Bonsai / Prism packs  
- Merging PrismML/llama.cpp into the pinned `llama-cpp-rs` used by `llamacpp`  
- Advertising a single global `spec_mode = dspark` that silently switches engines  
- Claiming TEE / sealed-memory speedups for proxied Prism  
- Treating DSpark as a multi-tenant default (single slot + no prompt cache)  
- Shipping Open WebUI / Bonsai-demo UI inside Power  
- Declaring victory from Qwen speculative benches without a Bonsai-specific capture

---

## 8. Acceptance gates

| Gate | Phase | Evidence |
| --- | --- | --- |
| G1 First principles | A0 | This doc; Option D/E rejected in writing |
| G2 Baseline profile documented | A1 | Runbook + packing advisory |
| G3 DSpark / Bonsai-2 speculation | A1 | **Bonsai-2 DSpark deferred** (no official pin; fail-closed). **DFlash peak 137.7 tok/s** e2e on patched SM89 Prism+DFlash2 (1.86× vs PQ2 same-binary). Stock Prism MTP fallback **89.9 tok/s**. |
| G4 Fail-closed Power speculative | A2 | Prism manifest + `spec_mode=mtp|dflash|…` errors |
| G5 Receipt/telemetry honesty | A2 | Upstream timings labeled; no false Power draft digests |
| G6 No llamacpp regress | A1–A4 | Existing speculative CI green without Prism feature coupling |
| G7 Optional in-process | A4 | Digested drafter + parity capture |

---

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Name collision “DSpark” | Separate docs/config keys (`prism_profile=dspark` vs `spec_mode=dspark`); never alias |
| Missing Bonsai-2 drafter | Provision pin or keep profile unavailable fail-closed |
| Users enable DSpark for multi-turn agents | Default profile = baseline; warn in docs and supervisor |
| Hardware where DSpark regresses | Require measured gate before calling a profile “recommended” |
| Operators point `prism_upstream` at a baseline server while believing DSpark is on | Health/observation should expose draft counters; mismatch → warn |

---

## 10. Decision summary

| Choice | Verdict |
| --- | --- |
| Accelerate Prism under Power? | **Yes**, as backend/runtime profiles — not as Power kernel work |
| First ship | **Option A** profiled upstream (`baseline` / `dspark` / `kv4`) |
| Next | **Option B** telemetry + fail-closed control honesty |
| Later | **Option C** in-process Prism DSpark only with its own artifact contract |
| Power MTP/DFlash on Bonsai | **Rejected** |
| Global `spec_mode` overload | **Rejected** |

Primary acceleration today for a 4090 host already using PTQ1_0: keep **Tier 1 packing/FA**, then measure **Tier 2 Prism DSpark** behind the same Power API once the Bonsai-2 drafter is pinned — without touching the pinned llamacpp speculative stack.
