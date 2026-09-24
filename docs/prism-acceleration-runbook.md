# Prism acceleration runbook (Power)

Operators serve Bonsai / Ternary packs through `PrismBackend` + an external Prism
`llama-server`. Acceleration is **profile-owned**, not Power `spec_mode`.

See also: [prism-acceleration-plan.md](prism-acceleration-plan.md),
[prism-backend-plan.md](prism-backend-plan.md).

## Profiles

| `prism_profile` | Intent | Upstream flags (via `tools/start-prism-upstream.ps1`) |
| --- | --- | --- |
| `baseline` (default) | Multi-turn / agentic; prompt-cache friendly | `-fa on -ngl 99 -np N` |
| `dspark` | Single-user DSpark decode peak | `-md <drafter> --spec-type draft-dspark --spec-draft-n-max 4 -np 1` |
| `mtp` | Single-user MTP (Bonsai-2 while DSpark absent) | `--spec-type draft-mtp --spec-draft-n-max N` on an MTP-capable GGUF |
| `dflash` | Community DFlash2 sidecar (Bonsai-2) | `-md <DFlash2.gguf> --spec-type draft-dflash -np 1` |
| `kv4` | Long-context VRAM relief | baseline + `--cache-type-k/v q4_0` |

**Serving tip (measured):** Bonsai-2 chat defaults burn tokens on thinking.
Pass `-Reasoning off -ReasoningBudget 0` for decode benches / code-only loads.
On RTX 4090 PTQ1 this alone lifts e2e from ~80 → **~85 tok/s**; combined with
MTP n-max=1 → **~90 tok/s** through Power.

**Do not** set Power `spec_mode = dspark|mtp|dflash` for Prism packs. That selects
the pinned `llamacpp` speculative adapters and is **fail-closed** on the prism
backend.

## Packing advisory (Tier 1)

| GPU class | Prefer (hypothesis until measured) |
| --- | --- |
| Ada (RTX 4090), L4 | `PTQ1_0` for decode |
| Hopper / Blackwell / Ampere | often `PQ2_0` for decode; PQ2_0 usually wins prefill |

**Measured on this repo’s RTX 4090 host (2026-09-24):** decode winner
`PTQ1_0` (81.4 vs 77.6 tok/s); prefill winner `PQ2_0`. Capture:
`docs/benchmarks/bonsai2-27b-packing-rtx4090/power-packing-compare-20260924-003352.json`.

Same-host evidence command:

```powershell
.\tools\run-prism-packing-compare-sequential.ps1 `
  -Ptq1ModelPath "<PTQ1_0.gguf>" -Pq2ModelPath "<PQ2_0.gguf>" `
  -BinDir "<prism-bin>" -OutDir docs\benchmarks\bonsai2-27b-packing-rtx4090
```

## Config sketch

```acl
host = "127.0.0.1"
port = 11435
prism_upstream = "http://127.0.0.1:8080"
prism_profile = "baseline"
# prism_profile = "dspark"
# prism_drafter = "D:/models/…-dspark-dflash-….gguf"  # required for dspark
# prism_profile = "mtp"   # Bonsai-2: in-file MTP graft; no prism_drafter
# prism_profile = "dflash"
# prism_drafter = "D:/models/…/Bonsai-2-27B-DFlash2-Q8_0.gguf"
tee_mode = false
spec_mode = "off"
```

## Start upstream

```powershell
.\tools\start-prism-upstream.ps1 -Profile baseline -Alias bonsai2-ptq1 `
  -Model "D:\Bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
  -BinDir "D:\Bonsai-demo\bin\cuda" -Port 8080 -Ctx 8192
```

For `dspark`, pass `-Drafter` to a target-matched `*dspark-dflash*.gguf`. **Bonsai 2
has no official DSpark file** on `prism-ml/Ternary-Bonsai-2-27B-gguf` (and
`Bonsai-demo/setup.ps1` sets `$drafterPattern = $null` for `bonsai2`).
`prism_profile=dspark` therefore **fails closed** without `prism_drafter`.

For Bonsai-2 speculation today, prefer **`prism_profile=mtp`** with an in-file
MTP graft (e.g. `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`) and reasoning off:

```powershell
.\tools\start-prism-upstream.ps1 -Profile mtp -Alias bonsai2-ptq1-mtp `
  -Model "…\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" `
  -BinDir "D:\Bonsai-demo\bin\cuda" -Port 8080 -DraftNMax 1 `
  -Reasoning off -ReasoningBudget 0
```

Optional community DFlash2 (not official DSpark). **Requires a patched
Prism+DFlash2 binary** (ProCreations `prism-dflash2-source.tar.gz` built for
local CUDA arch — SM89 on RTX 4090). Stock Prism rejects the draft GGUF
(tensor-count mismatch). Measured on this host: **~138 tok/s** mean
(`docs/benchmarks/bonsai2-27b-ptq1-rtx4090/`).

```powershell
.\tools\start-prism-upstream.ps1 -Profile dflash -Alias bonsai2-pq2-dflash `
  -Model "…\Ternary-Bonsai-2-27B-PQ2_0.gguf" `
  -Drafter "…\Bonsai-2-27B-DFlash2-Q8_0.gguf" `
  -BinDir "…\build-sm89\bin" -Port 8080 `
  -DraftNMax 3 -Reasoning off -ReasoningBudget 0
```

Do **not** point Bonsai-2 at a Bonsai-1 DSpark drafter — HF reports a large
decode regression (wrong target).

## Verify through Power

1. `POST /v1/models` register the GGUF path (filename must contain `PTQ1_0` / `PQ2_0` / Bonsai-2 markers).
2. `POST /v1/chat/completions` — response may include `prism_upstream` with
   `"source": "upstream-reported"` and llama-server `timings` (`predicted_per_second`,
   and `draft_n` / `draft_n_accepted` when DSpark is engaged).
3. Baseline bench (reproducible):

```powershell
.\tools\run-prism-baseline-bench.ps1 -Samples 3 -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090
```

4. Speculation compare (baseline vs mtp/dspark; requires matching upstream):

```powershell
.\tools\run-prism-spec-compare.ps1 -ProfileLabel mtp -Model bonsai2-ptq1-mtp `
  -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090
```

5. `prompt_cache_key` is rejected: Prism HTTP path does not implement Power’s
   keyed prefix-reuse contract. Enabling `dspark` / `mtp` also withdraws Prism’s
   own cross-request cache.

## Honesty rules

- `prism_upstream.timings` are **upstream-reported**, not Power-verified draft digests.
- Power does not claim MTP/DFlash speedups on Bonsai weights.
- Measure DSpark gains under `docs/benchmarks/bonsai2-…/` before calling a
  profile “recommended” on a given GPU.
