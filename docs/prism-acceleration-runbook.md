# Prism acceleration runbook (Power)

Operators serve Bonsai / Ternary packs through `PrismBackend` + an external Prism
`llama-server`. Acceleration is **profile-owned**, not Power `spec_mode`.

See also: [prism-acceleration-plan.md](prism-acceleration-plan.md),
[prism-backend-plan.md](prism-backend-plan.md).

## Profiles

| `prism_profile` | Intent | Upstream flags (via `tools/start-prism-upstream.ps1`) |
| --- | --- | --- |
| `baseline` (default) | Multi-turn / agentic; prompt-cache friendly | `-fa on -ngl 99 -np N` |
| `dspark` | Single-user decode peak | `-md <drafter> --spec-type draft-dspark --spec-draft-n-max 4 -np 1` |
| `kv4` | Long-context VRAM relief | baseline + `--cache-type-k/v q4_0` |

**Do not** set Power `spec_mode = dspark|mtp|dflash` for Prism packs. That selects
the pinned `llamacpp` speculative adapters and is **fail-closed** on the prism
backend.

## Packing advisory (Tier 1)

| GPU class | Prefer |
| --- | --- |
| Ada (RTX 4090), L4 | `PTQ1_0` for decode |
| Hopper / Blackwell / Ampere | often `PQ2_0` for decode; PQ2_0 usually wins prefill |

## Config sketch

```acl
host = "127.0.0.1"
port = 11435
prism_upstream = "http://127.0.0.1:8080"
prism_profile = "baseline"
# prism_profile = "dspark"
# prism_drafter = "D:/models/…-dspark-dflash-….gguf"  # required for dspark
tee_mode = false
spec_mode = "off"
```

## Start upstream

```powershell
.\tools\start-prism-upstream.ps1 -Profile baseline -Alias bonsai2-ptq1 `
  -Model "D:\Bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
  -BinDir "D:\Bonsai-demo\bin\cuda" -Port 8080 -Ctx 8192
```

For `dspark`, pass `-Drafter` to a target-matched `*dspark-dflash*.gguf`. As of
the Bonsai 2 PTQ1_0 pin used in this repo, an official Bonsai-2 DSpark drafter
was **not** published on Hugging Face — `prism_profile=dspark` therefore
**fails closed** until `prism_drafter` points at a real file.

## Verify through Power

1. `POST /v1/models` register the GGUF path (filename must contain `PTQ1_0` / `PQ2_0` / Bonsai-2 markers).
2. `POST /v1/chat/completions` — response may include `prism_upstream` with
   `"source": "upstream-reported"` and llama-server `timings` (`predicted_per_second`,
   and `draft_n` / `draft_n_accepted` when DSpark is engaged).
3. Baseline bench (reproducible):

```powershell
.\tools\run-prism-baseline-bench.ps1 -Samples 3 -OutDir docs\benchmarks\bonsai2-27b-ptq1-rtx4090
```

4. `prompt_cache_key` is rejected: Prism HTTP path does not implement Power’s
   keyed prefix-reuse contract. Enabling `dspark` also withdraws Prism’s own
   cross-request cache.

## Honesty rules

- `prism_upstream.timings` are **upstream-reported**, not Power-verified draft digests.
- Power does not claim MTP/DFlash speedups on Bonsai weights.
- Measure DSpark gains under `docs/benchmarks/bonsai2-…/` before calling a
  profile “recommended” on a given GPU.
