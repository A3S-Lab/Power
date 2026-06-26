# a3s-power — SAE Interpretability Integration Plan

Make the in-enclave SAE tap operational: during generation, tap the residual stream, encode it with a
Sparse Autoencoder, and emit a **confidential** `LlmActivations` NDJSON line (feature ids/activations
only — no prompt/completion text) that a3s-sentry's `SaeJudge` scores. White-box (judges the model's
internal concepts), confidential (TEE — only features leave), explainable (linear in named features).

## Status

- ✅ **`src/sae.rs`** — `SaeEncoder` (residual → sparse TopK/ReLU features) + `InterpTap` (encodes a
  layer's `hidden`, emits the confidential `LlmActivations` line). Unit-tested (4 tests).
- ✅ **a3s-sentry side** — `SaeJudge` tier + `Event::LlmActivations` + `Decision.explain` + the `sae{}`
  ACL block + pipeline routing. Tested (49 tests), pushed (`A3S-Lab/Sentry` `feat/sae-interpretability`).
- ⏳ **This plan** — wire the tap into the forward pass, load a real SAE, emit, attest.

## Steps

### 1. SAE weight loading (artifact)
- On-disk format: `safetensors` (or a simple binary) carrying `n_embd`, `n_features`, `top_k`, the
  JumpReLU threshold, `W_enc [n_features×n_embd]`, `b_enc [n_features]`.
- Add `SaeEncoder::from_safetensors(path)`. The SAE is **model+layer specific** → name it
  `<model>.L<layer>.sae`.
- The trained SAE is an offline ML artifact (see §6).

### 2. Config + build (which layer, which SAE, opt-in)
- New config block, **default OFF** (opt-in like the privacy gates), e.g.:
  ```hcl
  interp { sae = "models/minimax.L18.sae"  layer = 18  top_k = 32  enabled = true }
  ```
- At model load, build an `Option<InterpTap>` (encoder + layer) and thread it into `GenerateParams`
  (+ an `interp_sink: Option<Sender<String>>` for the emitted lines).
- Enable per-agent / per-request (only monitored agents pay the ~one-matmul cost).

### 3. Forward-pass tap (the hook) — `src/backend/picolm.rs`
- The residual stream is `let mut hidden = vec![0.0f32; n_embd]` in `forward_pass_streaming`
  (≈ line 525), updated per layer inside `for layer in 0..cfg.n_layers` by attention + ffn.
- **Insert the tap inside the layer loop, right after `hidden` is updated for the chosen layer:**
  ```rust
  if let Some(tap) = params.interp_tap.as_ref() {
      if layer as u32 == tap.layer() {
          if let Some(line) = tap.observe(pid, layer as u32, &hidden, agent, session) {
              if let Some(sink) = params.interp_sink.as_ref() { let _ = sink.send(line); }
          }
      }
  }
  ```
- **Token granularity:** tapping every generated token costs one encode/token. Recommend
  **one event per generation** — accumulate the completion tokens' residuals at layer L and emit once
  (last-token, or mean-pooled). Keeps volume sane + matches "one LlmIo per call".
- `pid` / `agent` / `session` come from the request context (the server already has the identity).

### 4. Emission sink (out of the enclave, confidential)
- The tap returns the NDJSON line; route it to the sentry-facing side channel (mirror the observer's
  NDJSON-to-sink pattern, or a dedicated `interp` stream the sentry daemon / AnySentry ingests).
- **Confidential by construction:** the `LlmActivations` struct has no text fields (the sae.rs test
  asserts no `prompt`/`completion` leaks). Prompts stay in the TEE on the response/audit path.

### 5. Attestation (verifiable interpretability)
- Seal/measure the SAE weights + layer config alongside the model via the existing
  `tee/model_seal.rs` + attestation chain; include the SAE digest in the attestation report.
- → a client can cryptographically prove **which** interpretability model scored the output.

### 6. ML track (offline, parallel)
- Add a debug/export mode that dumps layer-L residual activations from a3s-power.
- Train a TopK/JumpReLU SAE on harvested activations (or adopt a community SAE for the base family).
- Probe + label safety features → the feature dictionary JSON that sentry's `SaeJudge` loads
  (`sae { dict = "features.json" }`). Causal-validate each label (ablate feature → score moves).

## Dependencies / risks
- **picolm forward realism** — parts of picolm's math may still be stub; tapped activations are only
  meaningful once the forward is real. The **mistralrs** backend (candle) is real but needs
  hidden-state exposure (a separate hook) — do picolm first (it's our code, `hidden` is right there).
- **Per-model coupling** — the SAE is per checkpoint; version it; re-train on model swap; monitor
  calibration drift.
- **Cost** — gate per-request, tap one layer, prefer one event per generation.

## Build order
1. `SaeEncoder::from_safetensors` + `interp{}` config + `Option<InterpTap>` at model load.
2. picolm forward-pass hook (§3) + emission sink (§4).
3. E2E test: load a toy SAE, run a generation, assert an `LlmActivations` line is emitted and parses
   in a3s-sentry's `ObservedEvent::parse` → `SaeJudge`.
4. Attestation (§5).
5. mistralrs backend hook.
6. ML track (§6): train + label the real SAE → drop the dict into sentry's `sae{}` config.

Each step ships value; the chain is end-to-end after step 3 (with a toy SAE), production-grade after §6.
