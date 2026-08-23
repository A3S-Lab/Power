# Reviewed llama.cpp patch set

Power pins `llama-cpp-rs` at
`dfd12e4d334846367e4284a2a7763fe92c1bf676`, whose nested llama.cpp revision
is `e79e4bf660e19f2ad851e06c6913f7a8c5852621`. The files in this directory
extend only that exact source pair. The installer rejects every other revision.

Apply the set with:

```powershell
cargo fetch --locked
.\tools\apply-llamacpp-power-patches.ps1
```

The installer applies the binding, external-draft, DFlash2 runtime, MTP/FR,
and CUDA-priority patches in dependency order. It is idempotent and invalidates
cached `llama-cpp-2` and `llama-cpp-sys-2` artifacts whenever it changes the
checkout; otherwise Cargo can reuse a binary compiled before the patches.
Custom target directories populated before patching must also be rebuilt.

The DFlash2 runtime port is limited to the inference changes from upstream
llama.cpp commits `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4` and
`1deefcca395743049c3820ab8f9b15043f3e9446`. Conversion scripts, server UI,
and unrelated upstream changes are intentionally excluded. The separate
binding patch adds a typed discriminator and rejects DFlash v1/DFlash2
contract mismatches before native execution.

Do not apply these patches partially or copy them to a different dependency
revision. Update the pinned revisions, regenerate the complete patch set, and
repeat the native correctness and performance gates as one change.
