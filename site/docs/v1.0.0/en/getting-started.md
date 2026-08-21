---
title: Getting started
description: Choose an A3S Power surface, embed the bounded runtime, or start the OpenAI-compatible service.
---

# Getting started

Power has two inference entry points and one independent artifact surface. Pick
the narrowest boundary your product needs.

| If your product... | Start with |
| --- | --- |
| Owns a reviewed model graph inside a Rust crate | `embedded-inference` |
| Needs a hosted chat, completion, or embedding API | Default `server,mistralrs` profile |
| Installs an exact revision-locked artifact bundle | `artifact-provisioning` |
| Runs a layer-streaming GGUF service inside a constrained enclave | `tee-minimal` |

## 1. Embed the runtime

Add only the listener-free inference surface:

```toml
[dependencies]
a3s-power = { version = "1.0.0", default-features = false, features = ["embedded-inference"] }
```

Construct a runtime with an explicit device preference and resource limits:

```rust
use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};

fn main() -> Result<(), a3s_power::error::PowerError> {
    let runtime = EmbeddedRuntime::new(
        DevicePreference::Auto,
        InferenceLimits::default(),
    )?;

    println!("execution device: {}", runtime.device().name());
    Ok(())
}
```

Creating `EmbeddedRuntime` does not bind a socket, start a listener, download a
model, or invoke another process. The model crate supplies its reviewed graph
and retains semantic state; Power supplies the execution boundary.

Continue with [Architecture](/architecture) to see the ownership contract.

## 2. Run the hosted service

Install the default service profile and bind it to loopback:

```bash
cargo install a3s-power
a3s-power serve --host 127.0.0.1 --port 11434
```

In another terminal, pull and open a small GGUF model:

```bash
a3s-power models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
a3s-power chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
```

Model manifests and content-addressed blobs live under `~/.a3s/power` by
default. Set `A3S_POWER_HOME` to choose another store.

## 3. Send an OpenAI-compatible request

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Explain capability-based security."}],
    "stream": true
  }'
```

Chat and completion responses include an `attestation_receipt` and its SHA-256
digest. Streaming responses emit the receipt before `[DONE]`.

## 4. Make policy explicit

The service reads A3S ACL from `~/.a3s/power/config.acl`, or from the path
passed to `a3s-power serve --config`.

```text
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
keep_alive = "5m"

flash_attention = true
num_parallel = 1

gpu {
  gpu_layers = -1
  main_gpu = 0
}
```

Invalid ACL, ranges, strategies, hashes, or unsupported explicit backends fail
before inference. Production TEE policy adds verifier-owned measurements and
model hashes; see [Verification](/verification).

## Next steps

- [Understand the execution and ownership boundary](/architecture)
- [Inspect measured Qwen3.8 performance and quality evidence](/performance)
- [See how speculative decoding preserves exact target authority](/speculative-decoding)
- [Choose a backend and production build profile](/operations)
