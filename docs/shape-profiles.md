# Model-Owned Shape Profiles

Power can validate a finite set of optimized execution profiles without
learning what a model's dimensions mean. The model crate computes an opaque
SHA-256 shape-class identity; Power checks that identity against declared
batch, tensor-element, and scratch bounds.

This contract is the same for language, vision, OCR, embedding, and multimodal
graphs. It contains no Qwen, OCR, tokenizer, image-width, or sequence-length
branch.

## Boundary

The model crate owns:

- the mapping from an input to an opaque shape-class SHA-256;
- padding, bucketing, valid extents, and every dimension's meaning;
- each optimized implementation artifact and its SHA-256;
- the dynamic implementation used when an allowed fallback is selected;
- the current TEE policy identity.

Power owns:

- a canonical declaration capped at 256 unique shape classes;
- positive batch and tensor-element limits;
- host/device scratch admission against the bound runtime reservation;
- exact binding to weights, reviewed graph, resolved device topology, runtime
  memory reservations, and TEE policy;
- deterministic selection or a typed, explicit dynamic-fallback reason;
- digest-only receipt evidence tied to the exact input.

Power never derives a width bucket, context class, image size, or model family
from a tensor. The caller supplies the opaque class identity chosen by its own
reviewed logic.

## Declaration and selection

For a single-device `GraphExecutor`, derive the binding from the actual loaded
graph instead of repeating its identities:

```rust
use a3s_power::inference::{
    DynamicShapeFallback, RuntimeMemoryReservations, ShapeProfile,
    ShapeProfileDeclaration, ShapeProfileRequest,
};

let reservations = RuntimeMemoryReservations::default()
    .with_host_scratch_bytes(64 * 1024 * 1024)
    .with_device_scratch_bytes(256 * 1024 * 1024);

let binding = graph.shape_profile_binding(reservations, tee_policy_sha256)?;
let profile = ShapeProfile::new(
    optimized_implementation_sha256,
    model_owned_shape_class_sha256,
    8,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
)?;
let declaration = ShapeProfileDeclaration::new(
    binding.clone(),
    vec![profile],
    DynamicShapeFallback::allow(dynamic_implementation_sha256)?,
)?;

let request = ShapeProfileRequest::new(
    &input_digest.sha256,
    model_owned_shape_class_sha256,
    batch_size,
    aggregate_tensor_elements,
)?;
let selection = declaration.select(&binding, &request)?;
let implementation_sha256 = selection.implementation_sha256();
```

`GraphExecutor::shape_profile_binding` derives the weight digest, a canonical
digest over every reviewed graph-identity field, the runtime device, and the
single-device topology. A model-owned multi-device executor uses
`ShapeProfileBinding::new` with its canonical device-topology digest.

Profiles are sorted by class digest before their declaration identity is
computed. Duplicate classes are rejected. Input order therefore cannot create
two identities for the same finite declaration.

## Dynamic fallback

Fallback is never implicit:

- `DynamicShapeFallback::Deny` fails when the class is absent or an aggregate
  bound is exceeded.
- `DynamicShapeFallback::allow(sha256)` records the exact dynamic
  implementation and one of `shape-class-unavailable`, `batch-bound-exceeded`,
  or `tensor-element-bound-exceeded`.

A stale binding is not a fallback condition. Any change to weights, graph,
runtime device, device topology, fixed/scratch reservations, or TEE policy
rejects the declaration before profile selection.

## Receipt evidence

`EmbeddedRuntime::receipt_with_shape_profile` creates receipt v5 directly.
`EmbeddedRuntime::attach_shape_profile` can add the same evidence to a receipt
that already carries accelerator or microbatch evidence.

The extension contains only:

- declaration, binding, request, weight, and input SHA-256 values;
- the typed runtime device;
- the selected profile and implementation SHA-256; or
- the dynamic implementation SHA-256 and typed fallback reason.

It does not contain the class digest, tensor shape, dimensions, batch size,
tensor-element count, filesystem paths, model family, or input values. Debug
output also redacts all hashes and execution-envelope values. Digests are
binding material, not anonymization; callers should not derive class hashes
from low-entropy private labels if a declaration may leave its trust boundary.

## Validation

Run the contract tests from the Power crate:

```bash
cargo test --locked --no-default-features --features embedded-inference \
  --lib shape_profile_tests
cargo clippy --locked --all-targets --no-default-features \
  --features embedded-inference -- -D warnings
```

Coverage includes canonical ordering, all three fallback reasons, fallback
denial, every stale-binding dimension, scratch overcommit, profile-count and
duplicate bounds, serialized tampering, representation-neutral selection,
input/model/runtime receipt binding, extension composition, privacy-safe
evidence, and `Send + Sync` public contracts.
