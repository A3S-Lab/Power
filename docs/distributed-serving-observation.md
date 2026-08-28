# Distributed Serving Worker Observation

Power exposes execution facts; it does not decide where a request runs. The
`worker` object in `GET /health` is the versioned boundary used by a certified
control plane to build a scheduling snapshot for Gateway.

The current schema is `a3s.power.worker-observation.v1`. Each observation is a
closed JSON object containing:

- a random process epoch and a monotonic observation generation;
- observation and expiry timestamps;
- supported and currently ready execution phases;
- active and waiting request counts plus the configured active limit;
- aggregate prompt-cache support, occupancy, capacity, and bounded pressure;
- state-transfer capability and health.

The default server advertises only the `aggregated` phase. A composition root
may inject a typed `StateTransferService`; a valid adapter projects its
state-transfer capability and current health, while a missing, invalid, or
`unsupported` adapter fails closed. Transfer readiness does not add `prefill`
or `decode` to the execution phase set: those phases require a separate,
verified phase executor.

## State-transfer port

Power owns a model-neutral, pull-oriented transfer lifecycle:

1. the decode side prepares an exact destination state handle;
2. the prefill side publishes an exact source state handle against that target;
3. the decode side consumes the published source; and
4. either side can abort the bounded attempt and reclaim adapter state.

Targets, sources, and receipts bind one transfer ID, both process epochs, model,
execution and state-layout SHA-256 identities, state kind, token count, byte
count, protocol, and a maximum five-minute expiry. Adapter-owned connection
metadata is carried only in a trimmed, control-free, 16 KiB ticket. Local model
state handles are not serializable and redact their debug representation.

The protocol distinguishes direct device-memory pull from buffered host-memory
pull without making the Power core depend on NIXL, UCX, libfabric, or another
transport library. A concrete adapter owns memory registration, transport
timeouts, integrity, and cleanup. Gateway selects and orchestrates workers;
Cloud certifies compatible deployment generations. Neither receives KV bytes.

This port does not by itself activate request-level P/D execution. The default
Power backends do not inject a transfer service yet, and Gateway must continue
to reject P/D scheduling until the selected deployment truthfully reports a
compatible, ready adapter and the corresponding phase executor is installed.

## Immutable execution profile

Power accepts exactly one `serving_execution` block from A3S ACL. Omitting the
block is equivalent to the safe `aggregated` default. There is deliberately no
environment-variable override for this policy-bearing value.

A `prefill-decode` profile selects one local role and binds the exact model,
backend artifact, backend-owned phase execution contract, device declaration,
state layout, certified peer set, deployment generation, transfer protocol,
state kind, byte and concurrency limits, operation and cancellation deadlines,
privacy policy, and optional attestation policy. All SHA-256 values are
canonical lowercase hex. The profile has its own stable digest; an injected
adapter must report that digest in `execution_profile_sha256`, and the same
profile digest is included in Power's canonical inference-execution policy.

~~~acl
serving_execution {
  profile = "prefill-decode"
  role = "decode"
  model = "internal/model-v1"
  model_sha256 = "<64 lowercase hex characters>"
  backend = "llama.cpp"
  backend_sha256 = "<64 lowercase hex characters>"
  execution_sha256 = "<backend-owned phase contract SHA-256>"
  device_sha256 = "<device declaration SHA-256>"
  layout_sha256 = "<model-owned state layout SHA-256>"
  peer_set_sha256 = "<Cloud-certified peer set SHA-256>"
  generation = 7
  protocol = "direct-device-memory-pull-v1"
  state_kind = "kv-cache"
  max_state_bytes = 8589934592
  max_inflight_transfers = 32
  transfer_timeout_ms = 30000
  cancellation_timeout_ms = 5000
  privacy = "authenticated-encrypted-transport"
  privacy_policy_sha256 = "<64 lowercase hex characters>"
  attestation_policy_sha256 = "<64 lowercase hex characters>"
}
~~~

Static profile parsing and attestation binding do not make the phase runnable.
Startup remains fail-closed until the composition root supplies both an exact
profile-bound transfer adapter and a verified phase executor.

## Ownership boundary

The observation contains no target, deployment, replica, tenant, credential,
prompt, cache key, token, KV byte count, model identifier, or unbounded metric
label. Cloud owns the association between an observation and a rollout unit,
and Gateway accepts only a complete, unexpired snapshot with that association.

Power remains responsible for local admission, execution, cache/state safety,
and truthful measurement. Gateway owns request admission and endpoint choice;
Cloud owns placement, rollout, desired replicas, and autoscaling. Consumers
must fail closed when the schema, epoch/generation binding, transfer
compatibility, phase readiness, or expiry cannot be verified.

## Configuration

`worker_observation_ttl_seconds` controls the exclusive validity interval. Its
default is 15 seconds and the closed ACL validator accepts only `1..=300`.
This is a freshness bound, not a polling interval or a lease: no consumer may
extend it locally.

## Compatibility

The `worker` field is additive to the existing health response. New semantics
require a new schema value; existing fields in v1 are not repurposed. The
process epoch changes on restart and the generation increases for every
observation within that epoch, so consumers can reject replay and regression.
