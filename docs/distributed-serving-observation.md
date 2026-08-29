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

The default server advertises only the `aggregated` phase. A disaggregated
composition root must inject both a typed `StateTransferService` and a typed
`ServingPhaseExecutor`, and both must bind the exact execution-profile digest
and configured role. Only that complete pair projects `prefill` or `decode` as
a capability. The phase is ready only while both services can accept work. A
missing, mismatched, invalid, or unsupported member fails closed without
falling back to an aggregated capability that the process was not configured
to execute.

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
state handles are not serializable, and both local handles and wire tickets
redact their debug representation.

Server composition wraps every injected data path in
`BoundedStateTransferService`. The wrapper projects only the configured local
role and ACL limits even when the underlying transport supports more. It binds
commands to the current random worker epoch, admits no more than the configured
in-flight count, makes identical prepare/publish retries idempotent, retains
registered leases until consume or compensating abort, and reaps them at their
monotonic deadline without requiring another request. Operation drop, timeout,
explicit abort, and invalid adapter output all trigger abort under the separate
cancellation timeout. An unconfirmed cleanup marks the wrapper unavailable for
the rest of the process generation. Its snapshot contains only bounded,
content-free counters.

The protocol distinguishes direct device-memory pull from buffered host-memory
pull without making the Power core depend on NIXL, UCX, libfabric, or another
transport library. A concrete adapter owns memory registration, transport
integrity, and driver cleanup. The Power wrapper owns the common admission,
deadline, lease, and cleanup-verification policy. Gateway selects and
orchestrates workers; Cloud certifies compatible deployment generations.
Neither receives KV bytes.

When both ports are injected, the composition root assembles one
`DistributedServingRuntime`; AppState and worker observation retain that runtime
as the single source of distributed readiness. It reserves one bounded local
lease per execution ID, prepares a decode destination before state movement,
publishes prefill state only after phase execution, consumes verified state
before decode execution, and owns cancellation until the returned stream ends.
Expiry, caller cancellation, non-ready decisions, invalid adapter output, and
explicit abort all trigger compensating phase and transfer cleanup. An
unconfirmed cleanup permanently suppresses new work for that process epoch.

The default Power backends inject neither port, and this repository does not yet
ship a concrete distributed backend/transport pair. The internal request-flow
endpoint is present but fails closed without a matching composed runtime.
Gateway must therefore continue to reject P/D dispatch until a selected
deployment truthfully supplies that complete path.

## Cross-process conformance boundary

`tests/distributed_serving_cross_process.rs` provides executable evidence for
the process boundary itself. The parent test launches independent prefill and
decode Power HTTP processes, acts only as the request orchestrator, and passes
the opaque target and source tickets unchanged. A test-only backend owns the
fixture state and phase semantics; a test-only buffered-host adapter moves that
state through an authenticated AES-GCM loopback channel. The suite verifies a
successful decode stream, explicit cleanup, peer disappearance during transfer,
process restart, and rejection of the stale worker epoch.

This is conformance evidence for Power's composition and wire contracts. The
fixture adapter is not exported by the library, is not a high-speed network
implementation, and supplies no model-semantic parity evidence. A deployment
must still inject and certify a concrete model/backend adapter and a transport
that enforces the selected privacy profile before it advertises prefill/decode
readiness.

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
api_keys = ["<Gateway service-key SHA-256>"]

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

## Phase-executor port

`ServingPhaseExecutor` is the backend-owned boundary for one configured
`prefill` or `decode` role. It receives the existing local Power chat or text
request, but request values and process-local execution handles are not
serializable and their debug representation is redacted. The backend adapter,
not Power, owns tokenization, KV or recurrent layout, phase arithmetic,
reservation state, response generation, and cleanup.

The lifecycle separates preparation from execution. Preparation validates the
model, process epoch, profile digest, role, and an expiry no longer than the ACL
transfer timeout before any response bytes can be generated. A prefill
execution produces only an opaque local state handle plus the exact bounded
state binding; it is not transferable until the independent transfer adapter
publishes it. A decode preparation names the exact destination handle and
binding. `ImportedModelState` can be constructed only after the consume command
and receipt prove the same destination process, source, binding, protocol,
byte count, integrity result, and completion within the current local attempt.
Power bounds consumption by the profile timeout and invokes adapter abort under
the separately bound cancellation timeout before reporting a timeout.

Every pre-response operation returns one closed `PhaseDecision`: `ready`,
`recompute`, `retryable-unavailable` with an optional bounded delay, or
`terminal-failure`. Recompute and failure reasons are closed enums rather than
backend text. A transfer receipt proves state movement only; decode succeeds
only when the backend executor subsequently returns a response stream.

Phase abort accepts either an in-progress preparation without a backend handle
or a completed reservation with its opaque handle. Prepared phase lifetimes
must exactly equal the caller-bound execution lifetime; an adapter cannot
silently shorten or extend a distributed request lease.

## Internal request-flow protocol

Gateway calls the service-to-service boundary under
`/internal/v1/distributed-serving`. It is not a public OpenAI API and receives
no permissive CORS policy or public request-rate bucket. All four operations
require `Authorization: Bearer <key>` backed by `api_keys`; a
`prefill-decode` configuration with no keys is rejected at startup, and the
middleware still fails closed when a state is assembled manually without an
authentication provider.

The request schema is `a3s.power.distributed-serving.v1` and exposes four
`POST` operations:

- `/decode/prepare` reserves decode execution and returns a bounded opaque
  `StateTransferTarget`;
- `/prefill/execute` runs the selected model-owned prefill phase against that
  exact target and returns a bounded opaque `StateTransferSource`;
- `/decode/execute` consumes the source and produces
  `application/x-ndjson`; and
- `/abort` idempotently reclaims phase and transfer ownership.

Every document binds one non-nil execution ID, the selected process epoch, and
the exact execution-profile SHA-256. Schema mismatch, stale process identity,
profile mismatch, malformed JSON, missing runtime, and runtime failures use
closed, content-free error codes and messages. Bodies are capped at 8 MiB.
The request payload carries a closed chat-completions or completions envelope;
the presentation layer validates the supported OpenAI fields and translates it
into the existing process-local `PhaseRequest`. Neither serializable domain
commands nor a second inference request type are introduced.
If prefix reuse is requested, Gateway must replace the caller's cache key with
`a3s-gw-pcache-v1:<64 lowercase hex characters>` after domain-separating its
tenant/authentication identity, endpoint, model, and caller key. Power rejects
raw or malformed cache keys at this boundary so one Gateway service credential
cannot collapse multiple tenant cache domains.

Decode output uses `a3s.power.distributed-serving-stream.v1`. The first frame is
`ready`, followed by monotonically sequenced response chunks and exactly one
`completed` or `failed` terminal frame. The HTTP body polls the backend-owned
stream directly, preserving backpressure. Client disconnect drops the guarded
stream and triggers the same bounded compensating cleanup as local
cancellation. Wire DTO debug output redacts prompts, generated content, and
adapter tickets; golden fixtures under `tests/fixtures/distributed-serving-v1`
pin the JSON shape consumed by Gateway.

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
