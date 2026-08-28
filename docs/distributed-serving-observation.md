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

Power currently advertises only the `aggregated` phase. State transfer is
reported as unsupported until a model/backend adapter implements and validates
the opaque transfer contract. Publishing phase enum values does not claim that
prefill/decode disaggregation is available.

## Ownership boundary

The observation contains no target, deployment, replica, tenant, credential,
prompt, cache key, token, KV byte count, model identifier, or unbounded metric
label. Cloud owns the association between an observation and a rollout unit,
and Gateway accepts only a complete, unexpired snapshot with that association.

Power remains responsible for local admission, execution, cache/state safety,
and truthful measurement. Gateway owns request admission and endpoint choice;
Cloud owns placement, rollout, desired replicas, and autoscaling. Consumers
must fail closed when the schema, epoch/generation binding, phase readiness, or
expiry cannot be verified.

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
