# dynamo-gwp

Global Workload Plane routes OpenAI-compatible requests across Dynamo ingress
deployments. Envoy and Alyx own client streams; GWP provides scheduling,
lifecycle accounting, health, planner reflection, and shared session affinity.

Alyx calls `KVLocatorService` on the existing `DYN_GWP_GRPC_PORT` listener
(default 8091). Native placement and Envoy RPCs share topology, accounting,
readiness, and shutdown handling.

This crate is an experimental proof of concept. Start with:

- [Setup and operations](../../docs/components/gwp/README.md)
- [Design](../../docs/design-docs/global-workload-plane-design.md)
- [Kubernetes deployment](../../deploy/gwp/kubernetes/README.md)
- [Live Composer PoC](../../deploy/gwp/poc/README.md)

## Code map

- `config`, `models` — endpoint routes, aliases, policies, and hot reload.
- `reflector`, `identity` — planner polling and worker ownership.
- `session`, `tokens` — affinity backends and real/pseudo tokenization.
- `router`, `core` — selection, booking, reconciliation, and lifecycle state.
- `grpc`, `server`, `kvlocator` — Envoy and native placement RPCs and process wiring.
- `metrics`, `lifecycle` — observability, readiness, drain, and cleanup.

## Build

```console
cargo build -p dynamo-gwp --features server,etcd,redis --bin dynamo-gwp
cargo test -p dynamo-gwp --features server,etcd,redis
```

Features are opt-in: `server` enables the Tonic entrypoint, while `etcd` and
`redis` enable legacy shared affinity stores (`redis` also enables MCM discovery).
The `pubsub` affinity backend needs neither feature for affinity. The sidecar image
enables all three:

```console
docker build -f lib/gwp/Dockerfile -t dynamo-gwp:dev .
```

Pseudo tokenization is the default. A real tokenizer bundle is a directory
containing `tokenizer.json`, `chat_template.jinja`, and
`tokenizer_config.json`. A `template_only` policy loads only
`chat_template.jinja` and `tokenizer_config.json` — it renders the chat
template for chat-completions but pseudo-tokenizes the rendered text, keeping
the template's effect on prefix identity without the real tokenizer's startup
and per-request cost. Tokenizer selection and affinity backend selection are
startup configuration because they define stable routing identities.

```yaml
tokenization:
  models:
    glm-5.2:
      mode: template_only
      directory: /workspace/lib/gwp/vendored_tokenizers/glm5.2
```

## Call GWP with grpcurl

GWP exposes gRPC v1 reflection for native placement, lifecycle, and health on
port 8091. Use a recent `grpcurl`; no local proto files or generated client are
needed. For Kubernetes, pin one pod so lifecycle calls use the same replica:

```bash
kubectl -n gwp port-forward pod/<gwp-pod> 8091:8091
```

In another terminal:

```bash
# List services and native placement methods.
grpcurl -plaintext localhost:8091 list
grpcurl -plaintext localhost:8091 list dynamo.b10.generation.v1.KVLocatorService

# Inspect a request and print a JSON template.
grpcurl -plaintext -msg-template localhost:8091 describe dynamo.b10.generation.v1.LocateRequest

# Check whether this replica is ready for placement.
grpcurl -plaintext -d '{"service":"gwp-readiness"}' localhost:8091 grpc.health.v1.Health/Check

# Read-only placement lookup. Replace model and endpoint with configured IDs.
grpcurl -plaintext -d '{
  "modelId": "composer-2-5",
  "bid": {"tokens": [1, 2, 3, 4], "sessionId": "manual-test"},
  "eligibleEndpointIds": ["prod-fast"]
}' localhost:8091 dynamo.b10.generation.v1.KVLocatorService/Locate

# Export the protocol, including imported message definitions, for offline use.
grpcurl -plaintext -protoset-out /tmp/gwp.protoset localhost:8091 describe dynamo.b10.generation.v1.KVLocatorService
```

`Locate` needs ready, eligible workers to return candidates; reflection itself
is available while the process is warming up. `CommitIntend`,
`ResponseOpenOutcome`, and `ResponseFinish` use the same JSON command pattern;
inspect their request types with `describe`. These methods change accounting:
use a fresh attempt UUID per dispatch and the same pod for its lifecycle calls.

## MCM discovery through Redis

Set `discovery.type: mcm_redis` to use the Redis view already maintained by
MCM/Beefeater's Kafka consumers. This adapter feeds the existing GWP topology
independently of the request transport. It publishes complete worker snapshots
from Redis and never starts the workload-plane planner poller. The default
remains `discovery.type: planner`; select that backend explicitly to retain
direct planner discovery.

Set `discovery.namespace` to track every BIS deployment in one org namespace
(`deploy/gwp/mcm/gwp.example.yaml`). Every `rescan_interval_secs` (default 30)
GWP scans `rt:*` and keeps the route tables in that namespace whose deployment
publishes LSM samples. Each (deployment, workload plane) becomes endpoint
`<deployment>/<plane>`, routed under model `<deployment>`, and `endpoints` and
`routes` must not be configured. KVLocator callers send the deployment as
`model_id`; an empty `eligible_endpoint_ids` means every plane, and placements
name the bare plane. Unchanged topology is republished once per rescan interval.

For each tracked deployment it pipelines these reads in the background:

- `GET rt:<model_deployment_id>` for MCM's `RouteTable` protobuf JSON.
- `HGETALL bis:<model_deployment_id>:v1:metrics` for relayed LSM samples.
- Optionally, `HGETALL bis:<model_deployment_id>:v1:metrics:observed` for the
  newer samples Alyx has collected through responses.

Without `discovery.namespace`, the endpoint directory is operator
configuration: list `endpoints` and `routes`, and map every endpoint under
`discovery.endpoints` to one deployment and workload plane. Its canonical GWP
model ID must equal `model_deployment_id`; an optional per-endpoint `namespace`
additionally checks route and sample identity.

A destination must have an MCM route, fresh positive capacity, and usable worker
IDs in a fresh relayed sample.
Weighted routes require positive weight and replicas other than explicit zero;
fill-first routes only need to list the plane, because their replica counts lag
migrations; fresh relayed samples decide where workers are. This
adapter determines membership; it does not implement MCM traffic weights or
request-specific region/compliance policy. Existing GWP routing-requirement
filtering and worker scoring still apply.

The newest valid relayed/observed sample supplies capacity and drain markers.
Identity mismatches, malformed
samples, stale timestamps and regressing revisions cannot restore a removed
route. A missing route, expired hash field, Redis error or read timeout removes
the affected destinations from the published topology. Redis failures withdraw
all MCM destinations until a successful refresh. These empty snapshots propagate
to readiness; dispatchers can fall back to their existing routing. Sample age
is measured from LSM's timestamp, so reading a delayed Kafka event does not make
it fresh again.

Only the relayed sample's `workerIds` supplies inventory: Alyx's response-header
samples intentionally omit those IDs. Relayed inventory has independent age and
monotonicity checks, so fresh observed capacity cannot revive stale or removed
workers. A newer relayed sample replaces the previous set, including removals.
Missing or empty IDs yield no candidates: older LSM feeds that omit inventory
cannot be used for MCM discovery. IDs must be decimal u64 values; IDs advertised
by multiple endpoints are excluded. `drainingWorkers` maps to the existing drain
taints, whose routing behavior is controlled by `router_honor_draining`.

The current LSM publisher includes prefill and combined prefill/decode workers
in `workerIds`, and excludes decode-only workers. This is the candidate inventory
for prefill placement; it does not provide a separate decode candidate set or
per-worker DP/runtime metadata.

MCM discovery does not require per-worker load details. GWP uses its scheduler's
local and replicated attempt accounting for load estimates; no endpoint-level
capacity is divided into invented per-worker baselines. Inventory refreshes do
not reset that accounting. Planner URLs and credentials may be omitted in MCM
mode, and `ingress_url` is needed only for endpoints Envoy routes to.

Build with `server,redis,etcd`. `deploy/gwp/mcm` builds on the existing
GWP/Envoy deployment, supplies `DYN_GWP_MCM_REDIS_URL` from the
`dynamo-gwp-mcm` Secret's `redis-url` key, and needs the GWP and Envoy ConfigMaps and
service resources. Populate the GWP ConfigMap using the example adapted for
your deployment. Build the GWP image from this revision; see
`deploy/gwp/kubernetes/README.md` for the remaining deployment setup. Discovery
configuration is startup-only. No new Kafka consumer group is required. GWP's
existing etcd peer discovery and ZMQ/NATS scheduler synchronization remain
separate from MCM destination discovery.

Observe `gwp_discovery_polls_total{result="ok"|"error"}` and
`gwp_discovery_eligible_endpoints`, together with standard readiness and planner
metrics when using planner discovery. The Redis integration test uses an
isolated database and checks that MCM mode never contacts a planner:

```sh
DYN_GWP_TEST_REDIS=redis://127.0.0.1:6379/ cargo test -p dynamo-gwp \
  --features server,redis,etcd --lib b10_discovery:: -- --include-ignored
```

## Coordinator-backed session affinity

Set `session.backend.type: pubsub` to use Dynamo's `AffinityCoordinator`.
Confirmed successful placements refresh a local, bounded, TTL-based binding and
publish it through the existing event transport. Session IDs over 256 bytes are hashed
to 64-character internal keys; shorter and forwarded IDs remain unchanged. Locate reads do not refresh TTL.
The configured `session.ttl_secs` is fixed at startup for this backend.

Affinity updates use the same configured ZMQ or NATS transport as scheduler events.
etcd remains responsible for replica discovery. Redis is unnecessary for affinity;
MCM discovery, if enabled, still needs its separate Redis view.

Affinity is best effort: updates may be dropped and restarting replicas start with
empty bindings until new placements arrive. No durable snapshot or distributed
delete is provided. Existing worker liveness and model eligibility checks still
apply. Bindings retain worker/rank; optional previous-endpoint migration metrics
are unavailable. This does not provide cross-replica attempt accounting.
