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

## MCM discovery through Redis

Set `discovery.type: mcm_redis` to use the Redis view already maintained by
MCM/Beefeater's Kafka consumers. This adapter feeds the existing GWP topology
independently of the request transport. The default remains
`discovery.type: planner`.

For each configured deployment it pipelines these reads in the background:

- `GET rt:<model_deployment_id>` for MCM's `RouteTable` protobuf JSON.
- `HGETALL bis:<model_deployment_id>:v1:metrics` for relayed LSM samples.
- Optionally, `HGETALL bis:<model_deployment_id>:v1:metrics:observed` for the
  newer samples Alyx has collected through responses.

The endpoint directory and credentials remain operator configuration. Each
endpoint maps explicitly to one deployment and workload plane; its canonical
GWP model ID must equal `model_deployment_id`. Optional `namespace` additionally
checks both route and sample identity. Every configured endpoint needs a
mapping. See `deploy/gwp/mcm/gwp.example.yaml`.

A destination must have both an MCM route and fresh positive capacity.
Weighted routes require positive weight and replicas other than explicit zero;
fill-first routes require positive replicas and container concurrency. This
adapter determines membership; it does not implement MCM traffic weights or
request-specific region/compliance policy. Existing GWP routing-requirement
filtering and worker scoring still apply.

The newest valid relayed/observed sample wins. Identity mismatches, malformed
samples, stale timestamps and regressing revisions cannot restore a removed
route. A missing route, expired hash field, Redis error or read timeout removes
the affected destinations from the published topology. Redis failures withdraw
all MCM destinations until a successful refresh. These empty snapshots propagate
to readiness; dispatchers can fall back to their existing routing. Sample age
is measured from LSM's timestamp, so reading a delayed Kafka event does not make
it fresh again.

Planner polling continues to supply worker inventory and detailed load.
The current MCM capacity message lacks those details; its empty `workerIds`
field is not interpreted as an authoritative empty worker set. A Redis refresh
does not reset the scheduler's load-observation anchors. Extending LSM's worker
snapshot later can replace this planner source without changing request handlers.

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
metrics. The Redis integration test uses an isolated database:

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
