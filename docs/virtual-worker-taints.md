# Virtual worker taints

`b10_routing_config.virtual_taint_worker_fractions` partitions each router's discovered workers
into disjoint pools without changing worker registrations. Workers continue to run the
same engine configuration. Existing required/preferred taints control request routing.

Example with the Baseten Python frontend's header mapping:

```yaml
b10_routing_config:
  virtual_taint_worker_fractions:
    b10.virtual/fast: 0.2
    b10.virtual/throughput: 0.8
b10_priority_config:
  enabled: true
  service_tier_header: x-request-priority  # Set to the caller's actual header name.
  header_service_tier_mapping:
    interactive: fast
    background: flex
    system_background: flex
  service_tier_taint_mapping:
    fast:
      preferred_taints:
        b10.virtual/fast: 0.5
    flex:
      preferred_taints:
        b10.virtual/throughput: 1.0
```

Fractions specify the share of workers in each pool, independently of the request's
`preferred_taints` scores. Values are finite positive relative weights and are
normalized automatically: 2/8, 0.2/0.8, and 20/80 all target 20% fast and 80%
throughput over many worker identities. They do not need to sum to 1 and may exceed 1.
These are expected shares, not exact worker counts. Small
fleets can differ substantially from those fractions, and any pool can be empty.
Preferences allow overflow; a required taint with no eligible worker can queue or reject requests.

Each worker independently chooses a pool using weighted rendezvous hashing of its
`stable_routing_id` and the pool name, falling back to `worker_id` when no stable
identity is published. For each pool, the router derives a uniform hash value U in
(0, 1) and chooses the smallest exponential-race score `-ln(U) / fraction` (compared
in log space for numerical stability). Pool-name order breaks score ties.

With unchanged fractions and identities, adding or removing workers never changes
surviving workers' pools. Routers agree on shared workers even when their membership
views differ. Stable identities preserve assignments across re-registration with new
worker IDs; replacing a pod or changing its identity can change its own pool.
Adding a pool moves workers only into that pool; removing one moves only its members,
provided the relative fractions of the remaining pools stay unchanged. Increasing
one pool's relative weight can only move workers into it. General fraction changes
can reassign workers; renaming a pool changes its hash identity. No previous
assignment state or cross-router coordination is required.

`b10.virtual/` is reserved for router-assigned taints; do not register it on workers.
Registered taints, topology, and drain metadata are preserved. Every update derives from
the original discovery snapshot, so removed/renamed pools leave no stale virtual taints.

Set `virtual_taint_worker_fractions: {}` to disable. The Rust config reader polls its mounted file
every 15 seconds; the overlay observes published changes within another second. Kubernetes
ConfigMap projection delay is additional. This does not interrupt active requests.
With the preferred-only example, removal restores ordinary scoring. Required virtual
taints must also be removed from request constraints before disabling their pools.

This changes placement only. Keep engine batch capacity large enough for throughput
requests to batch, and compare latency separately by resolved service tier. The Python
request tracker receives the resolved tier; legacy ingress-tier latency series may label
requests by `x-baseten-service-tier` instead.
