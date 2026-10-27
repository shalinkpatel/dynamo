# baseten-mocker

Baseten extensions for the Dynamo SGLang mocker: a simulated HiCache host tier
and G2PB peer reads (G2Direct), with page hashes only.

## Config

`mocker.sglang.hicache` (`HiCacheArgs`). Sizes, bytes and bandwidths are
**per rank** (one GPU); the mocker derives the per-rank layout from the model.

| Field | Meaning |
|---|---|
| `size` | Host pool GB per rank (`--hicache-size`). |
| `write_policy` | `write_through` (default) or `write_back`. |
| `bandwidth_gb_per_sec` | Per-rank host-to-device load-back rate, GB/s. Unset: instant. |
| `peer_group` | Enables G2PB. Live mockers use the fleet-wide G2Direct index. |
| `peer_bandwidth_gb_per_sec` | Per-rank peer read rate achieved at real fetch sizes, GB/s. Unset: instant. |
| `tp_size`, `dcp_size`, `mla` | The engine's KV layout (defaults 1, 1, false). |

`kv_bytes_per_token` (top-level mocker setting) is the model's KV bytes per
token over all layers and heads. One rank holds:

| Layout | Bytes per token on a rank | Peer keys per page |
|---|---|---|
| MHA/GQA, TP N | `kv_bytes_per_token / N` | one per TP rank |
| MLA, TP N | `kv_bytes_per_token` (replicated) | one, shared by all TP ranks |
| MLA, DCP M | `kv_bytes_per_token / M` | one per DCP rank |

With fewer KV heads than TP ranks, heads are replicated: set `tp_size` to the
KV head count. Like G2Direct's key suffixes, replicas with different layouts
never share pages, and a page is a peer hit only if every rank finds its key.
Under DCP, G2Direct keys pages widened by `dcp_size`: set `block_size` to the
engine page times `dcp_size` (512 for DCP8 on 64-token pages).

## Calibrating the bandwidths

Transfer time is `pages x page_size x per-rank bytes / bandwidth`; ranks move
their shards in parallel. Use rates **achieved at your fetch sizes**, not link
rates: per-fetch overhead (posting, completion, leases) is folded into them.

- **Peer reads.** From a G2PB run with the target model, page size and
  transport, read the G2Direct core counters per rank
  (`dynamo_g2direct_core_total{tp_rank=...}`) and take
  `stat="fetched_bytes"` over `stat="rdma_ms"`. Use a run with one fetch in
  flight per rank, or summed durations double-count wall time. Sweep prompt
  sizes if fetch sizes vary a lot. Reference points: DSV4 Flash TP4 on GB200
  over RoCE gave ~4-7 GB/s per rank at ~9 MB fetches, well below the 50 GB/s
  line rate; Qwen2.5-7B TP2 on L4 over UCX TCP gave ~1.5 GB/s per rank at
  ~70 MB fetches, with lease, index and verify time adding ~8%.
- **Host load-back.** SGLang's `sglang:load_back_bytes_total` over
  `sglang:load_back_duration_seconds_sum`, per rank. The same L4 rig gave
  ~5-10 GB/s per rank, varying by node.

SGLang reports `load_back_*` and `storage_prefetch_*` on every TP rank but
`realtime_tokens_total` only on rank 0: read one rank, or sums double-count.

Not modeled: fixed per-read costs (a few ms), contention between concurrent
transfers, and the first layer's load that cannot overlap compute.
