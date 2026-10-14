# G2Direct Python transport

`from g2direct import HiCacheG2Direct` exposes the engine-owned host-cache
transport without installing KVBM, its allocators, or its connectors. The
Rust dependency graph includes `dynamo-g2direct`, `dynamo-runtime`,
`dynamo-memory`, and NIXL, but not `dynamo-llm`.

SGLang's storage adapter remains responsible for model/pool keys, page geometry,
and the lifetime of pinned host allocations. The constructor and methods match
the G2Direct API previously exposed by `kvbm._core`. That import remains
available for existing consumers and embeds the same binding implementation.

The standalone extension owns its Tokio runtime. When embedded in KVBM, its
runtime provider is linked into the same extension; Rust handles never cross
independently built Python extension boundaries.

## Build and consume

The binding shares the repository's Cargo workspace, lockfile, and build
profiles. Maturin enables the `python-extension` feature for wheel builds;
ordinary Cargo tests leave it disabled so test executables link to Python.
The wheel remains independently buildable and does not depend on KVBM.

In a Linux build environment with the repository's NIXL 1.4.0 and UCX libraries:

```sh
cd lib/bindings/g2direct
maturin build --release --locked
```

The framework-neutral Dynamo component image builds and exports
`/opt/dynamo/wheelhouse/g2direct-*.whl` alongside its other wheels, independently
of `ENABLE_KVBM`. The SGLang image can copy just this wheel and the matching
NIXL/UCX libraries from that component image without replacing its separately
pinned Dynamo serving runtime. Install the matching CUDA NIXL wheel first;
`pip install --no-deps g2direct-*.whl` then adds only this transport.
The Python package loads NIXL before the native extension so wheel-provided
`libnixl.so` is available without a system-wide library-path override.

For the Baseten SGLang adapter, replace
`from kvbm._core import HiCacheG2Direct` with
`from g2direct import HiCacheG2Direct`, and install this wheel instead of the
KVBM wheel. Existing NATS/ETCD, namespace, replica ID, NIC and host-memory
configuration remains necessary. Extraction does not add multinode orchestration
or change model/pool support.

## Contracts

G2Direct shares pages that live in the engine's host tier, which the engine
manages without telling the transport. These are the guarantees each side
relies on.

**Source leases are snapshots, not pins.** A fetch leases its pages on the
source replica: the source hands back the descriptors and published checksums
it holds for those keys, and keeps that snapshot until the reader releases the
lease or it expires (bounded by the reader's fetch deadline). The lease does
not stop HiCache from evicting or reusing the slot, and a lease granted before
the slot is reused still points at it. Every page read is therefore re-hashed at the
destination and compared with the checksum published for it; a mismatch fails
that page and the reader falls back to another peer or a miss. The verification
mode is `destination` by default; `off` exists for measurement only.

**Publish only complete, settled bytes.** `publish()` hashes the page's current
bytes and advertises that checksum under the given key, so the caller must call
it only after the page is fully written for that key, and must not rewrite the
slot until `publish()` returns. Destination verification proves a reader got
the bytes that were published; it cannot prove those bytes were right for the
key. In the SGLang adapter, HiCache provides this ordering: a node is backed up
to storage (and so published) only after its device-to-host write is
acknowledged, and its host pages hold a host lock until the backup completes.
The key is the node's token-chain hash.

**Withdraw a slot before refilling it.** HiCache reuses host slots without
telling the transport, so a replica's advertisement for a slot is kept in step
with the slot's contents by the calls that change them: `publish()` of the slot
under another key replaces its entry; `fetch()` withdraws whatever the replica
advertised at each destination slot before it posts any read into it; and
`invalidate_slots(ptrs, sizes, ptrs_per_page)`, which the caller must invoke
before writing any other bytes into host slots (an L3 load), withdraws the
entries at those slots and returns how many it removed. Slots are described
exactly as for `publish()`. The local withdrawal is immediate, so this replica
stops leasing the old key at once; peers' copies of the index drop it when the
removal event reaches them, and until then their leases for it fail.

**Destination pages stay owned until their reads settle.** `fetch()` returns
the leading pages that completed and, if reads were still in flight at the
deadline, a settlement ticket. The pages after the completed prefix may still be
written by those reads, so the caller must keep them allocated and out of reuse
until `is_settled(ticket)` returns true (it does so once per ticket); there is
no timeout, since nothing else proves a write stopped. The SGLang adapter hands
tickets to HiCache, whose deferred host release keeps those pages out of the
pool on every TP rank, including across a cache reset, and keeps holding them if
the backend is detached. `fetch()` raises only for invalid arguments, before any
read is posted, so an error never leaves pages to hold. A panic is a broken
internal invariant: treat the process as failed. The transport owns in-flight
requests in its reaper; dropping it while tickets are outstanding is only safe
at process exit.

**Discovery is incremental.** Each replica builds its index from NATS replica
events it receives after it subscribes, and there is no snapshot on join: a
replica that starts late learns only pages advertised afterwards, so pages
already resident on peers stay invisible to it until they are published again.
A source withdraws its entries on clean shutdown; a crashed source's entries
stay in the index but are skipped, because candidates are filtered to sources
that discovery still lists as live peers. A snapshot-on-join (a joining replica
asks each live peer for its current advertisements) or periodic re-advertisement
is planned and not part of this package yet.

## Validation

```sh
cargo test -p dynamo-g2direct --no-default-features
cargo test -p dynamo-g2direct
cargo test -p g2direct-py3
cargo check -p g2direct-py3 --features python-extension
```

The first command exercises lease/quarantine primitives without CUDA or NIXL.
Transport tests and wheel builds require the Linux native dependencies. RDMA,
destination checksum verification, recycled pages and deadline handling are
validated on deployment hardware with a two-worker SGLang test (see the PR).
