<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM sidecar

> [!WARNING]
> **Experimental.** This sidecar and its deployment examples are experimental
> and not yet packaged for distribution. The manifests, flags, and behavior may
> change without notice.

`dynamo-vllm-sidecar` connects a Dynamo worker to vLLM's native gRPC services:

- `vllm.Inference` for generation
- `vllm.Control` for model and server discovery
- Standard gRPC health for startup readiness

It is a standalone Rust executable.

## Supported

- Aggregated generation
- NIXL prefill/decode generation
- Token and text requests through Dynamo preprocessing
- Sampling, stop conditions, structured output, logprobs, cache options, and priority
- Opaque `kv_transfer_params` handoff
- Data-parallel rank routing and KV-event source discovery
- Image URL and data-URI inputs, including media UUIDs

The protocol does not support LoRA, encode workers, beam search, `n > 1`,
preprocessed multimodal features, audio/video media, or Dynamo tool-call and
reasoning parsers. Parser defaults returned by Control are intentionally not
advertised to the Dynamo frontend because the current inference protocol does
not preserve all parser-related request semantics.

## Run

Start a vLLM build with the split Inference and Control services and explicit
data-parallel-rank capability used by the vendored protocol:

```bash
vllm-rs serve Qwen/Qwen3-0.6B --host 127.0.0.1 --grpc-port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on
loopback or a private interface. Remote access requires network controls or a
secure proxy.

Start the Dynamo worker explicitly:

```bash
dynamo-vllm-sidecar \
  --vllm-endpoint 127.0.0.1:50051
```

Use `VLLM_GRPC_ENDPOINT` instead of `--vllm-endpoint` when the endpoint is
provided through the environment.

The sidecar discovers `model_id`, the served name, context length, KV capacity, scheduler limits, data-parallel topology, and KV-event sources through `vllm.Control`. `model_id` must be readable locally or fetchable by Dynamo for tokenization and chat templates.

Control reports the global data-parallel size, the rank hosted by the connected frontend, and whether explicit rank routing is supported. Dynamo forwards the selected rank on each generation request and uses discovered KV-event sources for KV-aware routing. Startup fails for data-parallel deployments when the server does not advertise explicit rank routing.

Aggregated serving is the default. Set the existing `--disaggregation-mode` to `prefill` or `decode` only for non-aggregated deployments; the current Control API does not report engine role.

The sidecar opens eight gRPC connections by default. This avoided
connection-level throttling in high-concurrency sidecar tests. Override the
pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`.

Connection startup uses a 30-second timeout per attempt, a one-second retry
interval, and a five-minute deadline for establishing the full connection
pool. Override them with `--grpc-connect-attempt-timeout-secs`,
`--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs`, or with the
corresponding `DYN_SIDECAR_GRPC_*` environment variables.

Each request owns its response stream but borrows a channel from the shared pool. Aggregate and prefill cancellation drops only that request's stream. Decode cancellation first submits the decode request and retains its stream through the first output token so a NIXL receiver can complete and release the transferred KV; it then drops the stream. vLLM automatically aborts the corresponding engine request while the pooled HTTP/2 connection remains available to other requests. The sidecar does not call the Control `Abort` RPC.

## Test without vLLM or a GPU

Use the CPU-only `dynamo-vllm-mocker-server` to exercise the same Inference, Control, and health contracts:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000}'

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --vllm-endpoint 127.0.0.1:50051
```

See [`../../mocker/servers/vllm/README.md`](../../mocker/servers/vllm/README.md)
for aggregated and prefill/decode examples, supported Mocker configuration,
and fidelity limits.

## Deploy on Kubernetes

Kubernetes examples are deferred until compatible vLLM and Dynamo sidecar images are published. The current `vllm/vllm-openai:v0.26.0` image exposes the older monolithic Generate service and cannot satisfy this sidecar's split Inference, Control, capability-discovery, and health contracts.

## Packaging

There is no published compatible image pair yet. Official packaging and Kubernetes manifests are deferred to a follow-up change.
