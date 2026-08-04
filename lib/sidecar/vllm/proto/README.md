<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Source: [`rust/proto/inference.proto`](https://github.com/connorcarpenter15/vllm/blob/3678360b3be3dc7795133276a0ab6ad51ce8875e/rust/proto/inference.proto) and [`rust/proto/control.proto`](https://github.com/connorcarpenter15/vllm/blob/3678360b3be3dc7795133276a0ab6ad51ce8875e/rust/proto/control.proto)
- Commit: `3678360b3be3dc7795133276a0ab6ad51ce8875e`
- `inference.proto` SHA-256: `a0d196dc240683e1c09abb54f324d4428d0c122a6802b44916ad2d96b491b06c`
- `control.proto` SHA-256: `390c88e94f1b68421c54c6d9440f2088d2709a432549c7a0fe94d35ce7b37476`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
