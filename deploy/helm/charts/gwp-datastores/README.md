<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GWP datastores

This optional Helm chart installs shared etcd and NATS instances for GWP
development and proof-of-concept deployments. Redis is disabled by default:

```console
helm install gwp-stores deploy/helm/charts/gwp-datastores --namespace gwp
```

Configure GWP with:

```text
ETCD_ENDPOINTS=http://gwp-stores-gwp-datastores-etcd:2379
```

```yaml
session:
  backend:
    type: pubsub
```

NATS Core pub-sub is available at:

```text
NATS_SERVER=nats://gwp-stores-gwp-datastores-nats:4222
```

GWP defaults to NATS for replica events. Configure `NATS_SERVER` as above;
JetStream is unnecessary. Set `DYN_EVENT_PLANE=zmq` to use direct pod-to-pod
events instead. Redis is optional (`redis.enabled=true`) for legacy affinity
or MCM discovery. etcd and NATS can be disabled independently with
`etcd.enabled=false` and `nats.enabled=false` when supplied externally.

The chart intentionally uses single replicas and ephemeral storage. Use
externally managed, persistent, highly available services for production.
