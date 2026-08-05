# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gpu_memory_service.cli.snapshot import loader as snapshot_loader
from gpu_memory_service.cli.snapshot import saver as snapshot_saver
from gpu_memory_service.v1 import loader, saver

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    ("module", "target"),
    [
        (snapshot_loader, "gpu_memory_service.v1.loader"),
        (snapshot_saver, "gpu_memory_service.v1.saver"),
    ],
)
def test_snapshot_cli_lazily_dispatches_v1(module, target, monkeypatch) -> None:
    calls = []

    def import_module(name):
        calls.append(("import", name))
        return SimpleNamespace(main=lambda argv: calls.append(("main", argv)))

    monkeypatch.setattr(module.importlib, "import_module", import_module)

    module.main(["--use-v1", "--checkpoint-dir", "/checkpoint", "--device", "3"])

    assert calls == [
        ("import", target),
        ("main", ["--checkpoint-dir", "/checkpoint", "--device", "3"]),
    ]


def test_v1_loader_passes_existing_backend_options(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        loader, "init_vmm", lambda device_type: calls.append(device_type)
    )
    monkeypatch.setattr(
        loader,
        "get_socket_path",
        lambda device, tag: f"/sockets/{device}-{tag}.sock",
    )
    monkeypatch.setattr(
        loader,
        "hydrate_weights",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    loader.main(
        [
            "--checkpoint-dir",
            "/checkpoint",
            "--device",
            "3",
            "--max-workers",
            "7",
            "--transfer-backend",
            "sharded-ssd",
            "--sharded-ssd-roots",
            "/ssd-a,/ssd-b",
            "--sharded-ssd-queues-per-root",
            "4",
            "--posix-backend-param",
            "ios_pool_size=64",
            "--posix-backend-param",
            "kernel_queue_size=16",
        ]
    )

    args, kwargs = calls[1]
    assert args == (
        "/checkpoint/device-3",
        "/sockets/3-weights.sock",
        3,
    )
    assert kwargs == {
        "max_workers": 7,
        "transfer_backend": "sharded-ssd",
        "sharded_ssd_roots": ["/ssd-a", "/ssd-b"],
        "sharded_ssd_queues_per_root": 4,
        "posix_backend_params": {
            "ios_pool_size": "64",
            "kernel_queue_size": "16",
        },
    }


def test_v1_loader_rejects_invalid_posix_backend_param(capsys) -> None:
    with pytest.raises(SystemExit):
        loader.main(
            [
                "--checkpoint-dir",
                "/checkpoint",
                "--posix-backend-param",
                "missing-separator",
            ]
        )
    assert "expected KEY=VALUE" in capsys.readouterr().err


def test_v1_saver_passes_existing_save_options(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        saver, "init_vmm", lambda device_type: calls.append(device_type)
    )
    monkeypatch.setattr(
        saver,
        "get_socket_path",
        lambda device, tag: f"/sockets/{device}-{tag}.sock",
    )
    monkeypatch.setattr(
        saver,
        "save_weights",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    saver.main(
        [
            "--checkpoint-dir",
            "/checkpoints/run/versions/1",
            "--device",
            "3",
            "--max-workers",
            "5",
            "--save-lock-timeout-ms",
            "9000",
            "--shard-size-bytes",
            "1024",
            "--sharded-ssd-roots",
            "/ssd-a,/ssd-b",
        ]
    )

    args, kwargs = calls[1]
    assert args == (
        "/checkpoints/run/versions/1/device-3",
        "/sockets/3-weights.sock",
        3,
    )
    assert kwargs == {
        "shard_size_bytes": 1024,
        "max_workers": 5,
        "admission_timeout": 9,
        "sharded_ssd_roots": [
            "/ssd-a/run/versions/1/device-3",
            "/ssd-b/run/versions/1/device-3",
        ],
    }


@pytest.mark.parametrize("module", [loader, saver])
@pytest.mark.parametrize(
    "option",
    [
        ["--device-type", "xpu"],
        ["--max-workers", "0"],
    ],
)
def test_v1_cli_rejects_unsupported_device_and_workers(
    module,
    option,
    capsys,
) -> None:
    with pytest.raises(SystemExit):
        module.main(["--checkpoint-dir", "/checkpoint", *option])
    error = capsys.readouterr().err
    assert "GMS V1" in error or "positive" in error


def test_v1_saver_rejects_non_positive_lock_timeout(capsys) -> None:
    with pytest.raises(SystemExit):
        saver.main(
            [
                "--checkpoint-dir",
                "/checkpoint",
                "--save-lock-timeout-ms",
                "0",
            ]
        )
    assert "must be a positive integer" in capsys.readouterr().err
