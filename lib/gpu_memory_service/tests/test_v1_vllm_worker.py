# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def vllm_modules():
    pytest.importorskip("vllm.device_allocator.sleep_mode_backend")
    pytest.importorskip("vllm.v1.worker.gpu_worker")
    backend = importlib.import_module("gpu_memory_service.v1.integrations.vllm.backend")
    worker = importlib.import_module("gpu_memory_service.v1.integrations.vllm.worker")
    return backend, worker


def test_worker_routes_managed_scopes_and_backend_orders_lifecycle(
    vllm_modules,
    monkeypatch,
) -> None:
    backend_module, worker_module = vllm_modules
    events = []
    final_model = object()

    class RoutedBackend:
        @contextmanager
        def capture_weights(self, model):
            events.append("weights_enter")
            yield
            events.append(("weights_exit", model()))

        @contextmanager
        def capture_kv_cache(self):
            events.append("kv_enter")
            yield
            events.append("kv_exit")

    routed_backend = RoutedBackend()

    def init_device(instance):
        events.append("vllm_init")
        instance.model_runner = SimpleNamespace(
            model=None,
            get_model=lambda: instance.model_runner.model,
        )

    monkeypatch.setattr(worker_module.Worker, "init_device", init_device)
    monkeypatch.setattr(
        worker_module.Worker,
        "_get_sleep_mode_backend",
        lambda _instance: routed_backend,
    )
    monkeypatch.setattr(
        worker_module.Worker,
        "_maybe_get_memory_pool_context",
        lambda _instance, tag: events.append(("super", tag)) or nullcontext(),
    )

    worker = object.__new__(worker_module.GMSV1Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            sleep_mode_backend="cumem",
        )
    )
    worker.init_device()
    with worker._maybe_get_memory_pool_context("weights"):
        worker.model_runner.model = final_model
    with worker._maybe_get_memory_pool_context("kv_cache"):
        pass
    with worker._maybe_get_memory_pool_context("activation"):
        pass

    backend = object.__new__(backend_module.GMSV1SleepModeBackend)
    backend_module.SleepModeBackend.__init__(backend)
    backend._device = 0
    info_messages = []
    monkeypatch.setattr(
        backend_module.logger,
        "info",
        lambda message, *_args: info_messages.append(message),
    )
    monkeypatch.setattr(backend_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    backend._raise_if_allocator_failed = lambda: events.append("allocator_ok")
    backend._weights = SimpleNamespace(
        unmap_all_vas=lambda: events.append("weights_unmap"),
        disconnect=lambda: events.append("weights_disconnect"),
        connect=lambda mode: events.append(("weights_connect", mode.value)),
        remap_all_vas=lambda: events.append("weights_remap"),
    )
    backend._kv_cache = SimpleNamespace(
        unmap_all_vas=lambda: events.append("kv_unmap"),
        disconnect=lambda: events.append("kv_disconnect"),
        connect=lambda mode: events.append(("kv_connect", mode.value)),
        reallocate_all_handles=lambda: events.append("kv_reallocate"),
        remap_all_vas=lambda: events.append("kv_remap"),
    )
    backend.suspend()
    backend.resume()

    assert worker.vllm_config.model_config.sleep_mode_backend == (
        backend_module.BACKEND_NAME
    )
    assert events == [
        "vllm_init",
        "weights_enter",
        ("weights_exit", final_model),
        "kv_enter",
        "kv_exit",
        ("super", "activation"),
        "gc",
        "allocator_ok",
        "weights_unmap",
        "weights_disconnect",
        "kv_unmap",
        "kv_disconnect",
        "empty_cache",
        "allocator_ok",
        ("kv_connect", "rw"),
        "kv_reallocate",
        "kv_remap",
        ("weights_connect", "ro"),
        "weights_remap",
    ]
    assert backend.state() == "RUNNING"
    assert info_messages == [
        (
            "GMS V1 KV wake device=%d connect_elapsed=%.3fs "
            "reallocate_elapsed=%.3fs remap_elapsed=%.3fs total_elapsed=%.3fs"
        ),
        (
            "GMS V1 weights wake device=%d connect_elapsed=%.3fs "
            "remap_elapsed=%.3fs total_elapsed=%.3fs"
        ),
        "GMS V1 wake complete device=%d total_elapsed=%.3fs",
    ]


def test_v1_allocator_has_one_owner(
    vllm_modules,
    monkeypatch,
) -> None:
    backend_module, _worker_module = vllm_modules
    shared_extension = importlib.import_module(
        "gpu_memory_service.client.torch.extensions._allocator_ext"
    )
    assert backend_module._allocator_ext is shared_extension

    registered = []
    extension = SimpleNamespace(
        init_module=lambda malloc, free: registered.append((malloc, free))
    )
    owner = object()
    malloc = object()
    free = object()

    monkeypatch.setattr(backend_module, "_allocator_ext", extension)
    monkeypatch.setattr(backend_module, "_allocator_owner", None)
    monkeypatch.setattr(backend_module, "_allocator_initializing", None)

    backend_module._reserve_allocator(owner)
    backend_module._claim_allocator(owner, malloc, free)

    assert registered == [(malloc, free)]
    with pytest.raises(RuntimeError, match="V1 supports exactly one"):
        backend_module._reserve_allocator(object())


def test_backend_pins_callbacks_and_rejects_a_second_instance(
    vllm_modules,
    monkeypatch,
) -> None:
    backend_module, _worker_module = vllm_modules
    registered = []
    managers = []

    class Manager:
        def __init__(self, *_args):
            managers.append(self)

        def connect(self, _mode):
            pass

        def disconnect(self):
            pass

    monkeypatch.setattr(backend_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "device",
        lambda _device: nullcontext(),
    )
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "CUDAPluggableAllocator",
        lambda *_args: SimpleNamespace(allocator=lambda: object()),
    )
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "MemPool",
        lambda *, allocator: allocator,
    )
    monkeypatch.setattr(backend_module, "get_vmm", object)
    monkeypatch.setattr(
        backend_module,
        "get_socket_path",
        lambda _device, domain: domain,
    )
    monkeypatch.setattr(backend_module, "GMSClientMemoryManager", Manager)
    monkeypatch.setattr(
        backend_module,
        "_allocator_ext",
        SimpleNamespace(
            __file__="shared-allocator.so",
            init_module=lambda malloc, free: registered.append((malloc, free)),
        ),
    )
    monkeypatch.setattr(backend_module, "_allocator_owner", None)
    monkeypatch.setattr(backend_module, "_allocator_initializing", None)

    owner = backend_module.GMSV1SleepModeBackend()

    assert registered == [(owner._malloc_callback, owner._free_callback)]
    assert registered[0][0] is owner._malloc_callback
    assert registered[0][1] is owner._free_callback
    assert owner._malloc_callback.__self__ is owner
    assert owner._free_callback.__self__ is owner
    assert backend_module._allocator_owner is owner

    with pytest.raises(RuntimeError, match="V1 supports exactly one"):
        backend_module.GMSV1SleepModeBackend()

    assert registered == [(owner._malloc_callback, owner._free_callback)]
    assert len(managers) == 2
    assert backend_module._allocator_owner is owner
    assert backend_module._allocator_initializing is None


def test_backend_constructor_failure_disconnects_both_sessions(
    vllm_modules,
    monkeypatch,
) -> None:
    backend_module, _worker_module = vllm_modules
    events = []

    class Manager:
        def __init__(self, path, _vmm, _device):
            self.path = path

        def connect(self, _mode):
            events.append(("connect", self.path))

        def disconnect(self):
            events.append(("disconnect", self.path))

    pools = 0

    def mem_pool(*, allocator):
        nonlocal pools
        pools += 1
        if pools == 2:
            raise RuntimeError("KV pool failed")
        return allocator

    monkeypatch.setattr(backend_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "device",
        lambda _device: nullcontext(),
    )
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "CUDAPluggableAllocator",
        lambda *_args: SimpleNamespace(allocator=lambda: object()),
    )
    monkeypatch.setattr(backend_module.torch.cuda, "MemPool", mem_pool)
    monkeypatch.setattr(backend_module, "get_vmm", object)
    monkeypatch.setattr(
        backend_module,
        "get_socket_path",
        lambda _device, domain: domain,
    )
    monkeypatch.setattr(backend_module, "GMSClientMemoryManager", Manager)
    monkeypatch.setattr(backend_module, "_allocator_owner", None)
    monkeypatch.setattr(backend_module, "_allocator_initializing", None)

    with pytest.raises(RuntimeError, match="KV pool failed"):
        backend_module.GMSV1SleepModeBackend()

    assert events == [
        ("connect", "weights"),
        ("connect", "kv_cache"),
        ("disconnect", "kv_cache"),
        ("disconnect", "weights"),
    ]
    assert backend_module._allocator_owner is None
    assert backend_module._allocator_initializing is None


@pytest.mark.parametrize(
    ("domain", "failure_point"),
    [
        ("weights", "yield"),
        ("weights", "finalize"),
        ("kv_cache", "yield"),
        ("kv_cache", "finalize"),
    ],
)
def test_capture_failure_disconnects_live_epochs(
    vllm_modules,
    monkeypatch,
    domain,
    failure_point,
) -> None:
    backend_module, _worker_module = vllm_modules
    backend = object.__new__(backend_module.GMSV1SleepModeBackend)
    events = []
    backend._weights = SimpleNamespace(
        disconnect=lambda: events.append("weights_disconnect"),
        mappings=(),
    )
    backend._kv_cache = SimpleNamespace(
        disconnect=lambda: events.append("kv_disconnect")
    )
    backend._weights_pool = object()
    backend._kv_cache_pool = object()

    @contextmanager
    def use_pool(*_args):
        yield

    backend._use_pool = use_pool
    if failure_point == "finalize":
        if domain == "weights":
            monkeypatch.setattr(
                backend_module,
                "copy_non_parameter_tensors_to_default_allocator",
                lambda *_args: (_ for _ in ()).throw(RuntimeError("finalize failed")),
            )
        else:
            backend._raise_if_allocator_failed = lambda: (_ for _ in ()).throw(
                RuntimeError("finalize failed")
            )

    capture = (
        backend.capture_weights(lambda: object())
        if domain == "weights"
        else backend.capture_kv_cache()
    )
    with pytest.raises(RuntimeError, match=failure_point), capture:
        if failure_point == "yield":
            raise RuntimeError("yield failed")

    assert events == ["kv_disconnect", "weights_disconnect"]


def test_suspend_empty_cache_allocator_failure_dispatches_fail_stop(
    vllm_modules,
    monkeypatch,
) -> None:
    backend_module, _worker_module = vllm_modules
    backend = object.__new__(backend_module.GMSV1SleepModeBackend)
    backend_module.SleepModeBackend.__init__(backend)
    backend._device = 0
    backend._allocator_failure = None
    backend._allocator_failure_lock = backend_module.threading.Lock()
    backend._weights = SimpleNamespace(
        owns=lambda _va: True,
        destroy_mapping=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("free failed")
        ),
        unmap_all_vas=lambda: None,
        disconnect=lambda: None,
    )
    backend._kv_cache = SimpleNamespace(
        owns=lambda _va: False,
        unmap_all_vas=lambda: None,
        disconnect=lambda: None,
    )
    monkeypatch.setattr(backend_module.gc, "collect", lambda: None)
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "empty_cache",
        lambda: backend._free(0x1000, 64, 0, 0),
    )
    fail_calls = []

    class ProcessFatal(BaseException):
        pass

    def fail(*args, **kwargs):
        fail_calls.append((args, kwargs))
        raise ProcessFatal

    monkeypatch.setattr(backend_module.common_utils, "fail", fail)

    with pytest.raises(ProcessFatal):
        backend.suspend()

    assert fail_calls == [
        (
            ("GMS V1 suspend failed; terminating the worker process",),
            {"exc_info": True},
        )
    ]
    assert backend.state() == "RUNNING"
