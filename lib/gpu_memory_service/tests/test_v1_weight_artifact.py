# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from time import monotonic

import msgspec
import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.snapshot.transfer import TransferBackendKind
from gpu_memory_service.v1 import weight_artifact
from gpu_memory_service.v1.protocol import AllocationRecord
from gpu_memory_service.v1.server import GMSRPCServer, GMSServerMemoryManager
from gpu_memory_service.v1.session import _GMSClientSession

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def _device_identity(monkeypatch):
    monkeypatch.setattr(
        weight_artifact.device_identity,
        "invalidate_device_uuid_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        weight_artifact.device_identity,
        "get_device_uuid",
        lambda _device: "GPU-0",
    )


@contextmanager
def _server(path: str, vmm: FakeVMM, manager=None):
    manager = manager or GMSServerMemoryManager("GPU-0", vmm, 0)
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield manager
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


class _ByteVMM(FakeVMM):
    def __init__(self):
        super().__init__(granularity=64)
        self._physical: dict[int, bytearray] = {}
        self._exported: dict[int, int] = {}
        self._imported: dict[int, int] = {}

    def create_tolerate_oom(self, size, device):
        allocated, handle = super().create_tolerate_oom(size, device)
        self._physical[handle] = bytearray(size)
        return allocated, handle

    def release(self, handle):
        if handle in self.server_handles:
            self._physical.pop(handle)
        self._imported.pop(handle, None)
        super().release(handle)

    def export_to_shareable_handle(self, handle):
        read_fd = super().export_to_shareable_handle(handle)
        self._exported[os.fstat(read_fd).st_ino] = handle
        return read_fd

    def import_shareable_handle_close_fd(self, fd):
        physical_handle = self._exported[os.fstat(fd).st_ino]
        imported_handle = super().import_shareable_handle_close_fd(fd)
        self._imported[imported_handle] = physical_handle
        return imported_handle

    def _allocation(self, va, size):
        mapped_size, imported_handle = self.mapped[va]
        assert size <= mapped_size
        return self._physical[self._imported[imported_handle]]

    def read(self, va, size):
        return bytes(self._allocation(va, size)[:size])

    def write(self, va, data):
        self._allocation(va, len(data))[: len(data)] = data


class _Writer:
    vmm: _ByteVMM

    def __init__(self, path, *, device):
        assert device == 0
        self._path = Path(path)
        self._data = bytearray()

    def __enter__(self):
        return self

    def write_device(self, src_ptr, byte_count):
        self._data.extend(self.vmm.read(src_ptr, byte_count))

    def __exit__(self, *_args):
        self._path.write_bytes(self._data)


class _Transfer:
    def __init__(
        self,
        vmm,
        sources,
        events,
        *,
        restore_failure=False,
        close_failure=False,
    ):
        self._vmm = vmm
        self._sources = sources
        self._events = events
        self._restore_failure = restore_failure
        self._close_failure = close_failure

    def restore(self, targets):
        if self._restore_failure:
            raise RuntimeError("restore failed")
        for source in self._sources:
            target = targets[source.allocation_id]
            shard = Path(source.file_path).read_bytes()
            data = shard[source.file_offset : source.file_offset + source.byte_count]
            assert len(data) == target.byte_count
            self._vmm.write(target.va, data)

    def close(self):
        self._events.append("transfer_close")
        if self._close_failure:
            raise RuntimeError("transfer close failed")


class _Backend:
    def __init__(
        self,
        vmm,
        events=None,
        *,
        restore_failure=False,
        transfer_close_failure=False,
        close_failure=False,
    ):
        self._vmm = vmm
        self._events = events if events is not None else []
        self._restore_failure = restore_failure
        self._transfer_close_failure = transfer_close_failure
        self._close_failure = close_failure
        self.closed = False

    def start_restore(self, sources):
        return _Transfer(
            self._vmm,
            sources,
            self._events,
            restore_failure=self._restore_failure,
            close_failure=self._transfer_close_failure,
        )

    def close(self):
        self.closed = True
        self._events.append("backend_close")
        if self._close_failure:
            raise RuntimeError("backend close failed")


def _write_manifest(
    path,
    allocations,
    *,
    version=1,
    shard_bytes=bytes(128),
):
    path.mkdir()
    manifest = weight_artifact.WeightArtifactManifest(
        version,
        tuple(allocations),
    )
    (path / "manifest.json").write_bytes(msgspec.json.encode(manifest))
    (path / "shard.bin").write_bytes(shard_bytes)


def _seed_weights(socket_path: str, vmm: _ByteVMM):
    writer = _GMSClientSession(socket_path, RequestedLockType.RW)
    record = AllocationRecord("weight", 64)
    writer.allocate(record.allocation_id, record.aligned_size)
    mapping = weight_artifact._map_export(
        writer,
        record,
        vmm,
        0,
        64,
        GrantedLockType.RW,
    )
    vmm.write(mapping[0].base, bytes(range(64)))
    return writer, mapping


Allocation = weight_artifact.WeightArtifactAllocation


@pytest.mark.timeout(10)
def test_byte_roundtrip_preserves_exact_allocations_on_fresh_server(
    tmp_path, monkeypatch
):
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    source_vmm = _ByteVMM()
    records = (
        AllocationRecord("weight-0", 64),
        AllocationRecord("weight-1", 128),
    )
    expected = {
        "weight-0": bytes(range(64)),
        "weight-1": bytes((255 - index) % 256 for index in range(128)),
    }
    _Writer.vmm = source_vmm
    monkeypatch.setattr(weight_artifact, "DeviceToFileWriter", _Writer)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: source_vmm)

    with _server(socket_path, source_vmm):
        writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        mappings = []
        for record in records:
            writer.allocate(record.allocation_id, record.aligned_size)
            mapping = weight_artifact._map_export(
                writer, record, source_vmm, 0, 64, GrantedLockType.RW
            )
            source_vmm.write(mapping[0].base, expected[record.allocation_id])
            mappings.append(mapping)
        writer.commit()
        manifest = weight_artifact.save_weights(
            str(artifact), socket_path, 0, shard_size_bytes=64
        )
        with pytest.raises(FileExistsError):
            weight_artifact.save_weights(str(artifact), socket_path, 0)
        for mapping in reversed(mappings):
            weight_artifact._release_mapping(source_vmm, mapping)
        writer.close()

    assert [
        (allocation.allocation_id, allocation.aligned_size)
        for allocation in manifest.allocations
    ] == [(record.allocation_id, record.aligned_size) for record in records]
    assert (artifact / "shards/shard_0000.bin").read_bytes() == expected["weight-0"]
    assert (artifact / "shards/shard_0001.bin").read_bytes() == expected["weight-1"]

    target_vmm = _ByteVMM()
    backend = _Backend(target_vmm)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: target_vmm)
    factory_calls = []

    def create_backend(name, config):
        factory_calls.append((name, config))
        return backend

    monkeypatch.setattr(weight_artifact, "create_transfer_backend", create_backend)
    with _server(socket_path, target_vmm):
        weight_artifact.hydrate_weights(
            str(artifact),
            socket_path,
            0,
            transfer_backend=TransferBackendKind.NIXL_GDS.value,
            max_workers=7,
            sharded_ssd_roots=["/ssd-a", "/ssd-b"],
            sharded_ssd_queues_per_root=3,
            posix_backend_params={"ios_pool_size": "64"},
        )
        reader = _GMSClientSession(socket_path, RequestedLockType.RO)
        assert weight_artifact._list_allocations(reader) == records
        restored = [
            weight_artifact._map_export(
                reader, record, target_vmm, 0, 64, GrantedLockType.RO
            )
            for record in records
        ]
        assert {
            mapping.allocation_id: target_vmm.read(mapping.base, mapping.aligned_size)
            for mapping, _handle in restored
        } == expected
        for mapping in reversed(restored):
            weight_artifact._release_mapping(target_vmm, mapping)
        reader.close()
    assert backend.closed
    assert factory_calls[0][0] == TransferBackendKind.NIXL_GDS.value
    config = factory_calls[0][1]
    assert config.device == 0
    assert config.max_workers == 7
    assert config.backend_config == {
        "sharded_ssd_roots": [
            str(Path("/ssd-a").resolve()),
            str(Path("/ssd-b").resolve()),
        ],
        "sharded_ssd_queues_per_root": 3,
        "posix_backend_params": {"ios_pool_size": "64"},
    }

    valid = Allocation("a", 64, "shard.bin", 0)
    for version, allocations, message in [
        (2, (valid,), "version"),
        (1, (), "no allocations"),
    ]:
        malformed = tmp_path / f"malformed-{version}-{len(allocations)}"
        _write_manifest(malformed, allocations, version=version)
        with pytest.raises(RuntimeError, match=message):
            weight_artifact.hydrate_weights(str(malformed), socket_path, 0)


class _FailingCleanupVMM(_ByteVMM):
    def __init__(self, events):
        super().__init__()
        self._events = events

    def unmap(self, va, size):
        self._events.append("unmap")
        super().unmap(va, size)
        raise RuntimeError("unmap cleanup failed")

    def release(self, handle):
        imported = handle in self.imports
        super().release(handle)
        if imported:
            raise RuntimeError("handle cleanup failed")

    def address_free(self, va, size):
        super().address_free(va, size)
        raise RuntimeError("VA cleanup failed")


@pytest.mark.timeout(10)
def test_hydrate_preserves_transfer_error_and_attempts_all_cleanup(
    tmp_path, monkeypatch, caplog
):
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    allocations = (
        Allocation("weight-0", 64, "shard.bin", 0),
        Allocation("weight-1", 64, "shard.bin", 64),
    )
    _write_manifest(artifact, allocations)
    events = []
    vmm = _FailingCleanupVMM(events)
    caplog.set_level("ERROR")
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: _Backend(vmm, events, restore_failure=True),
    )

    with _server(socket_path, vmm):
        with pytest.raises(RuntimeError, match="restore failed"):
            weight_artifact.hydrate_weights(str(artifact), socket_path, 0)
        assert not (vmm.imports or vmm.mapped or vmm.reservations or vmm.server_handles)
        _GMSClientSession(socket_path, RequestedLockType.RW).close()

    assert events.index("transfer_close") < events.index("unmap")
    assert events.index("unmap") < events.index("backend_close")
    assert events.count("unmap") == 2
    assert "resource cleanup failed" in caplog.text


@pytest.mark.timeout(10)
def test_sharded_save_records_absolute_paths_across_roots(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    roots = [tmp_path / "ssd-a", tmp_path / "ssd-b"]
    monkeypatch.chdir(tmp_path)
    vmm = _ByteVMM()
    records = (
        AllocationRecord("weight-0", 64),
        AllocationRecord("weight-1", 64),
    )
    _Writer.vmm = vmm
    monkeypatch.setattr(weight_artifact, "DeviceToFileWriter", _Writer)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)

    with _server(socket_path, vmm):
        writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        mappings = []
        for index, record in enumerate(records):
            writer.allocate(record.allocation_id, record.aligned_size)
            mapping = weight_artifact._map_export(
                writer,
                record,
                vmm,
                0,
                64,
                GrantedLockType.RW,
            )
            vmm.write(mapping[0].base, bytes([index]) * record.aligned_size)
            mappings.append(mapping)
        writer.commit()
        manifest = weight_artifact.save_weights(
            str(artifact),
            socket_path,
            0,
            shard_size_bytes=64,
            max_workers=2,
            sharded_ssd_roots=["ssd-a", "ssd-b"],
        )
        for mapping in reversed(mappings):
            weight_artifact._release_mapping(vmm, mapping)
        writer.close()

    shard_paths = [Path(allocation.shard) for allocation in manifest.allocations]
    assert [path.parent.parent.parent for path in shard_paths] == [
        roots[0],
        roots[1],
    ]
    assert [path.name for path in shard_paths] == [
        "shard_0000.bin",
        "shard_0001.bin",
    ]
    assert all(".attempt" in path.parent.parent.name for path in shard_paths)
    assert all(path.is_absolute() and path.exists() for path in shard_paths)
    _manifest, sources = weight_artifact._load_manifest(
        str(artifact),
        64,
        [str(root) for root in roots],
    )
    assert [source.file_path for source in sources] == [
        str(path) for path in shard_paths
    ]


@pytest.mark.timeout(10)
def test_save_lock_timeout_bounds_admission_and_retry_succeeds(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    vmm = _ByteVMM()
    _Writer.vmm = vmm
    monkeypatch.setattr(weight_artifact, "DeviceToFileWriter", _Writer)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)

    with _server(socket_path, vmm):
        writer, mapping = _seed_weights(socket_path, vmm)
        started_at = monotonic()
        with pytest.raises(ConnectionError, match="lock admission"):
            weight_artifact.save_weights(
                str(artifact),
                socket_path,
                0,
                admission_timeout=0.05,
            )
        assert monotonic() - started_at < 0.5
        assert not artifact.exists()
        writer.commit()
        weight_artifact.save_weights(
            str(artifact),
            socket_path,
            0,
            admission_timeout=1,
        )
        weight_artifact._release_mapping(vmm, mapping)
        writer.close()

    assert (artifact / "manifest.json").is_file()


@pytest.mark.timeout(10)
def test_failed_save_attempts_cleanup_and_retry(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    external_root = tmp_path / "ssd"
    external_root.mkdir()
    sentinel = external_root / "pre-existing"
    sentinel.write_text("keep")
    vmm = _ByteVMM()
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)

    with pytest.raises(ConnectionError):
        weight_artifact.save_weights(
            str(artifact),
            socket_path,
            0,
            connect_timeout=0.01,
            sharded_ssd_roots=[str(external_root)],
        )
    assert not artifact.exists()
    assert list(external_root.iterdir()) == [sentinel]

    class FailingWriter(_Writer):
        def __exit__(self, *_args):
            raise RuntimeError("shard write failed")

    with _server(socket_path, vmm):
        writer, mapping = _seed_weights(socket_path, vmm)
        writer.commit()
        _Writer.vmm = vmm
        FailingWriter.vmm = vmm
        monkeypatch.setattr(weight_artifact, "DeviceToFileWriter", FailingWriter)
        with pytest.raises(RuntimeError, match="shard write failed"):
            weight_artifact.save_weights(
                str(artifact),
                socket_path,
                0,
                sharded_ssd_roots=[str(external_root)],
            )
        assert not artifact.exists()
        assert list(external_root.iterdir()) == [sentinel]

        monkeypatch.setattr(weight_artifact, "DeviceToFileWriter", _Writer)
        weight_artifact.save_weights(
            str(artifact),
            socket_path,
            0,
            sharded_ssd_roots=[str(external_root)],
        )
        weight_artifact._release_mapping(vmm, mapping)
        writer.close()

    assert sentinel.read_text() == "keep"
    assert (artifact / "manifest.json").is_file()


def test_manifest_preflight_rejects_corruption_before_backend(
    tmp_path,
    monkeypatch,
) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(bytes(128))
    cases = {
        "duplicate": (
            "duplicate",
            Allocation("same", 64, "shard.bin", 0),
            Allocation("same", 64, "shard.bin", 64),
        ),
        "empty allocation": ("empty allocation", Allocation("", 64, "shard.bin", 0)),
        "zero size": ("invalid size", Allocation("a", 0, "shard.bin", 0)),
        "invalid size": ("invalid size", Allocation("a", 65, "shard.bin", 0)),
        "invalid offset": ("invalid offset", Allocation("a", 64, "shard.bin", -64)),
        "unaligned offset": ("invalid offset", Allocation("a", 64, "shard.bin", 1)),
        "not contiguous": (
            "not contiguous",
            Allocation("a", 64, "shard.bin", 0),
            Allocation("b", 64, "shard.bin", 0),
        ),
        "length": ("length", Allocation("a", 64, "shard.bin", 0)),
        "empty shard": ("empty shard", Allocation("a", 64, "", 0)),
        "missing": ("does not exist", Allocation("a", 64, "missing.bin", 0)),
        "directory": ("not a regular", Allocation("a", 64, "directory", 0)),
        "escapes": ("escapes", Allocation("a", 64, "../outside.bin", 0)),
        "configured roots": (
            "configured roots",
            Allocation("a", 64, str(outside), 0),
        ),
    }
    factory_called = False

    def create_backend(*_args, **_kwargs):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("backend constructed before preflight")

    monkeypatch.setattr(weight_artifact, "create_transfer_backend", create_backend)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: FakeVMM(granularity=64))

    for index, (case_name, values) in enumerate(cases.items()):
        message, *allocations = values
        artifact = tmp_path / f"malformed-{index}"
        shard_bytes = bytes(32) if case_name == "length" else bytes(128)
        _write_manifest(artifact, allocations, shard_bytes=shard_bytes)
        if case_name == "directory":
            (artifact / "directory").mkdir()
        with pytest.raises(RuntimeError, match=message):
            weight_artifact.hydrate_weights(str(artifact), "unused.sock", 0)
    assert not factory_called


@pytest.mark.parametrize("escape", ["file", "parent"])
def test_manifest_preflight_rejects_symlink_escape(tmp_path, escape) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "shard.bin").write_bytes(bytes(64))
    if escape == "file":
        (artifact / "shard.bin").symlink_to(outside / "shard.bin")
        shard = "shard.bin"
    else:
        (artifact / "linked").symlink_to(outside, target_is_directory=True)
        shard = "linked/shard.bin"
    manifest = weight_artifact.WeightArtifactManifest(
        1,
        (Allocation("a", 64, shard, 0),),
    )
    (artifact / "manifest.json").write_bytes(msgspec.json.encode(manifest))

    with pytest.raises(RuntimeError, match="symlink|escapes"):
        weight_artifact._load_manifest(str(artifact), 64)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("transfer", "transfer close failed"),
        ("backend", "backend close failed"),
        ("mapping", "unmap cleanup failed"),
    ],
)
@pytest.mark.timeout(10)
def test_hydrate_cleanup_failure_never_publishes(
    tmp_path,
    monkeypatch,
    failure,
    message,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    _write_manifest(
        artifact,
        (Allocation("weight", 64, "shard.bin", 0),),
        shard_bytes=bytes(64),
    )
    events = []
    vmm = _FailingCleanupVMM(events) if failure == "mapping" else _ByteVMM()
    backend = _Backend(
        vmm,
        events,
        transfer_close_failure=failure == "transfer",
        close_failure=failure == "backend",
    )
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )

    with _server(socket_path, vmm) as manager:
        with pytest.raises(RuntimeError, match=message):
            weight_artifact.hydrate_weights(str(artifact), socket_path, 0)
        assert not manager._sessions._committed
        assert not vmm.server_handles
        replacement = _GMSClientSession(socket_path, RequestedLockType.RW)
        replacement.close()


@pytest.mark.timeout(10)
def test_artifact_sessions_reject_wrong_physical_gpu(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    vmm = _ByteVMM()
    _Writer.vmm = vmm
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)
    monkeypatch.setattr(weight_artifact, "DeviceToFileWriter", _Writer)

    with _server(
        socket_path,
        vmm,
        GMSServerMemoryManager("GPU-1", vmm, 0),
    ):
        writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        writer.allocate("weight", 64)
        writer.commit()
        with pytest.raises(RuntimeError, match="another physical GPU"):
            weight_artifact.save_weights(str(artifact), socket_path, 0)
        writer.close()
    assert not artifact.exists()

    _write_manifest(
        artifact,
        (Allocation("weight", 64, "shard.bin", 0),),
        shard_bytes=bytes(64),
    )
    target_vmm = _ByteVMM()
    backend = _Backend(target_vmm)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: target_vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )
    with _server(
        socket_path,
        target_vmm,
        GMSServerMemoryManager("GPU-1", target_vmm, 0),
    ) as manager:
        with pytest.raises(RuntimeError, match="another physical GPU"):
            weight_artifact.hydrate_weights(str(artifact), socket_path, 0)
        assert not manager._sessions._committed
        assert not target_vmm.server_handles
    assert backend.closed


def test_public_api_normalizes_roots_and_rejects_invalid_workers(tmp_path) -> None:
    normalized = weight_artifact._normalize_roots(
        [str(tmp_path / "root"), str(tmp_path / "root")]
    )
    assert normalized == (str((tmp_path / "root").resolve()),)
    assert weight_artifact._normalize_roots([" "]) == ()
    assert weight_artifact._normalize_roots(str(tmp_path / "root")) == normalized
    with pytest.raises(ValueError, match="max_workers"):
        weight_artifact.save_weights("unused", "unused.sock", 0, max_workers=0)
    with pytest.raises(ValueError, match="max_workers"):
        weight_artifact.hydrate_weights("unused", "unused.sock", 0, max_workers=0)


def test_transfer_backend_names_remain_compatible() -> None:
    assert [backend.value for backend in TransferBackendKind] == [
        "nixl",
        "nixl-gds",
        "sharded-ssd",
    ]
