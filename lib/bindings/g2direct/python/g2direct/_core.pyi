from collections.abc import Callable

class HiCacheG2Direct:
    def __init__(
        self,
        tp_rank: int,
        tp_world_size: int,
        page_size: int,
        lease_ttl_sec: int = 30,
        max_transfer_mib: int = 512,
        ucx_devices: str | None = None,
        dp_rank: int = 0,
        cp_rank: int = 0,
        l3_directory_pages: int = 0,
    ) -> None: ...
    def register_host_region(self, addr: int, size: int, page_bytes: int) -> None: ...
    def publish(
        self,
        page_hashes: list[str],
        ptrs: list[int],
        sizes: list[int],
        segments_per_page: int,
    ) -> list[bool]: ...
    def unpublish(self, page_hashes: list[str]) -> None: ...
    def invalidate_slots(
        self, ptrs: list[int], sizes: list[int], ptrs_per_page: int
    ) -> int: ...
    def set_l3_stager(
        self, stager: Callable[[str, list[str], list[int]], list[bool]]
    ) -> None: ...
    def register_staging_pool(
        self, pool: str, ptrs: list[int], sizes: list[int], segments_per_page: int
    ) -> int: ...
    def publish_staged(
        self,
        pool: str,
        page_hashes: list[str],
        tokens: list[str],
        page_bytes: int,
        objects: list[list[str]] | None = None,
    ) -> int: ...
    def subscribe_l3_events(self, endpoint: str) -> None: ...
    def withdraw_staged(self, page_hashes: list[str]) -> int: ...
    def lookup_prefix(
        self, page_hashes: list[str], include_staged: bool = False
    ) -> int: ...
    def lookup_mask(
        self, page_hashes: list[str], include_staged: bool = False
    ) -> list[bool]: ...
    def fetch(
        self,
        page_hashes: list[str],
        ptrs: list[int],
        sizes: list[int],
        segments_per_page: int,
        staged_from: int | None = None,
    ) -> tuple[int, int | None]: ...
    def is_settled(self, ticket: int) -> bool: ...
    def stats(self) -> dict[str, int]: ...
