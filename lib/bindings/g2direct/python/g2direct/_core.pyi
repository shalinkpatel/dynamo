class HiCacheG2Direct:
    def __init__(
        self,
        tp_rank: int,
        tp_world_size: int,
        page_size: int,
        lease_ttl_sec: int = 30,
        max_transfer_mib: int = 512,
        ucx_devices: str | None = None,
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
    def lookup_prefix(self, page_hashes: list[str]) -> int: ...
    def lookup_mask(self, page_hashes: list[str]) -> list[bool]: ...
    def fetch(
        self,
        page_hashes: list[str],
        ptrs: list[int],
        sizes: list[int],
        segments_per_page: int,
    ) -> tuple[int, int | None]: ...
    def is_settled(self, ticket: int) -> bool: ...
    def stats(self) -> dict[str, int]: ...
