"""Client-side communications tracker."""

from __future__ import annotations


class ClientCommsTracker:
    """Accumulates network statistics for mechestim client requests."""

    def __init__(self) -> None:
        self._request_count: int = 0
        self._total_round_trip_ns: int = 0
        self._total_request_bytes: int = 0
        self._total_response_bytes: int = 0

    def record(
        self,
        *,
        round_trip_ns: int,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        """Record statistics for a single request/response round trip."""
        self._request_count += 1
        self._total_round_trip_ns += round_trip_ns
        self._total_request_bytes += request_bytes
        self._total_response_bytes += response_bytes

    @property
    def request_count(self) -> int:
        """Total number of requests recorded."""
        return self._request_count

    @property
    def total_round_trip_ns(self) -> int:
        """Sum of all round-trip times in nanoseconds."""
        return self._total_round_trip_ns

    @property
    def total_request_bytes(self) -> int:
        """Sum of all request payload sizes in bytes."""
        return self._total_request_bytes

    @property
    def total_response_bytes(self) -> int:
        """Sum of all response payload sizes in bytes."""
        return self._total_response_bytes
