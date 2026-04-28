"""CommsTracker — accumulates per-request timing and byte counts for a session."""

from __future__ import annotations


class CommsTracker:
    """Accumulates communication and compute statistics across requests in a session."""

    def __init__(self) -> None:
        self._request_count: int = 0
        self._fetch_count: int = 0
        self._total_bytes_sent: int = 0
        self._total_bytes_received: int = 0
        self._total_comms_overhead_ns: int = 0
        self._total_compute_time_ns: int = 0

    def record_request(
        self,
        *,
        bytes_received: int,
        bytes_sent: int,
        comms_overhead_ns: int,
        compute_time_ns: int,
        is_fetch: bool,
    ) -> None:
        """Accumulate statistics for a single request.

        Parameters
        ----------
        bytes_received:
            Number of bytes received in this request.
        bytes_sent:
            Number of bytes sent in this request.
        comms_overhead_ns:
            Communications overhead for this request, in nanoseconds.
        compute_time_ns:
            Compute time for this request, in nanoseconds.
        is_fetch:
            Whether this request is a fetch (array retrieval) request.
        """
        self._request_count += 1
        if is_fetch:
            self._fetch_count += 1
        self._total_bytes_sent += bytes_sent
        self._total_bytes_received += bytes_received
        self._total_comms_overhead_ns += comms_overhead_ns
        self._total_compute_time_ns += compute_time_ns

    def summary(self) -> dict:
        """Return a summary of accumulated statistics.

        Returns
        -------
        dict with keys:
            request_count: total number of requests recorded
            fetch_count: number of fetch requests recorded
            total_bytes_sent: total bytes sent across all requests
            total_bytes_received: total bytes received across all requests
            total_comms_overhead_ns: total communications overhead in nanoseconds
            total_compute_time_ns: total compute time in nanoseconds
            overhead_ratio: comms / (comms + compute); 0.0 if total is 0
        """
        total_ns = self._total_comms_overhead_ns + self._total_compute_time_ns
        if total_ns == 0:
            overhead_ratio = 0.0
        else:
            overhead_ratio = self._total_comms_overhead_ns / total_ns

        return {
            "request_count": self._request_count,
            "fetch_count": self._fetch_count,
            "total_bytes_sent": self._total_bytes_sent,
            "total_bytes_received": self._total_bytes_received,
            "total_comms_overhead_ns": self._total_comms_overhead_ns,
            "total_compute_time_ns": self._total_compute_time_ns,
            "overhead_ratio": overhead_ratio,
        }
