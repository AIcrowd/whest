"""Tests for mechestim._comms_tracker module."""

import pytest
from mechestim._comms_tracker import ClientCommsTracker


class TestClientCommsTrackerInitialState:
    def test_request_count_zero(self):
        t = ClientCommsTracker()
        assert t.request_count == 0

    def test_total_round_trip_ns_zero(self):
        t = ClientCommsTracker()
        assert t.total_round_trip_ns == 0

    def test_total_request_bytes_zero(self):
        t = ClientCommsTracker()
        assert t.total_request_bytes == 0

    def test_total_response_bytes_zero(self):
        t = ClientCommsTracker()
        assert t.total_response_bytes == 0


class TestClientCommsTrackerSingleRecord:
    def setup_method(self):
        self.tracker = ClientCommsTracker()
        self.tracker.record(
            round_trip_ns=1_000_000, request_bytes=128, response_bytes=256
        )

    def test_request_count_is_one(self):
        assert self.tracker.request_count == 1

    def test_round_trip_accumulated(self):
        assert self.tracker.total_round_trip_ns == 1_000_000

    def test_request_bytes_accumulated(self):
        assert self.tracker.total_request_bytes == 128

    def test_response_bytes_accumulated(self):
        assert self.tracker.total_response_bytes == 256


class TestClientCommsTrackerAccumulation:
    def setup_method(self):
        self.tracker = ClientCommsTracker()
        self.tracker.record(round_trip_ns=1_000, request_bytes=100, response_bytes=200)
        self.tracker.record(round_trip_ns=2_000, request_bytes=150, response_bytes=300)
        self.tracker.record(round_trip_ns=3_000, request_bytes=50, response_bytes=100)

    def test_request_count_three(self):
        assert self.tracker.request_count == 3

    def test_total_round_trip(self):
        assert self.tracker.total_round_trip_ns == 6_000

    def test_total_request_bytes(self):
        assert self.tracker.total_request_bytes == 300

    def test_total_response_bytes(self):
        assert self.tracker.total_response_bytes == 600


class TestClientCommsTrackerKeywordOnly:
    def test_record_requires_keyword_args(self):
        """record() must use keyword-only arguments."""
        t = ClientCommsTracker()
        with pytest.raises(TypeError):
            t.record(1_000, 128, 256)  # positional args should fail
