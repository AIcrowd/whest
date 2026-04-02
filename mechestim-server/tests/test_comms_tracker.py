"""Tests for CommsTracker — written first (TDD)."""

import pytest

from mechestim_server._comms_tracker import CommsTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tracker():
    return CommsTracker()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_request_count(tracker):
    assert tracker.summary()["request_count"] == 0


def test_initial_fetch_count(tracker):
    assert tracker.summary()["fetch_count"] == 0


def test_initial_total_bytes_sent(tracker):
    assert tracker.summary()["total_bytes_sent"] == 0


def test_initial_total_bytes_received(tracker):
    assert tracker.summary()["total_bytes_received"] == 0


def test_initial_total_comms_overhead_ns(tracker):
    assert tracker.summary()["total_comms_overhead_ns"] == 0


def test_initial_total_compute_time_ns(tracker):
    assert tracker.summary()["total_compute_time_ns"] == 0


def test_initial_overhead_ratio(tracker):
    assert tracker.summary()["overhead_ratio"] == 0.0


def test_initial_summary_has_all_keys(tracker):
    keys = tracker.summary().keys()
    expected = {
        "request_count",
        "fetch_count",
        "total_bytes_sent",
        "total_bytes_received",
        "total_comms_overhead_ns",
        "total_compute_time_ns",
        "overhead_ratio",
    }
    assert set(keys) == expected


# ---------------------------------------------------------------------------
# Record one non-fetch request
# ---------------------------------------------------------------------------

def test_record_non_fetch_increments_request_count(tracker):
    tracker.record_request(
        bytes_received=100,
        bytes_sent=200,
        comms_overhead_ns=1000,
        compute_time_ns=5000,
        is_fetch=False,
    )
    assert tracker.summary()["request_count"] == 1


def test_record_non_fetch_does_not_increment_fetch_count(tracker):
    tracker.record_request(
        bytes_received=100,
        bytes_sent=200,
        comms_overhead_ns=1000,
        compute_time_ns=5000,
        is_fetch=False,
    )
    assert tracker.summary()["fetch_count"] == 0


def test_record_non_fetch_accumulates_bytes_received(tracker):
    tracker.record_request(
        bytes_received=100,
        bytes_sent=200,
        comms_overhead_ns=1000,
        compute_time_ns=5000,
        is_fetch=False,
    )
    assert tracker.summary()["total_bytes_received"] == 100


def test_record_non_fetch_accumulates_bytes_sent(tracker):
    tracker.record_request(
        bytes_received=100,
        bytes_sent=200,
        comms_overhead_ns=1000,
        compute_time_ns=5000,
        is_fetch=False,
    )
    assert tracker.summary()["total_bytes_sent"] == 200


def test_record_non_fetch_accumulates_comms_overhead(tracker):
    tracker.record_request(
        bytes_received=100,
        bytes_sent=200,
        comms_overhead_ns=1000,
        compute_time_ns=5000,
        is_fetch=False,
    )
    assert tracker.summary()["total_comms_overhead_ns"] == 1000


def test_record_non_fetch_accumulates_compute_time(tracker):
    tracker.record_request(
        bytes_received=100,
        bytes_sent=200,
        comms_overhead_ns=1000,
        compute_time_ns=5000,
        is_fetch=False,
    )
    assert tracker.summary()["total_compute_time_ns"] == 5000


# ---------------------------------------------------------------------------
# Record one fetch request
# ---------------------------------------------------------------------------

def test_record_fetch_increments_fetch_count(tracker):
    tracker.record_request(
        bytes_received=50,
        bytes_sent=10,
        comms_overhead_ns=500,
        compute_time_ns=200,
        is_fetch=True,
    )
    assert tracker.summary()["fetch_count"] == 1


def test_record_fetch_also_increments_request_count(tracker):
    tracker.record_request(
        bytes_received=50,
        bytes_sent=10,
        comms_overhead_ns=500,
        compute_time_ns=200,
        is_fetch=True,
    )
    assert tracker.summary()["request_count"] == 1


# ---------------------------------------------------------------------------
# Multiple requests accumulate
# ---------------------------------------------------------------------------

def test_multiple_requests_accumulate_request_count(tracker):
    for _ in range(3):
        tracker.record_request(
            bytes_received=10,
            bytes_sent=20,
            comms_overhead_ns=100,
            compute_time_ns=400,
            is_fetch=False,
        )
    assert tracker.summary()["request_count"] == 3


def test_multiple_requests_accumulate_fetch_count(tracker):
    tracker.record_request(
        bytes_received=10, bytes_sent=20, comms_overhead_ns=100,
        compute_time_ns=400, is_fetch=True,
    )
    tracker.record_request(
        bytes_received=10, bytes_sent=20, comms_overhead_ns=100,
        compute_time_ns=400, is_fetch=False,
    )
    tracker.record_request(
        bytes_received=10, bytes_sent=20, comms_overhead_ns=100,
        compute_time_ns=400, is_fetch=True,
    )
    assert tracker.summary()["fetch_count"] == 2


def test_multiple_requests_accumulate_bytes_sent(tracker):
    tracker.record_request(
        bytes_received=0, bytes_sent=100,
        comms_overhead_ns=0, compute_time_ns=0, is_fetch=False,
    )
    tracker.record_request(
        bytes_received=0, bytes_sent=250,
        comms_overhead_ns=0, compute_time_ns=0, is_fetch=False,
    )
    assert tracker.summary()["total_bytes_sent"] == 350


def test_multiple_requests_accumulate_bytes_received(tracker):
    tracker.record_request(
        bytes_received=300, bytes_sent=0,
        comms_overhead_ns=0, compute_time_ns=0, is_fetch=False,
    )
    tracker.record_request(
        bytes_received=150, bytes_sent=0,
        comms_overhead_ns=0, compute_time_ns=0, is_fetch=False,
    )
    assert tracker.summary()["total_bytes_received"] == 450


def test_multiple_requests_accumulate_comms_overhead(tracker):
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=1000, compute_time_ns=0, is_fetch=False,
    )
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=2000, compute_time_ns=0, is_fetch=False,
    )
    assert tracker.summary()["total_comms_overhead_ns"] == 3000


def test_multiple_requests_accumulate_compute_time(tracker):
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=0, compute_time_ns=4000, is_fetch=False,
    )
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=0, compute_time_ns=6000, is_fetch=False,
    )
    assert tracker.summary()["total_compute_time_ns"] == 10000


# ---------------------------------------------------------------------------
# Overhead ratio calculation
# ---------------------------------------------------------------------------

def test_overhead_ratio_calculation(tracker):
    # comms=1000, compute=4000 → ratio = 1000/(1000+4000) = 0.2
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=1000, compute_time_ns=4000, is_fetch=False,
    )
    assert tracker.summary()["overhead_ratio"] == pytest.approx(0.2)


def test_overhead_ratio_all_comms(tracker):
    # comms=5000, compute=0 → ratio = 1.0
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=5000, compute_time_ns=0, is_fetch=False,
    )
    assert tracker.summary()["overhead_ratio"] == pytest.approx(1.0)


def test_overhead_ratio_all_compute(tracker):
    # comms=0, compute=5000 → ratio = 0.0
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=0, compute_time_ns=5000, is_fetch=False,
    )
    assert tracker.summary()["overhead_ratio"] == pytest.approx(0.0)


def test_overhead_ratio_accumulates_across_requests(tracker):
    # request 1: comms=1000, compute=1000
    # request 2: comms=3000, compute=1000
    # total comms=4000, compute=2000 → ratio = 4000/6000 ≈ 0.6667
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=1000, compute_time_ns=1000, is_fetch=False,
    )
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=3000, compute_time_ns=1000, is_fetch=False,
    )
    assert tracker.summary()["overhead_ratio"] == pytest.approx(4000 / 6000)


def test_overhead_ratio_is_zero_when_total_is_zero(tracker):
    """When both comms and compute are 0, overhead_ratio must be 0.0 (no ZeroDivisionError)."""
    assert tracker.summary()["overhead_ratio"] == 0.0


def test_overhead_ratio_is_float(tracker):
    tracker.record_request(
        bytes_received=0, bytes_sent=0,
        comms_overhead_ns=1000, compute_time_ns=3000, is_fetch=False,
    )
    assert isinstance(tracker.summary()["overhead_ratio"], float)
