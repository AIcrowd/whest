import json
from pathlib import Path

from benchmarks import _metadata
from benchmarks.overhead import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_collect_metadata_reuses_environment_payload(monkeypatch):
    sentinel = {
        "timestamp": "2026-04-19T00:00:00+00:00",
        "hardware": {"cpu_model": "test"},
        "software": {"python": "test"},
    }

    monkeypatch.setattr(_metadata, "collect_environment_metadata", lambda: sentinel)
    meta = _metadata.collect_metadata(dtype="float64", repeats=10, distributions=3)

    assert meta["timestamp"] == sentinel["timestamp"]
    assert meta["hardware"] is sentinel["hardware"]
    assert meta["software"] is sentinel["software"]
    assert "benchmark_config" in meta


def test_overhead_schema_version_is_integer():
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_static_threshold_and_exclusion_files_exist():
    threshold_path = ROOT / "benchmarks" / "overhead_thresholds.json"
    exclusion_path = ROOT / "benchmarks" / "overhead_exclusions.json"
    assert threshold_path.exists()
    assert exclusion_path.exists()

    thresholds = json.loads(threshold_path.read_text())
    exclusions = json.loads(exclusion_path.read_text())

    assert thresholds == {
        "schema_version": SCHEMA_VERSION,
        "policy_version": "bootstrap",
        "default": {"ratio_max": 6.0},
        "family": {
            "pointwise": {"ratio_max": 6.0},
            "reductions": {"ratio_max": 6.0},
            "contractions": {"ratio_max": 4.0},
            "linalg": {"ratio_max": 4.0},
            "fft": {"ratio_max": 4.0},
            "random": {"ratio_max": 7.0},
            "stats": {"ratio_max": 7.0},
        },
        "surface": {"operator": {"ratio_max": 7.0}},
    }

    assert exclusions == {
        "schema_version": SCHEMA_VERSION,
        "excluded": {
            "budget_summary": "diagnostic helper, not a NumPy baseline candidate",
            "budget_live": "diagnostic helper, not a NumPy baseline candidate",
            "configure": "configuration helper, not a runtime benchmark candidate",
            "namespace": "context helper, not a NumPy baseline candidate",
        },
        "unsupported_for_ratio": {
            "budget": "context-manager factory, not comparable to a single NumPy call"
        },
    }
