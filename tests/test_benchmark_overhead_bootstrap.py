import json
from pathlib import Path

from benchmarks._metadata import collect_environment_metadata, collect_metadata
from benchmarks.overhead import SCHEMA_VERSION


def test_collect_environment_metadata_has_required_sections():
    meta = collect_environment_metadata()
    assert "timestamp" in meta
    assert "hardware" in meta
    assert "software" in meta


def test_collect_metadata_reuses_environment_payload():
    meta = collect_metadata(dtype="float64", repeats=10, distributions=3)
    assert "benchmark_config" in meta
    assert "hardware" in meta
    assert "software" in meta


def test_overhead_schema_version_is_integer():
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_static_threshold_and_exclusion_files_exist():
    threshold_path = Path("benchmarks/overhead_thresholds.json")
    exclusion_path = Path("benchmarks/overhead_exclusions.json")
    assert threshold_path.exists()
    assert exclusion_path.exists()
    assert json.loads(threshold_path.read_text())["schema_version"] == SCHEMA_VERSION
    assert json.loads(exclusion_path.read_text())["schema_version"] == SCHEMA_VERSION
