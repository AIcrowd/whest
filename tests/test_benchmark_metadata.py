"""Tests for benchmark hardware/software metadata collection."""

from benchmarks._metadata import collect_metadata


def test_collect_metadata_has_required_keys():
    meta = collect_metadata(dtype="float64", repeats=10, distributions=3)
    assert "timestamp" in meta
    assert "hardware" in meta
    assert "software" in meta
    assert "benchmark_config" in meta


def test_hardware_keys():
    meta = collect_metadata(dtype="float64", repeats=10, distributions=3)
    hw = meta["hardware"]
    assert "cpu_model" in hw
    assert "cpu_cores" in hw
    assert "arch" in hw
    assert "ram_gb" in hw


def test_software_keys():
    meta = collect_metadata(dtype="float64", repeats=10, distributions=3)
    sw = meta["software"]
    assert "os" in sw
    assert "python" in sw
    assert "numpy" in sw


def test_benchmark_config_passes_through():
    meta = collect_metadata(dtype="float32", repeats=20, distributions=5)
    cfg = meta["benchmark_config"]
    assert cfg["dtype"] == "float32"
    assert cfg["repeats"] == 20
    assert cfg["distributions"] == 5
