from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_generate_api_docs_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_api_docs.py"
    spec = importlib.util.spec_from_file_location("generate_api_docs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_area_and_slug_for_supported_ops():
    mod = load_generate_api_docs_module()

    assert mod.normalize_area("numpy") == "core"
    assert mod.normalize_area("numpy.linalg") == "linalg"
    assert mod.normalize_area("numpy.fft") == "fft"
    assert mod.normalize_area("numpy.random") == "random"
    assert mod.normalize_area("whest.stats") == "stats"
    assert mod.slug_for_operation("absolute") == "absolute"
    assert mod.slug_for_operation("linalg.svd") == "linalg-svd"
    assert mod.slug_for_operation("stats.norm.pdf") == "stats-norm-pdf"


def test_display_type_for_category():
    mod = load_generate_api_docs_module()

    assert mod.display_type_for_category("free") == "free"
    assert mod.display_type_for_category("counted_unary") == "counted"
    assert mod.display_type_for_category("counted_custom") == "custom"
    assert mod.display_type_for_category("blacklisted") == "blocked"
