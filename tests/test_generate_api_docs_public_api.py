from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import flopscope as flops

MODULE_PATH = ROOT / "scripts" / "generate_api_docs.py"


_spec = importlib.util.spec_from_file_location(
    "generate_api_docs_public_api", MODULE_PATH
)
assert _spec is not None
assert _spec.loader is not None
generate_api_docs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = generate_api_docs
_spec.loader.exec_module(generate_api_docs)

canonical_api_href_for_name = generate_api_docs.canonical_api_href_for_name
collect_public_api_surface_names = generate_api_docs.collect_public_api_surface_names


def test_canonical_api_href_mirrors_import_path() -> None:
    assert canonical_api_href_for_name("einsum") == "/docs/api/einsum/"
    assert (
        canonical_api_href_for_name("random.symmetric") == "/docs/api/random/symmetric/"
    )
    assert canonical_api_href_for_name("stats.norm") == "/docs/api/stats/norm/"
    assert canonical_api_href_for_name("stats.norm.pdf") == "/docs/api/stats/norm/pdf/"
    assert (
        canonical_api_href_for_name("flops.einsum_cost")
        == "/docs/api/flops/einsum-cost/"
    )
    assert canonical_api_href_for_name("budget_summary") == "/docs/api/budget-summary/"


def test_canonical_api_href_uses_import_path_routes() -> None:
    assert (
        generate_api_docs.canonical_api_href("flopscope.numpy.einsum")
        == "/docs/api/numpy/einsum/"
    )
    assert (
        generate_api_docs.canonical_api_href("flopscope.numpy.linalg.svd")
        == "/docs/api/numpy/linalg/svd/"
    )
    assert (
        generate_api_docs.canonical_api_href("flopscope.symmetrize")
        == "/docs/api/flopscope/symmetrize/"
    )
    assert (
        generate_api_docs.canonical_api_href("flopscope.accounting.einsum_cost")
        == "/docs/api/accounting/einsum-cost/"
    )
    assert (
        generate_api_docs.canonical_api_href("flopscope.BudgetContext")
        == "/docs/api/flopscope/budget-context/"
    )


def test_stats_refs_use_stats_namespace_instead_of_numpy_namespace() -> None:
    assert (
        generate_api_docs.flopscope_ref("stats.norm.cdf", "flopscope.stats")
        == "flops.stats.norm.cdf"
    )
    assert (
        generate_api_docs.numpy_ref("stats.norm.cdf", "flopscope.stats")
        == "scipy.stats.norm.cdf"
    )


def test_public_api_surface_covers_new_symbol_namespaces() -> None:
    surface = collect_public_api_surface_names()

    assert "random.symmetric" in surface
    assert "testing.assert_allclose" in surface
    assert "stats.norm" in surface
    assert "flops.einsum_cost" in surface
    assert "budget_summary" in surface


def test_required_public_helpers_have_structured_docstrings() -> None:
    import flopscope.numpy as fnp

    required = {
        "budget": flops.budget,
        "budget_reset": flops.budget_reset,
        "budget_summary_dict": flops.budget_summary_dict,
        "budget_live": flops.budget_live,
        "budget_summary": flops.budget_summary,
        "configure": flops.configure,
        "clear_einsum_cache": fnp.clear_einsum_cache,
        "einsum_cache_info": fnp.einsum_cache_info,
    }

    for name, obj in required.items():
        doc = inspect.getdoc(obj) or ""
        assert "Parameters" in doc, name
        assert "Returns" in doc, name

    assert "Examples" in (inspect.getdoc(flops.budget_live) or "")
