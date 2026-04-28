from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "generate_api_docs.py"


spec = importlib.util.spec_from_file_location("generate_api_docs_public_api", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
generate_api_docs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generate_api_docs
spec.loader.exec_module(generate_api_docs)


def test_canonical_api_href_uses_import_path_routes():
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


def test_stats_refs_use_stats_namespace_instead_of_numpy_namespace():
    assert (
        generate_api_docs.flopscope_ref("stats.norm.cdf", "flopscope.stats")
        == "flopscope.stats.norm.cdf"
    )
    assert (
        generate_api_docs.numpy_ref("stats.norm.cdf", "flopscope.stats")
        == "scipy.stats.norm.cdf"
    )


def test_public_api_manifest_covers_representative_namespaces():
    manifest = generate_api_docs.build_public_api_manifest(
        generate_api_docs.load_registry()
    )

    assert manifest["flopscope.BudgetContext"]["href"] == (
        "/docs/api/flopscope/budget-context/"
    )
    assert manifest["flopscope.BudgetContext"]["kind"] == "class"

    assert manifest["flopscope.numpy.einsum"]["href"] == "/docs/api/numpy/einsum/"
    assert manifest["flopscope.numpy.einsum"]["kind"] == "function"

    assert manifest["flopscope.stats.norm"]["href"] == "/docs/api/stats/norm/"
    assert manifest["flopscope.stats.norm"]["kind"] == "object"

    assert manifest["flopscope.accounting.einsum_cost"]["href"] == (
        "/docs/api/accounting/einsum-cost/"
    )
    assert manifest["flopscope.accounting.einsum_cost"]["kind"] == "function"


def test_operation_cost_index_entries_include_numpy_and_stats_callables_only():
    manifest = generate_api_docs.build_public_api_manifest(
        generate_api_docs.load_registry()
    )
    entries = generate_api_docs.build_operation_cost_index_entries(manifest)
    import_paths = {entry["import_path"] for entry in entries}

    assert "flopscope.numpy.einsum" in import_paths
    assert "flopscope.stats.norm.pdf" in import_paths

    assert "flopscope.accounting.einsum_cost" not in import_paths
    assert "flopscope.BudgetContext" not in import_paths
    assert "flopscope.stats.norm" not in import_paths
