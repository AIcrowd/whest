from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in list(sys.modules):
    if name == "whest" or name.startswith("whest."):
        sys.modules.pop(name, None)

import whest as we
from scripts.generate_api_docs import (
    canonical_api_href_for_name,
    collect_public_api_surface_names,
)


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


def test_public_api_surface_covers_new_symbol_namespaces() -> None:
    surface = collect_public_api_surface_names()

    assert "random.symmetric" in surface
    assert "testing.assert_allclose" in surface
    assert "stats.norm" in surface
    assert "flops.einsum_cost" in surface
    assert "budget_summary" in surface


def test_generated_public_api_routes_cover_surface_and_reserve_authored_namespaces() -> (
    None
):
    routes = json.loads(
        (ROOT / "website/.generated/public-api-routes.json").read_text()
    )
    refs = json.loads((ROOT / "website/.generated/public-api-refs.json").read_text())
    surface = collect_public_api_surface_names()
    namespace_roots = {"random", "stats", "flops", "testing"}

    for name in surface:
        assert name in refs
        path_key = refs[name]["href"].removeprefix("/docs/api/").strip("/")
        assert path_key in routes
        assert routes[path_key]["href"] == refs[name]["href"]

    assert "random" not in routes
    assert "stats" not in routes
    assert "flops" not in routes
    assert "testing" not in routes

    expected_symbol_entries = {
        refs[name]["canonical_name"]: refs[name]["href"]
        for name in surface
        if refs[name]["kind"] != "op"
    }
    generated_symbol_refs = {
        entry["canonical_name"]: entry["href"]
        for entry in refs.values()
        if entry["kind"] != "op"
    }
    generated_symbol_routes = {
        entry["canonical_name"]: entry["href"]
        for entry in routes.values()
        if entry["kind"] == "symbol"
    }
    assert generated_symbol_refs == expected_symbol_entries
    assert generated_symbol_routes == expected_symbol_entries

    for name in (
        "random.symmetric",
        "stats.norm",
        "stats.norm.pdf",
        "flops.einsum_cost",
        "testing.assert_allclose",
    ):
        path_key = (
            canonical_api_href_for_name(name).removeprefix("/docs/api/").strip("/")
        )
        assert path_key in routes
        assert routes[path_key]["href"] == canonical_api_href_for_name(name)


def test_required_public_helpers_have_structured_docstrings() -> None:
    required = {
        "budget": we.budget,
        "budget_reset": we.budget_reset,
        "budget_summary_dict": we.budget_summary_dict,
        "budget_live": we.budget_live,
        "budget_summary": we.budget_summary,
        "configure": we.configure,
        "clear_einsum_cache": we.clear_einsum_cache,
        "einsum_cache_info": we.einsum_cache_info,
    }

    for name, obj in required.items():
        doc = inspect.getdoc(obj) or ""
        assert "Parameters" in doc, name
        assert "Returns" in doc, name

    assert "Examples" in (inspect.getdoc(we.budget_live) or "")
