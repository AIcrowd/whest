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
    assert mod.normalize_area("whest._polynomial") == "core"
    assert mod.normalize_area("whest._window") == "core"
    assert mod.normalize_area("whest._unwrap") == "core"
    assert mod.slug_for_operation("absolute") == "absolute"
    assert mod.slug_for_operation("linalg.svd") == "linalg-svd"
    assert mod.slug_for_operation("stats.norm.pdf") == "stats-norm-pdf"


def test_display_type_for_category():
    mod = load_generate_api_docs_module()

    assert mod.display_type_for_category("free") == "free"
    assert mod.display_type_for_category("counted_unary") == "counted"
    assert mod.display_type_for_category("counted_custom") == "custom"
    assert mod.display_type_for_category("blacklisted") == "blocked"


def test_write_generated_operation_artifacts(tmp_path):
    mod = load_generate_api_docs_module()

    records = [
        mod.OperationDocRecord(
            name="absolute",
            canonical_name="absolute",
            slug="absolute",
            href="/docs/api/ops/absolute",
            area="core",
            whest_ref="`we.absolute`",
            numpy_ref="`np.absolute`",
            category="counted_unary",
            display_type="counted",
            cost_formula="numel(output)",
            cost_formula_latex=r"$\text{numel}(\text{output})$",
            weight=1.0,
            notes="Element-wise absolute value.",
            aliases=["abs"],
            signature="we.absolute(...)",
            api_docs_html="",
            whest_examples_html="",
        )
    ]

    mod.write_operation_doc_artifacts(records, tmp_path)

    page_path = tmp_path / "content" / "docs" / "api" / "ops" / "absolute.mdx"
    docs_manifest_path = tmp_path / ".generated" / "op-docs.json"
    refs_manifest_path = tmp_path / ".generated" / "op-refs.json"

    assert page_path.exists()
    assert docs_manifest_path.exists()
    assert refs_manifest_path.exists()
    assert '<OperationDocPage name="absolute" />' in page_path.read_text()

    refs_manifest = mod.json.loads(refs_manifest_path.read_text())
    assert refs_manifest["abs"]["label"] == "`we.absolute`"
    assert refs_manifest["abs"]["href"] == "/docs/api/ops/absolute"
    assert refs_manifest["abs"]["canonical_name"] == "absolute"
