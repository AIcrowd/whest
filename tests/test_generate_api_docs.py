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
    page_text = page_path.read_text()
    assert 'title: "`we.absolute`"' in page_text
    assert '<OperationDocPage name="absolute" />' in page_text

    refs_manifest = mod.json.loads(refs_manifest_path.read_text())
    assert refs_manifest["abs"]["label"] == "`we.absolute`"
    assert refs_manifest["abs"]["href"] == "/docs/api/ops/absolute"
    assert refs_manifest["abs"]["canonical_name"] == "absolute"


def test_generate_ops_json_preserves_alias_weights(tmp_path):
    mod = load_generate_api_docs_module()

    registry = {
        "abs": {
            "category": "counted_unary",
            "module": "numpy",
            "notes": "Alias row.",
        },
        "absolute": {
            "category": "counted_unary",
            "module": "numpy",
            "notes": "Canonical row.",
        },
        "divmod": {
            "category": "counted_binary",
            "module": "numpy",
            "notes": "Alias row.",
        },
        "floor_divide": {
            "category": "counted_binary",
            "module": "numpy",
            "notes": "Canonical row.",
        },
    }

    mod.PUBLIC_DIR = tmp_path
    mod.load_alias_map = lambda _registry: {
        "abs": "absolute",
        "divmod": "floor_divide",
    }
    mod.load_operation_weights = lambda: {
        "abs": 7.0,
        "floor_divide": 16.0,
    }

    mod.generate_ops_json(registry)

    ops_manifest = mod.json.loads((tmp_path / "ops.json").read_text())
    operations = {entry["name"]: entry for entry in ops_manifest["operations"]}

    assert operations["absolute"]["weight"] == 7.0
    assert operations["abs"]["weight"] == 7.0
    assert operations["floor_divide"]["weight"] == 16.0
    assert operations["divmod"]["weight"] == 16.0


def test_example_coverage_marks_owned_examples(tmp_path):
    mod = load_generate_api_docs_module()

    example_dir = tmp_path / "content" / "api-examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    (example_dir / "absolute.mdx").write_text(
        "```python\nwith we.BudgetContext(...):\n    y = we.absolute(x)\n```"
    )

    coverage = mod.build_example_coverage(["absolute", "sum"], example_root=example_dir)

    assert coverage["absolute"]["has_whest_examples"] is True
    assert coverage["sum"]["has_whest_examples"] is False
    assert coverage["absolute"]["example_count"] == 1


def test_load_whest_example_html_renders_fenced_code(tmp_path):
    mod = load_generate_api_docs_module()

    example_dir = tmp_path / "content" / "api-examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    (example_dir / "absolute.mdx").write_text(
        "```python\nimport whest as we\n\nwith we.BudgetContext(...):\n    y = we.absolute(x)\n```"
    )

    html = mod.load_whest_example_html("absolute", example_root=example_dir)

    assert '<pre><code class="language-python">' in html
    assert "we.absolute(x)" in html


def test_parse_numpy_docstring_builds_structured_sections():
    mod = load_generate_api_docs_module()

    raw_doc = """
    Compute the histogram of a dataset.

    Parameters
    ----------
    a : array_like
        Input data.
    bins : int or sequence, optional
        Bin count or edges.

    Returns
    -------
    hist : ndarray
        Histogram values.
    bin_edges : ndarray
        Bin boundaries.

    See Also
    --------
    numpy.histogram_bin_edges, numpy.digitize

    Notes
    -----
    All but the last bin is half-open.

    Examples
    --------
    >>> import numpy as np
    >>> np.histogram([0, 1, 1, 2], bins=[0, 1, 2, 3])
    (array([1, 2, 1]), array([0, 1, 2, 3]))
    """

    doc = mod.parse_numpy_docstring(raw_doc)

    assert doc.summary == "Compute the histogram of a dataset."
    assert doc.parameters[0].name == "a"
    assert doc.parameters[0].type == "array_like"
    assert doc.returns[1].name == "bin_edges"
    assert doc.see_also[0].target == "numpy.histogram_bin_edges"
    assert "half-open" in doc.notes[0]
    assert "np.histogram" in doc.examples[0].code


def test_rewrite_api_refs_swaps_numpy_symbols_for_whest():
    mod = load_generate_api_docs_module()

    rewritten = mod.rewrite_api_refs(
        "Use np.histogram with numpy.digitize or numpy.histogram_bin_edges."
    )

    assert "np.histogram" not in rewritten
    assert "numpy.digitize" not in rewritten
    assert "we.histogram" in rewritten
    assert "we.digitize" in rewritten
    assert "we.histogram_bin_edges" in rewritten


def test_derive_example_from_doctest_keeps_code_and_output():
    mod = load_generate_api_docs_module()

    example = mod.derive_example_from_upstream(
        ">>> import numpy as np\n"
        ">>> np.histogram([0, 1, 1, 2], bins=[0, 1, 2, 3])\n"
        "(array([1, 2, 1]), array([0, 1, 2, 3]))"
    )

    assert "import whest as we" in example.code
    assert "we.histogram" in example.code
    assert "array([1, 2, 1])" in example.output
