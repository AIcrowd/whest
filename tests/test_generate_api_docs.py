from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_generate_api_docs_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_api_docs.py"
    )
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
            whest_ref="we.absolute",
            numpy_ref="np.absolute",
            category="counted_unary",
            display_type="counted",
            cost_formula="numel(output)",
            cost_formula_latex=r"$\text{numel}(\text{output})$",
            weight=1.0,
            notes="Element-wise absolute value.",
            aliases=["abs"],
            signature="we.absolute(...)",
            summary="Return the absolute value element-wise.",
            provenance_label="Adapted from NumPy docs",
            provenance_url="https://numpy.org/doc/stable/reference/generated/numpy.absolute.html",
            whest_source_url="https://github.com/AIcrowd/whest/blob/main/src/whest/_pointwise.py#L249",
            upstream_source_url="https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/code_generators/ufunc_docstrings.py#L1",
            parameters=[
                mod.DocField(
                    name="x",
                    type="array_like",
                    body=["Input array."],
                )
            ],
            returns=[
                mod.DocField(
                    name="absolute",
                    type="ndarray",
                    body=["Absolute values of `x`."],
                )
            ],
            see_also=[
                mod.DocLink(
                    label="we.fabs",
                    target="fabs",
                    description="Absolute value for floats.",
                )
            ],
            notes_sections=["For complex input, `absolute` returns the magnitude."],
            example=mod.DocExample(
                code="import whest as we\n\nwe.absolute([-1, 2, -3])",
                output="array([1, 2, 3])",
            ),
            previous=None,
            next=mod.OperationNavLink(href="/docs/api/ops/add", label="we.add"),
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
    assert 'title: "we.absolute"' in page_text
    assert '<OperationDocPage name="absolute" />' in page_text

    docs_manifest = mod.json.loads(docs_manifest_path.read_text())
    absolute = docs_manifest["absolute"]
    assert absolute["summary"] == "Return the absolute value element-wise."
    assert absolute["provenance_label"] == "Adapted from NumPy docs"
    assert absolute["parameters"][0]["name"] == "x"
    assert absolute["parameters"][0]["body"] == ["Input array."]
    assert absolute["returns"][0]["name"] == "absolute"
    assert absolute["see_also"][0]["target"] == "fabs"
    assert absolute["notes_sections"][0].startswith("For complex input")
    assert "we.absolute" in absolute["example"]["code"]
    assert absolute["next"]["href"] == "/docs/api/ops/add"

    refs_manifest = mod.json.loads(refs_manifest_path.read_text())
    assert refs_manifest["abs"]["label"] == "we.absolute"
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

    records = [
        mod.OperationDocRecord(
            name="absolute",
            canonical_name="absolute",
            slug="absolute",
            href="/docs/api/ops/absolute",
            area="core",
            whest_ref="we.absolute",
            numpy_ref="np.absolute",
            category="counted_unary",
            display_type="counted",
            cost_formula="numel(output)",
            cost_formula_latex=r"$\text{numel}(\text{output})$",
            weight=1.0,
            notes="Element-wise absolute value.",
            aliases=[],
            signature="we.absolute(x)",
            summary="Return the absolute value element-wise.",
            provenance_label="Adapted from NumPy docs",
            provenance_url="https://numpy.org/doc/stable/reference/generated/numpy.absolute.html",
            whest_source_url="https://github.com/AIcrowd/whest/blob/main/src/whest/_pointwise.py#L249",
            upstream_source_url="https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/code_generators/ufunc_docstrings.py#L1",
            parameters=[],
            returns=[],
            see_also=[],
            notes_sections=[],
            example=mod.DocExample(code="import whest as we\nwe.absolute(x)"),
            previous=None,
            next=None,
        ),
        mod.OperationDocRecord(
            name="sum",
            canonical_name="sum",
            slug="sum",
            href="/docs/api/ops/sum",
            area="core",
            whest_ref="we.sum",
            numpy_ref="np.sum",
            category="counted_reduction",
            display_type="counted",
            cost_formula="numel(input)",
            cost_formula_latex=r"$\text{numel}(\text{input})$",
            weight=1.0,
            notes="Reduction sum.",
            aliases=[],
            signature="we.sum(x)",
            summary="Sum array elements over a given axis.",
            provenance_label="Adapted from NumPy docs",
            provenance_url="https://numpy.org/doc/stable/reference/generated/numpy.sum.html",
            whest_source_url="https://github.com/AIcrowd/whest/blob/main/src/whest/_reductions.py#L1",
            upstream_source_url="https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/fromnumeric.py#L1",
            parameters=[],
            returns=[],
            see_also=[],
            notes_sections=[],
            example=mod.DocExample(code="import whest as we\nwe.sum(x)", output="6"),
            previous=None,
            next=None,
        ),
        mod.OperationDocRecord(
            name="zeros",
            canonical_name="zeros",
            slug="zeros",
            href="/docs/api/ops/zeros",
            area="core",
            whest_ref="we.zeros",
            numpy_ref="np.zeros",
            category="free",
            display_type="free",
            cost_formula="0",
            cost_formula_latex="$0$",
            weight=1.0,
            notes="Allocate zeros.",
            aliases=[],
            signature="we.zeros(shape)",
            summary="Return a new array of given shape and type, filled with zeros.",
            provenance_label="Adapted from NumPy docs",
            provenance_url="https://numpy.org/doc/stable/reference/generated/numpy.zeros.html",
            whest_source_url="https://github.com/AIcrowd/whest/blob/main/src/whest/__init__.py#L1",
            upstream_source_url="https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/numeric.py#L1",
            parameters=[],
            returns=[],
            see_also=[],
            notes_sections=[],
            example=None,
            previous=None,
            next=None,
        ),
    ]

    coverage = mod.build_example_coverage(records, example_root=example_dir)

    assert coverage["absolute"]["has_whest_examples"] is True
    assert coverage["absolute"]["coverage_status"] == "owned"
    assert coverage["sum"]["has_whest_examples"] is False
    assert coverage["sum"]["has_inherited_examples"] is True
    assert coverage["sum"]["coverage_status"] == "derived"
    assert coverage["zeros"]["coverage_status"] == "missing"
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


def test_parse_inline_nodes_emits_structured_role_references():
    mod = load_generate_api_docs_module()

    nodes = mod.parse_inline_nodes(
        "See :meth:`~numpy.ufunc.reduce`, :func:`custom text <numpy.absolute>`, and :func:`!numpy.sum`.",
        supported_ops={"absolute"},
        alias_map={},
    )

    reduce_ref = next(
        node
        for node in nodes
        if node.get("kind") == "role_reference" and node.get("role") == "meth"
    )
    absolute_ref = next(
        node
        for node in nodes
        if node.get("kind") == "role_reference"
        and node.get("role") == "func"
        and node.get("display_text") == "custom text"
    )
    suppressed_ref = next(
        node
        for node in nodes
        if node.get("kind") == "role_reference"
        and node.get("role") == "func"
        and node.get("suppress_link") is True
    )

    assert reduce_ref["display_text"] == "reduce"
    assert reduce_ref["external_url"].endswith("numpy.ufunc.reduce.html")
    assert absolute_ref["href"] == "/docs/api/ops/absolute"
    assert absolute_ref["explicit_title"] is True
    assert suppressed_ref["href"] == ""
    assert suppressed_ref["external_url"] == ""


def test_parse_rich_doc_blocks_preserves_directives_and_nested_blocks():
    mod = load_generate_api_docs_module()
    coverage = mod._new_doc_coverage()

    blocks = mod._parse_rich_doc_blocks(
        [
            "Elements to include in the variance. See :meth:`~numpy.ufunc.reduce` for details.",
            "",
            ".. versionadded:: 1.22.0",
            "",
            "'doane'",
            "    Estimator based on Doane's rule.",
            "",
            ".. note::",
            "    See ``histogram_bin_edges`` for additional context.",
            "",
            "* First item",
            "* Second item",
        ],
        supported_ops={"histogram_bin_edges"},
        alias_map={},
        coverage=coverage,
    )

    assert any(
        block["type"] == "directive_block" and block["directive"] == "versionadded"
        for block in blocks
    )
    assert any(block["type"] == "definition_list" for block in blocks)
    assert any(
        block["type"] == "list" and block["ordered"] is False for block in blocks
    )
    assert any(
        block["type"] == "directive_block"
        and block["directive"] == "note"
        and block["content_blocks"][0]["type"] == "paragraph"
        for block in blocks
    )
    assert coverage["unsupported_directives"] == []


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


def test_derive_example_from_doctest_ignores_plot_prose():
    mod = load_generate_api_docs_module()

    example = mod.derive_example_from_upstream(
        ">>> import numpy as np\n"
        ">>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])\n"
        "(array([0, 2, 1]), array([0, 1, 2, 3]))\n"
        "\n"
        "Automated Bin Selection Methods example.\n"
        "\n"
        ".. plot::\n"
        "    :include-source:\n"
        "\n"
        "    import matplotlib.pyplot as plt\n"
    )

    assert "plot::" not in example.output
    assert "Automated Bin Selection Methods" not in example.output
    assert "import whest as we" in example.code
    assert "we.histogram" in example.code


def test_resolve_doc_link_sets_internal_and_external_urls():
    mod = load_generate_api_docs_module()

    internal = mod.resolve_doc_link(
        mod.DocLink(
            label="numpy.histogram_bin_edges",
            target="numpy.histogram_bin_edges",
            description="Bin helper.",
        ),
        alias_map={},
        supported_ops={"histogram_bin_edges"},
    )
    external = mod.resolve_doc_link(
        mod.DocLink(
            label="scipy.linalg.svd",
            target="scipy.linalg.svd",
            description="Similar function in SciPy.",
        ),
        alias_map={},
        supported_ops={"svd"},
    )

    assert internal.href == "/docs/api/ops/histogram_bin_edges"
    assert internal.external_url == ""
    assert external.href == ""
    assert external.external_url == (
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html"
    )
    assert internal.description_inline


def test_write_generated_operation_artifacts_emit_structured_parity_fields(tmp_path):
    mod = load_generate_api_docs_module()

    records = [
        mod.OperationDocRecord(
            name="absolute",
            canonical_name="absolute",
            slug="absolute",
            href="/docs/api/ops/absolute",
            area="core",
            whest_ref="we.absolute",
            numpy_ref="np.absolute",
            category="counted_unary",
            display_type="counted",
            cost_formula="numel(output)",
            cost_formula_latex=r"$\text{numel}(\text{output})$",
            weight=1.0,
            notes="Element-wise absolute value.",
            aliases=["abs"],
            signature="we.absolute(x, /, out=None, *, where=True)",
            summary="Calculate the absolute value element-wise.",
            provenance_label="Adapted from NumPy docs",
            provenance_url="https://numpy.org/doc/stable/reference/generated/numpy.absolute.html",
            whest_source_url="https://github.com/AIcrowd/whest/blob/main/src/whest/_pointwise.py#L10",
            upstream_source_url="https://github.com/numpy/numpy/blob/main/numpy/_core/code_generators/ufunc_docstrings.py",
            parameters=[
                mod.DocField(name="x", type="array_like", body=["Input array."])
            ],
            returns=[
                mod.DocField(
                    name="absolute", type="ndarray", body=["Absolute value of x."]
                )
            ],
            see_also=[
                mod.DocLink(label="we.fabs", target="fabs", href="/docs/api/ops/fabs")
            ],
            notes_sections=["Supports broadcasting."],
            example=mod.DocExample(
                code="import whest as we\nwe.absolute([-1, 2])",
                output="array([1, 2])",
                source="derived",
            ),
            previous=None,
            next=mod.OperationNavLink(label="we.add", href="/docs/api/ops/add"),
        )
    ]

    mod.write_operation_doc_artifacts(records, tmp_path)

    manifest = mod.json.loads((tmp_path / ".generated" / "op-docs.json").read_text())
    absolute = manifest["absolute"]

    assert absolute["summary"] == "Calculate the absolute value element-wise."
    assert absolute["provenance_label"] == "Adapted from NumPy docs"
    assert absolute["parameters"][0]["name"] == "x"
    assert absolute["parameters"][0]["body"] == ["Input array."]
    assert absolute["returns"][0]["name"] == "absolute"
    assert absolute["see_also"][0]["href"] == "/docs/api/ops/fabs"
    assert absolute["notes_sections"][0] == "Supports broadcasting."
    assert absolute["example"]["source"] == "derived"
    assert absolute["next"]["href"] == "/docs/api/ops/add"


def test_write_op_doc_coverage_artifact(tmp_path):
    mod = load_generate_api_docs_module()

    record = mod.OperationDocRecord(
        name="absolute",
        canonical_name="absolute",
        slug="absolute",
        href="/docs/api/ops/absolute",
        area="core",
        whest_ref="we.absolute",
        numpy_ref="np.absolute",
        category="counted_unary",
        display_type="counted",
        cost_formula="numel(output)",
        cost_formula_latex=r"$\text{numel}(\text{output})$",
        weight=1.0,
        notes="Element-wise absolute value.",
        aliases=[],
        signature="we.absolute(x)",
        summary="Calculate the absolute value element-wise.",
        doc_coverage={
            "unresolved_references": [{"role": "ref", "target": "ufuncs.kwargs"}],
            "unsupported_directives": [],
            "raw_blocks": [],
        },
    )

    mod.write_op_doc_coverage_artifact([record], tmp_path)

    payload = mod.json.loads(
        (tmp_path / ".generated" / "op-doc-coverage.json").read_text()
    )
    assert payload["absolute"]["has_issues"] is True
    assert payload["absolute"]["unresolved_references"][0]["target"] == "ufuncs.kwargs"


def test_build_example_coverage_prefers_override_then_derived():
    mod = load_generate_api_docs_module()

    override = mod.DocExample(code="override()", output="", source="override")
    derived = mod.DocExample(code="derived()", output="", source="derived")

    coverage = mod.build_example_coverage(
        ["absolute", "sum", "mean"],
        overrides={"absolute": override},
        derived_examples={"sum": derived},
    )

    assert coverage["absolute"]["example_source"] == "override"
    assert coverage["sum"]["example_source"] == "derived"
    assert coverage["mean"]["example_source"] == "missing"


def test_build_structured_doc_provides_upstream_source_for_ufuncs():
    mod = load_generate_api_docs_module()

    _, _, _, _, upstream_source_url = mod.build_structured_doc("absolute", "numpy")

    assert upstream_source_url
    assert "ufunc_docstrings.py" in upstream_source_url
