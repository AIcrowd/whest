"""Verify scripts/numpy_audit.py covers Generator/RandomState method drift."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_audit_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "numpy_audit.py"
    spec = importlib.util.spec_from_file_location("numpy_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["numpy_audit"] = module
    spec.loader.exec_module(module)
    return module


class TestAuditRandomClass:
    def test_clean_against_current_numpy(self):
        audit = _load_audit_module()
        diffs = audit.audit_random_class(
            "Generator", np.random.Generator, "random.Generator."
        )
        assert diffs == [], f"Generator drift: {diffs}"

    def test_clean_random_state(self):
        audit = _load_audit_module()
        diffs = audit.audit_random_class(
            "RandomState", np.random.RandomState, "random.RandomState."
        )
        assert diffs == [], f"RandomState drift: {diffs}"


class TestDocsGenSmoke:
    """Verify the docs generator handles method-level ops without errors."""

    @staticmethod
    def _load_generator():
        import importlib.util
        import sys
        from pathlib import Path

        path = (
            Path(__file__).resolve().parent.parent / "scripts" / "generate_api_docs.py"
        )
        spec = importlib.util.spec_from_file_location("generate_api_docs", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["generate_api_docs"] = module
        spec.loader.exec_module(module)
        return module

    def test_resolve_live_objects_handles_dotted_names(self):
        m = self._load_generator()
        flops_obj, np_obj = m.resolve_live_objects(
            "random.Generator.standard_normal", "numpy.random"
        )
        assert callable(flops_obj)
        assert callable(np_obj)

    def test_categories_registered(self):
        m = self._load_generator()
        # Both new categories present in all four registries.
        assert "counted_random_method" in m.CATEGORY_LABELS
        assert "free_random_method" in m.CATEGORY_LABELS
        assert "counted_random_method" in m.CATEGORY_EMOJI
        assert "free_random_method" in m.CATEGORY_EMOJI
        assert "counted_random_method" in m.CATEGORY_COST_LATEX
        assert "free_random_method" in m.CATEGORY_COST_LATEX

    def test_display_type_for_free_random_method(self):
        m = self._load_generator()
        assert m.display_type_for_category("free_random_method") == "free"
        assert m.display_type_for_category("counted_random_method") == "counted"

    def test_cost_for_op_uses_per_formula_label(self):
        m = self._load_generator()
        plain, _ = m.cost_for_op("random.Generator.shuffle", "counted_random_method")
        assert "shape" in plain.lower()
        plain, _ = m.cost_for_op("random.Generator.normal", "counted_random_method")
        assert "numel" in plain.lower()
        plain, _ = m.cost_for_op("random.Generator.bit_generator", "free_random_method")
        assert plain == "0"
        # choice has its own composite label
        plain, _ = m.cost_for_op("random.Generator.choice", "counted_random_method")
        assert "replace" in plain.lower() or "numel" in plain.lower()
