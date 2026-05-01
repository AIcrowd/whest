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
