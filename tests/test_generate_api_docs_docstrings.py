from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "generate_api_docs.py"

spec = importlib.util.spec_from_file_location(
    "generate_api_docs_docstrings", MODULE_PATH
)
assert spec is not None
assert spec.loader is not None
generate_api_docs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generate_api_docs
spec.loader.exec_module(generate_api_docs)


def test_workflow_helpers_require_examples():
    required = {
        "flopscope.BudgetContext",
        "flopscope.budget",
        "flopscope.budget_live",
        "flopscope.namespace",
        "flopscope.as_symmetric",
        "flopscope.symmetrize",
        "flopscope.accounting.einsum_cost",
        "flopscope.accounting.pointwise_cost",
    }
    rules = generate_api_docs.public_doc_contract_rules()
    assert required.issubset(set(rules["require_examples_for"]))


def test_budgeting_helpers_are_in_the_required_examples_set():
    required = {
        "flopscope.BudgetContext",
        "flopscope.budget",
        "flopscope.budget_live",
        "flopscope.budget_summary",
        "flopscope.budget_summary_dict",
        "flopscope.budget_reset",
        "flopscope.namespace",
        "flopscope.configure",
        "flopscope.numpy.clear_einsum_cache",
        "flopscope.numpy.einsum_cache_info",
    }
    rules = generate_api_docs.public_doc_contract_rules()
    assert required.issubset(set(rules["require_examples_for"]))


def test_required_public_callables_must_have_parameters_and_returns():
    rules = generate_api_docs.public_doc_contract_rules()
    assert rules["require_parameters_and_returns_for_kind"] == {"function", "method"}


def test_stale_aliases_are_rejected_in_public_doc_examples():
    lines = ["import flopscope as we", "we.einsum('ij,j->i', W, x)"]
    problems = generate_api_docs.find_public_doc_contract_violations(
        import_path="flopscope.numpy.einsum",
        kind="function",
        summary="einsum summary",
        sections={"Parameters": ["x"], "Returns": ["value"], "Examples": lines},
    )
    assert any("stale alias" in problem for problem in problems)
