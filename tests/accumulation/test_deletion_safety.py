"""Verify that deleted modules / symbols are no longer importable."""

import pytest


def test_subgraph_symmetry_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _subgraph_symmetry  # noqa: F401


def test_subgraph_symmetry_oracle_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle  # noqa: F401
