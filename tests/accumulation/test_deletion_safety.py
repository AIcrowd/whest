"""Verify that deleted modules / symbols are no longer importable."""

import pytest


def test_subgraph_symmetry_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _subgraph_symmetry  # noqa: F401


def test_subgraph_symmetry_oracle_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle  # noqa: F401


def test_symmetric_flop_count_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum._symmetry import symmetric_flop_count  # noqa: F401


def test_unique_elements_in_opt_einsum_symmetry_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum._symmetry import unique_elements  # noqa: F401


def test_subset_symmetry_dataclass_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum._symmetry import SubsetSymmetry  # noqa: F401


def test_unique_elements_for_shape_in_symmetry_utils_is_kept():
    """Sanity check: the keeper helper (used by SymmetricTensor sizing) still works."""
    from flopscope._symmetry_utils import unique_elements_for_shape
    assert callable(unique_elements_for_shape)
