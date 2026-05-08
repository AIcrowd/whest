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


def test_symmetry_oracle_param_gone_from_contract_path():
    import inspect
    from flopscope._opt_einsum import contract_path
    sig = inspect.signature(contract_path)
    assert 'symmetry_oracle' not in sig.parameters


def test_symmetry_oracle_param_gone_from_paths_module():
    # _paths.py was deleted in Task 7; upstream opt_einsum.paths is used directly.
    import inspect
    import opt_einsum.paths as paths
    src = inspect.getsource(paths)
    assert 'symmetry_oracle' not in src


def test_symmetry_oracle_param_gone_from_path_random():
    # _path_random.py was deleted in Task 7; upstream opt_einsum.path_random is used directly.
    import inspect
    import opt_einsum.path_random as pr
    src = inspect.getsource(pr)
    assert 'symmetry_oracle' not in src


# ── Devendor task 7+8 deletions ─────────────────────────────────────────


def test_opt_einsum_paths_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _paths  # noqa: F401


def test_opt_einsum_path_random_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _path_random  # noqa: F401


def test_opt_einsum_blas_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _blas  # noqa: F401


def test_opt_einsum_testing_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _testing  # noqa: F401


def test_opt_einsum_typing_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _typing  # noqa: F401


def test_opt_einsum_parser_module_is_gone():
    with pytest.raises(ImportError):
        from flopscope._opt_einsum import _parser  # noqa: F401


def test_parse_einsum_input_reexported_from_init():
    """After Task 9: parse_einsum_input is importable from
    flopscope._opt_einsum (re-exported from upstream)."""
    from flopscope._opt_einsum import parse_einsum_input
    assert callable(parse_einsum_input)
