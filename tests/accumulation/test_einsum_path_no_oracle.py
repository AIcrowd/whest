"""Verify that _get_path_info no longer threads a symmetry oracle."""

import inspect

import flopscope._einsum as einsum_module


def test_symmetry_fingerprint_helper_removed():
    """_symmetry_fingerprint should be gone after Task 26."""
    assert not hasattr(einsum_module, '_symmetry_fingerprint')


def test_path_cache_signature_no_oracle_args():
    """The cached compute function no longer takes symmetry_fingerprint or use_inner_symmetry."""
    src = inspect.getsource(einsum_module)
    assert 'symmetry_fingerprint' not in src
    assert 'use_inner_symmetry' not in src
    assert 'SubgraphSymmetryOracle' not in src
