"""Unit tests for equal-operand detection in _einsum.py.

Tests:
- _is_valid_symmetry: the core validity check with self-mapping guard
- _enumerate_per_index_candidates / _enumerate_block_candidates: candidate sigmas
- _detect_induced_output_symmetry: top-level detection
"""

# ruff: noqa: F401
import numpy as np

from mechestim._einsum import (
    _detect_induced_output_symmetry,
    _enumerate_block_candidates,
    _enumerate_per_index_candidates,
    _is_valid_symmetry,
)
