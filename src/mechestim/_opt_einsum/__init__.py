"""Symmetry-aware einsum contraction path optimizer.

Forked from opt_einsum (https://github.com/dgasmith/opt_einsum).
See LICENSE and NOTICE in this directory for attribution.
"""

from mechestim._opt_einsum import _path_random, _paths
from mechestim._opt_einsum._contract import PathInfo, StepInfo, contract_path
from mechestim._opt_einsum._paths import BranchBound, DynamicProgramming

__all__ = [
    "contract_path",
    "PathInfo",
    "StepInfo",
    "BranchBound",
    "DynamicProgramming",
]

# Register random path functions
_paths.register_path_fn("random-greedy", _path_random.random_greedy)
_paths.register_path_fn("random-greedy-128", _path_random.random_greedy_128)
