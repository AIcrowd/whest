"""End-to-end tests that random_greedy is symmetry-aware."""

from __future__ import annotations

import numpy as np

from flopscope._opt_einsum import contract_path
from flopscope._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle


class TestRandomGreedyUsesSymmetry:
    def test_random_greedy_matches_greedy_on_symmetric_case(self):
        n = 10
        X = np.ones((n, n, n))
        # einsum('ijk,ilm->jklm', X, X) is a known block-S2 case.
        # The greedy optimizer finds ~101000 (symmetric) vs ~200000 (dense).
        _, greedy_info = contract_path(
            "ijk,ilm->jklm",
            (n, n, n),
            (n, n, n),
            shapes=True,
            optimize="greedy",
            symmetry_oracle=SubgraphSymmetryOracle(
                [X, X], ["ijk", "ilm"], [None, None], "jklm"
            ),
        )
        _, rg_info = contract_path(
            "ijk,ilm->jklm",
            (n, n, n),
            (n, n, n),
            shapes=True,
            optimize="random-greedy",
            symmetry_oracle=SubgraphSymmetryOracle(
                [X, X], ["ijk", "ilm"], [None, None], "jklm"
            ),
        )
        # random_greedy includes the deterministic r=0 trial which runs
        # the standard greedy, so its best result should match greedy's.
        assert rg_info.optimized_cost == greedy_info.optimized_cost

    def test_random_greedy_cost_less_than_dense(self):
        n = 10
        X = np.ones((n, n, n))
        oracle = SubgraphSymmetryOracle([X, X], ["ijk", "ilm"], [None, None], "jklm")
        _, info = contract_path(
            "ijk,ilm->jklm",
            (n, n, n),
            (n, n, n),
            shapes=True,
            optimize="random-greedy",
            symmetry_oracle=oracle,
        )
        dense_cost = 2 * n**5  # baseline: dense contraction of a 5-index step
        assert info.optimized_cost < dense_cost


class TestSsaPathComputeCostUsesOracle:
    def test_rescore_symmetric_less_than_dense(self):
        from flopscope._opt_einsum._path_random import ssa_path_compute_cost

        n = 10
        X = np.ones((n, n, n))
        inputs = [frozenset("ijk"), frozenset("ilm")]
        output = frozenset("jklm")
        size_dict = dict.fromkeys("ijklm", n)

        oracle = SubgraphSymmetryOracle([X, X], ["ijk", "ilm"], [None, None], "jklm")

        ssa_path = [(0, 1)]
        dense_cost, _ = ssa_path_compute_cost(
            ssa_path, inputs, output, size_dict, symmetry_oracle=None
        )
        sym_cost, _ = ssa_path_compute_cost(
            ssa_path, inputs, output, size_dict, symmetry_oracle=oracle
        )
        assert sym_cost < dense_cost
