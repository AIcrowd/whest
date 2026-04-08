"""End-to-end tests that DP is symmetry-aware."""

from __future__ import annotations

import numpy as np
import pytest

from mechestim._opt_einsum import contract_path
from mechestim._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle


class TestDPAcceptsOracle:
    def test_dp_with_oracle_does_not_crash(self):
        X = np.ones((3, 3, 3))
        _, info = contract_path(
            "ijk,ilm->jklm",
            (3, 3, 3),
            (3, 3, 3),
            shapes=True,
            optimize="dp",
            symmetry_oracle=SubgraphSymmetryOracle(
                [X, X], ["ijk", "ilm"], [None, None], "jklm"
            ),
        )
        assert info.optimized_cost > 0


class TestDPCostReducedUnderSymmetry:
    def test_dp_symmetric_less_than_dense(self):
        n = 6
        X = np.ones((n, n, n))
        # Without oracle: dense DP cost
        _, dense_info = contract_path(
            "ijk,ilm->jklm",
            (n, n, n),
            (n, n, n),
            shapes=True,
            optimize="dp",
            symmetry_oracle=None,
        )
        # With oracle: should be strictly less (DP applies the /2 heuristic)
        _, sym_info = contract_path(
            "ijk,ilm->jklm",
            (n, n, n),
            (n, n, n),
            shapes=True,
            optimize="dp",
            symmetry_oracle=SubgraphSymmetryOracle(
                [X, X], ["ijk", "ilm"], [None, None], "jklm"
            ),
        )
        assert sym_info.optimized_cost < dense_info.optimized_cost


class TestDPDoesNotCrashOnLargerEinsums:
    @pytest.mark.parametrize(
        "subscripts, shapes",
        [
            ("ij,jk->ik", ((3, 4), (4, 5))),
            ("ij,jk,kl->il", ((3, 4), (4, 5), (5, 6))),
            ("ai,bi,ci->abc", ((3, 4), (3, 4), (3, 4))),
        ],
    )
    def test_dp_runs(self, subscripts, shapes):
        operands = [np.ones(s) for s in shapes]
        input_parts = subscripts.split("->")[0].split(",")
        output = subscripts.split("->")[1]
        _, info = contract_path(
            subscripts,
            *shapes,
            shapes=True,
            optimize="dp",
            symmetry_oracle=SubgraphSymmetryOracle(
                operands, input_parts, [None] * len(operands), output
            ),
        )
        assert info.optimized_cost > 0
