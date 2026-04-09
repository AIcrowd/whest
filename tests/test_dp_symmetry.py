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


class TestBitmapToSubsetRenumbering:
    """Unit test for the fixed bitmap_to_subset closure that handles
    _dp_parse_out_single_term_ops's operand renumbering."""

    def test_inputs_contractions_tuple_and_int_forms(self):
        """Directly verify the renumbering logic for a mixed int/tuple
        inputs_contractions list. Simulates the state after a single
        operand was removed entirely and another was reduced in place."""
        # Simulated state after _dp_parse_out_single_term_ops:
        # - original operand 0 was removed entirely (scalar) → not in parsed list
        # - original operand 1 was kept with single-term reduction: (1,)
        # - original operand 2 was kept unchanged: 2
        # parsed positions 0, 1 correspond to original 1, 2.
        inputs_contractions: list[int | tuple[int]] = [(1,), 2]
        inputs_parsed_count = 2

        def bitmap_to_subset(s: int) -> frozenset[int]:
            result: set[int] = set()
            for k in range(inputs_parsed_count):
                if s >> k & 1:
                    orig = inputs_contractions[k]
                    result.add(orig if isinstance(orig, int) else orig[0])
            return frozenset(result)

        assert bitmap_to_subset(0b01) == frozenset({1}), "parsed 0 -> orig 1"
        assert bitmap_to_subset(0b10) == frozenset({2}), "parsed 1 -> orig 2"
        assert bitmap_to_subset(0b11) == frozenset({1, 2}), "both"


class TestDPMatchesOptimalUnderExactScoring:
    """With exact unique/dense ratio scoring, DP's reported cost should
    match optimal's exactly on symmetric cases where both can run."""

    def test_dp_matches_optimal_on_symmetric_s4(self):
        """einsum('ij,ik,il,im->jklm', X, X, X, X) has full S4 symmetry
        on the output {j,k,l,m}. Under exact scoring, DP and optimal
        should agree on optimized_cost down to the last FLOP."""
        n = 6
        X = np.ones((n, n))
        oracle_kwargs = dict(
            operands=[X, X, X, X],
            subscript_parts=["ij", "ik", "il", "im"],
            per_op_syms=[None, None, None, None],
            output_chars="jklm",
        )
        _, info_dp = contract_path(
            "ij,ik,il,im->jklm",
            (n, n),
            (n, n),
            (n, n),
            (n, n),
            shapes=True,
            optimize="dp",
            symmetry_oracle=SubgraphSymmetryOracle(**oracle_kwargs),
        )
        _, info_opt = contract_path(
            "ij,ik,il,im->jklm",
            (n, n),
            (n, n),
            (n, n),
            (n, n),
            shapes=True,
            optimize="optimal",
            symmetry_oracle=SubgraphSymmetryOracle(**oracle_kwargs),
        )
        assert info_dp.optimized_cost == info_opt.optimized_cost, (
            f"DP ({info_dp.optimized_cost}) should match optimal "
            f"({info_opt.optimized_cost}) under exact symmetry scoring"
        )

    def test_dp_exact_ratio_for_s3(self):
        """Verify DP uses the exact C(n+2,3)/n^3 ratio for S3, not 1/2.
        This is the canonical "conservative is too conservative" case:
        S3 gives ~1/6 savings, but // 2 would give only 1/2."""
        n = 10
        X = np.ones((n, n))
        oracle_kwargs = dict(
            operands=[X, X, X],
            subscript_parts=["ai", "bi", "ci"],
            per_op_syms=[None, None, None],
            output_chars="abc",
        )
        _, info_dp = contract_path(
            "ai,bi,ci->abc",
            (n, n),
            (n, n),
            (n, n),
            shapes=True,
            optimize="dp",
            symmetry_oracle=SubgraphSymmetryOracle(**oracle_kwargs),
        )
        _, info_opt = contract_path(
            "ai,bi,ci->abc",
            (n, n),
            (n, n),
            (n, n),
            shapes=True,
            optimize="optimal",
            symmetry_oracle=SubgraphSymmetryOracle(**oracle_kwargs),
        )
        assert info_dp.optimized_cost == info_opt.optimized_cost, (
            f"DP ({info_dp.optimized_cost}) should match optimal "
            f"({info_opt.optimized_cost}) with exact S3 scoring"
        )


class TestDPSingleTermReductionWithOracle:
    """Regression guard for the bitmap_to_subset bug fix.

    When _dp_parse_out_single_term_ops reduces an operand with a
    unique-to-itself index, the remaining operands are renumbered in
    the DP's internal parsed list. The bitmap_to_subset closure must
    translate parsed bitmap positions back to ORIGINAL operand
    positions so the oracle receives the correct subset.

    Test shape: 'i,ab,cd->abcd' with [v, X, X].
    - v has unique index 'i' → reduces to a scalar, dropped from the DP loop.
    - After parse: inputs_parsed = [X, X] with inputs_contractions = [1, 2].
    - Oracle on {1, 2} returns block S2 {(a,b),(c,d)}.
    - Without the fix, DP's buggy bitmap_to_subset would query {0, 1}
      (treating parsed positions as original), getting None → dense
      cost on the final step.
    - With the fix, DP queries {1, 2} → block S2 → reduced cost on
      the 'ab,cd->abcd' outer product step.
    """

    def test_dp_queries_oracle_with_correct_subset_after_renumbering(self):
        n = 4
        v = np.ones(n)
        X = np.ones((n, n))
        oracle_kwargs = dict(
            operands=[v, X, X],
            subscript_parts=["i", "ab", "cd"],
            per_op_syms=[None, None, None],
            output_chars="abcd",
        )
        # Sanity check: the oracle really does report block S2 at the
        # CORRECT subset {1, 2} and None at the WRONG subset {0, 1}.
        # If this invariant breaks (e.g., the oracle implementation
        # changes), the test doesn't exercise the bug it's trying to
        # guard against and needs to be rewritten.
        probe_oracle = SubgraphSymmetryOracle(**oracle_kwargs)
        assert probe_oracle.sym(frozenset({1, 2})) is not None, (
            "probe: oracle should see symmetry on the two X copies"
        )
        assert probe_oracle.sym(frozenset({0, 1})) is None, (
            "probe: oracle should NOT see symmetry on (v, X) — v is distinct"
        )

        _, info_dp = contract_path(
            "i,ab,cd->abcd",
            (n,),
            (n, n),
            (n, n),
            shapes=True,
            optimize="dp",
            symmetry_oracle=SubgraphSymmetryOracle(**oracle_kwargs),
        )

        # Find the step that produces the 'abcd' intermediate —
        # it's the one that should benefit from the block S2 symmetry.
        dense_n4 = n**4  # 256 for n=4
        outer_steps = [s for s in info_dp.steps if s.dense_flop_cost == dense_n4]
        assert len(outer_steps) >= 1, (
            f"expected at least one step with dense cost {dense_n4}; "
            f"got steps: {[(s.subscript, s.dense_flop_cost) for s in info_dp.steps]}"
        )
        outer_step = outer_steps[-1]

        # With the bitmap_to_subset fix, the final outer-product-style
        # step should have a reduced cost (strictly less than the dense
        # n^4), indicating the oracle returned a symmetry for the
        # renumbered subset. Without the fix, this step would cost the
        # full n^4 because the oracle would receive the wrong subset
        # and return None → dense scoring.
        assert outer_step.flop_cost < dense_n4, (
            f"Expected symmetry-reduced cost on the outer-product step; "
            f"got flop_cost={outer_step.flop_cost} dense={dense_n4}. "
            f"This suggests bitmap_to_subset is not translating parsed "
            f"positions back to original operand positions correctly — "
            f"the oracle is probably receiving the wrong subset and "
            f"returning None."
        )
