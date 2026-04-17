"""Unit tests for SubgraphSymmetryOracle and its algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from whest._opt_einsum._subgraph_symmetry import (
    EinsumBipartite,
    SubgraphSymmetryOracle,
    _build_bipartite,
    _induce_subgraph,
)
from whest._perm_group import PermutationGroup


def _sym_group(*labels: str) -> PermutationGroup:
    """Create a full symmetric PermutationGroup on the given labels."""
    k = len(labels)
    pg = PermutationGroup.symmetric(k, axes=tuple(range(k)))
    pg._labels = tuple(labels)
    return pg


def _has_labels(pg: PermutationGroup | None, *expected_labels: str) -> bool:
    """Check that a PermutationGroup covers the given labels (sorted)."""
    if pg is None:
        return False
    if pg._labels is None:
        return False
    return set(pg._labels) == set(expected_labels)


def _is_s_k(pg: PermutationGroup | None, k: int, *labels: str) -> bool:
    """Check that pg is S_k on the given labels."""
    if pg is None:
        return False
    if pg.degree != k:
        return False
    if labels and pg._labels is not None:
        if set(pg._labels) != set(labels):
            return False
    return pg.is_symmetric()


class TestBipartiteConstruction:
    def test_empty_graph_is_valid(self):
        g = EinsumBipartite(
            u_vertices=(),
            u_labels=(),
            u_operand=(),
            incidence=(),
            free_labels=frozenset(),
            summed_labels=frozenset(),
            identical_operand_groups=(),
            operand_labels=(),
            operand_subscripts=(),
            per_op_groups=(),
        )
        assert g.u_vertices == ()
        assert g.free_labels == frozenset()


class TestBuildBipartite:
    def test_single_dense_operand(self):
        A = np.zeros((3, 4))
        g = _build_bipartite(
            operands=[A],
            subscript_parts=["ij"],
            per_op_groups=[None],
            output_chars="ij",
        )
        assert len(g.u_vertices) == 2
        assert g.u_operand == (0, 0)
        assert g.free_labels == frozenset("ij")
        assert g.summed_labels == frozenset()
        assert g.identical_operand_groups == ()
        # Each U vertex has incidence 1 for its one label
        labels_per_u = [set(row.keys()) for row in g.incidence]
        assert {frozenset(s) for s in labels_per_u} == {
            frozenset({"i"}),
            frozenset({"j"}),
        }
        assert all(row[next(iter(row))] == 1 for row in g.incidence)

    def test_symmetric_operand_keeps_separate_u_vertices(self):
        T = np.zeros((3, 3))
        per_op = [[_sym_group("i", "j")]]  # T symmetric in (i, j)
        g = _build_bipartite(
            operands=[T],
            subscript_parts=["ij"],
            per_op_groups=per_op,
            output_chars="ij",
        )
        # Two U vertices — one per axis (no merging).
        assert len(g.u_vertices) == 2
        assert g.incidence[0] == {"i": 1}
        assert g.incidence[1] == {"j": 1}

    def test_repeated_axis_in_subscript_gives_multiplicity(self):
        # einsum('iij->ij', T) — axis i appears twice in T
        T = np.zeros((3, 3, 4))
        g = _build_bipartite(
            operands=[T],
            subscript_parts=["iij"],
            per_op_groups=[None],
            output_chars="ij",
        )
        # Two U vertices: one for the (class containing 'i' appearing twice),
        # one for 'j'. Since there's no declared symmetry, each axis is its
        # own class, but both i-axes share the same label so the labels sets
        # collide. We expect three U vertices (one per axis), with the two
        # i-axes having incidence {i: 1} each.
        assert len(g.u_vertices) == 3
        i_rows = [row for row in g.incidence if "i" in row]
        j_rows = [row for row in g.incidence if "j" in row]
        assert len(i_rows) == 2
        assert len(j_rows) == 1
        assert all(r == {"i": 1} for r in i_rows)
        assert j_rows[0] == {"j": 1}

    def test_free_vs_summed_partition(self):
        # einsum('ij,jk->ik', A, B): j is summed, i and k are free
        A = np.zeros((3, 4))
        B = np.zeros((4, 5))
        g = _build_bipartite(
            operands=[A, B],
            subscript_parts=["ij", "jk"],
            per_op_groups=[None, None],
            output_chars="ik",
        )
        assert g.free_labels == frozenset("ik")
        assert g.summed_labels == frozenset("j")

    def test_identical_operands_are_grouped(self):
        X = np.zeros((3, 3))
        g = _build_bipartite(
            operands=[X, X],
            subscript_parts=["ij", "jk"],
            per_op_groups=[None, None],
            output_chars="ik",
        )
        assert g.identical_operand_groups == ((0, 1),)

    def test_distinct_operands_same_shape_are_not_grouped(self):
        X = np.zeros((3, 3))
        Y = np.zeros((3, 3))  # different Python object
        g = _build_bipartite(
            operands=[X, Y],
            subscript_parts=["ij", "jk"],
            per_op_groups=[None, None],
            output_chars="ik",
        )
        assert g.identical_operand_groups == ()

    def test_wilson_worked_example(self):
        # einsum('ij,ai,bj->ab', T, S, S) with T symmetric in (i,j), S1 is S2
        T = np.zeros((3, 3))
        S = np.zeros((4, 3))
        g = _build_bipartite(
            operands=[T, S, S],
            subscript_parts=["ij", "ai", "bj"],
            per_op_groups=[[_sym_group("i", "j")], None, None],
            output_chars="ab",
        )
        # T (op 0) has two U vertices (one per axis, no merging): {i}, {j}
        # S1 (op 1) has two U vertices: {a}, {i}
        # S2 (op 2) has two U vertices: {b}, {j}
        assert len(g.u_vertices) == 6
        assert g.identical_operand_groups == ((1, 2),)
        assert g.free_labels == frozenset("ab")
        assert g.summed_labels == frozenset("ij")
        # T's rows have incidence {i: 1} and {j: 1}
        t_rows = [row for u, row in zip(g.u_operand, g.incidence, strict=False) if u == 0]
        assert len(t_rows) == 2
        assert {"i": 1} in t_rows
        assert {"j": 1} in t_rows


class TestSubsetInduction:
    def test_full_subset_matches_top_level_partition(self):
        A = np.zeros((3, 4))
        B = np.zeros((4, 5))
        g = _build_bipartite([A, B], ["ij", "jk"], [None, None], "ik")

        sub = _induce_subgraph(g, frozenset({0, 1}))
        assert sub.v_labels == frozenset("ik")
        assert sub.w_labels == frozenset("j")
        assert len(sub.u_local) == 4  # all U vertices
        assert sub.id_groups == ()

    def test_singleton_subset_crossing_labels_become_free(self):
        A = np.zeros((3, 4))
        B = np.zeros((4, 5))
        g = _build_bipartite([A, B], ["ij", "jk"], [None, None], "ik")

        # Contract only operand 0. Label i is top-level free (in output).
        # Label j is top-level summed but crosses the cut (also in op 1),
        # so it becomes a free-at-this-step label.
        sub = _induce_subgraph(g, frozenset({0}))
        assert sub.v_labels == frozenset("ij")
        assert sub.w_labels == frozenset()
        assert len(sub.u_local) == 2

    def test_mid_tree_subset_j_stays_summed(self):
        A = np.zeros((3, 4))
        B = np.zeros((4, 5))
        C = np.zeros((5, 6))
        g = _build_bipartite([A, B, C], ["ij", "jk", "kl"], [None, None, None], "il")

        # Contract ops 0 and 1. j is summed entirely within the subset
        # (not in any operand outside the subset, not in output), so j
        # belongs to W_S. k crosses the cut to op 2, so it's V_S.
        sub = _induce_subgraph(g, frozenset({0, 1}))
        assert sub.v_labels == frozenset("ik")
        assert sub.w_labels == frozenset("j")

    def test_identical_group_restricts_to_subset(self):
        X = np.zeros((3, 3))
        g = _build_bipartite([X, X, X], ["ai", "bi", "ci"], [None, None, None], "abc")

        sub_all = _induce_subgraph(g, frozenset({0, 1, 2}))
        assert sub_all.id_groups == ((0, 1, 2),)

        sub_pair = _induce_subgraph(g, frozenset({0, 2}))
        assert sub_pair.id_groups == ((0, 2),)

        sub_single = _induce_subgraph(g, frozenset({1}))
        assert sub_single.id_groups == ()  # no group with |intersection| >= 2


class TestPerIndexPairs:
    def test_wilson_t_s_s_example(self):
        # einsum('ij,ai,bj->ab', T, S, S) with T sym in (i,j), S1 is S2
        T = np.zeros((3, 3))
        S = np.zeros((4, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[T, S, S],
            subscript_parts=["ij", "ai", "bj"],
            per_op_groups=[[_sym_group("i", "j")], None, None],
            output_chars="ab",
        )
        result = oracle.sym(frozenset({0, 1, 2}))
        assert _is_s_k(result.output, 2, "a", "b")

    def test_three_identical_operands_finds_s3(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X, X],
            subscript_parts=["ai", "bi", "ci"],
            per_op_groups=[None, None, None],
            output_chars="abc",
        )
        result = oracle.sym(frozenset({0, 1, 2}))
        # All three of a, b, c should be in the group (S3)
        assert result.output is not None
        assert _has_labels(result.output, "a", "b", "c")
        assert result.output.order() == 6  # S3

    def test_distinct_operands_no_induced_symmetry(self):
        X = np.zeros((3, 3))
        Y = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, Y, Y],
            subscript_parts=["ai", "bi", "ci"],
            per_op_groups=[None, None, None],
            output_chars="abc",
        )
        result = oracle.sym(frozenset({0, 1, 2}))
        # Only Y, Y are identical (ops 1, 2), inducing a transposition on b, c
        assert result.output is not None
        assert result.output.order() == 2
        assert _has_labels(result.output, "a", "b", "c")

    def test_empty_v_returns_none(self):
        # einsum('i,i->', X, X) — scalar output, V is empty
        X = np.zeros((3,))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["i", "i"],
            per_op_groups=[None, None],
            output_chars="",
        )
        assert oracle.sym(frozenset({0, 1})).output is None


class TestHybridBlockPath:
    def test_gram_matrix_per_index_via_block(self):
        # einsum('ij,ik->jk', X, X) — a single-index swap the block path
        # also finds (as a 1-tuple block).
        X = np.zeros((3, 4))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["ij", "ik"],
            per_op_groups=[None, None],
            output_chars="jk",
        )
        result = oracle.sym(frozenset({0, 1}))
        assert _is_s_k(result.output, 2, "j", "k")

    def test_outer_product_block_s2(self):
        # einsum('ijk,ilm->jklm', X, X) — block symmetry on (j,k) and (l,m)
        X = np.zeros((3, 3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["ijk", "ilm"],
            per_op_groups=[None, None],
            output_chars="jklm",
        )
        result = oracle.sym(frozenset({0, 1}))
        assert result.output is not None
        # Output group covers j,k,l,m
        assert _has_labels(result.output, "j", "k", "l", "m")

    def test_distinct_operands_no_block(self):
        X = np.zeros((3, 3, 3))
        Y = np.zeros((3, 3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, Y],
            subscript_parts=["ijk", "ilm"],
            per_op_groups=[None, None],
            output_chars="jklm",
        )
        assert oracle.sym(frozenset({0, 1})).output is None


class TestOracleCaching:
    def test_same_subset_returns_cached_object(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["ij", "jk"],
            per_op_groups=[None, None],
            output_chars="ik",
        )
        sym1 = oracle.sym(frozenset({0, 1}))
        sym2 = oracle.sym(frozenset({0, 1}))
        assert sym1 is sym2

    def test_two_oracles_do_not_share_cache(self):
        X = np.zeros((3, 3))
        o1 = SubgraphSymmetryOracle([X, X], ["ij", "jk"], [None, None], "ik")
        o2 = SubgraphSymmetryOracle([X, X], ["ij", "jk"], [None, None], "ik")
        assert o1._cache is not o2._cache

    def test_empty_subset_returns_both_none(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle([X, X], ["ij", "jk"], [None, None], "ik")
        result = oracle.sym(frozenset())
        assert result.output is None
        assert result.inner is None


class TestMemoKeyIsSubsetOnly:
    def test_same_subset_via_different_query_paths(self):
        # Construct the same subset key via different frozenset constructions
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            [X, X, X], ["ai", "bi", "ci"], [None, None, None], "abc"
        )
        a = oracle.sym(frozenset({0, 1}))
        b = oracle.sym(frozenset([0, 1]))
        c = oracle.sym(frozenset({1, 0}))
        assert a is b is c


class TestOldSymIsSubsetOfNewSym:
    """Option B contract: the new oracle finds at least every symmetry
    the old _detect_induced_output_symmetry finds on every test case.

    This test is REMOVED in a follow-up commit along with the old
    reference implementation after the refactor has landed and stabilised.
    """

    @pytest.mark.parametrize(
        "subscripts, make_operands",
        [
            (
                "ij,ai,bj->ab",
                lambda: (np.zeros((3, 3)), np.zeros((4, 3)), np.zeros((4, 3))),
            ),
            ("ij,ik->jk", lambda: (np.zeros((3, 4)), np.zeros((3, 5)))),
            ("ijk,ilm->jklm", lambda: (np.zeros((3, 3, 3)), np.zeros((3, 3, 3)))),
            (
                "ai,bi,ci->abc",
                lambda: (np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))),
            ),
            ("ij,jk->ik", lambda: (np.zeros((3, 4)), np.zeros((4, 5)))),
        ],
    )
    def test_old_sym_subset_of_new(self, subscripts, make_operands):
        operands_tuple = make_operands()
        # Operands passed as the SAME object for repeated positions
        if subscripts == "ij,ai,bj->ab":
            T, S1, _ = operands_tuple
            operands = [T, S1, S1]
        elif subscripts == "ij,ik->jk":
            X, _ = operands_tuple
            operands = [X, X]
        elif subscripts == "ijk,ilm->jklm":
            X, _ = operands_tuple
            operands = [X, X]
        elif subscripts == "ai,bi,ci->abc":
            X, _, _ = operands_tuple
            operands = [X, X, X]
        elif subscripts == "ij,jk->ik":
            A, B = operands_tuple
            operands = [A, B]
        else:
            raise ValueError(subscripts)

        input_parts = subscripts.split("->")[0].split(",")
        output_chars = subscripts.split("->")[1]

        # Call the OLD reference implementation (still present in
        # whest._einsum until Commit 2). Skip gracefully if it has
        # already been removed.
        try:
            from whest._einsum import _detect_induced_output_symmetry
        except ImportError:
            pytest.skip("old detector already removed")

        old = _detect_induced_output_symmetry(
            operands=operands,
            subscript_parts=input_parts,
            output_chars=output_chars,
            per_op_groups=[None] * len(operands),
        )
        if old is None:
            old = []

        oracle = SubgraphSymmetryOracle(
            operands=operands,
            subscript_parts=input_parts,
            per_op_groups=[None] * len(operands),
            output_chars=output_chars,
        )
        result = oracle.sym(frozenset(range(len(operands))))
        new_pg = result.output  # PermutationGroup or None

        # For each old group, assert that the new PermutationGroup covers
        # those labels.
        if old:
            assert new_pg is not None, (
                f"Old detected groups {old} but new oracle returned None"
            )
            new_labels = set(new_pg._labels) if new_pg._labels else set()
            for old_g in old:
                old_chars = frozenset(c for block in old_g for c in block)
                assert old_chars <= new_labels, (
                    f"Old group {old_g} not covered by new oracle labels: {new_labels}"
                )


from whest._opt_einsum._symmetry import SubsetSymmetry


class TestSubsetSymmetryDataclass:
    def test_both_none(self):
        ss = SubsetSymmetry(output=None, inner=None)
        assert ss.output is None
        assert ss.inner is None

    def test_output_only(self):
        pg = PermutationGroup.symmetric(2)
        pg._labels = ("a", "b")
        ss = SubsetSymmetry(output=pg, inner=None)
        assert ss.output is pg
        assert ss.inner is None

    def test_both_populated(self):
        v = PermutationGroup.symmetric(2)
        v._labels = ("a", "b")
        w = PermutationGroup.symmetric(2)
        w._labels = ("i", "j")
        ss = SubsetSymmetry(output=v, inner=w)
        assert ss.output is v
        assert ss.inner is w

    def test_frozen(self):
        ss = SubsetSymmetry(output=None, inner=None)
        import pytest

        with pytest.raises(AttributeError):
            ss.output = PermutationGroup.symmetric(1)  # type: ignore[misc]


from whest._opt_einsum._subgraph_symmetry import (
    _derive_pi_canonical,
)


class TestDerivePiCanonical:
    def test_simple_swap(self):
        fp_to_labels = {
            (1, 0, 0, 0): {"a"},
            (0, 1, 0, 0): {"b"},
            (0, 0, 1, 0): {"c"},
            (0, 0, 0, 1): {"d"},
        }
        sigma_col_of = {
            "a": (0, 0, 1, 0),
            "b": (0, 0, 0, 1),
            "c": (1, 0, 0, 0),
            "d": (0, 1, 0, 0),
        }
        pi = _derive_pi_canonical(
            sigma_col_of,
            fp_to_labels,
            v_labels=frozenset("abcd"),
            w_labels=frozenset(),
        )
        assert pi == {"a": "c", "b": "d", "c": "a", "d": "b"}

    def test_no_match_returns_none(self):
        fp_to_labels = {(1, 0): {"a"}, (0, 1): {"b"}}
        sigma_col_of = {"a": (1, 1), "b": (0, 0)}
        pi = _derive_pi_canonical(
            sigma_col_of,
            fp_to_labels,
            v_labels=frozenset("ab"),
            w_labels=frozenset(),
        )
        assert pi is None

    def test_vw_crossing_returns_none(self):
        fp_to_labels = {(1,): {"a", "i"}}
        sigma_col_of = {"a": (1,), "i": (1,)}
        pi = _derive_pi_canonical(
            sigma_col_of,
            fp_to_labels,
            v_labels=frozenset("a"),
            w_labels=frozenset("i"),
        )
        assert pi is not None
        assert pi["a"] == "a"
        assert pi["i"] == "i"

    def test_collision_canonical_pick(self):
        fp_to_labels = {(1, 1): {"i", "j"}}
        sigma_col_of = {"i": (1, 1), "j": (1, 1)}
        pi = _derive_pi_canonical(
            sigma_col_of,
            fp_to_labels,
            v_labels=frozenset("ij"),
            w_labels=frozenset(),
        )
        assert pi is not None
        assert pi == {"i": "i", "j": "j"}


class TestPiBasedOracleRegression:
    """Oracle-level tests for the unified pi-based detection path."""

    def test_outer_product_block_s2(self):
        X = np.zeros((3, 4))
        oracle = SubgraphSymmetryOracle([X, X], ["ab", "cd"], [None, None], "abcd")
        result = oracle.sym(frozenset({0, 1}))
        assert result.output is not None
        assert _has_labels(result.output, "a", "b", "c", "d")
        assert result.inner is None

    def test_vector_outer_product_per_index(self):
        x = np.zeros((3,))
        oracle = SubgraphSymmetryOracle([x, x], ["a", "b"], [None, None], "ab")
        result = oracle.sym(frozenset({0, 1}))
        assert _is_s_k(result.output, 2, "a", "b")

    def test_gram_matrix(self):
        X = np.zeros((3, 4))
        oracle = SubgraphSymmetryOracle([X, X], ["ai", "bi"], [None, None], "ab")
        result = oracle.sym(frozenset({0, 1}))
        assert _is_s_k(result.output, 2, "a", "b")

    def test_matmul_no_symmetry(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle([X, X], ["ij", "jk"], [None, None], "ik")
        result = oracle.sym(frozenset({0, 1}))
        assert result.output is None

    def test_internal_symmetry_propagates(self):
        x = np.zeros((5,))
        Y = np.zeros((3, 3, 5))
        oracle = SubgraphSymmetryOracle(
            [x, Y],
            ["e", "abe"],
            [None, [_sym_group("a", "b")]],
            "ab",
        )
        result = oracle.sym(frozenset({0, 1}))
        assert _is_s_k(result.output, 2, "a", "b")

    def test_internal_sym_plus_repeated(self):
        T = np.zeros((3, 3, 4))
        oracle = SubgraphSymmetryOracle(
            [T, T],
            ["ijk", "ijl"],
            [[_sym_group("i", "j")]] * 2,
            "kl",
        )
        result = oracle.sym(frozenset({0, 1}))
        assert _is_s_k(result.output, 2, "k", "l")
        assert result.inner is not None
        assert _has_labels(result.inner, "i", "j")

    def test_rank3_block_s2(self):
        T = np.zeros((2, 3, 4))
        oracle = SubgraphSymmetryOracle([T, T], ["abc", "def"], [None, None], "abcdef")
        result = oracle.sym(frozenset({0, 1}))
        assert result.output is not None
        assert _has_labels(result.output, "a", "b", "c", "d", "e", "f")


class TestWSymmetryOracle:
    """Oracle-level tests for W-side symmetry detection."""

    def test_w_side_transposition(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle([X, X], ["ij", "ji"], [None, None], "")
        result = oracle.sym(frozenset({0, 1}))
        assert result.output is None
        assert _is_s_k(result.inner, 2, "i", "j")

    def test_w_empty_gives_inner_none(self):
        X = np.zeros((3, 4))
        oracle = SubgraphSymmetryOracle([X, X], ["ab", "cd"], [None, None], "abcd")
        result = oracle.sym(frozenset({0, 1}))
        assert result.inner is None

    def test_w_no_symmetry_gives_inner_none(self):
        X = np.zeros((3, 4))
        Y = np.zeros((4, 5))
        oracle = SubgraphSymmetryOracle([X, Y], ["ij", "jk"], [None, None], "ik")
        result = oracle.sym(frozenset({0, 1}))
        assert result.inner is None

    def test_internal_sym_w_collision(self):
        T = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            [T, T],
            ["ij", "ij"],
            [[_sym_group("i", "j")]] * 2,
            "ij",
        )
        result = oracle.sym(frozenset({0, 1}))
        assert _is_s_k(result.output, 2, "i", "j")


from whest._opt_einsum._symmetry import unique_elements


class TestExactGroupDetection:
    def test_trace_a_cubed_inner_is_c3(self):
        """einsum('ij,jk,ki->', A, A, A) — inner symmetry is C_3, not S_3."""
        A = np.ones((5, 5))
        oracle = SubgraphSymmetryOracle(
            operands=[A, A, A],
            subscript_parts=["ij", "jk", "ki"],
            per_op_groups=[None, None, None],
            output_chars="",
        )
        sym = oracle.sym(frozenset({0, 1, 2}))
        assert sym.inner is not None
        assert sym.inner.order() == 3  # C_3
        assert not sym.inner.is_symmetric()  # NOT S_3

    def test_gram_matrix_output_is_s2(self):
        """einsum('ij,ik->jk', X, X) — output symmetry is S_2."""
        X = np.ones((5, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["ij", "ik"],
            per_op_groups=[None, None],
            output_chars="jk",
        )
        sym = oracle.sym(frozenset({0, 1}))
        assert sym.output is not None
        assert sym.output.order() == 2
        assert sym.output.is_symmetric()

    def test_three_independent_operands_output_is_s3(self):
        """Independent subscripts give S_3 output."""
        X = np.ones((3, 4))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X, X],
            subscript_parts=["ij", "kl", "mn"],
            per_op_groups=[None, None, None],
            output_chars="jln",
        )
        sym = oracle.sym(frozenset({0, 1, 2}))
        assert sym.output is not None
        assert sym.output.is_symmetric()
        assert sym.output.order() == 6


class TestBurnsideFLOPCount:
    def test_c3_unique_via_perm_group(self):
        from whest._perm_group import PermutationGroup as PG

        n = 10
        c3 = PG.cyclic(3)
        result = unique_elements(
            frozenset({"i", "j", "k"}),
            {"i": n, "j": n, "k": n},
            perm_group=c3,
        )
        assert result == (n**3 + 2 * n) // 3

    def test_s3_unique_via_perm_group_gives_correct_value(self):
        from whest._perm_group import PermutationGroup as PG

        n = 10
        s3 = PG.symmetric(3)
        result_pg = unique_elements(
            frozenset({"i", "j", "k"}),
            {"i": n, "j": n, "k": n},
            perm_group=s3,
        )
        # C(n+2, 3) = C(12, 3) = 220
        from math import comb

        assert result_pg == comb(n + 2, 3)


class TestGroupDisplay:
    """Tests that the path info table renders exact group names."""

    def test_s2_displays_as_s2(self):
        import whest as we

        with we.BudgetContext(flop_budget=10**9, quiet=True):
            X = np.ones((5, 3))
            _, info = we.einsum_path("ij,ik->jk", X, X)
            table = info.format_table()
            assert "S2" in table

    def test_trace_a_cubed_shows_c3(self):
        import whest as we

        with we.BudgetContext(flop_budget=10**9, quiet=True):
            A = np.ones((5, 5))
            _, info = we.einsum_path("ij,jk,ki->", A, A, A)
            table = info.format_table()
            assert "C3" in table or "G(3)" in table


class TestDeclaredGroupNotPromoted:
    """Declared non-S_k groups must not be silently promoted to S_k."""

    def test_declared_c3_not_promoted_to_s3(self):
        """C_3 on T in 'ijk,ai->ajk': single-operand subset {0} should
        report C_3 on {i,j,k}, not S_3."""
        from whest._perm_group import PermutationGroup

        c3 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        c3._labels = ("i", "j", "k")

        T = np.ones((5, 5, 5))
        W = np.ones((5, 5))

        oracle = SubgraphSymmetryOracle(
            operands=[T, W],
            subscript_parts=["ijk", "ai"],
            per_op_groups=[[c3], None],
            output_chars="ajk",
        )
        # Single-operand subset: v_labels = {i,j,k} (j,k free + i crossing).
        sym = oracle.sym(frozenset({0}))
        assert sym.output is not None
        assert sym.output.order() == 3, (
            f"Expected C_3 (order 3), got order {sym.output.order()} — "
            f"declared group was promoted to S_k"
        )
        assert not sym.output.is_symmetric()

    def test_declared_s3_still_works(self):
        """S_3 declared on T should still be detected as S_3."""
        from whest._perm_group import PermutationGroup

        s3 = PermutationGroup.symmetric(3, axes=(0, 1, 2))
        s3._labels = ("i", "j", "k")

        T = np.ones((5, 5, 5))
        W = np.ones((5, 5))

        oracle = SubgraphSymmetryOracle(
            operands=[T, W],
            subscript_parts=["ijk", "ai"],
            per_op_groups=[[s3], None],
            output_chars="ajk",
        )
        sym = oracle.sym(frozenset({0}))
        assert sym.output is not None
        assert sym.output.order() == 6

    def test_no_declared_group_no_symmetry_detected(self):
        """Without declared groups on non-identical operands, no symmetry."""
        T = np.ones((5, 5, 5))
        W = np.ones((5, 5))

        oracle = SubgraphSymmetryOracle(
            operands=[T, W],
            subscript_parts=["ijk", "ai"],
            per_op_groups=[None, None],
            output_chars="ajk",
        )
        # Without declared symmetry, axes aren't merged so fingerprints
        # differ — no symmetry detected on the single-operand subset.
        sym = oracle.sym(frozenset({0}))
        assert sym.output is None


class TestC3AxisMergingBug:
    """Regression test: C3 orbit-based merging must not produce false S2."""

    def test_c3_self_contraction_no_false_s2(self):
        """einsum('ijk,jki->ik', T, T) with C3 on T must be trivial.

        Bug: orbit-based merging collapsed {i,j,k} into one U-vertex,
        causing the fingerprint fast path to falsely detect S2{i,k}.
        The result is numerically NOT symmetric: Result[i,k] != Result[k,i].
        """
        from whest._perm_group import PermutationGroup

        n = 4
        c3 = PermutationGroup.cyclic(3, axes=(0, 1, 2))
        c3._labels = ("i", "j", "k")

        T = np.ones((n, n, n))
        # Use identical Python objects for the two T operands
        oracle = SubgraphSymmetryOracle(
            operands=[T, T],
            subscript_parts=["ijk", "jki"],
            per_op_groups=[[c3], [c3]],
            output_chars="ik",
        )
        sym = oracle.sym(frozenset({0, 1}))
        # Must NOT detect S2 — result is not symmetric
        if sym.output is not None:
            assert sym.output.order() == 1, (
                f"Expected trivial (order 1), got order {sym.output.order()} — "
                f"C3 orbit merging produced false symmetry"
            )

    def test_c3_declared_uses_sigma_loop(self):
        """Declared C3 on T in 'aijk,ab->ijkb' should be found via σ-loop
        generators, not the (now-removed) fingerprint fast path."""
        from whest._perm_group import PermutationGroup

        n = 4
        c3 = PermutationGroup.cyclic(3, axes=(1, 2, 3))
        c3._labels = ("a", "i", "j", "k")

        T = np.ones((n, n, n, n))
        W = np.ones((n, n))

        oracle = SubgraphSymmetryOracle(
            operands=[T, W],
            subscript_parts=["aijk", "ab"],
            per_op_groups=[[c3], None],
            output_chars="ijkb",
        )
        sym = oracle.sym(frozenset({0, 1}))
        assert sym.output is not None
        assert sym.output.order() == 3, (
            f"Expected C3 (order 3), got order {sym.output.order()}"
        )
