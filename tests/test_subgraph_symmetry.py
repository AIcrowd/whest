"""Unit tests for SubgraphSymmetryOracle and its algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from mechestim._opt_einsum._subgraph_symmetry import (
    EinsumBipartite,
    SubgraphSymmetryOracle,
    _build_bipartite,
)


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
        )
        assert g.u_vertices == ()
        assert g.free_labels == frozenset()


class TestBuildBipartite:
    def test_single_dense_operand(self):
        A = np.zeros((3, 4))
        g = _build_bipartite(
            operands=[A],
            subscript_parts=["ij"],
            per_op_syms=[None],
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

    def test_fully_symmetric_operand_collapses_to_one_u(self):
        T = np.zeros((3, 3))
        per_op = [[frozenset({("i",), ("j",)})]]  # T symmetric in (i, j)
        g = _build_bipartite(
            operands=[T],
            subscript_parts=["ij"],
            per_op_syms=per_op,
            output_chars="ij",
        )
        # One U vertex for the class {i, j}
        assert len(g.u_vertices) == 1
        assert g.incidence[0] == {"i": 1, "j": 1}
        assert g.u_labels[0] == frozenset({"i", "j"})

    def test_repeated_axis_in_subscript_gives_multiplicity(self):
        # einsum('iij->ij', T) — axis i appears twice in T
        T = np.zeros((3, 3, 4))
        g = _build_bipartite(
            operands=[T],
            subscript_parts=["iij"],
            per_op_syms=[None],
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
            per_op_syms=[None, None],
            output_chars="ik",
        )
        assert g.free_labels == frozenset("ik")
        assert g.summed_labels == frozenset("j")

    def test_identical_operands_are_grouped(self):
        X = np.zeros((3, 3))
        g = _build_bipartite(
            operands=[X, X],
            subscript_parts=["ij", "jk"],
            per_op_syms=[None, None],
            output_chars="ik",
        )
        assert g.identical_operand_groups == ((0, 1),)

    def test_distinct_operands_same_shape_are_not_grouped(self):
        X = np.zeros((3, 3))
        Y = np.zeros((3, 3))  # different Python object
        g = _build_bipartite(
            operands=[X, Y],
            subscript_parts=["ij", "jk"],
            per_op_syms=[None, None],
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
            per_op_syms=[[frozenset({("i",), ("j",)})], None, None],
            output_chars="ab",
        )
        # T (op 0) has one U vertex for class {i, j}
        # S1 (op 1) has two U vertices: {a}, {i}
        # S2 (op 2) has two U vertices: {b}, {j}
        assert len(g.u_vertices) == 5
        assert g.identical_operand_groups == ((1, 2),)
        assert g.free_labels == frozenset("ab")
        assert g.summed_labels == frozenset("ij")
        # T's row has incidence {i: 1, j: 1}
        t_rows = [row for u, row in zip(g.u_operand, g.incidence) if u == 0]
        assert len(t_rows) == 1
        assert t_rows[0] == {"i": 1, "j": 1}


class TestSubsetInduction:
    def test_full_subset_matches_top_level_partition(self):
        A = np.zeros((3, 4))
        B = np.zeros((4, 5))
        g = _build_bipartite([A, B], ["ij", "jk"], [None, None], "ik")
        from mechestim._opt_einsum._subgraph_symmetry import _induce_subgraph

        sub = _induce_subgraph(g, frozenset({0, 1}))
        assert sub.v_labels == frozenset("ik")
        assert sub.w_labels == frozenset("j")
        assert len(sub.u_local) == 4  # all U vertices
        assert sub.id_groups == ()

    def test_singleton_subset_crossing_labels_become_free(self):
        A = np.zeros((3, 4))
        B = np.zeros((4, 5))
        g = _build_bipartite([A, B], ["ij", "jk"], [None, None], "ik")
        from mechestim._opt_einsum._subgraph_symmetry import _induce_subgraph

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
        from mechestim._opt_einsum._subgraph_symmetry import _induce_subgraph

        # Contract ops 0 and 1. j is summed entirely within the subset
        # (not in any operand outside the subset, not in output), so j
        # belongs to W_S. k crosses the cut to op 2, so it's V_S.
        sub = _induce_subgraph(g, frozenset({0, 1}))
        assert sub.v_labels == frozenset("ik")
        assert sub.w_labels == frozenset("j")

    def test_identical_group_restricts_to_subset(self):
        X = np.zeros((3, 3))
        g = _build_bipartite([X, X, X], ["ai", "bi", "ci"], [None, None, None], "abc")
        from mechestim._opt_einsum._subgraph_symmetry import _induce_subgraph

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
            per_op_syms=[[frozenset({("i",), ("j",)})], None, None],
            output_chars="ab",
        )
        sym = oracle.sym(frozenset({0, 1, 2}))
        assert sym == [frozenset({("a",), ("b",)})]

    def test_three_identical_operands_finds_s3(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X, X],
            subscript_parts=["ai", "bi", "ci"],
            per_op_syms=[None, None, None],
            output_chars="abc",
        )
        sym = oracle.sym(frozenset({0, 1, 2}))
        # All three of a, b, c should be in one merged group (S3)
        assert sym is not None
        assert len(sym) == 1
        only_group = sym[0]
        assert only_group == frozenset({("a",), ("b",), ("c",)})

    def test_distinct_operands_no_induced_symmetry(self):
        X = np.zeros((3, 3))
        Y = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, Y, Y],
            subscript_parts=["ai", "bi", "ci"],
            per_op_syms=[None, None, None],
            output_chars="abc",
        )
        sym = oracle.sym(frozenset({0, 1, 2}))
        # Only Y, Y are identical (ops 1, 2), inducing S2{b, c}
        assert sym == [frozenset({("b",), ("c",)})]

    def test_empty_v_returns_none(self):
        # einsum('i,i->', X, X) — scalar output, V is empty
        X = np.zeros((3,))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["i", "i"],
            per_op_syms=[None, None],
            output_chars="",
        )
        assert oracle.sym(frozenset({0, 1})) is None


class TestHybridBlockPath:
    def test_gram_matrix_per_index_via_block(self):
        # einsum('ij,ik->jk', X, X) — a single-index swap the block path
        # also finds (as a 1-tuple block).
        X = np.zeros((3, 4))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["ij", "ik"],
            per_op_syms=[None, None],
            output_chars="jk",
        )
        sym = oracle.sym(frozenset({0, 1}))
        assert sym == [frozenset({("j",), ("k",)})]

    def test_outer_product_block_s2(self):
        # einsum('ijk,ilm->jklm', X, X) — block symmetry on (j,k) and (l,m)
        X = np.zeros((3, 3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["ijk", "ilm"],
            per_op_syms=[None, None],
            output_chars="jklm",
        )
        sym = oracle.sym(frozenset({0, 1}))
        assert sym is not None
        # Expect a single block group: {(j,k), (l,m)}
        assert len(sym) == 1
        only = sym[0]
        assert frozenset({("j", "k"), ("l", "m")}) == only

    def test_distinct_operands_no_block(self):
        X = np.zeros((3, 3, 3))
        Y = np.zeros((3, 3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, Y],
            subscript_parts=["ijk", "ilm"],
            per_op_syms=[None, None],
            output_chars="jklm",
        )
        assert oracle.sym(frozenset({0, 1})) is None


class TestOracleCaching:
    def test_same_subset_returns_cached_object(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle(
            operands=[X, X],
            subscript_parts=["ij", "jk"],
            per_op_syms=[None, None],
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

    def test_empty_subset_returns_none(self):
        X = np.zeros((3, 3))
        oracle = SubgraphSymmetryOracle([X, X], ["ij", "jk"], [None, None], "ik")
        assert oracle.sym(frozenset()) is None


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
        # mechestim._einsum until Commit 2). Skip gracefully if it has
        # already been removed.
        try:
            from mechestim._einsum import _detect_induced_output_symmetry
        except ImportError:
            pytest.skip("old detector already removed")

        old = _detect_induced_output_symmetry(
            operands=operands,
            subscript_parts=input_parts,
            output_chars=output_chars,
            per_op_syms=[None] * len(operands),
        )
        if old is None:
            old = []

        oracle = SubgraphSymmetryOracle(
            operands=operands,
            subscript_parts=input_parts,
            per_op_syms=[None] * len(operands),
            output_chars=output_chars,
        )
        new = oracle.sym(frozenset(range(len(operands))))
        if new is None:
            new = []

        # For each old group, assert that there exists a new group that
        # covers it (superset by labels).
        for old_g in old:
            old_chars = frozenset(c for block in old_g for c in block)
            assert any(
                old_chars <= frozenset(c for block in new_g for c in block)
                for new_g in new
            ), f"Old group {old_g} not covered by new oracle: {new}"


from mechestim._opt_einsum._symmetry import SubsetSymmetry


class TestSubsetSymmetryDataclass:
    def test_both_none(self):
        ss = SubsetSymmetry(output=None, inner=None)
        assert ss.output is None
        assert ss.inner is None

    def test_output_only(self):
        sym = [frozenset({("a",), ("b",)})]
        ss = SubsetSymmetry(output=sym, inner=None)
        assert ss.output == sym
        assert ss.inner is None

    def test_both_populated(self):
        v = [frozenset({("a",), ("b",)})]
        w = [frozenset({("i",), ("j",)})]
        ss = SubsetSymmetry(output=v, inner=w)
        assert ss.output == v
        assert ss.inner == w

    def test_frozen(self):
        ss = SubsetSymmetry(output=None, inner=None)
        import pytest

        with pytest.raises(AttributeError):
            ss.output = [frozenset({("x",)})]  # type: ignore[misc]
