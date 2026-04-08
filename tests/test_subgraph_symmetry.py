"""Unit tests for SubgraphSymmetryOracle and its algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from mechestim._opt_einsum._subgraph_symmetry import (
    EinsumBipartite,
    SubgraphSymmetryOracle,
    _build_bipartite,
    _compute_subset_symmetry,
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
        g = _build_bipartite(
            [A, B, C], ["ij", "jk", "kl"], [None, None, None], "il"
        )
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
