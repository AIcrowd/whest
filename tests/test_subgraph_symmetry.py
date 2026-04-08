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
