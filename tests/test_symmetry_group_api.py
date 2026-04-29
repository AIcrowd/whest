import pytest

import flopscope as flops


def test_top_level_exports_only_symmetry_group():
    assert hasattr(flops, "SymmetryGroup")
    assert not hasattr(flops, "PermutationGroup")
    assert not hasattr(flops, "Permutation")
    assert not hasattr(flops, "Cycle")


def test_from_generators_uses_plain_literals_only():
    g = flops.SymmetryGroup.from_generators([[1, 0]], axes=(2, 5))
    assert g.axes == (2, 5)
    assert g.degree == 2
    assert g.order() == 2


def test_symmetric_degree_is_inferred_from_axes():
    g = flops.SymmetryGroup.symmetric(axes=(2, 5, 7))
    assert g.degree == 3
    assert g.axes == (2, 5, 7)


def test_cyclic_and_dihedral_infer_degree_from_axes():
    assert flops.SymmetryGroup.cyclic(axes=(0, 1, 2)).degree == 3
    assert flops.SymmetryGroup.dihedral(axes=(0, 1, 2, 3)).degree == 4


def test_axes_participate_in_group_identity():
    g1 = flops.SymmetryGroup.symmetric(axes=(0, 1))
    g2 = flops.SymmetryGroup.symmetric(axes=(2, 3))
    assert g1 != g2
    assert hash(g1) != hash(g2)


def test_axis_order_does_not_change_semantic_group_identity():
    a = flops.SymmetryGroup.symmetric(axes=(0, 2))
    b = flops.SymmetryGroup.symmetric(axes=(2, 0))
    assert a == b


def test_direct_product_rejects_overlapping_support():
    a = flops.SymmetryGroup.symmetric(axes=(0, 1))
    b = flops.SymmetryGroup.symmetric(axes=(1, 2))
    with pytest.raises(ValueError, match="disjoint"):
        flops.SymmetryGroup.direct_product(a, b)


def test_from_generators_rejects_non_bijective_literal():
    with pytest.raises(ValueError, match="bijection"):
        flops.SymmetryGroup.from_generators([[1, 1]], axes=(0, 1))


def test_from_generators_rejects_wrong_degree_literal():
    with pytest.raises(ValueError, match="degree"):
        flops.SymmetryGroup.from_generators([[1, 0, 2]], axes=(0, 1))
