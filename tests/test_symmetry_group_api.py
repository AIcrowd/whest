import pytest
import whest as we


def test_top_level_exports_only_symmetry_group():
    assert hasattr(we, "SymmetryGroup")
    assert not hasattr(we, "PermutationGroup")
    assert not hasattr(we, "Permutation")
    assert not hasattr(we, "Cycle")


def test_from_generators_uses_plain_literals_only():
    g = we.SymmetryGroup.from_generators([[1, 0]], axes=(2, 5))
    assert g.axes == (2, 5)
    assert g.degree == 2
    assert g.order() == 2


def test_symmetric_degree_is_inferred_from_axes():
    g = we.SymmetryGroup.symmetric(axes=(2, 5, 7))
    assert g.degree == 3
    assert g.axes == (2, 5, 7)


def test_cyclic_and_dihedral_infer_degree_from_axes():
    assert we.SymmetryGroup.cyclic(axes=(0, 1, 2)).degree == 3
    assert we.SymmetryGroup.dihedral(axes=(0, 1, 2, 3)).degree == 4


def test_axes_participate_in_group_identity():
    g1 = we.SymmetryGroup.symmetric(axes=(0, 1))
    g2 = we.SymmetryGroup.symmetric(axes=(2, 3))
    assert g1 != g2
    assert hash(g1) != hash(g2)


def test_axis_order_does_not_change_semantic_group_identity():
    a = we.SymmetryGroup.symmetric(axes=(0, 2))
    b = we.SymmetryGroup.symmetric(axes=(2, 0))
    assert a == b


def test_direct_product_rejects_overlapping_support():
    a = we.SymmetryGroup.symmetric(axes=(0, 1))
    b = we.SymmetryGroup.symmetric(axes=(1, 2))
    with pytest.raises(ValueError, match="disjoint"):
        we.SymmetryGroup.direct_product(a, b)


def test_from_generators_rejects_non_bijective_literal():
    with pytest.raises(ValueError, match="bijection"):
        we.SymmetryGroup.from_generators([[1, 1]], axes=(0, 1))


def test_from_generators_rejects_wrong_degree_literal():
    with pytest.raises(ValueError, match="degree"):
        we.SymmetryGroup.from_generators([[1, 0, 2]], axes=(0, 1))
