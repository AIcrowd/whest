import numpy as np

import whest as we
from whest._ndarray import WhestArray


def test_whestarray_ufunc_result_remains_subclass_and_can_be_view_backed():
    x = np.ones((2, 2)).view(WhestArray)

    result = np.add(x, x)

    assert isinstance(result, WhestArray)
    assert result.base is not None
    np.testing.assert_array_equal(result, np.full((2, 2), 2.0))


def test_symmetric_tensor_ufunc_result_preserves_symmetry():
    """Post-v2 (``__array_ufunc__`` enabled), ``np.add(A_sym, A_sym)``
    routes through ``me.add`` → ``_counted_binary`` → ``_pointwise_symmetry``,
    which preserves the surviving symmetry of both operands. The result
    therefore remains a ``SymmetricTensor`` carrying the same symmetry as
    the inputs — it does NOT drop to a plain ``WhestArray``.

    This supersedes the earlier ``..._can_drop_to_plain_view_backed_whestarray``
    test from PR #51, which pinned the pre-v2 behaviour where ufunc
    results lost symmetry metadata. With NEP-13 dispatch, that pathway
    no longer fires for whest-typed operands.
    """
    x = we.ones((2, 2))

    assert isinstance(x, we.SymmetricTensor)
    input_symmetry = x.symmetry
    assert input_symmetry is not None

    result = np.add(x, x)

    assert isinstance(result, we.SymmetricTensor)
    assert isinstance(result, WhestArray)
    assert result.symmetry == input_symmetry
    np.testing.assert_array_equal(result, np.full((2, 2), 2.0))
