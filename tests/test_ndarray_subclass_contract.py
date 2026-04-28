import numpy as np

import whest as we
from whest._ndarray import WhestArray


def test_whestarray_ufunc_result_remains_subclass_and_can_be_view_backed():
    x = np.ones((2, 2)).view(WhestArray)

    result = np.add(x, x)

    assert isinstance(result, WhestArray)
    assert result.base is not None
    np.testing.assert_array_equal(result, np.full((2, 2), 2.0))


def test_symmetric_tensor_ufunc_result_can_drop_to_plain_view_backed_whestarray():
    x = we.ones((2, 2))

    result = np.add(x, x)

    assert isinstance(result, WhestArray)
    assert type(result) is WhestArray
    assert result.base is not None
    assert not hasattr(result, "symmetry")
    np.testing.assert_array_equal(result, np.full((2, 2), 2.0))
