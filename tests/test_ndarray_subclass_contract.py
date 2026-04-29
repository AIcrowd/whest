import numpy as np

import flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray


def test_flopscopearray_ufunc_result_remains_subclass_and_can_be_view_backed():
    x = np.ones((2, 2)).view(FlopscopeArray)

    result = np.add(x, x)

    assert isinstance(result, FlopscopeArray)
    assert result.base is not None
    np.testing.assert_array_equal(result, np.full((2, 2), 2.0))


def test_symmetric_tensor_ufunc_result_can_drop_to_plain_view_backed_flopscopearray():
    x = fnp.ones((2, 2))

    result = np.add(x, x)

    assert isinstance(result, FlopscopeArray)
    assert type(result) is FlopscopeArray
    assert result.base is not None
    assert not hasattr(result, "symmetry")
    np.testing.assert_array_equal(result, np.full((2, 2), 2.0))
