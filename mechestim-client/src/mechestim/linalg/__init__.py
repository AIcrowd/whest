"""mechestim.linalg — linear algebra submodule proxy."""
from __future__ import annotations

from mechestim._remote_array import RemoteArray, RemoteScalar, _result_from_response, _encode_arg
from mechestim._connection import get_connection
from mechestim._protocol import encode_request
from mechestim._getattr import make_module_getattr


def _make_linalg_proxy(op_name: str):
    """Create a proxy function for ``linalg.<op_name>``."""
    qualified = f"linalg.{op_name}"

    def proxy(*args, **kwargs):
        conn = get_connection()
        encoded_args = [_encode_arg(a) for a in args]
        encoded_kwargs = {k: _encode_arg(v) for k, v in kwargs.items()}
        resp = conn.send_recv(encode_request(qualified, args=encoded_args, kwargs=encoded_kwargs))
        return _result_from_response(resp)

    proxy.__name__ = op_name
    proxy.__qualname__ = f"linalg.{op_name}"
    return proxy


# The only non-blacklisted linalg function in the registry is svd.
svd = _make_linalg_proxy("svd")

# Fall-through for blacklisted / unknown names
__getattr__ = make_module_getattr("linalg.", "mechestim.linalg")
