"""Transparent proxy classes for server-side arrays and scalar values.

``RemoteArray`` and ``RemoteScalar`` make the client-server split invisible
to participants: metadata is cached locally while data access and arithmetic
operations are dispatched to the server transparently.
"""
from __future__ import annotations

import functools
import math
import struct
from typing import Any, Dict, List, Tuple, Union

# ---------------------------------------------------------------------------
# dtype helpers  (NO numpy -- pure struct)
# ---------------------------------------------------------------------------

#: Maps dtype string to (struct format char, byte width).
_DTYPE_INFO: Dict[str, Tuple[str, int]] = {
    "float64": ("d", 8),
    "float32": ("f", 4),
    "int64": ("q", 8),
    "int32": ("i", 4),
    "int16": ("h", 2),
    "int8": ("b", 1),
    "uint64": ("Q", 8),
    "uint32": ("I", 4),
    "uint16": ("H", 2),
    "uint8": ("B", 1),
    "bool": ("?", 1),
}


def _bytes_to_list(
    data: bytes, shape: Tuple[int, ...], dtype: str
) -> Any:
    """Convert raw *data* bytes into a (possibly nested) Python list.

    Uses :mod:`struct` for unpacking -- no numpy dependency.

    Parameters
    ----------
    data:
        Raw little-endian bytes produced by the server.
    shape:
        Array shape.  An empty tuple means scalar.
    dtype:
        Element data-type string, e.g. ``"float64"``.

    Returns
    -------
    Scalar value, flat list, or nested list of lists depending on *shape*.
    """
    fmt_char, item_size = _DTYPE_INFO[dtype]
    total = max(math.prod(shape), 1) if shape else 1
    flat = list(struct.unpack(f"<{total}{fmt_char}", data))

    # Scalar
    if not shape:
        return flat[0]

    # 1-D
    if len(shape) == 1:
        return flat

    # N-D: reshape into nested lists
    return _reshape(flat, shape)


def _reshape(flat: list, shape: Tuple[int, ...]) -> Any:
    """Reshape a flat list into nested lists matching *shape*."""
    if len(shape) == 1:
        return flat

    stride = math.prod(shape[1:])
    return [
        _reshape(flat[i * stride : (i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


# ---------------------------------------------------------------------------
# RemoteScalar
# ---------------------------------------------------------------------------


class RemoteScalar:
    """Wraps a scalar value returned from the server.

    Behaves like a Python number for comparisons, arithmetic (via
    ``float()``), and hashing.

    Parameters
    ----------
    value:
        The scalar numeric value.
    dtype:
        Data-type string (e.g. ``"float64"``).
    """

    __slots__ = ("_value", "_dtype")

    def __init__(self, value: Union[int, float], dtype: str) -> None:
        self._value = value
        self._dtype = dtype

    # -- array-like metadata ------------------------------------------------

    @property
    def shape(self) -> tuple:
        return ()

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def handle_id(self) -> None:
        return None

    # -- conversions --------------------------------------------------------

    def tolist(self) -> Union[int, float]:
        return self._value

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RemoteScalar({self._value!r}, dtype={self._dtype!r})"

    def __str__(self) -> str:
        return str(self._value)

    # -- comparisons --------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value == other._value
        return self._value == other

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value < other._value
        return self._value < other  # type: ignore[operator]

    def __le__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value <= other._value
        return self._value <= other  # type: ignore[operator]

    def __gt__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value > other._value
        return self._value > other  # type: ignore[operator]

    def __ge__(self, other: object) -> bool:
        if isinstance(other, RemoteScalar):
            return self._value >= other._value
        return self._value >= other  # type: ignore[operator]

    def __hash__(self) -> int:
        return hash(self._value)


# ---------------------------------------------------------------------------
# RemoteArray
# ---------------------------------------------------------------------------


class RemoteArray:
    """Transparent proxy for a server-side numpy array.

    The constructor only stores metadata -- no data is transferred until
    explicitly requested (via :meth:`tolist`, :meth:`__repr__`, etc.).

    Parameters
    ----------
    handle_id:
        Opaque server handle for this array.
    shape:
        Array shape tuple.
    dtype:
        Element data-type string (e.g. ``"float64"``).
    """

    __slots__ = ("_handle_id", "_shape", "_dtype")

    def __init__(self, handle_id: str, shape: tuple, dtype: str) -> None:
        self._handle_id = handle_id
        self._shape = tuple(shape)
        self._dtype = dtype

    # -- cached metadata (no round-trip) ------------------------------------

    @property
    def handle_id(self) -> str:
        return self._handle_id

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return math.prod(self._shape) if self._shape else 1

    @property
    def nbytes(self) -> int:
        _, item_size = _DTYPE_INFO[self._dtype]
        return self.size * item_size

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError(
                "len() of unsized object (0-d array)"
            )
        return self._shape[0]

    # -- data access (auto-fetch from server) -------------------------------

    def _fetch_data(self) -> Tuple[bytes, tuple, str]:
        """Fetch the raw data from the server.

        Returns ``(raw_bytes, shape, dtype)``.
        """
        from mechestim._connection import get_connection
        from mechestim._protocol import encode_fetch

        resp = get_connection().send_recv(encode_fetch(self._handle_id))
        # Fetch responses may have data at top level or inside "result"
        if "data" in resp:
            return resp["data"], tuple(resp["shape"]), resp["dtype"]
        result = resp.get("result", {})
        return result["data"], tuple(result["shape"]), result["dtype"]

    def tolist(self) -> Any:
        """Fetch data and convert to a nested Python list."""
        data, shape, dtype = self._fetch_data()
        return _bytes_to_list(data, shape, dtype)

    def __repr__(self) -> str:
        try:
            values = self.tolist()
            return f"array({values!r})"
        except Exception:
            return (
                f"RemoteArray(handle_id={self._handle_id!r}, "
                f"shape={self._shape}, dtype={self._dtype!r})"
            )

    def __str__(self) -> str:
        return self.__repr__()

    def __float__(self) -> float:
        if self.size != 1:
            raise TypeError(
                "only size-1 arrays can be converted to Python scalars"
            )
        data, shape, dtype = self._fetch_data()
        result = _bytes_to_list(data, shape, dtype)
        # Unwrap single-element lists (e.g., shape (1,) returns [42.0])
        while isinstance(result, list):
            result = result[0]
        return float(result)

    def __int__(self) -> int:
        if self.size != 1:
            raise TypeError(
                "only size-1 arrays can be converted to Python scalars"
            )
        data, shape, dtype = self._fetch_data()
        result = _bytes_to_list(data, shape, dtype)
        while isinstance(result, list):
            result = result[0]
        return int(result)

    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous."
            )
        data, shape, dtype = self._fetch_data()
        result = _bytes_to_list(data, shape, dtype)
        while isinstance(result, list):
            result = result[0]
        return bool(result)

    def __iter__(self):
        values = self.tolist()
        if not self._shape:
            raise TypeError("iteration over a 0-d array")
        return iter(values)

    def __getitem__(self, key):
        values = self.tolist()
        if isinstance(values, list):
            return values[key]
        # scalar
        if key == () or key == Ellipsis:
            return values
        raise IndexError(f"too many indices for array with shape {self._shape}")

    # -- operator overloads (dispatch to server) ----------------------------

    def _dispatch_op(self, op_name: str, *args: Any) -> Any:
        """Encode and send an operation to the server, return the result."""
        from mechestim._connection import get_connection
        from mechestim._protocol import encode_request

        encoded = []
        for a in args:
            if isinstance(a, RemoteArray):
                encoded.append({"__handle__": a.handle_id})
            elif isinstance(a, RemoteScalar):
                encoded.append(a.tolist())
            else:
                encoded.append(a)
        resp = get_connection().send_recv(
            encode_request(op_name, args=encoded)
        )
        return _result_from_response(resp)

    # Arithmetic
    def __add__(self, other):
        return self._dispatch_op("add", self, other)

    def __radd__(self, other):
        return self._dispatch_op("add", other, self)

    def __sub__(self, other):
        return self._dispatch_op("subtract", self, other)

    def __rsub__(self, other):
        return self._dispatch_op("subtract", other, self)

    def __mul__(self, other):
        return self._dispatch_op("multiply", self, other)

    def __rmul__(self, other):
        return self._dispatch_op("multiply", other, self)

    def __truediv__(self, other):
        return self._dispatch_op("true_divide", self, other)

    def __rtruediv__(self, other):
        return self._dispatch_op("true_divide", other, self)

    def __floordiv__(self, other):
        return self._dispatch_op("floor_divide", self, other)

    def __mod__(self, other):
        return self._dispatch_op("remainder", self, other)

    def __pow__(self, other):
        return self._dispatch_op("power", self, other)

    def __matmul__(self, other):
        return self._dispatch_op("matmul", self, other)

    def __neg__(self):
        return self._dispatch_op("negative", self)

    def __abs__(self):
        return self._dispatch_op("abs", self)

    # Comparisons (dispatch to server -- element-wise, returning RemoteArray)
    def __eq__(self, other):
        # Only dispatch to server for array/RemoteArray comparisons
        if isinstance(other, (RemoteArray, RemoteScalar)):
            return self._dispatch_op("equal", self, other)
        # For plain scalars, also dispatch
        return self._dispatch_op("equal", self, other)

    def __ne__(self, other):
        return self._dispatch_op("not_equal", self, other)

    def __lt__(self, other):
        return self._dispatch_op("less", self, other)

    def __le__(self, other):
        return self._dispatch_op("less_equal", self, other)

    def __gt__(self, other):
        return self._dispatch_op("greater", self, other)

    def __ge__(self, other):
        return self._dispatch_op("greater_equal", self, other)

    # RemoteArray is not hashable (same as numpy arrays)
    __hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _result_from_response(resp: dict) -> Union[RemoteArray, RemoteScalar, tuple, dict]:
    """Convert a server response dict into the appropriate proxy object.

    Examines the ``"result"`` key:

    * ``"value"`` present  -> :class:`RemoteScalar`
    * ``"multi"`` present  -> ``tuple`` of :class:`RemoteArray`
    * ``"id"``   present  -> single :class:`RemoteArray`
    * otherwise           -> raw dict
    """
    result = resp.get("result", {})

    if "value" in result:
        return RemoteScalar(value=result["value"], dtype=result.get("dtype", "float64"))

    if "multi" in result:
        return tuple(
            RemoteArray(
                handle_id=item["id"],
                shape=tuple(item["shape"]),
                dtype=item["dtype"],
            )
            for item in result["multi"]
        )

    if "id" in result:
        return RemoteArray(
            handle_id=result["id"],
            shape=tuple(result["shape"]),
            dtype=result["dtype"],
        )

    return result
