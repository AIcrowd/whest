"""Client-side message encoding/decoding using msgpack."""
from __future__ import annotations

import msgpack


# Maximum byte length that we consider "short enough" to attempt ASCII decode.
# Handle IDs and status strings are short ASCII; binary array data may contain
# high bytes even when valid UTF-8.
_SHORT_THRESHOLD = 32


def _normalize(value: object) -> object:
    """Recursively normalize bytes keys/values to strings where appropriate.

    Binary data fields (raw array bytes) stay as bytes.  Only decode bytes
    that are short AND contain only ASCII printable characters (no bytes > 127).
    This catches handle IDs and dtype strings but never touches binary data.
    """
    if isinstance(value, bytes):
        if len(value) <= _SHORT_THRESHOLD and all(32 <= b < 128 for b in value):
            return value.decode("ascii")
        return value
    if isinstance(value, dict):
        return {_normalize(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        normalized = [_normalize(item) for item in value]
        return type(value)(normalized)
    return value


def encode_request(op: str, args=None, kwargs=None) -> bytes:
    """Msgpack-encode ``{"op": op, "args": args, "kwargs": kwargs}``."""
    return msgpack.packb(
        {"op": op, "args": args, "kwargs": kwargs},
        use_bin_type=True,
    )


def encode_create_from_data(data: bytes, shape: list, dtype: str) -> bytes:
    """Encode a create_from_data request."""
    return encode_request("create_from_data", args=[data, shape, dtype])


def encode_budget_open(flop_budget: int, flop_multiplier: float = 1.0) -> bytes:
    """Encode a budget_open request."""
    return encode_request("budget_open", kwargs={"flop_budget": flop_budget, "flop_multiplier": flop_multiplier})


def encode_budget_close() -> bytes:
    """Encode a budget_close request."""
    return encode_request("budget_close")


def encode_budget_status() -> bytes:
    """Encode a budget_status request."""
    return encode_request("budget_status")


def encode_fetch(handle_id: str) -> bytes:
    """Encode a fetch request."""
    return encode_request("fetch", kwargs={"handle_id": handle_id})


def encode_free(handles: list[str]) -> bytes:
    """Encode a free request."""
    return encode_request("free", kwargs={"handles": handles})


def decode_response(raw: bytes) -> dict:
    """Decode a msgpack response, normalizing bytes keys/values to strings.

    Binary data fields (raw array bytes) stay as bytes.
    """
    decoded = msgpack.unpackb(raw, raw=True)
    return _normalize(decoded)
