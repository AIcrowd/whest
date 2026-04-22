"""Serialization round-trip tests: client encode → server decode, without a network.

Verifies that objects survive the client-encode → server-decode cycle by
importing client and server code directly (no ZMQ/network required).

Path setup:
- Core: src/  (package: whest, found via root pyproject.toml)
- Client: whest-client/src/  (loaded with importlib for client-specific modules)
- Server: whest-server/src/  (added to sys.path for whest_server)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: add whest-server/src so whest_server is importable
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_SERVER_SRC = str(_ROOT / "whest-server" / "src")
_CLIENT_SRC = str(_ROOT / "whest-client" / "src")

if _SERVER_SRC not in sys.path:
    sys.path.insert(0, _SERVER_SRC)


# ---------------------------------------------------------------------------
# Import server modules (whest_server uses sys.path above)
# ---------------------------------------------------------------------------

from whest_server._request_handler import RequestHandler  # noqa: E402
from whest_server._session import Session  # noqa: E402

# ---------------------------------------------------------------------------
# Load client-specific modules via importlib (avoid package name collision)
# ---------------------------------------------------------------------------


def _load_client_module(rel_path: str, module_name: str):
    """Load a module from whest-client/src using importlib.

    The loaded module is registered in sys.modules under *module_name* so
    that intra-package imports (e.g. ``from whest._math_compat import ...``)
    resolve correctly within the client package namespace.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_file = Path(_CLIENT_SRC) / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load client sub-modules in dependency order so internal imports resolve.
# Each is registered under its canonical dotted name inside the whest package
# that the core already occupies, but these sub-modules don't exist in core so
# there's no collision risk.
_load_client_module("whest/_constants.py", "whest._constants")
_load_client_module("whest/_math_compat.py", "whest._math_compat")
_load_client_module("whest/_perm_group.py", "whest._perm_group")
_client_remote_array = _load_client_module(
    "whest/_remote_array.py", "whest._remote_array"
)
# Re-load perm_group so we have a local reference (it may already be cached)
_client_perm_group = sys.modules["whest._perm_group"]

_encode_arg = _client_remote_array._encode_arg
_result_from_response = _client_remote_array._result_from_response

ClientPermutation = _client_perm_group._Permutation
ClientSymmetryGroup = _client_perm_group.SymmetryGroup
ClientCycle = _client_perm_group._Cycle


# ---------------------------------------------------------------------------
# Core whest (from root pyproject.toml / src/)
# ---------------------------------------------------------------------------

import whest as we  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: create a fresh session + handler for each test
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler_session():
    """Yield a (session, handler) pair and close the session after the test."""
    session = Session(flop_budget=10_000_000)
    handler = RequestHandler(session)
    yield session, handler
    if session.is_open:
        session.close()


# ---------------------------------------------------------------------------
# TestPermutationRoundTrip
# ---------------------------------------------------------------------------


class TestPermutationRoundTrip:
    """Client hidden permutation helpers and SymmetryGroup round-trip cleanly."""

    def test_permutation_basic(self, handler_session):
        """Permutation([2,0,1]) survives the round-trip."""
        perm = ClientPermutation([2, 0, 1])
        encoded = _encode_arg(perm)

        # encoded must be the wire dict
        assert isinstance(encoded, dict)
        assert "__permutation__" in encoded
        assert encoded["__permutation__"] == [2, 0, 1]

        # Server resolves it back to a whest Permutation
        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert resolved.__class__.__name__ == "_Permutation"
        assert resolved.array_form == [2, 0, 1]

    def test_permutation_identity(self, handler_session):
        """Identity permutation of size 4 round-trips correctly."""
        perm = ClientPermutation([0, 1, 2, 3])
        encoded = _encode_arg(perm)
        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert resolved.__class__.__name__ == "_Permutation"
        assert resolved.array_form == [0, 1, 2, 3]
        assert resolved.is_identity

    def test_symmetry_group_symmetric(self, handler_session):
        """SymmetryGroup.symmetric encodes and decodes with correct degree."""
        group = ClientSymmetryGroup.symmetric(axes=(0, 1, 2))
        encoded = _encode_arg(group)

        assert isinstance(encoded, dict)
        assert "__symmetry_group__" in encoded
        pg_wire = encoded["__symmetry_group__"]
        assert pg_wire["axes"] == [0, 1, 2]
        assert len(pg_wire["generators"]) >= 1

        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert isinstance(resolved, we.SymmetryGroup)
        assert resolved.degree == 3
        # S_3 has 6 elements
        assert resolved.order() == 6

    def test_symmetry_group_with_axes(self, handler_session):
        """SymmetryGroup with axes survives round-trip."""
        group = ClientSymmetryGroup.symmetric(axes=(0, 1))
        encoded = _encode_arg(group)

        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert isinstance(resolved, we.SymmetryGroup)
        assert resolved.degree == 2
        assert resolved.axes == (0, 1)

    def test_cycle_composition(self, handler_session):
        """Cycle(0,2)(1,3) encodes and decodes to the correct permutation."""
        cycle = ClientCycle(0, 2)(1, 3)
        encoded = _encode_arg(cycle)

        assert isinstance(encoded, dict)
        assert "__permutation__" in encoded

        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert resolved.__class__.__name__ == "_Permutation"
        # Cycle(0,2)(1,3): 0->2, 2->0, 1->3, 3->1 → array form [2,3,0,1]
        assert resolved.array_form == [2, 3, 0, 1]

    def test_cycle_single(self, handler_session):
        """Single Cycle(0,1) encodes and decodes correctly."""
        cycle = ClientCycle(0, 1)
        encoded = _encode_arg(cycle)

        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert resolved.__class__.__name__ == "_Permutation"
        assert resolved.array_form == [1, 0]

    def test_permutation_in_list(self, handler_session):
        """Permutation nested inside a list encodes/decodes correctly."""
        perm = ClientPermutation([1, 0, 2])
        encoded = _encode_arg([perm, 42])

        assert isinstance(encoded, list)
        assert isinstance(encoded[0], dict)
        assert "__permutation__" in encoded[0]

        _, handler = handler_session
        resolved = handler._resolve_arg(encoded)

        assert isinstance(resolved, list)
        assert resolved[0].__class__.__name__ == "_Permutation"
        assert resolved[0].array_form == [1, 0, 2]
        assert resolved[1] == 42


# ---------------------------------------------------------------------------
# TestSymmetryInfoRoundTrip
# ---------------------------------------------------------------------------


class TestSymmetryInfoRoundTrip:
    """SymmetricTensor packed by server → unpacked by client → symmetry present."""

    def _make_symmetric_array(self) -> np.ndarray:
        """Create a small symmetric (in axes 0,1) numpy array."""
        rng = np.random.default_rng(42)
        data = rng.random((3, 3))
        # Make it symmetric: average with its transpose
        data = (data + data.T) / 2.0
        return data

    def test_symmetric_tensor_pack_unpack(self, handler_session):
        """as_symmetric → _pack_result → _result_from_response → symmetry present."""
        data = self._make_symmetric_array()
        sym_tensor = we.as_symmetric(
            data,
            symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
        )

        session, handler = handler_session
        packed = handler._pack_result(sym_tensor)

        assert packed["status"] == "ok"
        result_dict = packed["result"]
        assert "id" in result_dict, "Expected handle id in result"
        assert "symmetry" in result_dict, "Expected symmetry in result"

        remote_arr = _result_from_response(packed)

        assert remote_arr.symmetry is not None
        assert remote_arr.symmetry == ClientSymmetryGroup.from_payload(
            {"axes": [0, 1], "generators": [[1, 0]]}
        )

    def test_plain_array_no_symmetry(self, handler_session):
        """Plain ndarray packed → _result_from_response → symmetry is None."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        session, handler = handler_session
        packed = handler._pack_result(data)

        assert packed["status"] == "ok"
        assert "symmetry" not in packed["result"]

        remote_arr = _result_from_response(packed)
        assert remote_arr.symmetry is None

    def test_symmetric_tensor_shape_preserved(self, handler_session):
        """Shape is correctly preserved through the full pack/unpack cycle."""
        rng = np.random.default_rng(7)
        raw = rng.random((4, 4))
        data = (raw + raw.T) / 2.0
        sym_tensor = we.as_symmetric(
            data,
            symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
        )

        session, handler = handler_session
        packed = handler._pack_result(sym_tensor)
        remote_arr = _result_from_response(packed)

        assert remote_arr.shape == (4, 4)
        assert remote_arr.symmetry == ClientSymmetryGroup.from_payload(
            {"axes": [0, 1], "generators": [[1, 0]]}
        )

    def test_symmetric_tensor_multi_group(self, handler_session):
        """Tensor with two symmetry groups packs/unpacks with both groups present."""
        # Shape (3,3,2,2): axes (0,1) symmetric and axes (2,3) symmetric
        rng = np.random.default_rng(99)
        raw = rng.random((3, 3, 2, 2))
        # Symmetrize axes 0,1
        raw = (raw + raw.transpose(1, 0, 2, 3)) / 2.0
        # Symmetrize axes 2,3
        raw = (raw + raw.transpose(0, 1, 3, 2)) / 2.0
        sym_tensor = we.as_symmetric(
            raw,
            symmetry=we.SymmetryGroup.direct_product(
                we.SymmetryGroup.symmetric(axes=(0, 1)),
                we.SymmetryGroup.symmetric(axes=(2, 3)),
            ),
        )

        session, handler = handler_session
        packed = handler._pack_result(sym_tensor)
        remote_arr = _result_from_response(packed)

        assert remote_arr.symmetry is not None
        assert remote_arr.symmetry.to_payload() == {
            "axes": [0, 1, 2, 3],
            "generators": [[1, 0, 2, 3], [0, 1, 3, 2]],
        }
