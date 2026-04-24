"""Test that flopscope function signatures match numpy exactly.

Flopscope-only extensions (like svd.k) must be in the allowlist.
Ufuncs are skipped since numpy ufuncs are C-level objects whose
inspect.signature() returns (*args, **kwargs).
"""

import inspect

import numpy as np
import pytest

import flopscope as flops
import flopscope.numpy as fnp

fnp = fnp  # backwards-compat local alias for this test
# Flopscope-only keyword extensions that numpy doesn't have
ALLOWED_EXTRA_PARAMS = {
    "einsum": {"symmetric_axes", "symmetry", "subscripts"},
    "linalg.svd": {"k"},
    "linalg.svdvals": {"k"},
}

# Functions where flopscope intentionally differs (with reason)
SKIP_FUNCTIONS = {
    # einsum_path has different optimizer interface
    "einsum_path",
}

_NUMPY_GE_2_4 = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 4)
if _NUMPY_GE_2_4:
    # numpy 2.4 added C-level positional-only markers that flopscope's
    # (*args, **kwargs) wrappers can't match exactly.
    SKIP_FUNCTIONS |= {
        "dot",
        "packbits",
        "unpackbits",
        "shares_memory",
        "ravel_multi_index",
        "promote_types",
    }


def _iter_flopscope_functions():
    """Yield (dotted_name, flopscope_fn, numpy_fn) for all comparable functions."""
    # Top-level
    for name in sorted(dir(fnp)):
        if name.startswith("_"):
            continue
        we_fn = getattr(fnp, name, None)
        np_fn = getattr(np, name, None)
        if not callable(we_fn) or not callable(np_fn):
            continue
        if isinstance(np_fn, np.ufunc):
            continue
        yield name, we_fn, np_fn

    # Submodules
    for submod in ["linalg", "fft", "random"]:
        np_sub = getattr(np, submod, None)
        we_sub = getattr(fnp, submod, None)
        if np_sub is None or we_sub is None:
            continue
        for name in sorted(dir(we_sub)):
            if name.startswith("_"):
                continue
            we_fn = getattr(we_sub, name, None)
            np_fn = getattr(np_sub, name, None)
            if not callable(we_fn) or not callable(np_fn):
                continue
            if isinstance(np_fn, np.ufunc):
                continue
            yield f"{submod}.{name}", we_fn, np_fn


def _get_param_names(sig):
    """Extract parameter names, excluding *args and **kwargs."""
    return {
        name
        for name, p in sig.parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


@pytest.mark.parametrize(
    "dotted_name,we_fn,np_fn",
    [
        pytest.param(name, we_fn, np_fn, id=name)
        for name, we_fn, np_fn in _iter_flopscope_functions()
        if name not in SKIP_FUNCTIONS
    ],
)
def test_signature_matches_numpy(dotted_name, we_fn, np_fn):
    """flopscope function signature must be a superset of numpy's."""
    try:
        np_sig = inspect.signature(np_fn)
    except (ValueError, TypeError):
        pytest.skip(f"Cannot inspect numpy.{dotted_name}")
    try:
        we_sig = inspect.signature(we_fn)
    except (ValueError, TypeError):
        pytest.skip(f"Cannot inspect flopscope.{dotted_name}")

    np_params = _get_param_names(np_sig)
    we_params = _get_param_names(we_sig)
    allowed_extra = ALLOWED_EXTRA_PARAMS.get(dotted_name, set())

    missing = np_params - we_params
    unexpected_extra = (we_params - np_params) - allowed_extra

    assert not missing, (
        f"flopscope.{dotted_name} is missing numpy params: {missing}\n"
        f"  numpy:  {np_sig}\n"
        f"  flopscope:  {we_sig}"
    )
    assert not unexpected_extra, (
        f"flopscope.{dotted_name} has unexpected extra params: {unexpected_extra}\n"
        f"  (add to ALLOWED_EXTRA_PARAMS if intentional)\n"
        f"  numpy:  {np_sig}\n"
        f"  flopscope:  {we_sig}"
    )
