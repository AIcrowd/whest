"""Guardrail: every entry in _PATH_OPTIONS must accept symmetry_oracle."""

from __future__ import annotations

import inspect

from whest._opt_einsum._paths import _PATH_OPTIONS


def test_every_path_option_accepts_symmetry_oracle():
    failures = []
    for name, fn in _PATH_OPTIONS.items():
        sig = inspect.signature(fn)
        if "symmetry_oracle" not in sig.parameters:
            failures.append(name)
    assert not failures, (
        f"The following path optimizers do not accept symmetry_oracle: "
        f"{failures}. Every optimizer in _PATH_OPTIONS must accept this "
        f"kwarg to prevent silent symmetry drops."
    )
