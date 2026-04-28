"""Module-level ``__getattr__`` factory for helpful error messages.

Produces a ``__getattr__`` function that checks the registry and raises
:class:`AttributeError` with a descriptive message for blacklisted,
registered-but-unimplemented, or completely unknown names.
"""

from __future__ import annotations

from flopscope._registry import BLACKLISTED, get_category


def make_module_getattr(module_prefix: str, module_label: str):
    """Return a ``__getattr__`` suitable for assignment at module scope.

    Parameters
    ----------
    module_prefix:
        Prefix prepended before looking up the registry, e.g. ``"fft."``
        for the ``flopscope.numpy.fft`` submodule.  Pass ``""`` for the top-level
        package.
    module_label:
        Human-readable module name used in error messages, e.g.
        ``"flopscope.numpy.fft"``.
    """

    def __getattr__(name: str):
        # Skip dunder/private names to avoid interfering with import machinery
        if name.startswith("_"):
            raise AttributeError(f"module '{module_label}' has no attribute '{name}'")

        qualified = f"{module_prefix}{name}" if module_prefix else name
        category = get_category(qualified)

        if category == BLACKLISTED:
            raise AttributeError(
                f"'{module_label}.{name}' is intentionally not supported "
                f"in flopscope. This function is blacklisted because it "
                f"involves I/O, string formatting, or system-level "
                f"operations that are not meaningful in a remote-compute "
                f"environment."
            )
        elif category is not None:
            raise AttributeError(
                f"'{module_label}.{name}' is registered but not yet "
                f"implemented as a client-side proxy. "
                f"Category: {category}."
            )
        else:
            raise AttributeError(f"module '{module_label}' has no attribute '{name}'")

    return __getattr__
