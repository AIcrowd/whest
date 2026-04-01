"""Docstring inheritance helper for mechestim wrappers."""
from __future__ import annotations


def attach_docstring(wrapper, np_func, category: str, cost_description: str) -> None:
    """Attach a mechestim + numpy combined docstring to a wrapper function."""
    header = f"[mechestim] Cost: {cost_description} | Category: {category}"
    np_doc = getattr(np_func, "__doc__", None) or ""
    if np_doc:
        wrapper.__doc__ = f"{header}\n\n--- numpy docstring ---\n{np_doc}"
    else:
        wrapper.__doc__ = header
