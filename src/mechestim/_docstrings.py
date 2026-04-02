"""Docstring inheritance helper for mechestim wrappers."""
from __future__ import annotations


def attach_docstring(wrapper, np_func, category: str, cost_description: str) -> None:
    """Attach NumPy's docstring to a mechestim wrapper, injecting cost info into Notes."""
    np_doc = getattr(np_func, "__doc__", None) or ""

    cost_note = f"**mechestim cost:** {cost_description}\n"

    if not np_doc:
        wrapper.__doc__ = (
            f"Counted wrapper for ``numpy.{np_func.__name__}``.\n\n"
            f"Notes\n-----\n{cost_note}"
        )
        return

    # Inject cost info into existing Notes section, or add one
    if "\nNotes\n-----" in np_doc:
        wrapper.__doc__ = np_doc.replace(
            "\nNotes\n-----\n",
            f"\nNotes\n-----\n{cost_note}\n",
            1,
        )
    else:
        wrapper.__doc__ = f"{np_doc.rstrip()}\n\nNotes\n-----\n{cost_note}"
