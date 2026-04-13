"""Docstring inheritance helper for mechestim wrappers."""

from __future__ import annotations


def attach_docstring(wrapper, np_func, category: str, cost_description: str) -> None:
    """Attach NumPy's docstring to a mechestim wrapper with a FLOP Cost section.

    Inserts a dedicated "FLOP Cost" section after the summary line(s),
    before the Parameters section. This keeps cost info prominent and
    separate from NumPy's own Notes.
    """
    np_doc = getattr(np_func, "__doc__", None) or ""

    cost_section = f"FLOP Cost\n---------\n{cost_description}\n"

    if not np_doc:
        wrapper.__doc__ = (
            f"Counted wrapper for ``numpy.{np_func.__name__}``.\n\n{cost_section}"
        )
        return

    # Find the first standard section header (Parameters, Returns, etc.)
    # and insert the cost section before it.
    lines = np_doc.split("\n")
    section_headers = {
        "Parameters",
        "Returns",
        "Raises",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "Yields",
        "Warns",
        "Other Parameters",
        "Attributes",
        "Methods",
    }

    insert_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped in section_headers
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("---")
        ):
            insert_idx = i
            break

    if insert_idx is not None:
        # Insert cost section before the first standard section
        before = "\n".join(lines[:insert_idx]).rstrip()
        after = "\n".join(lines[insert_idx:])
        wrapper.__doc__ = f"{before}\n\n{cost_section}\n{after}"
    else:
        # No standard sections found — append cost section
        wrapper.__doc__ = f"{np_doc.rstrip()}\n\n{cost_section}"
