#!/usr/bin/env python
"""generate_api_docs.py — Generate API reference data and verify coverage.

Usage
-----
    python scripts/generate_api_docs.py              # generate API data
    python scripts/generate_api_docs.py --verify     # verify API data only
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import inspect
import json
import re
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "website"
PUBLIC_DIR = WEBSITE / "public"
GENERATED_DIR = WEBSITE / ".generated"
API_INDEX_PATH = WEBSITE / "content" / "docs" / "api" / "index.mdx"
OP_DOCS_DIR = WEBSITE / "content" / "docs" / "api" / "ops"
API_EXAMPLES_DIR = WEBSITE / "content" / "api-examples"
# Legacy MkDocs paths kept for helper functions that are no longer invoked by
# the current docs pipeline. The active build path only emits website/public/ops.json.
DOCS = ROOT / "docs"
API_DIR = DOCS / "api"
REF_DIR = DOCS / "reference"
WEIGHTS_PATH = ROOT / "src" / "whest" / "data" / "weights.json"
WEIGHTS_CSV_PATH = ROOT / "src" / "whest" / "data" / "weights.csv"

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def load_registry() -> dict[str, dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from whest._registry import REGISTRY  # type: ignore

    return REGISTRY


# ---------------------------------------------------------------------------
# Module mapping: which whest source modules cover which registry ops
# ---------------------------------------------------------------------------

# Maps (registry module, category prefix) → list of mkdocstrings directives
# and the doc page they belong to.
#
# Existing pages (not generated):
#   counted-ops.md  → ::: whest._pointwise, ::: whest._einsum.einsum
#   free-ops.md     → ::: whest._free_ops
#
# Generated pages:
GENERATED_PAGES: dict[str, dict] = {
    "api/linalg.md": {
        "title": "Linear Algebra",
        "description": (
            "Operations from `whest.linalg`. "
            "Cost formulas vary per operation — see each function's docstring."
        ),
        "directives": [
            "whest.linalg._svd",
            "whest.linalg._decompositions",
            "whest.linalg._solvers",
            "whest.linalg._properties",
            "whest.linalg._compound",
            "whest.linalg._aliases",
        ],
        "registry_modules": {"numpy.linalg"},
    },
    "api/fft.md": {
        "title": "FFT",
        "description": textwrap.dedent("""\
            Fast Fourier Transform operations from `whest.fft`. All FFT
            transforms are counted. Real-valued transforms (`rfft`) cost roughly
            half of complex transforms.

            ## Cost Summary

            | Operation | Cost Formula |
            |-----------|-------------|
            | `fft`, `ifft` | $5n \\cdot \\lceil\\log_2 n\\rceil$ |
            | `fft2`, `ifft2` | $5N \\cdot \\lceil\\log_2 N\\rceil$ where $N = n_1 \\cdot n_2$ |
            | `fftn`, `ifftn` | $5N \\cdot \\lceil\\log_2 N\\rceil$ where $N = \\prod_i n_i$ |
            | `rfft`, `irfft` | $5(n/2) \\cdot \\lceil\\log_2 n\\rceil$ |
            | `rfft2`, `irfft2` | $5(N/2) \\cdot \\lceil\\log_2 N\\rceil$ where $N = n_1 \\cdot n_2$ |
            | `rfftn`, `irfftn` | $5(N/2) \\cdot \\lceil\\log_2 N\\rceil$ where $N = \\prod_i n_i$ |
            | `hfft`, `ihfft` | $5n_{\\text{out}} \\cdot \\lceil\\log_2 n_{\\text{out}}\\rceil$ |
            | `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` | $0$ (free) |

            ## Examples

            ```python
            import whest as we

            with we.BudgetContext(flop_budget=1_000_000) as budget:
                signal = we.random.randn(1024)    # free
                spectrum = we.fft.fft(signal)     # 5 * 1024 * 10 = 51,200 FLOPs
                freqs = we.fft.fftfreq(1024)      # free
                print(f"FFT cost: {budget.flops_used:,}")  # 51,200
            ```

            ## API Reference
        """),
        "directives": [
            "whest.fft._transforms",
            "whest.fft._free",
        ],
        "registry_modules": {"numpy.fft"},
    },
    "api/random.md": {
        "title": "Random",
        "description": (
            "Random number generation from `whest.random`. "
            "Sampling operations are **counted** — each sample costs "
            "`numel(output)` FLOPs, and shuffle/permutation costs "
            "`n * ceil(log2(n))` FLOPs. Only configuration helpers "
            "(`seed`, `get_state`, `set_state`, `default_rng`) are free (0 FLOPs)."
        ),
        "directives": [
            "whest.random",
        ],
        "registry_modules": {"numpy.random"},
    },
    "api/polynomial.md": {
        "title": "Polynomial",
        "description": textwrap.dedent("""\
            Polynomial operations from `whest`. These wrap NumPy's polynomial
            functions with FLOP counting.

            ## Cost Summary

            | Operation | Cost Formula |
            |-----------|-------------|
            | `polyval` | $m \\cdot \\text{deg}$ (Horner's method, FMA=1) |
            | `polyadd`, `polysub` | $\\max(n_1, n_2)$ |
            | `polymul`, `polydiv` | $n_1 \\cdot n_2$ |
            | `polyfit` | $2m \\cdot (\\text{deg}+1)^2$ |
            | `poly` | $n^2$ |
            | `roots` | $10n^3$ (companion matrix eigendecomposition) |
            | `polyder`, `polyint` | $n$ |

            ## Examples

            ```python
            import whest as we

            with we.BudgetContext(flop_budget=1_000_000) as budget:
                coeffs = we.array([1.0, -3.0, 2.0])  # x^2 - 3x + 2 (free)
                x = we.linspace(0, 5, 100)            # free
                y = we.polyval(coeffs, x)             # 2 * 100 * 2 = 400 FLOPs
                print(f"polyval cost: {budget.flops_used}")  # 400
            ```

            ## API Reference
        """),
        "directives": [
            "whest._polynomial",
        ],
        "registry_modules": {"whest._polynomial"},
    },
    "api/stats.md": {
        "title": "Statistical Distributions",
        "description": textwrap.dedent("""\
            Statistical distributions from `whest.stats`. This submodule
            provides a **subset of scipy.stats** — each distribution is a
            singleton with `.pdf()`, `.cdf()`, and `.ppf()` methods that
            match the scipy API exactly (same signatures, same numerical
            results to within 1e-12).

            Unlike NumPy operations which are direct wrappers, these functions
            reproduce the `scipy.stats` interface so that participants can use
            standard statistical distributions without importing scipy.

            ## Cost Summary

            | Distribution | pdf | cdf | ppf |
            |-------------|-----|-----|-----|
            | `norm` | $10n$ | $20n$ | $40n$ |
            | `uniform` | $3n$ | $3n$ | $3n$ |
            | `expon` | $5n$ | $5n$ | $5n$ |
            | `cauchy` | $6n$ | $5n$ | $5n$ |
            | `logistic` | $8n$ | $5n$ | $5n$ |
            | `laplace` | $5n$ | $5n$ | $5n$ |
            | `lognorm` | $15n$ | $25n$ | $45n$ |
            | `truncnorm` | $30n$ | $30n$ | $50n$ |

            where $n$ = `numel(x)` (or `numel(q)` for ppf).

            ## Examples

            ```python
            import whest as we

            with we.BudgetContext(flop_budget=1_000_000) as budget:
                x = we.linspace(-3, 3, 1000)         # free
                pdf_vals = we.stats.norm.pdf(x)       # 10 * 1000 = 10,000 FLOPs
                cdf_vals = we.stats.norm.cdf(x)       # 20 * 1000 = 20,000 FLOPs
                q = we.array([0.025, 0.975])           # free
                bounds = we.stats.norm.ppf(q)          # 40 * 2 = 80 FLOPs
                print(f"Total: {budget.flops_used:,}")  # 30,080
            ```

            ## Compatibility

            All outputs are verified against `scipy.stats` in the test suite.
            See `tests/test_stats_*.py`.

            ## API Reference
        """),
        "directives": [
            "whest.stats._norm",
            "whest.stats._uniform",
            "whest.stats._expon",
            "whest.stats._cauchy",
            "whest.stats._logistic",
            "whest.stats._laplace",
            "whest.stats._lognorm",
            "whest.stats._truncnorm",
        ],
        "registry_modules": set(),  # stats ops are not in the numpy registry
    },
    "api/window.md": {
        "title": "Window Functions",
        "description": textwrap.dedent("""\
            Window function wrappers from `whest`. These generate window
            arrays used in signal processing (e.g., for windowed FFTs).

            ## Cost Summary

            | Operation | Cost Formula | Notes |
            |-----------|-------------|-------|
            | `bartlett` | $n$ | Linear taper |
            | `hamming` | $n$ | One cosine term |
            | `hanning` | $n$ | One cosine term |
            | `blackman` | $3n$ | Three cosine terms |
            | `kaiser` | $3n$ | Bessel function evaluation |

            ## Examples

            ```python
            import whest as we

            with we.BudgetContext(flop_budget=1_000_000) as budget:
                win = we.hamming(256)   # 256 FLOPs
                win2 = we.kaiser(256)   # 768 FLOPs (3 * 256)
                print(f"Window cost: {budget.flops_used}")  # 1024
            ```

            ## API Reference
        """),
        "directives": [
            "whest._window",
        ],
        "registry_modules": {"whest._window"},
    },
}

# Existing (non-generated) pages and what registry modules they cover
EXISTING_PAGES: dict[str, set[str]] = {
    "api/counted-ops.md": set(),  # covers numpy counted ops + einsum + unwrap
    "api/free-ops.md": set(),  # covers numpy free ops
    "api/symmetric.md": set(),  # covers SymmetricTensor, SymmetryInfo, as_symmetric
}


# ---------------------------------------------------------------------------
# Generate API reference pages
# ---------------------------------------------------------------------------

HEADER = (
    "<!-- Auto-generated by scripts/generate_api_docs.py. Do not edit manually. -->"
)


def generate_api_page(page_path: str, page_info: dict) -> None:
    """Generate a single API reference markdown page."""
    out = DOCS / page_path
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        HEADER,
        f"# {page_info['title']}",
        "",
        page_info["description"],
        "",
    ]

    for directive in page_info["directives"]:
        lines.append(f"::: {directive}")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"  Generated {page_path}")


# ---------------------------------------------------------------------------
# Generate operation audit page
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "free",
    "counted_unary",
    "counted_binary",
    "counted_reduction",
    "counted_custom",
    "blacklisted",
]

CATEGORY_LABELS = {
    "free": ("Free Operations (0 FLOPs)", "0 FLOPs"),
    "counted_unary": ("Counted Unary Operations", r"$\text{numel}(\text{output})$"),
    "counted_binary": ("Counted Binary Operations", r"$\text{numel}(\text{output})$"),
    "counted_reduction": (
        "Counted Reduction Operations",
        r"$\text{numel}(\text{input})$",
    ),
    "counted_custom": ("Counted Custom Operations", "Per-operation formula"),
    "blacklisted": ("Blacklisted Operations", "Not available"),
}

CATEGORY_EMOJI = {
    "free": "\U0001f7e2",  # green circle
    "counted_unary": "\U0001f7e1",  # yellow circle
    "counted_binary": "\U0001f7e1",
    "counted_reduction": "\U0001f7e1",
    "counted_custom": "\U0001f7e0",  # orange circle
    "blacklisted": "\U0001f534",  # red circle
}

# ---------------------------------------------------------------------------
# Per-operation cost formulas (plain text + LaTeX)
# ---------------------------------------------------------------------------

CUSTOM_COSTS: dict[str, tuple[str, str]] = {
    "einsum": (
        "op_factor * product of all index dims",
        r"$\text{op\_factor} \cdot \prod_i d_i$",
    ),
    "einsum_path": ("0 (planning only)", "$0$"),
    "dot": ("m * k * n (FMA=1)", r"$m \cdot k \cdot n$"),
    "matmul": ("m * k * n (FMA=1)", r"$m \cdot k \cdot n$"),
    "inner": ("n", "$n$"),
    "outer": ("m * n", r"$m \cdot n$"),
    "tensordot": ("product of contracted dims * output size", r"$\prod_i d_i$"),
    "vdot": ("n", "$n$"),
    "vecdot": ("n", "$n$"),
    "kron": ("m1*m2 * n1*n2", r"$m_1 m_2 \cdot n_1 n_2$"),
    "clip": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "cross": ("numel(output)", r"$\text{numel}(\text{output})$"),
    "diff": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "ediff1d": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "gradient": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "convolve": ("n * m", r"$n \cdot m$"),
    "correlate": ("n * m", r"$n \cdot m$"),
    "corrcoef": ("n^2 * m", r"$n^2 \cdot m$"),
    "cov": ("n^2 * m", r"$n^2 \cdot m$"),
    "interp": ("n * log(m)", r"$n \cdot \log m$"),
    "trapezoid": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "trapz": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "linalg.svd": ("m * n * k", r"$m \cdot n \cdot k$"),
    "linalg.svdvals": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.cholesky": ("n^3", r"$n^3$"),
    "linalg.qr": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.eig": ("n^3", r"$n^3$"),
    "linalg.eigh": ("n^3", r"$n^3$"),
    "linalg.eigvals": ("n^3", r"$n^3$"),
    "linalg.eigvalsh": ("n^3", r"$n^3$"),
    "linalg.cross": ("delegates to we.cross", r"delegates to `cross`"),
    "linalg.matmul": ("delegates to we.matmul", r"delegates to `matmul`"),
    "linalg.outer": ("delegates to we.outer", r"delegates to `outer`"),
    "linalg.tensordot": ("delegates to we.tensordot", r"delegates to `tensordot`"),
    "linalg.vecdot": ("delegates to we.vecdot", r"delegates to `vecdot`"),
    "linalg.solve": ("n^3", r"$n^3$"),
    "linalg.inv": ("n^3", r"$n^3$"),
    "linalg.lstsq": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.pinv": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.tensorsolve": ("n^3", r"$n^3$"),
    "linalg.tensorinv": ("n^3", r"$n^3$"),
    "linalg.trace": ("n", "$n$"),
    "linalg.det": ("n^3", r"$n^3$"),
    "linalg.slogdet": ("n^3", r"$n^3$"),
    "linalg.norm": ("depends on ord", r"varies"),
    "linalg.vector_norm": ("numel", r"$n$"),
    "linalg.matrix_norm": ("depends on ord", r"varies"),
    "linalg.cond": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.matrix_rank": ("m * n * min(m,n)", r"$m \cdot n \cdot \min(m,n)$"),
    "linalg.multi_dot": ("optimal chain cost", r"optimal chain"),
    "linalg.matrix_power": ("log2(k) * n^3", r"$\lfloor\log_2 k\rfloor \cdot n^3$"),
    "fft.fft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "fft.ifft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "fft.fft2": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.ifft2": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.fftn": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.ifftn": ("5N * ceil(log2(N))", r"$5N \cdot \lceil\log_2 N\rceil$"),
    "fft.rfft": ("5(n/2) * ceil(log2(n))", r"$5(n/2) \cdot \lceil\log_2 n\rceil$"),
    "fft.irfft": ("5(n/2) * ceil(log2(n))", r"$5(n/2) \cdot \lceil\log_2 n\rceil$"),
    "fft.rfft2": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.irfft2": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.rfftn": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.irfftn": ("5(N/2) * ceil(log2(N))", r"$5(N/2) \cdot \lceil\log_2 N\rceil$"),
    "fft.hfft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "fft.ihfft": ("5n * ceil(log2(n))", r"$5n \cdot \lceil\log_2 n\rceil$"),
    "polyval": ("m * deg (FMA=1)", r"$m \cdot \text{deg}$"),
    "polyadd": ("max(n1, n2)", r"$\max(n_1, n_2)$"),
    "polysub": ("max(n1, n2)", r"$\max(n_1, n_2)$"),
    "polyder": ("n", "$n$"),
    "polyint": ("n", "$n$"),
    "polymul": ("n1 * n2", r"$n_1 \cdot n_2$"),
    "polydiv": ("n1 * n2", r"$n_1 \cdot n_2$"),
    "polyfit": ("2m * (deg+1)^2", r"$2m \cdot (\text{deg}+1)^2$"),
    "poly": ("n^2", r"$n^2$"),
    "roots": ("10n^3", r"$10n^3$"),
    "bartlett": ("n", "$n$"),
    "blackman": ("3n", "$3n$"),
    "hamming": ("n", "$n$"),
    "hanning": ("n", "$n$"),
    "kaiser": ("3n", "$3n$"),
    "unwrap": ("numel(input)", r"$\text{numel}(\text{input})$"),
    # stats distributions (not in numpy registry — these are scipy-compatible)
    "stats.norm.pdf": ("10n", r"$10n$"),
    "stats.norm.cdf": ("20n", r"$20n$"),
    "stats.norm.ppf": ("40n", r"$40n$"),
    "stats.uniform.pdf": ("3n", r"$3n$"),
    "stats.uniform.cdf": ("3n", r"$3n$"),
    "stats.uniform.ppf": ("3n", r"$3n$"),
    "stats.expon.pdf": ("5n", r"$5n$"),
    "stats.expon.cdf": ("5n", r"$5n$"),
    "stats.expon.ppf": ("5n", r"$5n$"),
    "stats.cauchy.pdf": ("6n", r"$6n$"),
    "stats.cauchy.cdf": ("5n", r"$5n$"),
    "stats.cauchy.ppf": ("5n", r"$5n$"),
    "stats.logistic.pdf": ("8n", r"$8n$"),
    "stats.logistic.cdf": ("5n", r"$5n$"),
    "stats.logistic.ppf": ("5n", r"$5n$"),
    "stats.laplace.pdf": ("5n", r"$5n$"),
    "stats.laplace.cdf": ("5n", r"$5n$"),
    "stats.laplace.ppf": ("5n", r"$5n$"),
    "stats.lognorm.pdf": ("15n", r"$15n$"),
    "stats.lognorm.cdf": ("25n", r"$25n$"),
    "stats.lognorm.ppf": ("45n", r"$45n$"),
    "stats.truncnorm.pdf": ("30n", r"$30n$"),
    "stats.truncnorm.cdf": ("30n", r"$30n$"),
    "stats.truncnorm.ppf": ("50n", r"$50n$"),
}

CATEGORY_COST_LATEX: dict[str, tuple[str, str]] = {
    "free": ("0", "$0$"),
    "counted_unary": ("numel(output)", r"$\text{numel}(\text{output})$"),
    "counted_binary": ("numel(output)", r"$\text{numel}(\text{output})$"),
    "counted_reduction": ("numel(input)", r"$\text{numel}(\text{input})$"),
    "counted_custom": ("per-operation", "varies"),
    "blacklisted": ("N/A", "N/A"),
}


# ---------------------------------------------------------------------------
# Operation doc model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OperationDocRecord:
    name: str
    canonical_name: str
    slug: str
    href: str
    area: str
    whest_ref: str
    numpy_ref: str
    category: str
    display_type: str
    cost_formula: str
    cost_formula_latex: str
    weight: float
    notes: str
    aliases: list[str]
    signature: str
    summary: str = ""
    provenance_label: str = ""
    provenance_url: str = ""
    whest_source_url: str = ""
    upstream_source_url: str = ""
    parameters: list["DocField"] | None = None
    returns: list["DocField"] | None = None
    see_also: list["DocLink"] | None = None
    notes_sections: list[str] | None = None
    example: "DocExample | None" = None
    previous: "OperationNavLink | None" = None
    next: "OperationNavLink | None" = None
    api_docs_html: str = ""
    whest_examples_html: str = ""


@dataclass(slots=True)
class DocField:
    name: str
    type: str
    description: list[str]


@dataclass(slots=True)
class DocLink:
    label: str
    target: str
    description: str = ""
    href: str | None = None
    external_url: str | None = None


@dataclass(slots=True)
class DocExample:
    code: str
    output: str = ""
    source: str = "derived"


@dataclass(slots=True)
class OperationNavLink:
    name: str
    href: str
    label: str


@dataclass(slots=True)
class ParsedDoc:
    summary: str
    parameters: list[DocField]
    returns: list[DocField]
    see_also: list[DocLink]
    notes: list[str]
    examples: list[DocExample]


# ---------------------------------------------------------------------------
# Helper functions for op references and cost lookup
# ---------------------------------------------------------------------------


def normalize_area(module: str) -> str:
    """Map a registry module name to its doc area."""
    if module in {"numpy", "whest"}:
        return "core"
    if module.startswith("numpy."):
        return module.removeprefix("numpy.")
    if module == "whest.stats":
        return "stats"
    return "core"


def slug_for_operation(name: str) -> str:
    """Return a URL-safe slug for an operation name."""
    return name.replace(".", "-")


def display_type_for_category(category: str) -> str:
    """Return the UI display type for a registry category."""
    if category == "blacklisted":
        return "blocked"
    if category == "free":
        return "free"
    if category.startswith("counted_") and category != "counted_custom":
        return "counted"
    return "custom"


def whest_ref(name: str, module: str) -> str:
    """Derive the whest call reference from an op name and registry module."""
    if module == "numpy.linalg":
        return f"`we.linalg.{name.removeprefix('linalg.')}`"
    if module == "numpy.fft":
        return f"`we.fft.{name.removeprefix('fft.')}`"
    if module == "numpy.random":
        return f"`we.random.{name.removeprefix('random.')}`"
    return f"`we.{name}`"


def numpy_ref(name: str, module: str) -> str:
    """Derive the NumPy call reference from an op name and registry module."""
    if module == "numpy.linalg":
        return f"`np.linalg.{name.removeprefix('linalg.')}`"
    if module == "numpy.fft":
        return f"`np.fft.{name.removeprefix('fft.')}`"
    if module == "numpy.random":
        return f"`np.random.{name.removeprefix('random.')}`"
    return f"`np.{name}`"


def cost_for_op(name: str, category: str) -> tuple[str, str]:
    """Return (plain_text, latex) cost formula for an operation."""
    if name in CUSTOM_COSTS:
        return CUSTOM_COSTS[name]
    return CATEGORY_COST_LATEX.get(category, ("unknown", "unknown"))


NUMPY_REF_PATTERN = re.compile(r"\b(?:np|numpy)\.(?=[A-Za-z_])")


def rewrite_api_refs(text: str) -> str:
    """Rewrite NumPy API references to their whest equivalents."""
    return NUMPY_REF_PATTERN.sub("we.", text)


def _split_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
            continue
        if current:
            paragraphs.append(" ".join(part.strip() for part in current).strip())
            current = []
    if current:
        paragraphs.append(" ".join(part.strip() for part in current).strip())
    return paragraphs


def derive_example_from_upstream(example_block: str) -> DocExample:
    """Derive a single whest example from a doctest-style upstream snippet."""
    code_lines: list[str] = []
    output_lines: list[str] = []
    seen_output = False

    for raw_line in textwrap.dedent(example_block).strip("\n").splitlines():
        stripped = raw_line.lstrip()
        if stripped.startswith(">>>"):
            seen_output = False
            code = stripped[3:]
            if code.startswith(" "):
                code = code[1:]
            code_lines.append(code.rstrip())
            continue
        if stripped.startswith("..."):
            code = stripped[3:]
            if code.startswith(" "):
                code = code[1:]
            code_lines.append(code.rstrip())
            continue
        if stripped == "":
            if seen_output:
                output_lines.append("")
            elif code_lines:
                code_lines.append("")
            continue

        seen_output = True
        output_lines.append(raw_line.rstrip())

    code = rewrite_api_refs("\n".join(code_lines).strip())
    code = code.replace("import numpy as np", "import whest as we")
    output = "\n".join(output_lines).strip()
    return DocExample(code=code, output=output)


def parse_numpy_docstring(raw_doc: str) -> ParsedDoc:
    """Parse a NumPy-style docstring into a structured internal model."""
    try:
        from numpydoc.docscrape import NumpyDocString
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("numpydoc is required to parse upstream docstrings") from exc

    doc = NumpyDocString(textwrap.dedent(raw_doc).strip("\n"))

    summary = " ".join(part.strip() for part in doc["Summary"]).strip()

    parameters = [
        DocField(name=name, type=type_, description=[line.rstrip() for line in desc])
        for name, type_, desc in doc["Parameters"]
    ]
    returns = [
        DocField(name=name, type=type_, description=[line.rstrip() for line in desc])
        for name, type_, desc in doc["Returns"]
    ]
    see_also: list[DocLink] = []
    for entry in doc["See Also"]:
        if not entry:
            continue
        targets, desc = entry
        description = " ".join(desc).strip() if desc else ""
        if isinstance(targets, list):
            for target, _ in targets:
                see_also.append(
                    DocLink(label=target, target=target, description=description)
                )
            continue
        see_also.append(DocLink(label=targets, target=targets, description=description))
    notes = _split_paragraphs(list(doc["Notes"]))

    example_text = "\n".join(line.rstrip() for line in doc["Examples"]).strip()
    examples = [DocExample(code=example_text)] if example_text else []

    return ParsedDoc(
        summary=summary,
        parameters=parameters,
        returns=returns,
        see_also=see_also,
        notes=notes,
        examples=examples,
    )


def docs_url_for_operation(name: str, module: str) -> str:
    """Return the upstream reference docs URL for a supported operation."""
    if module == "numpy.linalg":
        return (
            "https://numpy.org/doc/stable/reference/generated/"
            f"numpy.linalg.{name.removeprefix('linalg.')}.html"
        )
    if module == "numpy.fft":
        return (
            "https://numpy.org/doc/stable/reference/generated/"
            f"numpy.fft.{name.removeprefix('fft.')}.html"
        )
    if module == "numpy.random":
        return (
            "https://numpy.org/doc/stable/reference/generated/"
            f"numpy.random.{name.removeprefix('random.')}.html"
        )
    if module == "whest.stats":
        dist_name = name.split(".")[1]
        return (
            "https://docs.scipy.org/doc/scipy/reference/generated/"
            f"scipy.stats.{dist_name}.html"
        )
    return f"https://numpy.org/doc/stable/reference/generated/numpy.{name}.html"


def _repo_source_url(
    obj: object, *, repo_blob_root: str, package_root: Path | None = None
) -> str:
    """Build a GitHub source URL for an imported object when inspect can locate it."""
    target = inspect.unwrap(obj)
    try:
        source_path = Path(
            inspect.getsourcefile(target) or inspect.getfile(target)
        ).resolve()
        _, start_line = inspect.getsourcelines(target)
    except (OSError, TypeError):  # pragma: no cover - inspect support varies by object
        return ""

    if package_root is None:
        try:
            relative = source_path.relative_to(ROOT)
        except ValueError:
            return ""
    else:
        try:
            relative = source_path.relative_to(package_root.parent)
        except ValueError:
            return ""

    return f"{repo_blob_root}/{relative.as_posix()}#L{start_line}"


def _resolve_attr(root: object, dotted_name: str) -> object:
    current = root
    for part in dotted_name.split("."):
        current = getattr(current, part)
    return current


def resolve_live_objects(name: str, module: str) -> tuple[object, object | None]:
    """Resolve the live whest object and its upstream NumPy/SciPy counterpart."""
    import numpy as np

    import whest as we

    if module == "numpy.linalg":
        short_name = name.removeprefix("linalg.")
        return getattr(we.linalg, short_name), getattr(np.linalg, short_name, None)
    if module == "numpy.fft":
        short_name = name.removeprefix("fft.")
        return getattr(we.fft, short_name), getattr(np.fft, short_name, None)
    if module == "numpy.random":
        short_name = name.removeprefix("random.")
        return getattr(we.random, short_name), getattr(np.random, short_name, None)
    if module == "whest.stats":
        try:
            from scipy import stats as scipy_stats
        except Exception:  # pragma: no cover - scipy availability is environment specific
            scipy_stats = None
        return _resolve_attr(we, name), (
            _resolve_attr(scipy_stats, name.removeprefix("stats."))
            if scipy_stats is not None
            else None
        )
    return getattr(we, name), getattr(np, name, None)


def _rewrite_doc_field(field: DocField) -> DocField:
    return DocField(
        name=field.name,
        type=rewrite_api_refs(field.type),
        description=[rewrite_api_refs(line) for line in field.description],
    )


def _rewrite_doc_link(link: DocLink) -> DocLink:
    rewritten_target = rewrite_api_refs(link.target)
    return DocLink(
        label=rewrite_api_refs(link.label),
        target=rewritten_target.removeprefix("we."),
        description=rewrite_api_refs(link.description),
        href="",
        external_url="",
    )


def build_structured_doc(
    name: str, module: str, owned_example_html: str = ""
) -> tuple[str, ParsedDoc, DocExample | None, str, str]:
    """Resolve live objects and build the structured doc model for one op."""
    import numpy as np

    whest_obj, upstream_obj = resolve_live_objects(name, module)
    raw_doc = inspect.getdoc(upstream_obj) or inspect.getdoc(whest_obj) or ""
    parsed = parse_numpy_docstring(raw_doc)

    parsed.summary = rewrite_api_refs(parsed.summary)
    parsed.parameters = [_rewrite_doc_field(field) for field in parsed.parameters]
    parsed.returns = [_rewrite_doc_field(field) for field in parsed.returns]
    parsed.see_also = [_rewrite_doc_link(link) for link in parsed.see_also]
    parsed.notes = [rewrite_api_refs(note) for note in parsed.notes]

    example: DocExample | None = None
    if owned_example_html:
        example = None
    elif parsed.examples:
        example = derive_example_from_upstream(parsed.examples[0].code)

    try:
        signature = f"{whest_ref(name, module).strip('`')}{inspect.signature(whest_obj)}"
    except (TypeError, ValueError):
        signature = f"{whest_ref(name, module).strip('`')}(...)"

    whest_source_url = _repo_source_url(
        whest_obj,
        repo_blob_root="https://github.com/AIcrowd/whest/blob/main",
    )

    upstream_source_url = ""
    if upstream_obj is not None:
        numpy_root = Path(np.__file__).resolve().parent
        upstream_source_url = _repo_source_url(
            upstream_obj,
            repo_blob_root=f"https://github.com/numpy/numpy/blob/v{np.__version__}",
            package_root=numpy_root,
        )

    return signature, parsed, example, whest_source_url, upstream_source_url


ALIAS_NOTE_PATTERN = re.compile(r"alias for ([\w./]+)", re.IGNORECASE)
ALIAS_REASON_PATTERN = re.compile(r"Alias of ([\w./]+)", re.IGNORECASE)
EXAMPLE_FENCE_PATTERN = re.compile(
    r"```(?P<lang>[A-Za-z0-9_-]+)?\n(?P<body>[\s\S]*?)```", re.MULTILINE
)


def choose_alias_target(raw_target: str, registry: dict[str, dict]) -> str | None:
    """Return the first canonical registry target encoded in alias metadata."""
    cleaned = raw_target.strip().rstrip(").,;:")
    candidates = [part.strip().rstrip(").,;:") for part in cleaned.split("/")]
    for candidate in candidates:
        if candidate in registry:
            return candidate
    return None


def load_alias_map(registry: dict[str, dict]) -> dict[str, str]:
    """Derive alias -> canonical mappings from registry notes and weights metadata."""
    alias_map: dict[str, str] = {}

    for name, info in registry.items():
        match = ALIAS_NOTE_PATTERN.search(info.get("notes", ""))
        if not match:
            continue
        target = choose_alias_target(match.group(1), registry)
        if target and target != name:
            alias_map[name] = target

    if WEIGHTS_CSV_PATH.exists():
        with WEIGHTS_CSV_PATH.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Status") != "alias":
                    continue
                name = row.get("Operation", "").strip()
                if not name or name in alias_map or name not in registry:
                    continue
                reason = row.get("Exclusion Reason", "")
                match = ALIAS_REASON_PATTERN.search(reason)
                if not match:
                    continue
                target = choose_alias_target(match.group(1), registry)
                if not target or target == name:
                    continue
                # Prefer registry-authored canonical targets when metadata disagrees.
                if alias_map.get(target) == name:
                    continue
                alias_map[name] = target

    return alias_map


def resolve_canonical_name(name: str, alias_map: dict[str, str]) -> str:
    """Resolve an alias chain to its canonical target."""
    current = name
    seen = {current}
    while current in alias_map and alias_map[current] not in seen:
        current = alias_map[current]
        seen.add(current)
    return current


def load_operation_weights() -> dict[str, float]:
    """Load per-operation weights for docs manifests."""
    if not WEIGHTS_PATH.exists():
        return {}
    raw = json.loads(WEIGHTS_PATH.read_text())
    return raw.get("weights", {})


def example_file_for(name: str, example_root: Path) -> Path:
    """Return the owned-example path for a canonical operation name."""
    return example_root / f"{name}.mdx"


def build_example_coverage(
    records: list[OperationDocRecord], example_root: Path
) -> dict[str, dict]:
    """Compute example coverage for canonical operation pages."""
    coverage: dict[str, dict] = {}
    for record in records:
        name = record.name
        path = example_file_for(name, example_root)
        if not path.exists():
            if record.example is not None:
                coverage[name] = {
                    "has_whest_examples": False,
                    "has_inherited_examples": True,
                    "example_count": 1,
                    "example_sources": [record.provenance_url] if record.provenance_url else [],
                    "coverage_status": "derived",
                }
                continue
            coverage[name] = {
                "has_whest_examples": False,
                "has_inherited_examples": False,
                "example_count": 0,
                "example_sources": [],
                "coverage_status": "missing",
            }
            continue

        source = path.read_text()
        example_count = len(EXAMPLE_FENCE_PATTERN.findall(source)) or 1
        coverage[name] = {
            "has_whest_examples": True,
            "has_inherited_examples": False,
            "example_count": example_count,
            "example_sources": [str(path)],
            "coverage_status": "owned",
        }

    return coverage


def render_example_markdown_html(source: str) -> str:
    """Render simple MDX example snippets into HTML for the op-doc manifest."""
    rendered: list[str] = []
    cursor = 0
    for match in EXAMPLE_FENCE_PATTERN.finditer(source):
        prose = source[cursor : match.start()].strip()
        if prose:
            rendered.append(f"<p>{html.escape(prose)}</p>")

        language = (match.group("lang") or "").strip()
        class_attr = f' class="language-{html.escape(language)}"' if language else ""
        body = html.escape(match.group("body").rstrip("\n"))
        rendered.append(f"<pre><code{class_attr}>{body}</code></pre>")
        cursor = match.end()

    trailing = source[cursor:].strip()
    if trailing:
        rendered.append(f"<p>{html.escape(trailing)}</p>")

    return "\n".join(rendered)


def load_whest_example_html(name: str, example_root: Path) -> str:
    """Load and render owned whest examples for a canonical operation."""
    path = example_file_for(name, example_root)
    if not path.exists():
        return ""

    return render_example_markdown_html(path.read_text())


def build_alias_groups(
    registry: dict[str, dict], alias_map: dict[str, str]
) -> dict[str, list[str]]:
    """Group aliases by resolved canonical name."""
    alias_groups: dict[str, list[str]] = {}
    for alias in sorted(alias_map):
        canonical = resolve_canonical_name(alias, alias_map)
        if canonical == alias or canonical not in registry:
            continue
        alias_groups.setdefault(canonical, []).append(alias)
    return alias_groups


def resolve_operation_weight(
    name: str,
    registry: dict[str, dict],
    weights: dict[str, float],
    alias_map: dict[str, str],
    alias_groups: dict[str, list[str]] | None = None,
) -> float:
    """Resolve a stable weight for canonical and alias rows."""
    canonical = resolve_canonical_name(name, alias_map)
    if canonical in weights:
        return weights[canonical]

    aliases = alias_groups.get(canonical, []) if alias_groups is not None else []
    for alias in aliases:
        if alias in weights:
            return weights[alias]

    if canonical not in registry and name in weights:
        return weights[name]

    return 1.0


def build_operation_doc_records(registry: dict[str, dict]) -> list[OperationDocRecord]:
    """Build canonical operation doc records for supported operations."""
    alias_map = load_alias_map(registry)
    alias_groups = build_alias_groups(registry, alias_map)
    weights = load_operation_weights()
    records: list[OperationDocRecord] = []
    for name, info in sorted(registry.items()):
        if info["category"] == "blacklisted":
            continue
        if resolve_canonical_name(name, alias_map) != name:
            continue

        aliases = sorted(alias_groups.get(name, []))
        weight = resolve_operation_weight(
            name=name,
            registry=registry,
            weights=weights,
            alias_map=alias_map,
            alias_groups=alias_groups,
        )

        module = info["module"]
        cost_plain, cost_latex = cost_for_op(name, info["category"])
        owned_example_html = load_whest_example_html(name, API_EXAMPLES_DIR)
        signature, parsed_doc, derived_example, whest_source_url, upstream_source_url = (
            build_structured_doc(name, module, owned_example_html)
        )
        records.append(
            OperationDocRecord(
                name=name,
                canonical_name=name,
                slug=slug_for_operation(name),
                href=f"/docs/api/ops/{slug_for_operation(name)}",
                area=normalize_area(module),
                whest_ref=whest_ref(name, module),
                numpy_ref=numpy_ref(name, module),
                category=info["category"],
                display_type=display_type_for_category(info["category"]),
                cost_formula=cost_plain,
                cost_formula_latex=cost_latex,
                weight=weight,
                notes=info.get("notes", ""),
                aliases=aliases,
                signature=signature,
                summary=parsed_doc.summary,
                provenance_label="Adapted from NumPy docs",
                provenance_url=docs_url_for_operation(name, module),
                whest_source_url=whest_source_url,
                upstream_source_url=upstream_source_url,
                parameters=parsed_doc.parameters,
                returns=parsed_doc.returns,
                see_also=parsed_doc.see_also,
                notes_sections=parsed_doc.notes,
                example=derived_example,
                api_docs_html="",
                whest_examples_html=owned_example_html,
            )
        )

    for index, record in enumerate(records):
        if index > 0:
            previous = records[index - 1]
            record.previous = OperationNavLink(
                name=previous.name,
                href=previous.href,
                label=previous.whest_ref.strip("`"),
            )
        if index + 1 < len(records):
            nxt = records[index + 1]
            record.next = OperationNavLink(
                name=nxt.name,
                href=nxt.href,
                label=nxt.whest_ref.strip("`"),
            )

    return records


def render_operation_stub(op: OperationDocRecord) -> str:
    """Render a generated standalone MDX page stub for one canonical operation."""
    return (
        f'---\n'
        f'title: "{op.whest_ref.strip("`")}"\n'
        f'---\n\n'
        f'<OperationDocPage name="{op.name}" />\n'
    )


def write_json(path: Path, payload: object) -> None:
    """Write a deterministic JSON artifact with trailing newline."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_operation_doc_artifacts(
    records: list[OperationDocRecord], website_root: Path
) -> None:
    """Emit standalone MDX stubs plus generated operation manifests."""
    op_docs_dir = website_root / "content" / "docs" / "api" / "ops"
    generated_dir = website_root / ".generated"
    op_docs_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    expected_pages = {record.slug for record in records}
    for existing_page in op_docs_dir.glob("*.mdx"):
        if existing_page.stem not in expected_pages:
            existing_page.unlink()

    docs_manifest: dict[str, dict[str, object]] = {}
    refs_manifest: dict[str, dict[str, str]] = {}

    for record in sorted(records, key=lambda op: op.name):
        stub_path = op_docs_dir / f"{record.slug}.mdx"
        stub_path.write_text(render_operation_stub(record))
        docs_manifest[record.name] = asdict(record)
        refs_manifest[record.name] = {
            "label": record.whest_ref,
            "href": record.href,
            "canonical_name": record.canonical_name,
        }
        for alias in record.aliases:
            refs_manifest[alias] = {
                "label": record.whest_ref,
                "href": record.href,
                "canonical_name": record.canonical_name,
            }

    write_json(generated_dir / "op-docs.json", docs_manifest)
    write_json(generated_dir / "op-refs.json", refs_manifest)
    print(f"  Generated {len(records)} standalone operation stubs")
    print(f"  Generated {generated_dir / 'op-docs.json'}")
    print(f"  Generated {generated_dir / 'op-refs.json'}")


def write_example_coverage_artifact(
    coverage: dict[str, dict], website_root: Path
) -> None:
    """Write the owned-example coverage manifest."""
    generated_dir = website_root / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    write_json(generated_dir / "api-example-coverage.json", coverage)
    print(f"  Generated {generated_dir / 'api-example-coverage.json'}")


def assert_supported_docs_env() -> None:
    """Fail fast unless the docs generator is running on supported NumPy."""
    try:
        import numpy as np
        from numpy.lib import NumpyVersion
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise RuntimeError("generate_api_docs.py requires NumPy >= 2.0,<2.3") from exc

    version = NumpyVersion(np.__version__)
    if not (NumpyVersion("2.0.0") <= version < NumpyVersion("2.3.0")):
        raise RuntimeError(
            f"generate_api_docs.py requires NumPy >= 2.0,<2.3; found {np.__version__}"
        )


def generate_audit_page(registry: dict[str, dict]) -> None:
    """Generate docs/reference/operation-audit.md — 7-column searchable table."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[str]] = {cat: [] for cat in CATEGORY_ORDER}
    for name, info in sorted(registry.items()):
        cat = info["category"]
        if cat in by_cat:
            by_cat[cat].append(name)

    lines = [
        HEADER,
        "# Operation Audit",
        "",
        "Complete inventory of every NumPy operation and its whest status.",
        "Generated from the operation registry (`_registry.py`).",
        "",
        "## Summary",
        "",
        "| Category | Count | Cost |",
        "|----------|-------|------|",
    ]
    total = 0
    for cat in CATEGORY_ORDER:
        label, cost = CATEGORY_LABELS[cat]
        count = len(by_cat[cat])
        total += count
        lines.append(f"| {cat} | {count} | {cost} |")
    lines.append(f"| **Total** | **{total}** | |")
    lines.append("")

    lines.append("## All Operations")
    lines.append("")
    lines.append("| Operation | whest | NumPy | Category | Cost | Status | Notes |")
    lines.append("|-----------|-----------|-------|----------|------|--------|-------|")
    for name, info in sorted(registry.items()):
        cat = info["category"]
        mod = info["module"]
        me_ref = whest_ref(name, mod)
        np_ref = numpy_ref(name, mod)
        _, latex = cost_for_op(name, cat)
        emoji = CATEGORY_EMOJI.get(cat, "")
        status = "blocked" if cat == "blacklisted" else "supported"
        status_display = f"{emoji} {status}"
        notes = info.get("notes", "")
        if cat == "blacklisted":
            me_ref = "\u2014"
        lines.append(
            f"| `{name}` | {me_ref} | {np_ref} | {cat} | {latex} | {status_display} | {notes} |"
        )
    lines.append("")
    out = REF_DIR / "operation-audit.md"
    out.write_text("\n".join(lines))
    print(f"  Generated reference/operation-audit.md ({total} operations)")


def generate_ops_json(registry: dict[str, dict]) -> None:
    """Generate website/public/ops.json — machine-readable operation manifest."""
    weights = load_operation_weights()
    alias_map = load_alias_map(registry)
    alias_groups = build_alias_groups(registry, alias_map)

    ops = []
    for name, info in sorted(registry.items()):
        cat = info["category"]
        mod = info["module"]
        plain, latex = cost_for_op(name, cat)
        ops.append(
            {
                "name": name,
                "module": mod,
                "whest_ref": whest_ref(name, mod).strip("`"),
                "numpy_ref": numpy_ref(name, mod).strip("`"),
                "category": cat,
                "cost_formula": plain,
                "cost_formula_latex": latex,
                "free": cat == "free",
                "blocked": cat == "blacklisted",
                "status": "blocked" if cat == "blacklisted" else "supported",
                "notes": info.get("notes", ""),
                "weight": resolve_operation_weight(
                    name=name,
                    registry=registry,
                    weights=weights,
                    alias_map=alias_map,
                    alias_groups=alias_groups,
                ),
            }
        )
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    out = PUBLIC_DIR / "ops.json"
    out.write_text(json.dumps({"operations": ops, "total": len(ops)}, indent=2))
    print(f"  Generated ops.json ({len(ops)} operations)")


def generate_cheat_sheet(registry: dict[str, dict]) -> None:
    """Generate docs/reference/cheat-sheet.md — agent-optimized cost reference."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[str]] = {cat: [] for cat in CATEGORY_ORDER}
    for name, info in sorted(registry.items()):
        cat = info["category"]
        if cat in by_cat:
            by_cat[cat].append(name)

    lines = [
        HEADER,
        "# FLOP Cost Cheat Sheet",
        "",
        "> **whest is NOT NumPy.** All computation requires a `BudgetContext`.",
        "> Some operations cost FLOPs, some are free, and 32 are blocked entirely.",
        "",
        "## Key Constraints",
        "",
        "- All counted operations require an active `BudgetContext`",
        "- Budget is checked *before* execution — exceeding it raises `BudgetExhaustedError`",
        "- 32 operations are blocked (I/O, config, state functions)",
        "- `sort`, `argsort`, `trace`, and random sampling are **counted** with analytical FLOP costs",
        "",
        "## Cost by Category",
        "",
        "| Category | Count | Cost Formula |",
        "|----------|-------|-------------|",
    ]
    for cat in CATEGORY_ORDER:
        _, latex = CATEGORY_COST_LATEX[cat]
        label = cat.removeprefix("counted_").replace("_", " ").title()
        if cat == "free":
            label = "Free"
        elif cat == "blacklisted":
            label = "Blocked"
        lines.append(f"| {label} | {len(by_cat[cat])} | {latex} |")
    lines.append("")

    custom_ops = by_cat.get("counted_custom", [])
    if custom_ops:
        lines.append("## Custom Operation Costs")
        lines.append("")
        lines.append("| Operation | Cost Formula | Notes |")
        lines.append("|-----------|-------------|-------|")
        for name in custom_ops:
            _, latex = cost_for_op(name, "counted_custom")
            notes = registry[name].get("notes", "")
            lines.append(f"| `{name}` | {latex} | {notes} |")
        lines.append("")

    lines.append("## Free Operations (complete list)")
    lines.append("")
    free_ops = [f"`{op}`" for op in by_cat["free"]]
    for i in range(0, len(free_ops), 8):
        lines.append(", ".join(free_ops[i : i + 8]))
    lines.append("")

    lines.append("## Blocked Operations (complete list)")
    lines.append("")
    blocked_ops = [f"`{op}`" for op in by_cat["blacklisted"]]
    for i in range(0, len(blocked_ops), 8):
        lines.append(", ".join(blocked_ops[i : i + 8]))
    lines.append("")

    out = REF_DIR / "cheat-sheet.md"
    out.write_text("\n".join(lines))
    print("  Generated reference/cheat-sheet.md")


# ---------------------------------------------------------------------------
# Update counted-ops.md to include polynomial, window, unwrap
# ---------------------------------------------------------------------------


def update_counted_ops_page(registry: dict[str, dict]) -> None:
    """Ensure counted-ops.md includes directives for polynomial, window, unwrap."""
    page = API_DIR / "counted-ops.md"
    content = page.read_text()

    additions = []
    for module_directive, label in [
        ("whest._polynomial", "polynomial"),
        ("whest._window", "window"),
        ("whest._unwrap", "unwrap"),
    ]:
        if f"::: {module_directive}" not in content:
            additions.append(f"\n::: {module_directive}")

    if additions:
        new_content = content.rstrip() + "\n" + "\n".join(additions) + "\n"
        page.write_text(new_content)
        print(
            f"  Updated api/counted-ops.md with: {', '.join(a.strip().removeprefix('::: ') for a in additions)}"
        )
    else:
        print("  api/counted-ops.md already up to date")


# ---------------------------------------------------------------------------
# Verify coverage
# ---------------------------------------------------------------------------


def get_module_public_names(module_path: str) -> set[str]:
    """Import a module and return its public callable names."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return set()

    if hasattr(mod, "__all__"):
        return {n for n in mod.__all__ if callable(getattr(mod, n, None))}

    return {
        n
        for n, obj in inspect.getmembers(mod)
        if not n.startswith("_") and callable(obj)
    }


def extract_directives_from_file(path: Path) -> list[str]:
    """Extract ::: directives from a markdown file."""
    directives = []
    if not path.exists():
        return directives
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("::: "):
            directives.append(stripped[4:].strip())
    return directives


def verify_coverage(registry: dict[str, dict]) -> bool:
    """Verify the generated API reference data matches the registry."""
    if not API_INDEX_PATH.exists():
        print(f"\nAPI index NOT FOUND at {API_INDEX_PATH}")
        return False

    api_index = API_INDEX_PATH.read_text()
    if "<ApiReference />" not in api_index:
        print(f"\nAPI index at {API_INDEX_PATH} no longer renders <ApiReference />")
        return False

    print("API index renders the interactive ApiReference component.")

    ops_json_path = PUBLIC_DIR / "ops.json"
    if not ops_json_path.exists():
        print(f"\nops.json NOT FOUND at {ops_json_path}")
        return False

    ops_data = json.loads(ops_json_path.read_text())
    if "operations" not in ops_data:
        print(f"\nops.json missing 'operations' key at {ops_json_path}")
        return False

    ops_names = {op["name"] for op in ops_data["operations"]}
    registry_names = set(registry.keys())
    missing_from_json = registry_names - ops_names
    if missing_from_json:
        print(f"\nMISSING from ops.json ({len(missing_from_json)} ops):")
        for name in sorted(missing_from_json):
            print(f"  {name}")
        return False

    print(f"ops.json covers all {len(ops_names)} operations.")

    op_docs_path = GENERATED_DIR / "op-docs.json"
    if not op_docs_path.exists():
        print(f"\nop-docs.json NOT FOUND at {op_docs_path}")
        return False

    op_refs_path = GENERATED_DIR / "op-refs.json"
    if not op_refs_path.exists():
        print(f"\nop-refs.json NOT FOUND at {op_refs_path}")
        return False

    example_coverage_path = GENERATED_DIR / "api-example-coverage.json"
    if not example_coverage_path.exists():
        print(f"\napi-example-coverage.json NOT FOUND at {example_coverage_path}")
        return False

    sample_op_page = OP_DOCS_DIR / "absolute.mdx"
    if not sample_op_page.exists():
        print(f"\nGenerated operation page NOT FOUND at {sample_op_page}")
        return False

    op_docs = json.loads(op_docs_path.read_text())
    if "absolute" not in op_docs:
        print(f"\nop-docs.json missing canonical entry for 'absolute' at {op_docs_path}")
        return False

    op_refs = json.loads(op_refs_path.read_text())
    abs_ref = op_refs.get("abs")
    if (
        not isinstance(abs_ref, dict)
        or abs_ref.get("label") != "`we.absolute`"
        or abs_ref.get("href") != "/docs/api/ops/absolute"
        or abs_ref.get("canonical_name") != "absolute"
    ):
        print(
            "\nop-refs.json missing structured alias entry for "
            "'abs' -> '/docs/api/ops/absolute'"
        )
        return False

    print("Generated operation doc manifests, stub pages, and example coverage are present.")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate API reference data from the whest registry"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify coverage only (no generation)"
    )
    args = parser.parse_args()

    assert_supported_docs_env()
    registry = load_registry()

    if args.verify:
        ok = verify_coverage(registry)
        sys.exit(0 if ok else 1)

    print("Generating API reference data...")
    generate_ops_json(registry)
    records = build_operation_doc_records(registry)
    write_operation_doc_artifacts(records, WEBSITE)
    example_coverage = build_example_coverage(records, API_EXAMPLES_DIR)
    write_example_coverage_artifact(example_coverage, WEBSITE)

    print("\nDone. Run with --verify to check coverage.")


if __name__ == "__main__":
    main()
