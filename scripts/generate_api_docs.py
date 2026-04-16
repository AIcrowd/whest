#!/usr/bin/env python
"""generate_api_docs.py — Generate API reference data and verify coverage.

Usage
-----
    python scripts/generate_api_docs.py              # generate API data
    python scripts/generate_api_docs.py --verify     # verify API data only
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
import textwrap
import warnings
from pathlib import Path

from numpydoc.docscrape import NumpyDocString

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "website"
PUBLIC_DIR = WEBSITE / "public"
API_DATA_DIR = PUBLIC_DIR / "api-data" / "ops"
GENERATED_DIR = WEBSITE / ".generated"
API_INDEX_PATH = WEBSITE / "content" / "docs" / "api" / "index.mdx"
# Legacy MkDocs paths kept for helper functions that are no longer invoked by
# the current docs pipeline. The active build path only emits website/public/ops.json.
DOCS = ROOT / "docs"
API_DIR = DOCS / "api"
REF_DIR = DOCS / "reference"

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
# Helper functions for op references and cost lookup
# ---------------------------------------------------------------------------


def whest_ref(name: str, module: str) -> str:
    """Derive the whest call reference from an op name and registry module."""
    if module == "numpy.linalg":
        return f"`we.linalg.{name.removeprefix('linalg.')}`"
    if module == "numpy.fft":
        return f"`we.fft.{name.removeprefix('fft.')}`"
    if module == "numpy.random":
        return f"`we.random.{name.removeprefix('random.')}`"
    if module == "whest.stats":
        return f"`we.{name}`"
    return f"`we.{name}`"


def numpy_ref(name: str, module: str) -> str:
    """Derive the NumPy call reference from an op name and registry module."""
    if module == "numpy.linalg":
        return f"`np.linalg.{name.removeprefix('linalg.')}`"
    if module == "numpy.fft":
        return f"`np.fft.{name.removeprefix('fft.')}`"
    if module == "numpy.random":
        return f"`np.random.{name.removeprefix('random.')}`"
    if module == "whest.stats":
        return f"`scipy.{name}`"
    return f"`np.{name}`"


def cost_for_op(name: str, category: str) -> tuple[str, str]:
    """Return (plain_text, latex) cost formula for an operation."""
    if name in CUSTOM_COSTS:
        return CUSTOM_COSTS[name]
    return CATEGORY_COST_LATEX.get(category, ("unknown", "unknown"))


def slug_for_op(name: str) -> str:
    """Return the canonical slug for an operation."""
    return name


def detail_href_for_slug(slug: str) -> str:
    """Return the public op detail page URL."""
    return f"/docs/api/ops/{slug}/"


def detail_json_href_for_slug(slug: str) -> str:
    """Return the public per-op JSON URL."""
    return f"/api-data/ops/{slug}.json"


def load_weights() -> dict[str, float]:
    """Load empirical weights if available."""
    weights_path = ROOT / "src" / "whest" / "data" / "weights.json"
    if not weights_path.exists():
        return {}
    raw = json.loads(weights_path.read_text())
    return raw.get("weights", {})


def resolve_whest_callable(name: str, module: str):
    """Resolve a registry operation name to the exported whest callable."""
    whest = importlib.import_module("whest")

    if module == "numpy.linalg":
        return getattr(whest.linalg, name.removeprefix("linalg."))
    if module == "numpy.fft":
        return getattr(whest.fft, name.removeprefix("fft."))
    if module == "numpy.random":
        return getattr(whest.random, name.removeprefix("random."))
    if module == "whest.stats":
        current = whest
        for part in name.split("."):
            current = getattr(current, part)
        return current
    return getattr(whest, name)


def resolve_numpy_callable(name: str, module: str):
    """Resolve a registry operation name to the corresponding NumPy callable."""
    numpy = importlib.import_module("numpy")

    if module == "numpy.linalg":
        return getattr(numpy.linalg, name.removeprefix("linalg."))
    if module == "numpy.fft":
        return getattr(numpy.fft, name.removeprefix("fft."))
    if module == "numpy.random":
        return getattr(numpy.random, name.removeprefix("random."))
    if module == "whest.stats":
        return None
    return getattr(numpy, name)


def numpy_docs_url(name: str, module: str) -> str:
    """Return the NumPy generated-doc page for an operation."""
    if module == "numpy.linalg":
        full_name = f"numpy.linalg.{name.removeprefix('linalg.')}"
    elif module == "numpy.fft":
        full_name = f"numpy.fft.{name.removeprefix('fft.')}"
    elif module == "numpy.random":
        full_name = f"numpy.random.{name.removeprefix('random.')}"
    elif module == "whest.stats":
        return ""
    else:
        full_name = f"numpy.{name}"
    return f"https://numpy.org/doc/stable/reference/generated/{full_name}.html"


def github_source_url(obj, *, repo: str, version: str | None = None) -> str | None:
    """Build a GitHub source URL for a Python object when its source file is known."""
    try:
        source_file = Path(inspect.getsourcefile(obj) or "")
        if not source_file.exists():
            return None
        source_lines, start_line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None

    if repo == "whest":
        try:
            rel_path = source_file.resolve().relative_to(ROOT)
        except ValueError:
            return None
        branch = "main"
        end_line = start_line + len(source_lines) - 1
        return (
            f"https://github.com/AIcrowd/whest/blob/{branch}/{rel_path.as_posix()}"
            f"#L{start_line}-L{end_line}"
        )

    if repo == "numpy":
        numpy = importlib.import_module("numpy")
        numpy_root = Path(numpy.__file__).resolve().parent
        try:
            rel_path = source_file.resolve().relative_to(numpy_root.parent)
        except ValueError:
            return None
        tag = f"v{version or numpy.__version__}"
        end_line = start_line + len(source_lines) - 1
        return (
            f"https://github.com/numpy/numpy/blob/{tag}/{rel_path.as_posix()}"
            f"#L{start_line}-L{end_line}"
        )

    return None


def block_from_lines(lines: list[str]) -> dict[str, str] | None:
    """Convert a chunk of docstring lines into a normalized render block."""
    if not lines:
        return None
    first = lines[0].strip()
    if not first:
        return None

    if first.startswith(">>>") or first.startswith("..."):
        return {"type": "code", "language": "text", "code": "\n".join(lines).rstrip()}

    if first.startswith(".. "):
        directive_name = first[3:].rstrip(":").strip()
        body = "\n".join(lines[1:]).rstrip()
        return {
            "type": "directive",
            "name": directive_name,
            "body": body,
            "raw": "\n".join(lines).rstrip(),
        }

    return {"type": "paragraph", "text": "\n".join(lines).rstrip()}


def lines_to_blocks(lines: list[str]) -> list[dict[str, str]]:
    """Normalize section text lines into paragraph/code/directive blocks."""
    blocks: list[dict[str, str]] = []
    chunk: list[str] = []

    def flush() -> None:
        nonlocal chunk
        block = block_from_lines(chunk)
        if block is not None:
            blocks.append(block)
        chunk = []

    for line in lines:
        if line.strip():
            chunk.append(line.rstrip())
            continue
        flush()
    flush()
    return blocks


def field_items_to_payload(items) -> list[dict[str, object]]:
    """Normalize numpydoc parameter/return entries into JSON-safe payloads."""
    payload = []
    for item in items:
        payload.append(
            {
                "name": item.name,
                "type": item.type,
                "desc_blocks": lines_to_blocks(list(item.desc)),
            }
        )
    return payload


def see_also_to_blocks(entries) -> list[dict[str, str]]:
    """Normalize numpydoc See Also tuples into paragraph blocks."""
    blocks: list[dict[str, str]] = []
    for refs, desc in entries:
        ref_text = ", ".join(name for name, _role in refs if name)
        description = " ".join(line.strip() for line in desc if line.strip())
        text = ref_text if not description else f"{ref_text} — {description}"
        if text:
            blocks.append({"type": "paragraph", "text": text})
    return blocks


def normalize_signature(
    whest_call_ref: str, callable_obj, np_doc_signature: str
) -> str:
    """Return a display signature preferring explicit numpydoc signatures when present."""
    if np_doc_signature:
        return re.sub(r"^[A-Za-z0-9_.]+", whest_call_ref, np_doc_signature, count=1)
    return f"{whest_call_ref}{inspect.signature(callable_obj)}"


def parse_docstring_payload(callable_obj) -> tuple[str, list[dict[str, object]]]:
    """Parse a callable docstring into a normalized summary and section list."""
    docstring = inspect.getdoc(callable_obj) or ""
    if not docstring:
        return "", []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_doc = NumpyDocString(docstring)

    summary_lines = [line.strip() for line in np_doc["Summary"] if line.strip()]
    extended_summary_lines = [line.rstrip() for line in np_doc["Extended Summary"]]
    summary = " ".join(summary_lines).strip()

    sections: list[dict[str, object]] = []
    summary_blocks = lines_to_blocks(
        summary_lines
        + ([""] + extended_summary_lines if extended_summary_lines else [])
    )
    if summary_blocks:
        sections.append(
            {"kind": "summary", "title": "Summary", "blocks": summary_blocks}
        )

    section_specs = [
        ("Parameters", "parameters", "Parameters"),
        ("Returns", "returns", "Returns"),
        ("Notes", "notes", "Notes"),
        ("Examples", "examples", "Examples"),
        ("See Also", "see-also", "See Also"),
    ]

    for np_key, kind, title in section_specs:
        value = np_doc[np_key]
        if not value:
            continue
        if np_key in {"Parameters", "Returns"}:
            items = field_items_to_payload(value)
            if items:
                sections.append({"kind": kind, "title": title, "items": items})
            continue

        if np_key == "See Also":
            blocks = see_also_to_blocks(value)
            if blocks:
                sections.append({"kind": kind, "title": title, "blocks": blocks})
            continue

        blocks = lines_to_blocks(list(value))
        if blocks:
            sections.append({"kind": kind, "title": title, "blocks": blocks})

    return summary, sections


def build_op_detail_payload(
    name: str, info: dict, *, weights: dict[str, float]
) -> dict[str, object]:
    """Build the full per-op JSON payload for an operation."""
    module = info["module"]
    category = info["category"]
    slug = slug_for_op(name)
    numpy_callable = resolve_numpy_callable(name, module)
    whest_callable = None
    if category != "blacklisted":
        whest_callable = resolve_whest_callable(name, module)

    whest_call_ref = whest_ref(name, module).strip("`")
    numpy_call_ref = numpy_ref(name, module).strip("`")
    doc_callable = whest_callable or numpy_callable
    if doc_callable is None:
        raise ValueError(f"Could not resolve a doc callable for {name!r}")
    summary, sections = parse_docstring_payload(doc_callable)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_doc = NumpyDocString(inspect.getdoc(doc_callable) or "")

    signature = normalize_signature(whest_call_ref, doc_callable, np_doc["Signature"])
    detail_href = detail_href_for_slug(slug)
    detail_json_href = detail_json_href_for_slug(slug)
    plain_cost, latex_cost = cost_for_op(name, category)

    whest_source = (
        github_source_url(whest_callable, repo="whest")
        if whest_callable is not None
        else None
    )
    numpy_source = (
        github_source_url(numpy_callable, repo="numpy")
        if numpy_callable is not None
        else None
    )
    if numpy_source is None:
        numpy_source = numpy_docs_url(name, module)

    return {
        "schema_version": 1,
        "slug": slug,
        "detail_href": detail_href,
        "detail_json_href": detail_json_href,
        "source": {
            "whest": whest_source,
            "numpy": numpy_source,
        },
        "op": {
            "name": name,
            "module": module,
            "whest_ref": whest_call_ref,
            "numpy_ref": numpy_call_ref,
            "category": category,
            "status": "blocked" if category == "blacklisted" else "supported",
            "weight": weights.get(name, 1.0),
            "cost_formula": plain_cost,
            "cost_formula_latex": latex_cost,
            "notes": info.get("notes", ""),
            "summary": summary,
            "signature": signature,
        },
        "docs": {
            "sections": sections,
        },
    }


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


def generate_op_doc_import_map(slugs: list[str]) -> None:
    """Generate the internal slug->lazy import map used by static op pages."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "import type { OpDocPayload } from '@/components/api-reference/op-doc-types';",
        "",
        "export const opDocImports: Record<string, () => Promise<{ default: OpDocPayload }>> = {",
    ]
    for slug in slugs:
        lines.append(f'  "{slug}": () => import("../public/api-data/ops/{slug}.json"),')
    lines.extend(
        [
            "};",
            "",
            "export const opDocSlugs = [",
        ]
    )
    for slug in slugs:
        lines.append(f'  "{slug}",')
    lines.extend(["] as const;", ""])

    out = GENERATED_DIR / "op-doc-imports.ts"
    out.write_text("\n".join(lines))
    print(f"  Generated .generated/op-doc-imports.ts ({len(slugs)} ops)")


def generate_ops_json(registry: dict[str, dict]) -> None:
    """Generate the slim global index plus per-op JSON payloads."""
    weights = load_weights()

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    API_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for existing_json in API_DATA_DIR.glob("*.json"):
        existing_json.unlink()

    ops = []
    slugs: list[str] = []
    seen_slugs: dict[str, str] = {}

    for name, info in sorted(registry.items()):
        slug = slug_for_op(name)
        if slug in seen_slugs:
            raise ValueError(
                f"Slug collision for {slug!r}: {seen_slugs[slug]!r} and {name!r}"
            )
        seen_slugs[slug] = name
        slugs.append(slug)

        payload = build_op_detail_payload(name, info, weights=weights)
        (API_DATA_DIR / f"{slug}.json").write_text(json.dumps(payload, indent=2))

        cat = info["category"]
        mod = info["module"]
        plain, latex = cost_for_op(name, cat)
        ops.append(
            {
                "name": name,
                "slug": slug,
                "detail_href": detail_href_for_slug(slug),
                "detail_json_href": detail_json_href_for_slug(slug),
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
                "summary": payload["op"]["summary"],
                "weight": weights.get(name, 1.0),
            }
        )

    out = PUBLIC_DIR / "ops.json"
    out.write_text(json.dumps({"operations": ops, "total": len(ops)}, indent=2))
    generate_op_doc_import_map(slugs)
    print(f"  Generated ops.json ({len(ops)} operations)")
    print(f"  Generated api-data/ops/*.json ({len(ops)} operations)")


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

    missing_detail_fields = [
        op["name"]
        for op in ops_data["operations"]
        if not {"slug", "detail_href", "detail_json_href", "summary"} <= set(op)
    ]
    if missing_detail_fields:
        print("\nops.json entries missing detail fields:")
        for name in missing_detail_fields[:20]:
            print(f"  {name}")
        return False

    detail_dir = PUBLIC_DIR / "api-data" / "ops"
    if not detail_dir.exists():
        print(f"\nper-op detail dir NOT FOUND at {detail_dir}")
        return False

    sample_missing = []
    for op in ops_data["operations"][:10]:
        detail_path = detail_dir / f"{op['slug']}.json"
        if not detail_path.exists():
            sample_missing.append(str(detail_path))
    if sample_missing:
        print("\nMissing per-op detail payloads:")
        for path in sample_missing:
            print(f"  {path}")
        return False

    import_map_path = GENERATED_DIR / "op-doc-imports.ts"
    if not import_map_path.exists():
        print(f"\nGenerated import map NOT FOUND at {import_map_path}")
        return False

    print("Per-op detail payloads and import map are present.")

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

    registry = load_registry()

    if args.verify:
        ok = verify_coverage(registry)
        sys.exit(0 if ok else 1)

    print("Generating API reference data...")
    generate_ops_json(registry)

    print("\nDone. Run with --verify to check coverage.")


if __name__ == "__main__":
    main()
