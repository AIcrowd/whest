"""Benchmark case specifications for overhead measurements."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from benchmarks._bitwise import BITWISE_OPS
from benchmarks._complex import COMPLEX_OPS
from benchmarks._fft import FFT_OPS
from benchmarks._linalg import LINALG_OPS
from benchmarks._linalg_delegates import LINALG_DELEGATE_OPS
from benchmarks._misc import MISC_OPS
from benchmarks._pointwise import BINARY_OPS, UNARY_OPS
from benchmarks._polynomial import POLYNOMIAL_OPS
from benchmarks._random import RANDOM_OPS
from benchmarks._reductions import REDUCTION_OPS
from benchmarks._window import WINDOW_OPS

Surface = Literal["api", "operator"]
FactorySpec = Callable[..., object] | str

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_OPS_PATH = _REPO_ROOT / "website" / "public" / "ops.json"
_DOCS_OP_PAYLOAD_DIR = _REPO_ROOT / "website" / "public" / "api-data" / "ops"
_WHEST_BLOB_RE = re.compile(r"/blob/[^/]+/(?P<path>src/[^#]+)")

_FFT_NAMES = {name.split(".", 1)[1] for name in FFT_OPS}
_LINALG_NAMES = {name.split(".", 1)[1] for name in LINALG_OPS}
_LINALG_DELEGATE_NAMES = {name.split(".", 1)[1] for name in LINALG_DELEGATE_OPS}
_RANDOM_NAMES = {name.split(".", 1)[1] for name in RANDOM_OPS}
_WINDOW_NAMES = set(WINDOW_OPS)
_POLYNOMIAL_NAMES = set(POLYNOMIAL_OPS)
_BITWISE_NAMES = set(BITWISE_OPS)
_COMPLEX_NAMES = set(COMPLEX_OPS)
_UNARY_NAMES = set(UNARY_OPS)
_BINARY_NAMES = set(BINARY_OPS)
_REDUCTION_NAMES = set(REDUCTION_OPS)
_SORTING_PROFILE_KINDS = {
    "sort": "sort_like",
    "argsort": "sort_like",
    "unique": "sort_like",
    "unique_all": "sort_like",
    "unique_counts": "sort_like",
    "unique_inverse": "sort_like",
    "unique_values": "sort_like",
    "partition": "partition_like",
    "argpartition": "partition_like",
    "searchsorted": "searchsorted",
    "in1d": "set_binary",
    "isin": "set_binary",
    "intersect1d": "set_binary",
    "setdiff1d": "set_binary",
    "setxor1d": "set_binary",
    "union1d": "set_binary",
}
_CUSTOM_PROFILE_KINDS = {
    "matmul": "matmul",
    "allclose": "allclose",
    "array_equal": "array_compare",
    "array_equiv": "array_compare",
    "clip": "clip",
    "array": "array_constructor",
    "arange": "arange",
    "linspace": "linspace",
    "logspace": "logspace",
    "geomspace": "geomspace",
    "trace": "trace",
}
_MISC_PROFILE_KINDS = {
    "diff": "diff_like",
    "ediff1d": "diff_like",
    "gradient": "gradient",
    "unwrap": "gradient",
    "convolve": "convolve_like",
    "correlate": "convolve_like",
    "corrcoef": "corrcoef_like",
    "cov": "corrcoef_like",
    "cross": "cross_like",
    "histogram": "histogram",
    "histogram2d": "histogram2d",
    "histogramdd": "histogramdd",
    "histogram_bin_edges": "histogram_bin_edges",
    "digitize": "digitize",
    "bincount": "bincount",
    "interp": "interp",
    "trapezoid": "trapezoid",
    "vander": "vander",
}
_FREE_PROFILE_KINDS = {
    "array_split": "split_array",
    "asarray": "array_unary_free",
    "astype": "astype_free",
    "atleast_1d": "atleast_free",
    "atleast_2d": "atleast_free",
    "atleast_3d": "atleast_free",
    "broadcast_arrays": "broadcast_arrays_free",
    "broadcast_shapes": "broadcast_shapes_free",
    "broadcast_to": "broadcast_to_free",
    "can_cast": "dtype_relation_free",
    "column_stack": "stack_free",
    "common_type": "dtype_relation_free",
    "copy": "array_unary_free",
    "diag_indices": "diag_indices_free",
    "diag_indices_from": "diag_indices_from_free",
    "diagonal": "diagonal_free",
    "dsplit": "split_array",
    "empty": "shape_constructor_free",
    "empty_like": "like_constructor_free",
    "expand_dims": "expand_dims_free",
    "eye": "eye_free",
    "flip": "array_unary_free",
    "fliplr": "array_unary_free",
    "flipud": "array_unary_free",
    "hsplit": "split_array",
    "hstack": "stack_free",
    "identity": "eye_free",
    "isdtype": "dtype_relation_free",
    "isfortran": "array_predicate_free",
    "isscalar": "scalar_predicate_free",
    "issubdtype": "dtype_relation_free",
    "iterable": "scalar_predicate_free",
    "matrix_transpose": "transpose_free",
    "may_share_memory": "memory_relation_free",
    "min_scalar_type": "dtype_relation_free",
    "mintypecode": "dtype_relation_free",
    "moveaxis": "moveaxis_free",
    "ndim": "array_predicate_free",
    "ones": "shape_constructor_free",
    "ones_like": "like_constructor_free",
    "permute_dims": "moveaxis_free",
    "promote_types": "dtype_relation_free",
    "ravel": "array_unary_free",
    "ravel_multi_index": "ravel_multi_index_free",
    "require": "array_unary_free",
    "reshape": "reshape_free",
    "result_type": "dtype_relation_free",
    "rollaxis": "moveaxis_free",
    "rot90": "array_unary_free",
    "shape": "array_predicate_free",
    "shares_memory": "memory_relation_free",
    "size": "array_predicate_free",
    "split": "split_array",
    "squeeze": "array_unary_free",
    "swapaxes": "moveaxis_free",
    "transpose": "transpose_free",
    "tri": "tri_free",
    "tril": "triangular_matrix_free",
    "tril_indices": "tri_indices_free",
    "tril_indices_from": "tri_indices_from_free",
    "triu": "triangular_matrix_free",
    "triu_indices": "tri_indices_free",
    "triu_indices_from": "tri_indices_from_free",
    "unravel_index": "unravel_index_free",
    "vsplit": "split_array",
    "zeros": "shape_constructor_free",
    "zeros_like": "like_constructor_free",
}
_LINALG_RHS_NAMES = {"solve", "lstsq"}
_FFT_MATRIX_NAMES = {
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
}
_FFT_FREE_PROFILE_KINDS = {
    "fftfreq": "fft_freqs",
    "rfftfreq": "fft_freqs",
    "fftshift": "fft_shift",
    "ifftshift": "fft_shift",
}
_EXCLUDED_PROFILE_STATUSES = {
    "default_rng": "excluded",
    "from_dlpack": "excluded",
    "frombuffer": "excluded",
    "get_state": "excluded",
    "seed": "excluded",
    "set_state": "excluded",
}


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class BenchmarkCase:
    """Static description of one benchmarked operation variant."""

    case_id: str
    op_name: str
    qualified_name: str | None
    family: str
    surface: Surface
    dtype: str
    size_name: str
    startup_mode: str
    source_file: str
    operand_shapes: tuple[tuple[int, ...], ...] = ()
    numpy_factory: FactorySpec = ""
    whest_factory: FactorySpec = ""
    slug: str | None = None
    area: str | None = None
    category: str | None = None
    profile_kind: str | None = None
    profile_params: dict[str, object] = field(default_factory=dict)


def _source_file_for_slug(slug: str) -> str:
    payload_path = _DOCS_OP_PAYLOAD_DIR / f"{slug}.json"
    if not payload_path.exists():
        return ""
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    whest_source = payload.get("source", {}).get("whest")
    if not whest_source:
        return ""
    match = _WHEST_BLOB_RE.search(str(whest_source))
    return match.group("path") if match else ""


def _qualified_name_from_ref(whest_ref: str | None) -> str | None:
    if not whest_ref or not whest_ref.startswith("we."):
        return None
    return "whest." + whest_ref[3:]


def _numpy_path_from_ref(numpy_ref: str | None) -> str | None:
    if not numpy_ref or not numpy_ref.startswith("np."):
        return None
    return "numpy." + numpy_ref[3:]


def _family_for_operation(name: str, module: str, category: str) -> str:
    leaf_name = _leaf_name(name)
    if module == "numpy.linalg" or leaf_name in _LINALG_NAMES:
        return "linalg"
    if module == "numpy.fft" or leaf_name in _FFT_NAMES:
        return "fft"
    if module == "numpy.random" or leaf_name in _RANDOM_NAMES:
        return "random"
    if leaf_name == "matmul":
        return "contractions"
    if leaf_name in _UNARY_NAMES or leaf_name in _BINARY_NAMES:
        return "pointwise"
    if leaf_name in _REDUCTION_NAMES:
        return "reductions"
    if leaf_name in _SORTING_PROFILE_KINDS:
        return "sorting"
    if leaf_name in _WINDOW_NAMES:
        return "window"
    if leaf_name in _POLYNOMIAL_NAMES:
        return "polynomial"
    if leaf_name in MISC_OPS or category == "counted_custom":
        return "misc"
    if category == "free":
        return "free"
    return "misc"


def _generation_plan(
    name: str, module: str, category: str
) -> tuple[str, str | None, dict[str, object]]:
    leaf_name = _leaf_name(name)
    if leaf_name in _EXCLUDED_PROFILE_STATUSES:
        return _EXCLUDED_PROFILE_STATUSES[leaf_name], None, {"op_name": leaf_name}
    if module == "whest.stats":
        return "unsupported", None, {"op_name": leaf_name}
    if module == "numpy.random" and leaf_name in {"bytes", "random_integers"}:
        return (
            "generated",
            "random_call",
            {"op_name": leaf_name, "random_name": leaf_name},
        )
    if module == "numpy.linalg" and leaf_name in _LINALG_NAMES:
        kind = (
            "linalg_matrix_rhs" if leaf_name in _LINALG_RHS_NAMES else "linalg_matrix"
        )
        return "generated", kind, {"op_name": leaf_name}
    if module == "numpy.linalg" and leaf_name in _LINALG_DELEGATE_NAMES:
        return "generated", "linalg_delegate", {"op_name": leaf_name}
    if module == "numpy.fft" and leaf_name in _FFT_NAMES:
        kind = "fft_matrix" if leaf_name in _FFT_MATRIX_NAMES else "fft_vector"
        return "generated", kind, {"op_name": leaf_name}
    if module == "numpy.fft" and leaf_name in _FFT_FREE_PROFILE_KINDS:
        return "generated", _FFT_FREE_PROFILE_KINDS[leaf_name], {"op_name": leaf_name}
    if module == "numpy.random" and leaf_name in _RANDOM_NAMES:
        return (
            "generated",
            "random_call",
            {"op_name": leaf_name, "random_name": leaf_name},
        )
    if leaf_name in _BITWISE_NAMES:
        if "shift" in leaf_name:
            return "generated", "bitwise_shift", {"op_name": leaf_name}
        if leaf_name in {"bitwise_and", "bitwise_or", "bitwise_xor", "gcd", "lcm"}:
            return "generated", "bitwise_binary", {"op_name": leaf_name}
        return "generated", "bitwise_unary", {"op_name": leaf_name}
    if leaf_name in _COMPLEX_NAMES:
        kind = "complex_sort" if leaf_name == "sort_complex" else "complex_unary"
        return "generated", kind, {"op_name": leaf_name}
    if leaf_name == "absolute":
        return "generated", "vector_unary", {"op_name": leaf_name}
    if leaf_name == "isclose":
        return "generated", "vector_binary", {"op_name": leaf_name}
    if leaf_name in _MISC_PROFILE_KINDS:
        return "generated", _MISC_PROFILE_KINDS[leaf_name], {"op_name": leaf_name}
    if category == "free" and leaf_name in _FREE_PROFILE_KINDS:
        return "generated", _FREE_PROFILE_KINDS[leaf_name], {"op_name": leaf_name}
    if leaf_name in _POLYNOMIAL_NAMES:
        return "generated", "polynomial_call", {"op_name": leaf_name}
    if leaf_name in _UNARY_NAMES:
        return "generated", "vector_unary", {"op_name": leaf_name}
    if leaf_name in _BINARY_NAMES:
        return "generated", "vector_binary", {"op_name": leaf_name}
    if leaf_name in _REDUCTION_NAMES:
        return "generated", "vector_reduction", {"op_name": leaf_name}
    if leaf_name in _SORTING_PROFILE_KINDS:
        return "generated", _SORTING_PROFILE_KINDS[leaf_name], {"op_name": leaf_name}
    if leaf_name in _CUSTOM_PROFILE_KINDS:
        return "generated", _CUSTOM_PROFILE_KINDS[leaf_name], {"op_name": leaf_name}
    if leaf_name in _WINDOW_NAMES:
        return "generated", "window", {"op_name": leaf_name}
    return "profile_missing", None, {}


def _profile_dtype(profile_kind: str | None) -> str:
    if profile_kind in {"bitwise_unary", "bitwise_binary", "bitwise_shift"}:
        return "int64"
    if profile_kind in {"complex_unary", "complex_sort"}:
        return "complex128"
    return "float64"


def _case_id(slug: str, surface: Surface, size_name: str) -> str:
    return f"{slug}-{surface}-{size_name}"


def _numpy_add_api(a, b):
    return np.add(a, b)


def _whest_add_api(a, b):
    import whest as we

    return we.add(a, b)


def _numpy_add_operator(a, b):
    return a + b


def _whest_add_operator(a, b):
    import whest as we

    return we.array(a) + we.array(b)


def _numpy_matmul_api(a, b):
    return np.matmul(a, b)


def _whest_matmul_api(a, b):
    import whest as we

    return we.matmul(a, b)


def _numpy_matmul_operator(a, b):
    return a @ b


def _whest_matmul_operator(a, b):
    import whest as we

    return we.array(a) @ we.array(b)


def _add_shapes(size_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if size_name == "tiny":
        return ((8,), (8,))
    if size_name == "medium":
        return ((1024,), (1024,))
    raise ValueError(f"unsupported add size_name: {size_name}")


def _matmul_shapes(size_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if size_name == "tiny":
        return ((8, 8), (8, 8))
    if size_name == "medium":
        return ((512, 512), (512, 512))
    raise ValueError(f"unsupported matmul size_name: {size_name}")


@lru_cache(maxsize=1)
def seed_cases() -> tuple[BenchmarkCase, ...]:
    """Return the curated representative benchmark matrix used by `ci`."""

    dtype = "float64"
    cases = [
        BenchmarkCase(
            case_id=_case_id("add", "api", "tiny"),
            op_name="add",
            qualified_name="whest.add",
            family="pointwise",
            surface="api",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_add_shapes("tiny"),
            numpy_factory=_numpy_add_api,
            whest_factory=_whest_add_api,
            slug="add",
            category="counted_binary",
        ),
        BenchmarkCase(
            case_id=_case_id("add", "api", "medium"),
            op_name="add",
            qualified_name="whest.add",
            family="pointwise",
            surface="api",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_add_shapes("medium"),
            numpy_factory=_numpy_add_api,
            whest_factory=_whest_add_api,
            slug="add",
            category="counted_binary",
        ),
        BenchmarkCase(
            case_id=_case_id("add", "operator", "tiny"),
            op_name="add",
            qualified_name=None,
            family="pointwise",
            surface="operator",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_add_shapes("tiny"),
            numpy_factory=_numpy_add_operator,
            whest_factory=_whest_add_operator,
            slug="add",
            category="counted_binary",
        ),
        BenchmarkCase(
            case_id=_case_id("add", "operator", "medium"),
            op_name="add",
            qualified_name=None,
            family="pointwise",
            surface="operator",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_add_shapes("medium"),
            numpy_factory=_numpy_add_operator,
            whest_factory=_whest_add_operator,
            slug="add",
            category="counted_binary",
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "api", "tiny"),
            op_name="matmul",
            qualified_name="whest.matmul",
            family="contractions",
            surface="api",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_matmul_shapes("tiny"),
            numpy_factory=_numpy_matmul_api,
            whest_factory=_whest_matmul_api,
            slug="matmul",
            category="counted_custom",
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "api", "medium"),
            op_name="matmul",
            qualified_name="whest.matmul",
            family="contractions",
            surface="api",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_matmul_shapes("medium"),
            numpy_factory=_numpy_matmul_api,
            whest_factory=_whest_matmul_api,
            slug="matmul",
            category="counted_custom",
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "operator", "tiny"),
            op_name="matmul",
            qualified_name=None,
            family="contractions",
            surface="operator",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_matmul_shapes("tiny"),
            numpy_factory=_numpy_matmul_operator,
            whest_factory=_whest_matmul_operator,
            slug="matmul",
            category="counted_custom",
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "operator", "medium"),
            op_name="matmul",
            qualified_name=None,
            family="contractions",
            surface="operator",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_matmul_shapes("medium"),
            numpy_factory=_numpy_matmul_operator,
            whest_factory=_whest_matmul_operator,
            slug="matmul",
            category="counted_custom",
        ),
    ]
    return tuple(cases)


@lru_cache(maxsize=1)
def documented_operations() -> tuple[dict[str, object], ...]:
    """Return documented supported operations enriched with generation metadata."""

    payload = json.loads(_DOCS_OPS_PATH.read_text(encoding="utf-8"))
    operations: list[dict[str, object]] = []
    for entry in payload.get("operations", []):
        slug = str(entry["slug"])
        name = str(entry["name"])
        module = str(entry.get("module", ""))
        category = str(entry.get("category", ""))
        generation_status, profile_kind, profile_params = _generation_plan(
            name,
            module,
            category,
        )
        operations.append(
            {
                **entry,
                "qualified_name": _qualified_name_from_ref(entry.get("whest_ref")),
                "numpy_path": _numpy_path_from_ref(entry.get("numpy_ref")),
                "source_file": _source_file_for_slug(slug),
                "family": _family_for_operation(name, module, category),
                "generation_status": generation_status,
                "profile_kind": profile_kind,
                "profile_params": profile_params,
                "profile_dtype": _profile_dtype(profile_kind),
                "generated_case_ids": [
                    _case_id(slug, "api", size_name) for size_name in ("tiny", "medium")
                ]
                if generation_status == "generated"
                else [],
            }
        )
    return tuple(operations)


@lru_cache(maxsize=1)
def full_cases() -> tuple[BenchmarkCase, ...]:
    """Return generated API benchmark cases for the exhaustive `full` sweep."""

    cases: list[BenchmarkCase] = []
    for operation in documented_operations():
        if operation["generation_status"] != "generated":
            continue

        qualified_name = operation["qualified_name"]
        numpy_path = operation["numpy_path"]
        if qualified_name is None or numpy_path is None:
            continue

        for size_name in ("tiny", "medium"):
            cases.append(
                BenchmarkCase(
                    case_id=_case_id(str(operation["slug"]), "api", size_name),
                    op_name=str(operation["name"]),
                    qualified_name=qualified_name,
                    family=str(operation["family"]),
                    surface="api",
                    dtype=str(operation.get("profile_dtype", "float64")),
                    size_name=size_name,
                    startup_mode="warmup",
                    source_file=str(operation["source_file"]),
                    numpy_factory=numpy_path,
                    whest_factory=qualified_name,
                    slug=str(operation["slug"]),
                    area=operation.get("area"),
                    category=str(operation["category"]),
                    profile_kind=str(operation["profile_kind"]),
                    profile_params=dict(operation["profile_params"]),
                )
            )
    return tuple(cases)
