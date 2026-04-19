"""Declarative benchmark case specifications for overhead measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

import whest as we

Surface = Literal["api", "operator"]


@dataclass(frozen=True)
class BenchmarkCase:
    """Static description of one benchmarked operation variant."""

    case_id: str
    op_name: str
    family: str
    surface: Surface
    dtype: str
    size_name: str
    startup_mode: str
    source_file: str
    operand_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    numpy_factory: Callable[[object, object], object]
    whest_factory: Callable[[object, object], object]


def _case_id(op_name: str, surface: Surface, size_name: str) -> str:
    return f"{op_name}-{surface}-{size_name}"


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


def _numpy_add_api(a, b):
    return np.add(a, b)


def _whest_add_api(a, b):
    return we.add(a, b)


def _numpy_add_operator(a, b):
    return a + b


def _whest_add_operator(a, b):
    return we.array(a) + we.array(b)


def _numpy_matmul_api(a, b):
    return np.matmul(a, b)


def _whest_matmul_api(a, b):
    return we.matmul(a, b)


def _numpy_matmul_operator(a, b):
    return a @ b


def _whest_matmul_operator(a, b):
    return we.array(a) @ we.array(b)


def seed_cases() -> tuple[BenchmarkCase, ...]:
    """Return the initial overhead benchmark matrix."""

    dtype = "float64"
    cases = [
        BenchmarkCase(
            case_id=_case_id("add", "api", "tiny"),
            op_name="add",
            family="pointwise",
            surface="api",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_add_shapes("tiny"),
            numpy_factory=_numpy_add_api,
            whest_factory=_whest_add_api,
        ),
        BenchmarkCase(
            case_id=_case_id("add", "api", "medium"),
            op_name="add",
            family="pointwise",
            surface="api",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_add_shapes("medium"),
            numpy_factory=_numpy_add_api,
            whest_factory=_whest_add_api,
        ),
        BenchmarkCase(
            case_id=_case_id("add", "operator", "tiny"),
            op_name="add",
            family="pointwise",
            surface="operator",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_add_shapes("tiny"),
            numpy_factory=_numpy_add_operator,
            whest_factory=_whest_add_operator,
        ),
        BenchmarkCase(
            case_id=_case_id("add", "operator", "medium"),
            op_name="add",
            family="pointwise",
            surface="operator",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_add_shapes("medium"),
            numpy_factory=_numpy_add_operator,
            whest_factory=_whest_add_operator,
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "api", "tiny"),
            op_name="matmul",
            family="contractions",
            surface="api",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_matmul_shapes("tiny"),
            numpy_factory=_numpy_matmul_api,
            whest_factory=_whest_matmul_api,
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "api", "medium"),
            op_name="matmul",
            family="contractions",
            surface="api",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_pointwise.py",
            operand_shapes=_matmul_shapes("medium"),
            numpy_factory=_numpy_matmul_api,
            whest_factory=_whest_matmul_api,
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "operator", "tiny"),
            op_name="matmul",
            family="contractions",
            surface="operator",
            dtype=dtype,
            size_name="tiny",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_matmul_shapes("tiny"),
            numpy_factory=_numpy_matmul_operator,
            whest_factory=_whest_matmul_operator,
        ),
        BenchmarkCase(
            case_id=_case_id("matmul", "operator", "medium"),
            op_name="matmul",
            family="contractions",
            surface="operator",
            dtype=dtype,
            size_name="medium",
            startup_mode="warmup",
            source_file="src/whest/_ndarray.py",
            operand_shapes=_matmul_shapes("medium"),
            numpy_factory=_numpy_matmul_operator,
            whest_factory=_whest_matmul_operator,
        ),
    ]
    return tuple(cases)
