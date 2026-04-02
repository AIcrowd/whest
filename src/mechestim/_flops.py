"""FLOP cost calculators for mechestim operations."""
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mechestim._symmetric import SymmetryInfo


def parse_einsum_subscripts(subscripts: str) -> tuple[list[list[str]], list[str]]:
    """Parse an einsum subscript string into input and output index lists."""
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        input_part, output_part = subscripts.split("->")
        output = list(output_part)
    else:
        input_part = subscripts
        all_labels: list[str] = []
        for part in input_part.split(","):
            all_labels.extend(list(part))
        counts = Counter(all_labels)
        output = sorted(l for l, c in counts.items() if c == 1)
    inputs = [list(part) for part in input_part.split(",")]
    return inputs, output


def einsum_cost(subscripts: str, shapes: list[tuple[int, ...]], repeated_operand_indices: list[int] | None = None, symmetric_dims: list[tuple[int, ...]] | None = None, operand_symmetries: "list[SymmetryInfo | None] | None" = None) -> int:
    """Calculate the FLOP cost of an einsum operation."""
    inputs, output = parse_einsum_subscripts(subscripts)
    label_dims: dict[str, int] = {}
    for operand_labels, shape in zip(inputs, shapes):
        for label, dim in zip(operand_labels, shape):
            if label in label_dims:
                if label_dims[label] != dim:
                    raise ValueError(f"Inconsistent dimension for label '{label}': {label_dims[label]} vs {dim}")
            else:
                label_dims[label] = dim
    all_labels = set()
    for operand_labels in inputs:
        all_labels.update(operand_labels)
    all_labels.update(output)
    base_cost = 1
    for label in all_labels:
        base_cost *= label_dims[label]
    symmetry_factor = 1
    if repeated_operand_indices and len(repeated_operand_indices) >= 2:
        symmetry_factor *= math.factorial(len(repeated_operand_indices))
    if symmetric_dims:
        for group in symmetric_dims:
            symmetry_factor *= math.factorial(len(group))
    cost = base_cost // symmetry_factor

    # Apply input operand symmetry savings
    if operand_symmetries:
        for i, sym_info in enumerate(operand_symmetries):
            if sym_info is None:
                continue
            total = 1
            for d in sym_info.shape:
                total *= d
            if total > 0:
                unique = sym_info.unique_elements
                cost = cost * unique // total

    return max(cost, 1)


def pointwise_cost(shape: tuple[int, ...], symmetry_info: "SymmetryInfo | None" = None) -> int:
    """Calculate the FLOP cost of a pointwise operation."""
    if symmetry_info is not None:
        return max(symmetry_info.unique_elements, 1)
    result = 1
    for dim in shape:
        result *= dim
    return max(result, 1)


def reduction_cost(input_shape: tuple[int, ...], axis: int | None = None, symmetry_info: "SymmetryInfo | None" = None) -> int:
    """Calculate the FLOP cost of a reduction operation."""
    if symmetry_info is not None:
        return max(symmetry_info.unique_elements, 1)
    result = 1
    for dim in input_shape:
        result *= dim
    return max(result, 1)


def svd_cost(m: int, n: int, k: int | None = None) -> int:
    """Calculate the FLOP cost of a truncated SVD."""
    if k is None:
        k = min(m, n)
    return m * n * k
