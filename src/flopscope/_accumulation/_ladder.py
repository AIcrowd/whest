"""Regime ladder dispatcher and supporting data types.

Port of website/components/symmetry-aware-einsum-contractions/engine/accumulationCount.js
plus the regime contract from regimes/index.js.

The ladder runs per-component:
    Stage 1: trivial short-circuit (|G| ≤ 1)
    Stage 2a: functionalProjection takes priority (covers shape ∈ {allVisible, allSummed,
              and mixed-but-functional})
    Stage 2b: mixed regimes ladder — singleton, young, partitionCount
    Fallthrough: 'unavailable' (brute-force orbit B.8 is excluded by policy)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from flopscope._perm_group import _Permutation as Permutation

from ._shape import Shape

RegimeId = Literal[
    'trivial',
    'functionalProjection',
    'singleton',
    'young',
    'partitionCount',
    'unavailable',
]

Decision = Literal['fired', 'refused']


@dataclass(frozen=True)
class RegimeContext:
    """Input to a regime's recognize() and compute()."""
    labels: tuple[str, ...]
    va: tuple[str, ...]
    wa: tuple[str, ...]
    elements: tuple[Permutation, ...]
    generators: tuple[Permutation, ...]
    sizes: tuple[int, ...]
    visible_positions: tuple[int, ...]
    partition_budget: int


@dataclass(frozen=True)
class Verdict:
    """Output of regime.recognize()."""
    fired: bool
    reason: str


@dataclass(frozen=True)
class RegimeOutput:
    """Output of regime.compute()."""
    count: int
    sub_steps: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Regime:
    """A regime is identified by id and consists of recognize + compute callables."""
    id: RegimeId
    recognize: Callable[[RegimeContext], Verdict]
    compute: Callable[[RegimeContext], RegimeOutput]


@dataclass(frozen=True)
class RegimeStep:
    """One entry in the regime-dispatch trace."""
    regime_id: RegimeId
    decision: Decision
    reason: str
    sub_steps: tuple[dict, ...] = ()


@dataclass(frozen=True)
class AccumulationResult:
    """Output of compute_accumulation() — the ladder's primitive output for one
    component. Reused by future reduction-cost.

    `count` is None when regime_id == 'unavailable' (partition budget exceeded
    with brute-force disabled by policy)."""
    count: int | None
    regime_id: RegimeId
    shape: Shape
    trace: tuple[RegimeStep, ...]
