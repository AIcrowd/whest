"""Typed set partition utilities for the partitionCount regime.

Port of website/components/symmetry-aware-einsum-contractions/engine/partition/typedPartitions.js.

A typed equality pattern is a partition of label positions where blocks may only merge
positions sharing a domain class (same numeric size n_s). The partitionCount regime sums
over typed-partition orbits; each contribution is `falling_factorial(n_s, b_s) / |Ḡ_x̃|`
times the number of stored output representatives reachable.

This module currently provides the basic utilities. Orbit dedup and induced-block-action
helpers land in Task 6.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence


def falling_factorial(n: int, m: int) -> int:
    """Falling factorial n*(n-1)*...*(n-m+1). Zero when m > n; identity 1 when m = 0."""
    if m < 0:
        raise ValueError(f"falling_factorial received negative m={m}")
    if m > n:
        return 0
    result = 1
    for i in range(m):
        result *= n - i
    return result


def normalize_partition(partition: Sequence[int]) -> list[int]:
    """Renumber block IDs by first appearance, so equivalent partitions get equal lists."""
    remap: dict[int, int] = {}
    next_id = 0
    out: list[int] = []
    for block in partition:
        if block not in remap:
            remap[block] = next_id
            next_id += 1
        out.append(remap[block])
    return out


def partition_key(partition: Sequence[int]) -> str:
    """Stable key for partition equality, after normalization. Mirrors JS partitionKey."""
    return "|".join(str(b) for b in normalize_partition(partition))


def num_blocks(partition: Sequence[int]) -> int:
    """Number of distinct block IDs in `partition`."""
    return len(set(partition))


def block_domains(
    partition: Sequence[int], sizes: Sequence[int]
) -> dict[int, int]:
    """Map block ID → common domain size of all positions in that block.
    Raises ValueError if any block contains positions of unequal size.
    """
    domains: dict[int, int] = {}
    for position, block in enumerate(partition):
        domain = sizes[position]
        existing = domains.get(block)
        if existing is not None and existing != domain:
            raise ValueError(
                f"partition block {block} mixes dimensions {existing} and {domain}"
            )
        domains[block] = domain
    return domains


def typed_labeling_count(
    partition: Sequence[int], sizes: Sequence[int]
) -> int:
    """Number of injective concrete labelings of a typed partition.

    For each domain class with ``b_s`` blocks, contributes ``falling(n_s, b_s)``;
    multiplied across domains.
    """
    domains = block_domains(partition, sizes)
    counts_by_domain: dict[int, int] = {}
    for domain in domains.values():
        counts_by_domain[domain] = counts_by_domain.get(domain, 0) + 1

    result = 1
    for domain_size, block_count in counts_by_domain.items():
        result *= falling_factorial(domain_size, block_count)
    return result


def generate_typed_set_partitions(
    sizes: Sequence[int],
) -> list[list[int]]:
    """Enumerate all typed equality patterns over `sizes`.

    A typed partition only merges positions with equal `sizes[i]`. Returns a
    list of normalized partitions (block IDs renumbered by first appearance).
    Result is deduplicated by partition_key.

    Cached by sizes tuple — most repeated einsum calls share shapes.
    """
    return _generate_typed_set_partitions_cached(tuple(sizes))


@functools.lru_cache(maxsize=256)
def _generate_typed_set_partitions_cached(
    sizes: tuple[int, ...],
) -> list[list[int]]:
    results: list[list[int]] = []
    current: list[int] = []

    def visit(position: int, block_count: int) -> None:
        if position == len(sizes):
            results.append(normalize_partition(current))
            return
        # Try merging into each existing block of the same domain
        for block in range(block_count):
            first_position_in_block = next(
                (i for i, b in enumerate(current) if b == block), -1
            )
            if (
                first_position_in_block >= 0
                and sizes[first_position_in_block] == sizes[position]
            ):
                current.append(block)
                visit(position + 1, block_count)
                current.pop()
        # Open a fresh block
        current.append(block_count)
        visit(position + 1, block_count + 1)
        current.pop()

    visit(0, 0)

    # Deduplicate by normalized key (the JS uses Map<key, partition>).
    by_key: dict[str, list[int]] = {}
    for partition in results:
        by_key[partition_key(partition)] = partition
    return list(by_key.values())
