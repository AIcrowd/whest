"""Tests for _partition.py — port of partition/typedPartitions.js (Task 5: basic utilities)."""

import pytest

from flopscope._accumulation._partition import (
    block_domains,
    falling_factorial,
    generate_typed_set_partitions,
    normalize_partition,
    num_blocks,
    partition_key,
    typed_labeling_count,
)


def test_falling_factorial_base_cases():
    assert falling_factorial(5, 0) == 1
    assert falling_factorial(5, 1) == 5
    assert falling_factorial(5, 2) == 20  # 5 * 4
    assert falling_factorial(5, 3) == 60  # 5 * 4 * 3


def test_falling_factorial_zero_when_m_exceeds_n():
    assert falling_factorial(1, 2) == 0
    assert falling_factorial(3, 5) == 0


def test_falling_factorial_rejects_negative_m():
    with pytest.raises(ValueError, match="negative m"):
        falling_factorial(5, -1)


def test_normalize_partition_renumbers_blocks_by_first_appearance():
    assert normalize_partition([2, 5, 2, 5]) == [0, 1, 0, 1]
    assert normalize_partition([0, 1, 2]) == [0, 1, 2]
    assert normalize_partition([3, 3, 3]) == [0, 0, 0]


def test_partition_key_uses_normalized_form():
    assert partition_key([2, 5, 2, 5]) == "0|1|0|1"
    assert partition_key([0, 1, 0, 1]) == "0|1|0|1"  # already normalized


def test_num_blocks_counts_distinct_block_ids():
    assert num_blocks([0, 1, 0, 1]) == 2
    assert num_blocks([0, 1, 2, 3]) == 4
    assert num_blocks([0, 0, 0]) == 1


def test_block_domains_groups_by_size():
    # Partition [0, 1, 0, 1] over sizes [3, 5, 3, 5] → block 0 has size 3, block 1 has size 5
    domains = block_domains([0, 1, 0, 1], (3, 5, 3, 5))
    assert domains == {0: 3, 1: 5}


def test_block_domains_rejects_mixed_sizes():
    # Block 0 mixes positions of sizes 3 and 5 — invalid typed partition.
    with pytest.raises(ValueError, match="mixes dimensions"):
        block_domains([0, 0, 1], (3, 5, 4))


def test_typed_labeling_count_uniform_sizes():
    # Two blocks, both domain 5: 5 * 4 = 20
    assert typed_labeling_count([0, 1], (5, 5)) == 20


def test_typed_labeling_count_mixed_domains():
    # Two blocks of domain 3 (need 3*2=6) and one block of domain 5 (need 5)
    # Partition [0,1,2] over (3,3,5): block 0 size 3, block 1 size 3, block 2 size 5
    # countsByDomain: {3: 2, 5: 1}
    # falling(3, 2) * falling(5, 1) = 6 * 5 = 30
    assert typed_labeling_count([0, 1, 2], (3, 3, 5)) == 30


def test_generate_typed_set_partitions_disjoint_when_sizes_differ():
    # Sizes (2, 3) — different domains can't merge → only the discrete partition.
    partitions = generate_typed_set_partitions((2, 3))
    assert partitions == [[0, 1]]


def test_generate_typed_set_partitions_full_when_sizes_match():
    # Sizes (4, 4) — both partitions of 2 are valid.
    partitions = generate_typed_set_partitions((4, 4))
    keys = sorted(partition_key(p) for p in partitions)
    assert keys == ["0|0", "0|1"]


def test_generate_typed_set_partitions_three_position_uniform():
    # Sizes (n, n, n) — all 5 set partitions of {1,2,3} (Bell(3) = 5).
    partitions = generate_typed_set_partitions((4, 4, 4))
    keys = sorted(partition_key(p) for p in partitions)
    assert keys == ["0|0|0", "0|0|1", "0|1|0", "0|1|1", "0|1|2"]


def test_generate_typed_set_partitions_heterogeneous_three():
    # Sizes (3, 3, 5) — block of size-3 positions can merge or not; size-5 stands alone.
    # Possible partitions: (0,1,2)=all-distinct, (0,0,1)=first two merge.
    partitions = generate_typed_set_partitions((3, 3, 5))
    keys = sorted(partition_key(p) for p in partitions)
    assert keys == ["0|0|1", "0|1|2"]
