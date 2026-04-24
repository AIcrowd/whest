"""FLOP cost model constants.

This codebase counts a fused multiply-add (FMA) as **1 operation**.
Hardware FMA units compute  a × b + c  in a single instruction;
counting it as 2 (one multiply + one add) is a common but arbitrary
convention we intentionally do not follow.
"""

FMA_COST: int = 1
"""Cost of one fused multiply-add operation.

Set to 1 because FMA is a single hardware instruction.
The opt_einsum / textbook convention of 2 counts the multiply
and add separately; we reject that convention.
"""
