"""FLOP cost model.

Counts a fused multiply-add (FMA) as 1 operation by default. The convention
is configurable via ``flopscope.configure(fma_cost=N)`` where N is 1 or 2.
"""

from flopscope._config import get_setting


def fma_cost() -> int:
    """Return the configured FMA cost (1 by default).

    A fused multiply-add (FMA) computes ``a × b + c`` in a single hardware
    instruction. We count this as 1 operation by default, mirroring hardware
    semantics. The textbook / opt_einsum convention of 2 (one multiply +
    one add) is also available via ``flopscope.configure(fma_cost=2)``.
    """
    return int(get_setting('fma_cost'))
