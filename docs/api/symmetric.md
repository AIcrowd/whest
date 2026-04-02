# Symmetric Tensors

First-class symmetric tensor support for automatic FLOP cost reductions.

`SymmetricTensor` is an `ndarray` subclass that carries symmetry metadata
through operations. When passed to any mechestim operation, the cost is
automatically reduced based on the number of unique elements.

See [Exploit Symmetry Savings](../how-to/exploit-symmetry.md) for usage patterns.

::: mechestim._symmetric
