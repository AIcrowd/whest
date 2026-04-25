"""flopscope.numpy.fft — FFT submodule stub (all functions blacklisted)."""

from __future__ import annotations

from flopscope._getattr import make_module_getattr

# Every fft.* function in the registry is blacklisted.
# __getattr__ provides helpful error messages.
__getattr__ = make_module_getattr("fft.", "flopscope.numpy.fft")
