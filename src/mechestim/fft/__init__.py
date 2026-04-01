"""FFT submodule stub for mechestim.

All numpy.fft functions are currently blacklisted because their
O(n log n) cost model has not been implemented.
"""
from mechestim._registry import make_module_getattr as _make_module_getattr

__getattr__ = _make_module_getattr(module_prefix="fft.", module_label="mechestim.fft")
