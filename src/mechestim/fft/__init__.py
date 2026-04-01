"""FFT submodule for mechestim."""
from mechestim.fft._transforms import (  # noqa: F401
    fft, ifft, rfft, irfft,
    fft2, ifft2, rfft2, irfft2,
    fftn, ifftn, rfftn, irfftn,
    hfft, ihfft,
    fft_cost, rfft_cost, fftn_cost, rfftn_cost, hfft_cost,
)
from mechestim.fft._free import (  # noqa: F401
    fftfreq, rfftfreq, fftshift, ifftshift,
)
from mechestim._registry import make_module_getattr as _make_module_getattr

__all__ = [
    "fft", "ifft", "rfft", "irfft",
    "fft2", "ifft2", "rfft2", "irfft2",
    "fftn", "ifftn", "rfftn", "irfftn",
    "hfft", "ihfft",
    "fftfreq", "rfftfreq", "fftshift", "ifftshift",
]

__getattr__ = _make_module_getattr(module_prefix="fft.", module_label="mechestim.fft")
