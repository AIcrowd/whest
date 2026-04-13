"""FFT submodule for mechestim."""

from mechestim._registry import make_module_getattr as _make_module_getattr
from mechestim.fft._free import (  # noqa: F401
    fftfreq,
    fftshift,
    ifftshift,
    rfftfreq,
)
from mechestim.fft._transforms import (  # noqa: F401
    fft,
    fft2,
    fft_cost,
    fftn,
    fftn_cost,
    hfft,
    hfft_cost,
    ifft,
    ifft2,
    ifftn,
    ihfft,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfft_cost,
    rfftn,
    rfftn_cost,
)

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]

__getattr__ = _make_module_getattr(module_prefix="fft.", module_label="mechestim.fft")
