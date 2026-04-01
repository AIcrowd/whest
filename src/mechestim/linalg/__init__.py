"""Linear algebra submodule for mechestim."""
from mechestim.linalg._svd import svd  # noqa: F401

__all__ = ["svd"]

def __getattr__(name):
    raise AttributeError(
        f"mechestim.linalg does not provide '{name}'. "
        f"Currently supported: svd. "
        f"Request new ops at https://github.com/AIcrowd/mechestim/issues"
    )
