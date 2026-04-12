"""Tests for version identity and runtime banner."""

import re
import warnings

import mechestim as me
from mechestim._budget import BudgetContext
from mechestim._registry import REGISTRY_META


def test_version_includes_numpy_suffix():
    assert "+np" in me.__version__
    assert re.match(r"\d+\.\d+\.\d+\+np\d+\.\d+\.\d+", me.__version__)


def test_numpy_version_attribute():
    import numpy

    assert me.__numpy_version__ == numpy.__version__


def test_numpy_pinned_attribute():
    assert me.__numpy_pinned__ == REGISTRY_META["numpy_version"]


def test_budget_context_prints_banner(capsys):
    with BudgetContext(flop_budget=1_000_000_000):
        pass
    captured = capsys.readouterr()
    assert "mechestim" in captured.err
    assert "numpy" in captured.err
    assert "backend" in captured.err


def test_budget_context_quiet_suppresses_banner(capsys):
    with BudgetContext(flop_budget=1_000_000_000, quiet=True):
        pass
    captured = capsys.readouterr()
    assert captured.err == ""


def test_numpy_supported_attribute():
    assert hasattr(me, "__numpy_supported__")
    assert ">=" in me.__numpy_supported__
    assert "<" in me.__numpy_supported__


def test_numpy_version_out_of_range_warning():
    from mechestim._version_check import check_numpy_version

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_numpy_version(">=99.0.0,<100.0.0")
        assert len(w) == 1
        assert "mechestim supports numpy" in str(w[0].message)


def test_numpy_version_in_range_no_warning():
    import numpy

    from mechestim._version_check import check_numpy_version

    major, minor = numpy.__version__.split(".")[:2]
    range_str = f">={major}.{minor}.0,<{major}.{int(minor)+1}.0"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_numpy_version(range_str)
        mechestim_warnings = [x for x in w if issubclass(x.category, me.MechEstimWarning)]
        assert len(mechestim_warnings) == 0
