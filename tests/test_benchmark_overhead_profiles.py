import warnings

import numpy as np
import pytest

from benchmarks.overhead.execution import _build_case_closures
from benchmarks.overhead.profiles import materialize_case_inputs
from benchmarks.overhead.specs import BenchmarkCase


def _payload(op_name: str, profile_kind: str) -> dict[str, object]:
    return {
        "op_name": op_name,
        "size_name": "medium",
        "dtype": "float64",
        "profile_kind": profile_kind,
        "profile_params": {"op_name": op_name},
    }


@pytest.mark.parametrize("op_name", ["arccos", "arcsin"])
def test_materialize_case_inputs_bounds_inverse_trig_domains(op_name):
    (values,), _ = materialize_case_inputs(_payload(op_name, "vector_unary"))

    assert np.all(np.abs(values) <= 1.0)


@pytest.mark.parametrize(
    ("op_name", "lower_bound", "strict"),
    [
        ("arccosh", 1.0, False),
        ("log", 0.0, True),
        ("log10", 0.0, True),
        ("log1p", -1.0, True),
        ("log2", 0.0, True),
        ("sqrt", 0.0, True),
        ("reciprocal", 0.0, True),
    ],
)
def test_materialize_case_inputs_uses_valid_unary_domains(op_name, lower_bound, strict):
    (values,), _ = materialize_case_inputs(_payload(op_name, "vector_unary"))

    if strict:
        assert np.all(values > lower_bound)
    else:
        assert np.all(values >= lower_bound)


def test_materialize_case_inputs_bounds_arctanh_domain():
    (values,), _ = materialize_case_inputs(_payload("arctanh", "vector_unary"))

    assert np.all(np.abs(values) < 1.0)


@pytest.mark.parametrize(
    "op_name",
    ["divide", "true_divide", "floor_divide", "fmod", "mod", "remainder"],
)
def test_materialize_case_inputs_avoids_zero_denominators(op_name):
    _, denominator = materialize_case_inputs(_payload(op_name, "vector_binary"))[0]

    assert np.all(denominator != 0)


@pytest.mark.parametrize("op_name", ["power", "float_power"])
def test_materialize_case_inputs_uses_positive_bases_for_power_ops(op_name):
    base, exponent = materialize_case_inputs(_payload(op_name, "vector_binary"))[0]

    assert np.all(base > 0)
    assert np.all(np.isfinite(exponent))


def test_materialize_case_inputs_uses_integer_exponents_for_ldexp():
    values, exponents = materialize_case_inputs(_payload("ldexp", "vector_binary"))[0]

    assert np.all(np.isfinite(values))
    assert np.issubdtype(exponents.dtype, np.integer)


def test_materialize_case_inputs_builds_linalg_matrices_without_runtime_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (matrix,), _ = materialize_case_inputs(_payload("cholesky", "linalg_matrix"))

    assert caught == []
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize(
    ("op_name", "expected_q"),
    [
        ("percentile", 50),
        ("nanpercentile", 50),
        ("quantile", 0.5),
        ("nanquantile", 0.5),
    ],
)
def test_materialize_case_inputs_uses_keyword_q_for_percentile_families(
    op_name, expected_q
):
    args, kwargs = materialize_case_inputs(_payload(op_name, "vector_reduction"))

    assert len(args) == 1
    assert kwargs["q"] == expected_q


@pytest.mark.parametrize(
    ("op_name", "profile_kind"),
    [
        ("arccos", "vector_unary"),
        ("log", "vector_unary"),
        ("divide", "vector_binary"),
        ("power", "vector_binary"),
        ("ldexp", "vector_binary"),
        ("in1d", "set_binary"),
        ("percentile", "vector_reduction"),
    ],
)
def test_generated_profiles_run_without_runtime_warnings(op_name, profile_kind):
    case = BenchmarkCase(
        case_id=f"{op_name}-api-medium",
        op_name=op_name,
        qualified_name=f"whest.{op_name}",
        family="pointwise",
        surface="api",
        dtype="float64",
        size_name="medium",
        startup_mode="warmup",
        source_file="src/whest/_pointwise.py",
        numpy_factory=f"numpy.{op_name}",
        whest_factory=f"whest.{op_name}",
        profile_kind=profile_kind,
        profile_params={"op_name": op_name},
    )

    _, whest_callable = _build_case_closures(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        whest_callable()

    assert caught == []
