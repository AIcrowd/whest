from benchmarks.overhead.specs import (
    BenchmarkCase,
    documented_operations,
    full_cases,
    seed_cases,
)


def test_seed_cases_include_api_and_operator_surfaces():
    cases = seed_cases()
    surfaces = {(case.op_name, case.surface) for case in cases}
    assert ("add", "api") in surfaces
    assert ("add", "operator") in surfaces
    assert ("matmul", "api") in surfaces
    assert ("matmul", "operator") in surfaces
    assert len({case.case_id for case in cases}) == len(cases)
    assert {
        case.case_id
        for case in cases
    } == {
        "add-api-tiny",
        "add-api-medium",
        "add-operator-tiny",
        "add-operator-medium",
        "matmul-api-tiny",
        "matmul-api-medium",
        "matmul-operator-tiny",
        "matmul-operator-medium",
    }
    assert {
        case.op_name: case.family for case in cases if case.surface == "api"
    }["add"] == "pointwise"
    assert {
        case.op_name: case.family for case in cases if case.surface == "api"
    }["matmul"] == "contractions"
    assert {
        case.op_name: case.qualified_name
        for case in cases
        if case.surface == "api"
    } == {
        "add": "whest.add",
        "matmul": "whest.matmul",
    }
    assert all(case.qualified_name is None for case in cases if case.surface == "operator")


def test_seed_cases_have_required_fields():
    cases = seed_cases()
    assert cases

    for case in cases:
        assert isinstance(case, BenchmarkCase)
        assert case.case_id
        assert case.op_name in {"add", "matmul"}
        assert case.qualified_name in {"whest.add", "whest.matmul", None}
        assert case.family
        assert case.surface in {"api", "operator"}
        assert case.dtype == "float64"
        assert case.size_name in {"tiny", "medium"}
        assert case.startup_mode
        assert case.source_file
        assert case.operand_shapes
        assert all(isinstance(shape, tuple) for shape in case.operand_shapes)
        assert callable(case.numpy_factory)
        assert callable(case.whest_factory)


def test_documented_operations_include_generated_and_profile_missing_rows():
    operations = documented_operations()
    by_slug = {entry["slug"]: entry for entry in operations}

    assert by_slug["add"]["generation_status"] == "generated"
    assert by_slug["mean"]["generation_status"] == "generated"
    assert by_slug["sort"]["generation_status"] == "generated"
    assert by_slug["fft-fft"]["generation_status"] == "generated"
    assert by_slug["fft-fftshift"]["generation_status"] == "generated"
    assert by_slug["linalg-cholesky"]["generation_status"] == "generated"
    assert by_slug["linalg-matrix_power"]["generation_status"] == "generated"
    assert by_slug["random-beta"]["generation_status"] == "generated"
    assert by_slug["random-power"]["profile_kind"] == "random_call"
    assert by_slug["random-bytes"]["generation_status"] == "generated"
    assert by_slug["random-default_rng"]["generation_status"] == "excluded"
    assert by_slug["asarray"]["generation_status"] == "generated"
    assert by_slug["frombuffer"]["generation_status"] == "excluded"
    assert by_slug["zeros"]["generation_status"] == "generated"
    assert by_slug["stats-cauchy-cdf"]["generation_status"] == "unsupported"
    assert by_slug["polyfit"]["generation_status"] == "generated"
    assert by_slug["bitwise_and"]["generation_status"] == "generated"
    assert by_slug["angle"]["generation_status"] == "generated"
    assert by_slug["linalg-matmul"]["family"] == "linalg"
    assert by_slug["apply_along_axis"]["generation_status"] == "profile_missing"
    assert by_slug["add"]["qualified_name"] == "whest.add"
    assert by_slug["sort"]["source_file"] == "src/whest/_sorting_ops.py"


def test_full_cases_expand_docs_inventory_with_tiny_and_medium_profiles():
    cases = full_cases()
    case_ids = {case.case_id for case in cases}

    assert "add-api-tiny" in case_ids
    assert "add-api-medium" in case_ids
    assert "mean-api-tiny" in case_ids
    assert "mean-api-medium" in case_ids
    assert "sort-api-tiny" in case_ids
    assert "sort-api-medium" in case_ids
    assert "fft-fft-api-tiny" in case_ids
    assert "fft-fftshift-api-tiny" in case_ids
    assert "linalg-cholesky-api-medium" in case_ids
    assert "linalg-matrix_power-api-medium" in case_ids
    assert "random-beta-api-tiny" in case_ids
    assert "random-bytes-api-tiny" in case_ids
    assert "asarray-api-tiny" in case_ids
    assert "zeros-api-tiny" in case_ids
    assert "polyfit-api-medium" in case_ids
    assert "bitwise_and-api-tiny" in case_ids
    assert "angle-api-tiny" in case_ids

    add_cases = [case for case in cases if case.op_name == "add" and case.surface == "api"]
    assert {case.size_name for case in add_cases} == {"tiny", "medium"}
    assert all(case.slug == "add" for case in add_cases)
