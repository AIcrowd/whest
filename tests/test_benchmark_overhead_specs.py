from benchmarks.overhead.specs import BenchmarkCase, seed_cases


def test_seed_cases_include_api_and_operator_surfaces():
    cases = seed_cases()
    surfaces = {(case.op_name, case.surface) for case in cases}
    assert ("add", "api") in surfaces
    assert ("add", "operator") in surfaces
    assert ("matmul", "api") in surfaces
    assert ("matmul", "operator") in surfaces


def test_seed_cases_have_required_fields():
    cases = seed_cases()
    assert cases

    for case in cases:
        assert isinstance(case, BenchmarkCase)
        assert case.case_id
        assert case.op_name in {"add", "matmul"}
        assert case.family
        assert case.surface in {"api", "operator"}
        assert case.dtype == "float64"
        assert case.size_name in {"tiny", "medium"}
        assert case.startup_mode
        assert case.source_file
        assert callable(case.numpy_factory)
        assert callable(case.whest_factory)
