from benchmarks.overhead.specs import BenchmarkCase, seed_cases


def test_seed_cases_include_api_and_operator_surfaces():
    cases = seed_cases()
    surfaces = {(case.op_name, case.surface) for case in cases}
    assert ("add", "api") in surfaces
    assert ("add", "operator") in surfaces
    assert ("matmul", "api") in surfaces
    assert ("matmul", "operator") in surfaces


def test_seed_cases_have_required_fields():
    case = seed_cases()[0]
    assert isinstance(case, BenchmarkCase)
    assert case.family
    assert case.case_id
    assert case.size_name in {"tiny", "medium"}
    assert case.surface in {"api", "operator"}
    assert callable(case.numpy_factory)
    assert callable(case.whest_factory)
