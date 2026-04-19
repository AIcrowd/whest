from benchmarks.overhead.specs import BenchmarkCase, seed_cases


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
