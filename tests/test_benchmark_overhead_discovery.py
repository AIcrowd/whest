from benchmarks.overhead.discovery import classify_public_operations


def test_classification_records_unclassified_operations_explicitly():
    result = classify_public_operations()
    assert "unclassified" in result
    assert isinstance(result["unclassified"], list)


def test_operator_surface_cases_are_present_for_key_dunders():
    result = classify_public_operations()
    operator_cases = {
        entry["op_name"]
        for entry in result["benchmarked"]
        if entry["surface"] == "operator"
    }
    assert "add" in operator_cases
    assert "matmul" in operator_cases


def test_stats_surface_entries_are_accounted_for():
    result = classify_public_operations()
    entries = (
        result["benchmarked"]
        + result["excluded"]
        + result["unsupported"]
        + result["unclassified"]
    )
    assert any(
        entry["surface"] == "stats"
        and entry["qualified_name"].endswith((".pdf", ".cdf", ".ppf"))
        for entry in entries
    )


def test_alias_exports_remain_visible_per_surface():
    result = classify_public_operations()
    matmul_exports = {
        entry["qualified_name"]
        for entry in result["inventory"]
        if entry["op_name"] == "matmul"
    }
    assert "whest.matmul" in matmul_exports
    assert "whest.linalg.matmul" in matmul_exports
