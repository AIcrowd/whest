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
    assert any(entry["surface"] == "stats" for entry in entries)
