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
    stats_entries = [
        entry
        for entry in result["inventory"]
        if entry["surface"] == "stats" and entry["op_name"] == "pdf"
    ]
    qualified_names = {entry["qualified_name"] for entry in stats_entries}
    assert "whest.stats.norm.pdf" in qualified_names
    assert "whest.stats.uniform.pdf" in qualified_names
    assert all(entry["status"] == "unclassified" for entry in stats_entries)


def test_alias_exports_remain_visible_per_surface():
    result = classify_public_operations()
    matmul_statuses = {
        entry["qualified_name"]: entry["status"]
        for entry in result["inventory"]
        if entry["op_name"] == "matmul"
    }
    assert matmul_statuses["whest.matmul"] == "benchmarked"
    assert matmul_statuses["whest.linalg.matmul"] == "unclassified"
