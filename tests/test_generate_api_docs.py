from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS_INDEX_PATH = ROOT / "website" / "public" / "ops.json"
OPS_DETAIL_DIR = ROOT / "website" / "public" / "api-data" / "ops"
OP_IMPORT_MAP_PATH = ROOT / "website" / ".generated" / "op-doc-imports.ts"


def load_ops_index() -> dict:
    assert OPS_INDEX_PATH.exists(), f"ops.json not found at {OPS_INDEX_PATH}"
    return json.loads(OPS_INDEX_PATH.read_text())


def test_ops_index_exposes_detail_links_and_slugs():
    data = load_ops_index()
    operations = data["operations"]
    assert operations, "ops.json should contain at least one operation"

    sample = operations[0]
    assert sample["slug"] == sample["name"]
    assert sample["detail_href"] == f"/docs/api/ops/{sample['slug']}/"
    assert sample["detail_json_href"] == f"/api-data/ops/{sample['slug']}.json"
    assert "summary" in sample


def test_per_op_detail_payloads_exist_for_index_entries():
    data = load_ops_index()
    operations = data["operations"]
    assert OPS_DETAIL_DIR.exists(), f"per-op detail dir not found at {OPS_DETAIL_DIR}"

    sample_names = ["abs", "add", "einsum", "sum"]
    operations_by_name = {entry["name"]: entry for entry in operations}

    for name in sample_names:
        assert name in operations_by_name, f"{name} missing from ops.json"
        entry = operations_by_name[name]
        detail_path = OPS_DETAIL_DIR / f"{entry['slug']}.json"
        assert detail_path.exists(), f"missing detail payload for {name}: {detail_path}"

        payload = json.loads(detail_path.read_text())
        assert payload["schema_version"] == 1
        assert payload["slug"] == entry["slug"]
        assert payload["detail_href"] == entry["detail_href"]
        assert payload["detail_json_href"] == entry["detail_json_href"]
        assert payload["op"]["name"] == name
        assert payload["op"]["whest_ref"] == entry["whest_ref"]
        assert payload["docs"]["sections"], f"{name} detail payload should contain docs sections"


def test_generated_import_map_contains_known_op_slugs():
    assert OP_IMPORT_MAP_PATH.exists(), (
        f"generated op import map not found at {OP_IMPORT_MAP_PATH}"
    )
    source = OP_IMPORT_MAP_PATH.read_text()

    assert "export const opDocImports" in source
    assert '"abs":' in source
    assert '"einsum":' in source
