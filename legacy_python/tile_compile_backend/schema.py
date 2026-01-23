import json
from pathlib import Path


def load_schema_json(schema_path: str | None = None) -> dict:
    if schema_path is None:
        schema_path = str(Path(__file__).resolve().parents[1] / "tile_compile.schema.json")

    p = Path(schema_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("schema must be a JSON object")
    return data
