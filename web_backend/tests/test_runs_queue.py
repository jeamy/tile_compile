from __future__ import annotations

from app.services.queue_utils import extract_queue_specs, normalize_queue_item


def test_normalize_queue_item() -> None:
    assert normalize_queue_item("/data/L") == {"input_dir": "/data/L"}
    assert normalize_queue_item({"input_dir": "/data/R", "filter": "R"}) == {
        "input_dir": "/data/R",
        "filter": "R",
    }
    assert normalize_queue_item({"filter": "G"}) is None
    assert normalize_queue_item(42) is None


def test_extract_queue_specs_prefers_explicit_queue() -> None:
    payload = {
        "queue": [{"input_dir": "/data/L", "filter": "L"}, {"input_dir": "/data/R", "filter": "R"}],
        "input_dirs": ["/ignored/a", "/ignored/b"],
    }
    queue = extract_queue_specs(payload)
    assert len(queue) == 2
    assert queue[0]["filter"] == "L"
    assert queue[1]["filter"] == "R"


def test_extract_queue_specs_from_input_dirs() -> None:
    payload = {"input_dirs": ["/data/L", {"input_dir": "/data/R", "filter": "R"}]}
    queue = extract_queue_specs(payload)
    assert len(queue) == 2
    assert queue[0]["input_dir"] == "/data/L"
    assert queue[1]["filter"] == "R"
