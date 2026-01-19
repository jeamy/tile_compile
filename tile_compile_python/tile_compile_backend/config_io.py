from pathlib import Path


def load_config_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def save_config_text(path: str, yaml_text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml_text, encoding="utf-8")
