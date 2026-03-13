#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
CONFIG_PATH = REPO / "tile_compile_cpp" / "tile_compile.yaml"
SCHEMA_PATH = REPO / "tile_compile_cpp" / "tile_compile.schema.yaml"
OUT_MD = ROOT / "parameter_katalog.md"

PANEL_BY_TOP = {
    "run_dir": "Run Setup",
    "log_level": "Run Setup",
    "pipeline": "Pipeline",
    "output": "Output",
    "data": "Data",
    "input": "Input & Scan",
    "linearity": "Linearity",
    "calibration": "Calibration",
    "assumptions": "Assumptions",
    "normalization": "Normalization",
    "registration": "Registration",
    "dithering": "Dithering",
    "tile_denoise": "Tile Denoise",
    "chroma_denoise": "Chroma Denoise",
    "global_metrics": "Global Metrics",
    "tile": "Tile",
    "local_metrics": "Local Metrics",
    "synthetic": "Synthetic",
    "debayer": "Debayer",
    "astrometry": "Astrometry",
    "bge": "BGE",
    "pcc": "PCC",
    "stacking": "Stacking",
    "validation": "Validation",
    "runtime_limits": "Runtime Limits",
}

GUI_RUNTIME_PARAMS = [
    ("scan.input_dirs[]", "string[]", "Input & Scan", "Mehrere Eingabeordner in Reihenfolge"),
    ("scan.frames_min", "int", "Input & Scan", "Mindestanzahl Frames fuer den Scan"),
    ("scan.with_checksums", "bool", "Input & Scan", "Optionaler Checksummen-Scan"),
    ("scan.confirmed_color_mode", "enum", "Input & Scan", "Manuelle Farbmodus-Bestaetigung"),
    ("run.working_dir", "string", "Run", "Basisarbeitsverzeichnis"),
    ("run.runs_dir", "string", "Run", "Ausgabeverzeichnis fuer Runs"),
    ("run.input_subdirs[]", "string[]", "Run", "Subfolder je Input-Ordner"),
    ("run.dry_run", "bool", "Run", "Simulation ohne echte Verarbeitung"),
    ("run.pattern", "string", "Run", "Dateimuster fuer Inputdateien"),
    ("run.filter_queue[]", "object[]", "Run", "Serielle MONO-Filter-Queue (LRGB/SHO)"),
    ("run.filter_queue[].filter_name", "string", "Run", "Filterbezeichner, z. B. L/R/G/B/Ha/OIII/SII"),
    ("run.filter_queue[].input_dir", "string", "Run", "Input-Ordner fuer den Filtereintrag"),
    ("run.filter_queue[].run_label", "string", "Run", "Optionales Label/Subfolder je Filtereintrag"),
    ("run.filter_queue[].pattern", "string", "Run", "Optionales Pattern je Filtereintrag"),
    ("run.filter_queue[].enabled", "bool", "Run", "Eintrag aktiv/inaktiv"),
    ("run.filter_queue[].status", "enum", "Run Monitor", "Queue-Status: pending/running/ok/error/skipped"),
    ("run.active_filter_index", "int", "Run Monitor", "Aktueller Filterindex in der seriellen Queue"),
    ("current_run.logs_tail", "int", "Current Run", "Anzahl Event-Zeilen"),
    ("current_run.logs_filter", "string", "Current Run", "Filter fuer Event-Logs"),
    ("history.selected_run", "string", "History", "Aktuell gewaehlter Lauf"),
    ("astrometry.catalog_selection", "enum", "Astrometry", "D05/D20/D50/D80 Auswahl"),
    ("astrometry.solve_file", "string", "Astrometry", "Plate-Solve Eingabedatei"),
    ("pcc.quick.fits_path", "string", "PCC", "RGB-FITS fuer Schnelltest"),
    ("pcc.quick.wcs_path", "string", "PCC", "WCS-Datei fuer Schnelltest"),
    ("ui.locale", "enum", "Global UI", "Aktive Sprache der GUI (`de`/`en`)"),
    ("ui.active_scenarios[]", "string[]", "Parameter Studio", "Aktive Situation-Assistent-Presets"),
]

MANUAL_SHORT_HELP = {
    "registration.allow_rotation": "Erlaubt Rotationsanteile im Registrierungsmodell.",
    "registration.engine": "Waehlt die Hauptmethode fuer die Bildregistrierung.",
    "registration.star_topk": "Anzahl der zu pruefenden Top-Sterne fuer Matching.",
    "registration.reject_shift_px_min": "Untergrenze fuer Shift-Outlier-Regel in Pixeln.",
    "pcc.k_max": "Begrenzt die maximale lineare PCC-Gain-Verstaerkung.",
    "pcc.mag_bright_limit": "Obergrenze fuer sehr helle Sterne im PCC-Fit.",
    "bge.fit.rbf_lambda": "Regularisierung des RBF-Fits gegen Ueberschwingen.",
    "bge.sample_quantile": "Quantil fuer robuste Hintergrund-Samples.",
    "assumptions.frames_reduced_threshold": "Schwelle fuer Reduced-Mode statt Full-Mode.",
    "synthetic.clustering.cluster_count_range": "Min/Max-Anzahl fuer Synthetic-Cluster.",
}


def normalize_value(v):
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=True)


def scalar_type(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def flatten(prefix: str, node, out: list[tuple[str, str, str]]):
    if isinstance(node, dict):
        for k, v in node.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            flatten(new_prefix, v, out)
        return

    out.append((prefix, scalar_type(node), normalize_value(node)))


def flatten_schema(prefix: str, node, out: dict[str, dict]):
    if not isinstance(node, dict):
        return

    if "properties" in node and isinstance(node["properties"], dict):
        for k, child in node["properties"].items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            flatten_schema(child_prefix, child, out)
        return

    if prefix:
        out[prefix] = node


def panel_for_path(path: str) -> str:
    top = path.split(".", 1)[0]
    return PANEL_BY_TOP.get(top, "Misc")


def clean_text(s: str) -> str:
    s = " ".join(s.split())
    if len(s) > 140:
        return s[:137].rstrip() + "..."
    return s


def auto_short_help(path: str) -> str:
    key = path.split(".")[-1]
    context = path.rsplit(".", 1)[0] if "." in path else "global"
    key_words = key.replace("_", " ")
    context_words = context.replace(".", " > ").replace("_", " ")

    if key == "enabled":
        return f"Aktiviert oder deaktiviert den Bereich {context_words}."
    if key.endswith("_dir"):
        return f"Pfad zum Verzeichnis fuer {context_words}."
    if key.endswith("_bin"):
        return f"Pfad zur ausfuehrbaren Datei fuer {context_words}."
    if key.endswith("_range"):
        return f"Legt den Wertebereich fuer {context_words} fest."
    if key.endswith("_limit"):
        return f"Grenzwert fuer {context_words}."
    if key == "pattern" or key.endswith("_pattern"):
        return f"Datei- oder Suchmuster fuer {context_words}."
    if key.startswith("min_"):
        return f"Mindestwert fuer {key_words[4:]} im Bereich {context_words}."
    if key.startswith("max_"):
        return f"Hoechstwert fuer {key_words[4:]} im Bereich {context_words}."
    if key.startswith("use_"):
        return f"Schaltet die Nutzung von {key_words[4:]} in {context_words}."
    if "sigma" in key:
        return f"Steuert die Sigma-Schwelle fuer {context_words}."
    if "frames" in key:
        return f"Steuert die Frame-Anforderung in {context_words}."
    if "radius" in key:
        return f"Steuert den Radius fuer {context_words}."
    if "weight" in key:
        return f"Gewichtung fuer den Teilbereich {context_words}."
    if key == "mode" or key.endswith("_mode"):
        return f"Waehlt den Modus fuer den Bereich {context_words}."
    return f"Steuert {key_words} im Bereich {context_words}."


def short_help_for(path: str, schema_map: dict[str, dict]) -> str:
    if path in MANUAL_SHORT_HELP:
        return MANUAL_SHORT_HELP[path]

    schema_node = schema_map.get(path, {})
    desc = schema_node.get("description")
    if isinstance(desc, str) and desc.strip():
        return clean_text(desc)

    return auto_short_help(path)


def scenario_tags_for(path: str) -> str:
    tags: list[str] = []

    if path.startswith("registration.") or path.startswith("dithering."):
        tags.extend(["Alt/Az", "Starke Rotation"])
    if path.startswith("pcc."):
        tags.append("Helle Sterne")
    if path.startswith("bge."):
        tags.append("Starker Gradient")
    if path.startswith("assumptions.") or path.startswith("synthetic."):
        tags.append("Wenige Frames")
    if path.startswith("stacking.sigma_clip") or path.startswith("stacking.cosmetic"):
        tags.append("Helle Sterne")
    if path.startswith("runtime_limits."):
        tags.append("Langlauf/Batch")

    dedup: list[str] = []
    for t in tags:
        if t not in dedup:
            dedup.append(t)
    return ", ".join(dedup) if dedup else "-"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))

    rows: list[tuple[str, str, str]] = []
    flatten("", cfg, rows)
    rows.sort(key=lambda r: r[0])

    schema_map: dict[str, dict] = {}
    flatten_schema("", schema, schema_map)

    lines: list[str] = []
    lines.append("# GUI 2 Parameter-Katalog")
    lines.append("")
    lines.append("Quelle: `tile_compile_cpp/tile_compile.yaml`, `tile_compile_cpp/tile_compile.schema.yaml` und GUI-Laufzeitparameter aus `gui_cpp`.")
    lines.append("")
    lines.append("## 1) Konfigurationsparameter (YAML)")
    lines.append("")
    lines.append("| Parameter | Typ | Default | Kurz-Erklaerung | Szenario-Hinweis | Zielbereich GUI 2 |")
    lines.append("|---|---|---|---|---|---|")
    for path, typ, value in rows:
        value_one_line = value.replace("\n", " ").replace("|", "\\|")
        short_help = short_help_for(path, schema_map).replace("|", "\\|")
        scenarios = scenario_tags_for(path)
        panel_cell = panel_for_path(path)
        lines.append(
            f"| `{path}` | `{typ}` | `{value_one_line}` | {short_help} | {scenarios} | {panel_cell} |"
        )

    lines.append("")
    lines.append("## 2) GUI-Laufzeitparameter (nicht im YAML)")
    lines.append("")
    lines.append("| Parameter | Typ | GUI-Bereich | Kurz-Erklaerung |")
    lines.append("|---|---|---|---|")
    for path, typ, panel, desc in GUI_RUNTIME_PARAMS:
        lines.append(f"| `{path}` | `{typ}` | {panel} | {desc} |")

    lines.append("")
    lines.append("## 3) Situation Assistant (Pflicht in Linie B)")
    lines.append("")
    lines.append("- Alt/Az")
    lines.append("- Starke Rotation")
    lines.append("- Helle Sterne im Feld")
    lines.append("- Wenige Frames / kurze Session")
    lines.append("- Starker Hintergrundgradient")
    lines.append("- Details siehe `szenario_empfehlungen.md`.")

    lines.append("")
    lines.append("## 4) Eingabeprinzip in GUI 2")
    lines.append("")
    lines.append("- Jeder Parameter ist sowohl per Formular als auch als YAML sichtbar.")
    lines.append("- Suchleiste filtert ueber alle Parameterpfade inkl. verschachtelter Keys.")
    lines.append("- Guardrails markieren Werte ausserhalb empfohlener Bereiche sofort inline.")
    lines.append("- Jeder Parameter hat eine Kurz-Erklaerung im Explain-Panel.")
    lines.append("- Presets koennen auf Teilbereiche angewendet werden (z. B. nur `bge.*` oder `pcc.*`).")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
