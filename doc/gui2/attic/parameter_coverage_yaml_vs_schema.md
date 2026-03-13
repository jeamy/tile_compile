# Parameter-Coverage: `tile_compile.yaml` vs `tile_compile.schema.yaml`

Stand: 2026-03-07

## Zusammenfassung

- Schema-Leaf-Parameter: **181**
- YAML-Leaf-Parameter: **187**
- In Schema definiert, aber im YAML nicht aktiv gesetzt: **4**
- Im YAML gesetzt, aber nicht im Schema definiert: **10**

Hinweis: Die 4 fehlenden `data.*`-Felder sind im YAML aktuell auskommentiert.

## In Schema, aber nicht aktiv in YAML

- `data.bayer_pattern`
- `data.color_mode`
- `data.image_height`
- `data.image_width`

## In YAML, aber nicht in Schema

- `run_dir`
- `log_level`
- `input.pattern`
- `input.sort`
- `input.max_frames`
- `bge.min_valid_sample_fraction_for_apply`
- `bge.min_valid_samples_for_apply`
- `pcc.apply_attenuation`
- `pcc.chroma_strength`
- `pcc.k_max`

## GUI2-Folgerung

- Der Clickdummy-Editor ist auf volle Editierbarkeit umgestellt:
  - alle Parameterpfade sind ueber Kategorien editierbar
  - Suche springt direkt zu editierbaren Feldern
- Datenbasis ist ein vereinigter Editor-Index aus Schema + YAML.
