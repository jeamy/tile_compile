# Tile Compile – GUI Controls Mapping

**Zweck:** Abbildung von `tile_compile.schema.yaml` auf konkrete GUI-Controls

---

## 1. Grundregeln

* `editable: false` → read-only Anzeige
* `editable: confirm` → einmal bestätigbar vor Run
* `const:` → gesperrt, nicht editierbar
* Validierung erfolgt **vor Run-Start**

---

## 2. Mapping-Tabelle

### Pipeline

| Feld                   | Typ     | Control  | Hinweise           |
| ---------------------- | ------- | -------- | ------------------ |
| pipeline.mode          | enum    | Dropdown | production / test  |
| pipeline.abort_on_fail | boolean | Toggle   | sofortiger Abbruch |

---

### Input (automatisch)

| Feld                  | Typ  | Control        | Hinweise           |
| --------------------- | ---- | -------------- | ------------------ |
| input.image_width     | int  | Label          | aus FITS (NAXIS1)  |
| input.image_height    | int  | Label          | aus FITS (NAXIS2)  |
| input.frames_detected | int  | Label          | Dateisystem        |
| input.color_mode      | enum | Confirm-Select | einmal bestätigbar |

---

### Input Constraints

| Feld                         | Typ | Control       | Hinweise               |
| ---------------------------- | --- | ------------- | ---------------------- |
| input_constraints.frames_min | int | Numeric Input | Hard-Abbruch unterhalb |

---

### Normalisierung

| Feld                      | Typ  | Control  | Hinweise            |
| ------------------------- | ---- | -------- | ------------------- |
| normalization.enabled     | bool | Label    | const=true          |
| normalization.mode        | enum | Dropdown | background / median |
| normalization.per_channel | bool | Toggle   | OSC getrennt        |

---

### Registrierung

| Feld                                     | Typ    | Control       | Hinweise            |
| ---------------------------------------- | ------ | ------------- | ------------------- |
| registration.engine                      | enum   | Dropdown      | siril               |
| registration.reference                   | enum   | Dropdown      | auto                |
| registration.output_dir                  | string | Text Input    | relativ zum Projekt |
| registration.registered_filename_pattern | string | Text Input    | Formatstring        |
| registration.min_star_matches            | int    | Numeric Input | Validierung         |
| registration.allow_rotation              | bool   | Toggle        | Alt-Az              |

---

### Globale Metriken

| Feld                     | Typ   | Control     | Hinweise         |
| ------------------------ | ----- | ----------- | ---------------- |
| global_metrics.weights.* | float | Slider      | Summe = 1 prüfen |
| global_metrics.clamp     | array | Range Input | [-3, +3] Default |

---

### Tile-Geometrie

| Feld                  | Typ   | Control       | Hinweise   |
| --------------------- | ----- | ------------- | ---------- |
| tile.size_factor      | int   | Numeric Input | Default 32 |
| tile.min_size         | int   | Numeric Input | ≥ 64       |
| tile.max_divisor      | int   | Numeric Input | Default 6  |
| tile.overlap_fraction | float | Slider        | max 0.5    |
| tile.star_min_count   | int   | Numeric Input | ≥ 0        |

---

### Lokale Metriken

| Feld                     | Typ   | Control     | Hinweise  |
| ------------------------ | ----- | ----------- | --------- |
| local_metrics.clamp      | array | Range Input | [-3,+3]   |
| star_mode.fwhm_transform | enum  | Label       | const=log |
| star_mode.weights.*      | float | Slider      | Summe = 1 |
| structure_mode.metric    | enum  | Label       | const     |
| structure_mode.*_weight  | float | Slider      | Summe = 1 |

---

### Synthetische Frames

| Feld                 | Typ | Control       | Hinweise     |
| -------------------- | --- | ------------- | ------------ |
| synthetic.frames_min | int | Numeric Input | ≥ 1          |
| synthetic.frames_max | int | Numeric Input | ≥ frames_min |

---

### Rekonstruktion

| Feld                              | Typ  | Control | Hinweise |
| --------------------------------- | ---- | ------- | -------- |
| reconstruction.weighting_function | enum | Label   | const    |
| reconstruction.window_function    | enum | Label   | const    |

---

### Stacking

| Feld                 | Typ    | Control    | Hinweise |
| -------------------- | ------ | ---------- | -------- |
| stacking.engine      | enum   | Dropdown   | siril    |
| stacking.method      | enum   | Label      | const    |
| stacking.output_file | string | Text Input | FITS     |

---

### Validierung

| Feld         | Typ         | Control          | Hinweise           |
| ------------ | ----------- | ---------------- | ------------------ |
| validation.* | number/bool | Numeric / Toggle | Hard-Abbruchregeln |
