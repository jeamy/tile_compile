# TLR Implementation Guide - Schritt-für-Schritt Anleitung

**Datum:** 2026-01-20  
**Status:** Bereit für manuelle Implementierung  
**Ziel:** Vollständige Integration von Tile-wise Local Registration (TLR) in die Pipeline

---

## Was bereits erledigt ist ✅

1. **TLR-Kernmodul:** `runner/tile_local_registration.py` - vollständig implementiert
2. **Config-Update:** `tile_compile.yaml` - registration.local_tiles konfiguriert
3. **Phase-Liste:** dry_run Phase-Liste aktualisiert (REGISTRATION entfernt)
4. **Migrationsplan:** `MIGRATION_TO_TLR_V4.md` - detaillierte Architektur-Dokumentation
5. **Registrierungs-Fixes:** 7 kritische Bugs behoben (in TLR integriert)

---

## Was noch zu tun ist 🔧

### KRITISCH: phases_impl.py Änderungen

Die Datei `tile_compile_python/runner/phases_impl.py` muss wie folgt geändert werden:

#### Schritt 1: REGISTRATION Phase komplett entfernen

**Zeilen zu löschen:** ~1669-2217 (ca. 550 Zeilen)

**Von:**
```python
phase_id = 1
phase_name = "REGISTRATION"
if should_skip_phase(phase_id):
    ...
elif reg_engine == "opencv_cfa":
    # Gesamte opencv_cfa Registrierung
    ...
else:
    # Siril-basierte Registrierung
    ...

reg_out_dir = outputs_dir / reg_out_name
registered_files = sorted([...])
...
```

**Zu:**
```python
# Phase 1 (REGISTRATION) removed in Methodik v4
# Registration is now performed tile-wise locally during TILE_RECONSTRUCTION_TLR

# For backward compatibility: use raw frames directly
registered_files = frames
```

**Wichtig:** Alle Hilfsfunktionen innerhalb der REGISTRATION Phase müssen ebenfalls entfernt werden:
- `_compose_affine()`
- `_warp_cfa()`
- `_warp_cfa_mask()`
- `_load_clean_and_lum01()`
- `_check_step_warp_sanity()`
- `_get_bright_centroid()`
- `_verify_warp_by_centroid()`

#### Schritt 2: Phase-IDs aktualisieren

Alle `phase_id` Zuweisungen nach der entfernten REGISTRATION Phase um 1 reduzieren:

```python
# Alt → Neu
phase_id = 2  # CHANNEL_SPLIT → phase_id = 1
phase_id = 3  # NORMALIZATION → phase_id = 2
phase_id = 4  # GLOBAL_METRICS → phase_id = 3
phase_id = 5  # TILE_GRID → phase_id = 4
phase_id = 6  # LOCAL_METRICS → phase_id = 5
phase_id = 7  # TILE_RECONSTRUCTION → phase_id = 6 (wird zu TILE_RECONSTRUCTION_TLR)
phase_id = 8  # STATE_CLUSTERING → phase_id = 7
phase_id = 9  # SYNTHETIC_FRAMES → phase_id = 8
phase_id = 10 # STACKING → phase_id = 9
phase_id = 11 # DEBAYER → phase_id = 10
phase_id = 12 # DONE → phase_id = 11
```

**Suchen und Ersetzen:**
- `phase_id = 2` → `phase_id = 1` (CHANNEL_SPLIT)
- `phase_id = 3` → `phase_id = 2` (NORMALIZATION)
- usw.

**ACHTUNG:** Auch in `should_skip_phase(phase_id)` Aufrufen und allen Referenzen!

#### Schritt 3: CHANNEL_SPLIT anpassen

**Alt (Zeile ~2220):**
```python
phase_id = 2
phase_name = "CHANNEL_SPLIT"
...
# Arbeitet auf registered_files
for reg_file in registered_files:
    ...
```

**Neu:**
```python
phase_id = 1
phase_name = "CHANNEL_SPLIT"
...
# Arbeitet auf raw frames (nach Calibration, vor Registrierung)
for frame in frames:
    ...
```

**Wichtig:** CHANNEL_SPLIT muss jetzt auf `frames` statt `registered_files` arbeiten!

#### Schritt 4: TILE_RECONSTRUCTION_TLR implementieren

**Alt (Zeile ~3130):**
```python
phase_id = 7
phase_name = "TILE_RECONSTRUCTION"
...
# Lädt registered frames und rekonstruiert Tiles
for ch in ["R", "G", "B"]:
    for f_idx, ch_file in enumerate(channel_files[ch]):
        f = fits.getdata(str(ch_file))  # Bereits registriert
        # Tile-Rekonstruktion ohne weitere Registrierung
        ...
```

**Neu:**
```python
phase_id = 6
phase_name = "TILE_RECONSTRUCTION_TLR"
if should_skip_phase(phase_id):
    phase_start(run_id, log_fp, phase_id, phase_name)
    phase_end(run_id, log_fp, phase_id, phase_name, "skipped", {"reason": "resume_from_phase", "resume_from": resume_from_phase})
else:
    phase_start(run_id, log_fp, phase_id, phase_name)
    if stop_requested(run_id, log_fp, phase_id, phase_name, stop_flag):
        return False

    # Import TLR module
    from .tile_local_registration import tile_local_reconstruct_all_channels

    # Prepare tile grid for TLR
    tile_size_recon = grid_cfg.get("tile_size", tile_size)
    overlap_px_recon = grid_cfg.get("overlap_px", int(overlap * tile_size_recon))
    step_recon = tile_size_recon - overlap_px_recon
    
    # Load first frame to get dimensions
    first_frame = fits.getdata(str(channel_files["R"][0])).astype("float32", copy=False)
    h0, w0 = first_frame.shape[:2]
    del first_frame
    
    # Compute tile grid
    n_tiles_y = max(1, (h0 - tile_size_recon) // step_recon + 1)
    n_tiles_x = max(1, (w0 - tile_size_recon) // step_recon + 1)
    
    tiles = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y_start = ty * step_recon
            x_start = tx * step_recon
            y_end = min(y_start + tile_size_recon, h0)
            x_end = min(x_start + tile_size_recon, w0)
            tiles.append({
                "y_start": y_start,
                "y_end": y_end,
                "x_start": x_start,
                "x_end": x_end,
                "tile_idx": len(tiles)
            })
    
    tile_grid = {"tiles": tiles}
    
    # Prepare frames by channel
    frames_by_channel = {
        "R": channel_files.get("R", []),
        "G": channel_files.get("G", []),
        "B": channel_files.get("B", [])
    }
    
    # Prepare weights
    weights_global = {}
    weights_local = {}
    
    for ch in ["R", "G", "B"]:
        if ch in channel_metrics:
            gfc = np.asarray(channel_metrics[ch]["global"].get("G_f_c") or [], dtype=np.float32)
            weights_global[ch] = gfc
            
            tiles_data = channel_metrics[ch].get("tiles", {})
            l_local = tiles_data.get("L_local", [])
            
            # Convert L_local to array format
            if l_local and len(l_local) > 0:
                num_frames = len(l_local)
                num_tiles = len(l_local[0]) if isinstance(l_local[0], list) else 0
                L_array = np.zeros((num_frames, num_tiles), dtype=np.float32)
                for f_idx, l_f in enumerate(l_local):
                    if isinstance(l_f, list):
                        for t_idx, val in enumerate(l_f):
                            if t_idx < num_tiles:
                                L_array[f_idx, t_idx] = float(val)
                weights_local[ch] = L_array
            else:
                # Fallback: uniform weights
                num_frames = len(frames_by_channel[ch])
                num_tiles = len(tiles)
                weights_local[ch] = np.ones((num_frames, num_tiles), dtype=np.float32)
    
    # Call TLR
    print(f"[TLR] Starting tile-wise local registration and reconstruction")
    reconstructed = tile_local_reconstruct_all_channels(
        frames_by_channel,
        tile_grid,
        weights_global,
        weights_local,
        cfg
    )
    
    # Save reconstructed channels
    hdr0 = None
    try:
        hdr0 = fits.getheader(str(frames[0]), ext=0)
    except Exception:
        pass
    
    for ch in ["R", "G", "B"]:
        if ch in reconstructed:
            out_path = outputs_dir / f"reconstructed_{ch}.fits"
            fits.writeto(str(out_path), reconstructed[ch], header=hdr0, overwrite=True)
            print(f"[TLR] Saved reconstructed channel {ch} to {out_path}")
    
    phase_end(run_id, log_fp, phase_id, phase_name, "ok", {
        "method": "tile_local_registration",
        "channels": list(reconstructed.keys()),
        "num_tiles": len(tiles)
    })
```

#### Schritt 5: STACKING Phase anpassen

**Alt:**
```python
stack_input_dir = work_dir / reg_out_name  # registered/
stack_files = sorted(stack_input_dir.glob(stack_input_pattern))
```

**Neu:**
```python
# Stacking arbeitet direkt auf rekonstruierten Kanälen
stack_files = []
for ch in ["R", "G", "B"]:
    recon_file = outputs_dir / f"reconstructed_{ch}.fits"
    if recon_file.exists():
        stack_files.append(recon_file)

if not stack_files:
    phase_end(run_id, log_fp, phase_id, phase_name, "error", {"error": "no reconstructed channels found for stacking"})
    return False
```

#### Schritt 6: Entferne veraltete Config-Variablen

**Zu entfernen aus dem Code:**
- `reg_engine` Checks
- `reg_out_name` Verwendung
- `reg_pattern` Verwendung
- `allow_rotation` (wird jetzt in TLR intern gehandhabt)
- `min_star_matches` (wird jetzt in TLR intern gehandhabt)

**Zeilen ~980-1010:** Config-Parsing für Registration kann vereinfacht werden:
```python
# Alt:
registration_cfg = cfg.get("registration") if isinstance(cfg.get("registration"), dict) else {}
reg_engine = str(registration_cfg.get("engine") or "")
reg_script_cfg = registration_cfg.get("siril_script")
...

# Neu:
registration_cfg = cfg.get("registration") if isinstance(cfg.get("registration"), dict) else {}
# TLR config wird direkt in tile_local_registration.py gelesen
```

---

## Schema-Update

**Datei:** `tile_compile_python/tile_compile.schema.yaml`

### Zu entfernen:
```yaml
registration:
  engine:
    type: string
    enum: [opencv_cfa, siril]
  reference:
    type: string
  allow_rotation:
    type: boolean
  min_star_matches:
    type: integer
  border_mode:
    type: string
  border_value:
    type: number
  expand_canvas:
    type: boolean
  output_dir:
    type: string
  registered_filename_pattern:
    type: string
  siril_script:
    type: string
```

### Hinzuzufügen:
```yaml
registration:
  local_tiles:
    type: object
    properties:
      model:
        type: string
        enum: [translation]
        default: translation
      ecc_cc_min:
        type: number
        minimum: 0.0
        maximum: 1.0
        default: 0.2
      min_valid_frames:
        type: integer
        minimum: 1
        default: 10
      reference_method:
        type: string
        enum: [median_time, min_gradient]
        default: median_time
      max_tile_size:
        type: integer
        minimum: 32
        maximum: 512
        default: 128
```

---

## Tests

**Datei:** `tests/test_tile_local_registration.py` (neu erstellen)

```python
import pytest
import numpy as np
from pathlib import Path
from runner.tile_local_registration import (
    select_tile_reference_frame,
    estimate_tile_local_translation,
    tile_local_register_and_reconstruct
)

def test_select_tile_reference_median():
    """Test median_time reference selection."""
    frames = [Path(f"frame_{i}.fits") for i in range(10)]
    tile_bounds = (0, 100, 0, 100)
    ref_idx = select_tile_reference_frame(frames, tile_bounds, method="median_time")
    assert ref_idx == 5  # Median of 10 frames

def test_estimate_tile_local_translation_identity():
    """Test translation estimation with identical tiles."""
    tile = np.random.rand(64, 64).astype(np.float32)
    warp, cc = estimate_tile_local_translation(tile, tile, ecc_cc_min=0.2)
    
    assert warp is not None
    assert cc > 0.9  # High correlation for identical tiles
    
    # Check it's identity transform
    assert abs(warp[0, 0] - 1.0) < 0.01
    assert abs(warp[1, 1] - 1.0) < 0.01
    assert abs(warp[0, 2]) < 0.5  # Small translation
    assert abs(warp[1, 2]) < 0.5

def test_estimate_tile_local_translation_shifted():
    """Test translation estimation with known shift."""
    tile_ref = np.random.rand(64, 64).astype(np.float32)
    
    # Shift by (5, 3) pixels
    tile_mov = np.zeros_like(tile_ref)
    tile_mov[3:, 5:] = tile_ref[:-3, :-5]
    
    warp, cc = estimate_tile_local_translation(tile_mov, tile_ref, ecc_cc_min=0.2)
    
    assert warp is not None
    assert cc > 0.5
    
    # Check translation is approximately (-5, -3)
    assert abs(warp[0, 2] - (-5.0)) < 2.0
    assert abs(warp[1, 2] - (-3.0)) < 2.0
```

---

## Dokumentation v4

### Methodik v4 (DE)

**Datei:** `doc/tile_basierte_qualitatsrekonstruktion_methodik_v_4.md`

**Hauptänderungen gegenüber v3:**

1. **Abschnitt 0: Motivation für v4**
   ```markdown
   ## 0. Motivation für v4
   
   Methodik v3 definierte globale Registrierung als separaten Schritt.
   
   Methodik v4 ersetzt dies durch **Tile-weise lokale Registrierung (TLR)**:
   - Keine globale Referenz mehr
   - Jedes Tile registriert unabhängig
   - Geeignet für Alt/Az-Montierungen mit Feldrotation
   - Auch für EQ-Montierungen optimal
   ```

2. **Abschnitt 2: Annahmen**
   ```markdown
   ### 2.1 Harte Annahmen
   
   - Daten sind **linear**
   - **keine Frame-Selektion**
   - Verarbeitung **kanalgetrennt**
   - Pipeline ist **streng linear**
   - **Tile-weise lokale Registrierung** (neu in v4)
   ```

3. **Neuer Abschnitt: Tile-weise lokale Registrierung**
   ```markdown
   ## 3. Tile-weise lokale Registrierung (TLR)
   
   ### 3.1 Grundprinzip
   
   Statt globaler Transformation A_f für jedes Frame f:
   
   A_f,t für jedes Frame f und Tile t
   
   ### 3.2 Bewegungsmodell
   
   Translation-only:
   A_f,t = [[1, 0, Δx_f,t], [0, 1, Δy_f,t]]
   
   Rotation wird durch lokale Approximation absorbiert.
   
   ### 3.3 Algorithmus
   
   Für jedes Tile t:
   1. Wähle Referenz-Frame (median_time oder min_gradient)
   2. Für jedes Frame f:
      a. Extrahiere Tile-Region
      b. ECC-Preprocessing (Hochpass)
      c. Phase Correlation → Initial-Shift
      d. ECC Translation-only → Feinschätzung
      e. Validiere ECC-Korrelation (cc >= cc_min)
   3. Rekonstruiere: I_t(p) = Σ W_f,t · I_f(A_f,t(p)) / Σ W_f,t
   ```

4. **Pipeline-Phasen aktualisieren**
   ```markdown
   ## 4. Pipeline-Phasen
   
   0. SCAN_INPUT
   1. CHANNEL_SPLIT (neu: vor Registrierung)
   2. NORMALIZATION
   3. GLOBAL_METRICS
   4. TILE_GRID
   5. LOCAL_METRICS
   6. TILE_RECONSTRUCTION_TLR (neu: integriert Registrierung)
   7. STATE_CLUSTERING
   8. SYNTHETIC_FRAMES
   9. STACKING
   10. DEBAYER
   ```

### Methodik v4 (EN)

Analog zur deutschen Version.

### process_flow_v4/

**Neue Struktur:**
```
doc/process_flow_v4/
├── README.md
├── phase_0_overview.md
├── phase_1_channel_split.md (NEU)
├── phase_2_normalization.md
├── phase_3_global_metrics.md
├── phase_4_tile_grid.md
├── phase_5_local_metrics.md
├── phase_6_tlr.md (NEU - detaillierte TLR-Beschreibung)
├── phase_7_clustering.md
└── ...
```

**phase_6_tlr.md Inhalt:**
```markdown
# Phase 6: Tile-wise Local Registration (TLR)

## Ziel

Registrierung und Rekonstruktion in einem Schritt, tile-weise lokal.

## Eingabe

- Channel-split frames (R, G, B)
- Tile-Grid Definition
- Global weights (G_f)
- Local weights (L_f,t)

## Ausgabe

- Rekonstruierte Kanäle (reconstructed_R.fits, reconstructed_G.fits, reconstructed_B.fits)

## Algorithmus

[Detaillierte Beschreibung wie oben]

## Konfiguration

```yaml
registration:
  local_tiles:
    model: translation
    ecc_cc_min: 0.2
    min_valid_frames: 10
    reference_method: median_time
```
```

### README.md

**GUI-Workflow aktualisieren:**

```markdown
## GUI Workflow (Methodik v4)

1. **Scan Input** - Frames einlesen
2. **Channel Split** - R/G/B trennen
3. **Normalization** - Hintergrund normalisieren
4. **Global Metrics** - Globale Qualität berechnen
5. **Tile Grid** - Tile-Raster definieren
6. **Local Metrics** - Lokale Qualität pro Tile
7. **TLR Reconstruction** - Tile-weise Registrierung + Rekonstruktion
8. **Stacking** - Finale Kombination
9. **Debayer** - Farbrekonstruktion

**Wichtig:** Es gibt keine separate "Registration" Phase mehr.
Die Registrierung erfolgt automatisch während der Tile-Rekonstruktion.
```

---

## Zusammenfassung der Änderungen

| Komponente | Status | Aktion |
|------------|--------|--------|
| TLR-Modul | ✅ Fertig | `runner/tile_local_registration.py` |
| Config | ✅ Fertig | `tile_compile.yaml` |
| Phase-Liste | ✅ Fertig | dry_run aktualisiert |
| phases_impl.py | ⚠️ Manuell | REGISTRATION entfernen, TLR integrieren |
| Schema | ⏳ TODO | `tile_compile.schema.yaml` |
| Tests | ⏳ TODO | `tests/test_tile_local_registration.py` |
| Docs v4 (DE) | ⏳ TODO | `doc/tile_basierte_qualitatsrekonstruktion_methodik_v_4.md` |
| Docs v4 (EN) | ⏳ TODO | `doc/tile_basierte_qualitatsrekonstruktion_methodik_v_4_en.md` |
| process_flow_v4 | ⏳ TODO | `doc/process_flow_v4/` |
| README | ⏳ TODO | Workflow aktualisieren |

---

## Nächste Schritte (Empfohlen)

1. **Backup erstellen:** `git commit -am "WIP: TLR migration"`
2. **phases_impl.py manuell bearbeiten** (siehe Schritt 1-6 oben)
3. **Schema aktualisieren**
4. **Syntax-Check:** `python -m py_compile tile_compile_python/runner/phases_impl.py`
5. **Tests schreiben und ausführen**
6. **Dokumentation schreiben**
7. **Integration testen** mit echten Daten

---

**Ende der Implementierungsanleitung**
