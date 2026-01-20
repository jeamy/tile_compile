# Migration zu Tile-wise Local Registration (TLR) - Methodik v4

**Status:** In Arbeit  
**Datum:** 2026-01-20  
**Ziel:** Vollständiger Ersatz der globalen Registrierung durch tile-weise lokale Registrierung

---

## Übersicht der Änderungen

### Architektur-Paradigmenwechsel

**Alt (Methodik v3):**
```
SCAN → REGISTRATION (global) → CHANNEL_SPLIT → ... → TILE_RECONSTRUCTION (nutzt registered frames)
```

**Neu (Methodik v4):**
```
SCAN → CHANNEL_SPLIT → ... → TILE_RECONSTRUCTION_TLR (registriert lokal pro Tile)
```

**Kernprinzip:** Jedes Tile registriert sich unabhängig gegen seine eigene Referenz. Keine globalen registered frames mehr.

---

## Implementierungsstatus

### ✅ Abgeschlossen

1. **TLR-Kernmodul** (`runner/tile_local_registration.py`)
   - `select_tile_reference_frame()` - Referenz-Selektion (median_time / min_gradient)
   - `estimate_tile_local_translation()` - Phase Corr + ECC Translation-only
   - `tile_local_register_and_reconstruct()` - Hauptfunktion pro Kanal
   - `tile_local_reconstruct_all_channels()` - Wrapper für R/G/B

2. **Config-Update** (`tile_compile.yaml`)
   ```yaml
   registration:
     local_tiles:
       model: translation
       ecc_cc_min: 0.2
       min_valid_frames: 10
       reference_method: median_time
       max_tile_size: 128
   ```

3. **Phase-Liste aktualisiert**
   - REGISTRATION Phase (alt: Phase 1) entfernt
   - Alle nachfolgenden Phasen um 1 nach vorne verschoben
   - TILE_RECONSTRUCTION → TILE_RECONSTRUCTION_TLR (Phase 6)

4. **Registrierungs-Fixes** (vorherige Session)
   - 7 kritische Bugs behoben (min_star_matches, ECC-Preprocessing, Mutual NN, Scale-Rejection, etc.)
   - Diese Fixes sind in TLR integriert

### 🔄 In Arbeit

5. **Pipeline-Integration in `phases_impl.py`**
   - Phase 1 (REGISTRATION) komplett entfernen (Zeilen ~1668-2165)
   - Phase 6 (TILE_RECONSTRUCTION_TLR) umschreiben
   - Alle Phase-IDs aktualisieren

### ⏳ Ausstehend

6. **Schema-Update** (`tile_compile.schema.yaml`)
   - Alte `registration.engine`, `registration.allow_rotation`, etc. entfernen
   - Neue `registration.local_tiles.*` Felder definieren

7. **Tests** (`tests/test_tile_local_registration.py`)
   - Unit-Tests für TLR-Funktionen
   - Integration-Tests

8. **Dokumentation**
   - Methodik v4 (DE + EN)
   - process_flow_v4/
   - README.md

---

## Detaillierte Implementierungsschritte

### Schritt 5: Pipeline-Integration (KRITISCH)

#### 5.1 Phase 1 (REGISTRATION) entfernen

**Zu löschen:** Zeilen ~1668-2165 in `phases_impl.py`

```python
# Alt:
phase_id = 1
phase_name = "REGISTRATION"
if should_skip_phase(phase_id):
    ...
elif reg_engine == "opencv_cfa":
    # Gesamte opencv_cfa Registrierung (~400 Zeilen)
    ...
else:
    # Siril-basierte Registrierung (~50 Zeilen)
    ...
```

**Ersetzen durch:** Nichts - Phase komplett entfernen

**Konsequenz:**
- `registered_files` existiert nicht mehr
- `reg_out` Verzeichnis wird nicht mehr erstellt
- Alle nachfolgenden Phasen müssen auf `channel_files` zugreifen statt `registered_files`

#### 5.2 Phase-IDs aktualisieren

Alle `phase_id` Zuweisungen nach der entfernten REGISTRATION Phase um 1 reduzieren:

```python
# Alt:
phase_id = 2  # CHANNEL_SPLIT
phase_id = 3  # NORMALIZATION
phase_id = 4  # GLOBAL_METRICS
phase_id = 5  # TILE_GRID
phase_id = 6  # LOCAL_METRICS
phase_id = 7  # TILE_RECONSTRUCTION
phase_id = 8  # STATE_CLUSTERING
...

# Neu:
phase_id = 1  # CHANNEL_SPLIT
phase_id = 2  # NORMALIZATION
phase_id = 3  # GLOBAL_METRICS
phase_id = 4  # TILE_GRID
phase_id = 5  # LOCAL_METRICS
phase_id = 6  # TILE_RECONSTRUCTION_TLR
phase_id = 7  # STATE_CLUSTERING
...
```

#### 5.3 CHANNEL_SPLIT anpassen

**Alt:** CHANNEL_SPLIT arbeitet auf `registered_files`

```python
# Zeile ~2167
phase_id = 2
phase_name = "CHANNEL_SPLIT"
...
for reg_file in registered_files:
    # Split registered frames
```

**Neu:** CHANNEL_SPLIT arbeitet direkt auf `frames` (aus SCAN_INPUT)

```python
phase_id = 1
phase_name = "CHANNEL_SPLIT"
...
for frame in frames:
    # Split raw frames (nach Calibration, vor Registrierung)
```

**Wichtig:** Dies bedeutet, dass CHANNEL_SPLIT jetzt **vor** jeglicher Registrierung erfolgt.

#### 5.4 TILE_RECONSTRUCTION_TLR umschreiben

**Alt:** Phase 7, arbeitet auf `registered_files` + `channel_files`

```python
phase_id = 7
phase_name = "TILE_RECONSTRUCTION"
...
for ch in ["R", "G", "B"]:
    for f_idx, ch_file in enumerate(channel_files[ch]):
        f = fits.getdata(str(ch_file))  # Bereits registriert
        # Tile-Rekonstruktion ohne weitere Registrierung
```

**Neu:** Phase 6, ruft TLR auf

```python
phase_id = 6
phase_name = "TILE_RECONSTRUCTION_TLR"
...
from .tile_local_registration import tile_local_reconstruct_all_channels

# Prepare tile grid
tile_grid = {
    "tiles": [
        {"y_start": y0, "y_end": y1, "x_start": x0, "x_end": x1, ...}
        for each tile
    ]
}

# Prepare frames by channel
frames_by_channel = {
    "R": channel_files["R"],
    "G": channel_files["G"],
    "B": channel_files["B"]
}

# Prepare weights
weights_global = gfc  # From GLOBAL_METRICS
weights_local = {
    "R": L_local_R,  # From LOCAL_METRICS
    "G": L_local_G,
    "B": L_local_B
}

# Call TLR
reconstructed = tile_local_reconstruct_all_channels(
    frames_by_channel,
    tile_grid,
    weights_global,
    weights_local,
    cfg
)

# Save reconstructed channels
for ch in ["R", "G", "B"]:
    if ch in reconstructed:
        out_path = outputs_dir / f"reconstructed_{ch}.fits"
        fits.writeto(str(out_path), reconstructed[ch], header=hdr0, overwrite=True)
```

#### 5.5 Nachfolgende Phasen anpassen

**STACKING Phase:** Muss auf `reconstructed_{R,G,B}.fits` zugreifen statt auf `registered/*.fit`

```python
# Alt:
stack_input_dir = work_dir / reg_out_name  # registered/
stack_files = sorted(stack_input_dir.glob(stack_input_pattern))

# Neu:
# Stacking arbeitet direkt auf rekonstruierten Kanälen
stack_files = [
    outputs_dir / "reconstructed_R.fits",
    outputs_dir / "reconstructed_G.fits",
    outputs_dir / "reconstructed_B.fits"
]
```

---

## Kritische Designentscheidungen

### 1. Wann erfolgt Channel-Split?

**Entscheidung:** **Vor** TLR (Phase 1)

**Begründung:**
- TLR arbeitet pro Kanal unabhängig
- Jeder Kanal hat eigene lokale Metriken
- Vermeidet redundante Registrierung für R/G/B

**Konsequenz:**
- CHANNEL_SPLIT muss auf **raw calibrated frames** arbeiten
- Nicht auf registered frames (die existieren nicht mehr)

### 2. Was ist mit Siril-basierten Workflows?

**Entscheidung:** Siril-Pfad wird **entfernt**

**Begründung:**
- Siril kann keine tile-weise Registrierung
- TLR ist universell (EQ + Alt/Az)
- Clean Break für Methodik v4

**Migration:** Nutzer müssen auf TLR umsteigen

### 3. Wie werden Overlaps behandelt?

**Entscheidung:** Einfaches Averaging in Overlap-Regionen

**Implementierung in TLR:**
```python
# In tile_local_register_and_reconstruct():
reconstructed[y0:y1, x0:x1] += tile_reconstructed

# Später normalisieren durch Anzahl überlappender Tiles
# (Oder Hanning-Window wie in alter TILE_RECONSTRUCTION)
```

**TODO:** Hanning-Window aus alter TILE_RECONSTRUCTION in TLR integrieren

### 4. Wie wird die Tile-Referenz gewählt?

**Zwei Modi:**

1. **median_time** (default): Temporaler Median
   - Schnell
   - Robust
   - Keine I/O-Last

2. **min_gradient**: Frame mit minimaler Gradientenenergie
   - Wählt bestes Seeing für jedes Tile
   - Langsamer (muss Tiles lesen)
   - Optimal für Qualität

**Config:**
```yaml
registration:
  local_tiles:
    reference_method: median_time  # oder min_gradient
```

---

## Risiken und Mitigationen

### Risiko 1: Performance

**Problem:** TLR liest jedes Frame N_tiles mal (statt 1x bei globaler Registrierung)

**Mitigation:**
- Tile-Größe optimieren (nicht zu klein)
- Caching erwägen (RAM-intensiv)
- Parallele Verarbeitung (TODO)

### Risiko 2: Overlap-Artefakte

**Problem:** Einfaches Averaging kann Kanten erzeugen

**Mitigation:**
- Hanning-Window aus v3 übernehmen
- Oder: Feathering mit größeren Overlap-Regionen

### Risiko 3: Tile-Referenz-Wahl

**Problem:** Schlechte Referenz → schlechte Registrierung für gesamtes Tile

**Mitigation:**
- `min_gradient` Modus für kritische Daten
- Fallback auf median_time bei Fehlern

### Risiko 4: Numerische Instabilität

**Problem:** Tiles mit wenigen gültigen Frames

**Mitigation:**
- `min_valid_frames: 10` Guard (bereits implementiert)
- Tile als invalid markieren bei Unterschreitung
- Interpolation aus Nachbar-Tiles (TODO)

---

## Testing-Strategie

### Unit-Tests

```python
# tests/test_tile_local_registration.py

def test_select_tile_reference_median():
    # Test median_time Modus
    
def test_estimate_tile_local_translation():
    # Test Translation-Schätzung
    
def test_tile_local_register_synthetic():
    # Test mit synthetischen Daten (bekannte Translation)
```

### Integration-Tests

```python
def test_full_pipeline_with_tlr():
    # Test komplette Pipeline mit TLR
    # Vergleich mit v3 Ergebnissen (sollte ähnlich sein für EQ)
```

### Regression-Tests

- Bestehende Alt/Az-Daten mit v3 vs v4 vergleichen
- Metriken: FWHM, Sternzahl, SNR, visuelle Qualität

---

## Dokumentations-Updates

### Methodik v4 (DE)

**Hauptänderungen:**
- Abschnitt 2: "Registrierung" → "Tile-weise lokale Registrierung"
- Abschnitt 3: Formel für A_{f,t} statt A_f
- Abschnitt 4: Pipeline-Phasen aktualisieren
- Abschnitt 5: TLR-Algorithmus beschreiben

### Methodik v4 (EN)

- Analog zu DE

### process_flow_v4/

**Neue Struktur:**
```
process_flow_v4/
├── README.md                    # Übersicht
├── phase_0_overview.md          # Calibration (unverändert)
├── phase_1_channel_split.md     # NEU: Vor Registrierung
├── phase_2_normalization.md     # Renummeriert
├── phase_3_global_metrics.md    # Renummeriert
├── phase_4_tile_grid.md         # Renummeriert
├── phase_5_local_metrics.md     # Renummeriert
├── phase_6_tlr.md               # NEU: TLR-Beschreibung
├── phase_7_clustering.md        # Renummeriert
└── ...
```

### README.md

**GUI-Workflow aktualisieren:**
- Tab "Registration" entfernen oder umbenennen zu "TLR Settings"
- Workflow-Beschreibung anpassen

---

## Nächste Schritte (Priorisiert)

1. **[KRITISCH]** Phase 1 (REGISTRATION) aus `phases_impl.py` entfernen
2. **[KRITISCH]** Phase-IDs in `phases_impl.py` aktualisieren
3. **[KRITISCH]** CHANNEL_SPLIT auf raw frames umstellen
4. **[KRITISCH]** TILE_RECONSTRUCTION_TLR implementieren
5. **[WICHTIG]** Hanning-Window in TLR integrieren
6. **[WICHTIG]** Schema-Update
7. **[WICHTIG]** Tests schreiben
8. **[NORMAL]** Dokumentation aktualisieren

---

## Rollback-Plan

Falls TLR nicht funktioniert:

1. Git-Branch für v4 erstellen
2. v3 auf main/master behalten
3. Schrittweise Migration testen
4. Bei Problemen: zurück zu v3

**Empfehlung:** Erst v3 mit den 7 Registrierungs-Fixes testen, bevor TLR-Migration abgeschlossen wird.

---

## Offene Fragen

1. **Parallele Verarbeitung:** Sollen Tiles parallel verarbeitet werden?
   - Pro: Schneller
   - Contra: RAM-Verbrauch, Komplexität

2. **Tile-Interpolation:** Was tun mit invalid tiles?
   - Option A: Als NaN markieren
   - Option B: Aus Nachbarn interpolieren
   - Option C: Warnung ausgeben, aber behalten

3. **Backward Compatibility:** Sollen alte registered/ Verzeichnisse noch gelesen werden?
   - Vermutlich nein (Clean Break)

4. **GUI-Integration:** Wie zeigt GUI TLR-Fortschritt an?
   - Pro Tile? Pro Kanal? Pro Frame?

---

**Ende des Migrations-Plans**
