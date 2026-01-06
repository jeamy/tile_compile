# Methodik v3 – Integration, Kurzfassung und Phasen-Enum

**Dokumenttyp:** Integrations- und Ableitungsdokument  
**Bezug:** `tile_basierte_qualitatsrekonstruktion_methodik_v3.md`  
**Ziel:** Konsistenter Abgleich zwischen Methodik, Konfiguration, Backend und GUI

---

## Teil 1 – Abgleich Methodik v3 ↔ tile_compile.yaml

Diese Tabelle ist **normativ**. Jede Phase der Methodik v3 muss eindeutig einer Konfigurationssektion entsprechen.

| Methodik v3 – Phase | Beschreibung | tile_compile.yaml – Sektion | Pflicht |
|--------------------|--------------|-----------------------------|--------|
| Assumptions | Invariante/Weiche Annahmen, Reduced Mode | `assumptions` | optional |
| Phase 0 | Registrierung & Geometrie | `registration` | ja |
| A.2.1 | Siril Debayer + Register | `registration.engine: siril` | optional |
| B.2.1 | CFA-Luminanz-Estimator | `registration.engine: opencv_cfa` | optional |
| B.2.2 | CFA-aware Warp | `registration.engine: opencv_cfa` | optional |
| Phase 1 | Kanaltrennung | implizit nach `registration` | ja |
| Phase 2 | Globale Normalisierung | `normalization` | ja |
| Phase 3 | Globale Frame-Metriken | `global_metrics` | ja |
| Phase 4 | Tile-Geometrie | `tile` | ja |
| Phase 5 | Lokale Tile-Metriken | `local_metrics` | ja |
| Phase 6 | Tile-basierte Rekonstruktion | `reconstruction` | ja |
| Phase 7 | Zustands-Clusterung | `synthetic` | optional |
| Phase 8 | Synthetische Frames | `synthetic` | optional |
| Phase 9 | Finales lineares Stacking | `stacking` | ja |
| Phase 10 | Kombination (RGB/LRGB) | **außerhalb** | nein |

**Wichtig:** Kombination ist bewusst **nicht** Teil von `tile_compile.yaml`.

---

## Teil 2 – Kurzfassung für README / Paper

### Tile-basierte Qualitätsrekonstruktion für DSO (Kurzfassung)

Die Tile-basierte Qualitätsrekonstruktion ist ein lokales Rekonstruktionsverfahren für Deep-Sky-Aufnahmen mit sehr vielen Einzelbelichtungen.

Im Gegensatz zu klassischen globalen Stacking-Ansätzen wird die Bildqualität **lokal (Tile-weise)** und **kanalgetrennt** bewertet und rekonstruiert.

Der Workflow besteht aus folgenden Kernschritten:

1. **Registrierung** aller Frames auf eine gemeinsame Geometrie (Siril oder CFA-basiert).
2. **Kanaltrennung** der registrierten Frames (R/G/B oder Mono).
3. **Globale lineare Normalisierung** und Frame-Metriken.
4. **Erzeugung eines festen Tile-Rasters** über das Bildfeld.
5. **Lokale Qualitätsbewertung pro Tile und Kanal** (FWHM, Kontrast, SNR).
6. **Tile-basierte Rekonstruktion** mit lokalen Gewichten.
7. Optional: **zustandsbasierte Clusterung** und synthetische Frames.
8. **Finales lineares Stacking pro Kanal**.

Farbliche Kombination (RGB/LRGB), Farbkalibrierung und Stretch sind **nicht Teil der Rekonstruktionsmethodik** und erfolgen in einem separaten Nachverarbeitungsschritt.

Dieser Ansatz erlaubt es, lokale Seeing-, Fokus- und SNR-Unterschiede optimal auszunutzen und übertrifft globale Selektions- und Stacking-Methoden insbesondere bei großen Datensätzen.

---

## Teil 3 – Phasen-Enum für Backend & GUI

Das folgende Phasen-Enum ist **verbindlich** für Backend-Status, Logging, GUI-Fortschritt und Resume-Logik.

### 3.1 PhaseEnum (normativ)

```text
SCAN_INPUT
REGISTRATION
CHANNEL_SPLIT
NORMALIZATION
GLOBAL_METRICS
TILE_GRID
LOCAL_METRICS
TILE_RECONSTRUCTION
STATE_CLUSTERING
SYNTHETIC_FRAMES
STACKING
DONE
FAILED
```

---

### 3.2 Bedeutung der Phasen

| Phase | Bedeutung |
|------|----------|
| SCAN_INPUT | Frames lesen, Header prüfen, Zählung |
| REGISTRATION | Geometrische Registrierung (Siril oder CFA) |
| CHANNEL_SPLIT | Trennung in R/G/B oder Mono |
| NORMALIZATION | Globale lineare Normalisierung |
| GLOBAL_METRICS | Frame-weite Qualitätsmetriken |
| TILE_GRID | Erzeugung des festen Tile-Rasters |
| LOCAL_METRICS | Tile-lokale Qualitätsmetriken |
| TILE_RECONSTRUCTION | Tile-basierte Rekonstruktion |
| STATE_CLUSTERING | Zustands-Clusterung (optional) |
| SYNTHETIC_FRAMES | Rekonstruktion synthetischer Frames |
| STACKING | Finales lineares Stacking |
| DONE | Erfolgreicher Abschluss |
| FAILED | Abbruch mit Fehler |

---

### 3.3 GUI-Implikationen

* Jede Phase ist **atomar** und restart-fähig.
* GUI zeigt:
  * aktuelle Phase
  * Fortschritt innerhalb der Phase
  * letzte abgeschlossene Phase
* Abbruch ist **nur zwischen Phasen** zulässig.

---

## Teil 4 – Assumptions & Reduced Mode (Methodik v3 §1)

### 4.1 Konfigurationssektion `assumptions`

Diese Sektion ist optional. Wenn nicht gesetzt, gelten Backend-Defaults.

Beispiel:

```yaml
assumptions:
  frames_min: 50
  frames_optimal: 800
  frames_reduced_threshold: 200
  exposure_time_tolerance_percent: 5
  registration_residual_warn_px: 0.5
  registration_residual_max_px: 1.0
  elongation_warn: 0.3
  elongation_max: 0.4
  reduced_mode_skip_clustering: true
  reduced_mode_cluster_range: [5, 10]
```

### 4.2 Reduced Mode Verhalten

Wenn `frame_count < assumptions.frames_reduced_threshold` (und `>= frames_min`):

* `STATE_CLUSTERING` wird übersprungen (oder optional mit reduziertem Cluster-Bereich)
* `SYNTHETIC_FRAMES` wird übersprungen
* Pipeline läuft deterministisch weiter mit direktem tile-gewichteten Output

Diese Entscheidung ist im Phase-Output mit `reduced_mode=true` und `skipped=true` sichtbar.

---

## Abschluss

Dieses Dokument stellt sicher, dass:

* Methodik v3
* Konfigurationsschema (`tile_compile.yaml`)
* Backend-Implementierung
* GUI-Statusmodell

**dieselbe Pipeline-Semantik teilen**.

Abweichungen erfordern eine neue Methodik-Version.

---

**Ende des Dokuments**

