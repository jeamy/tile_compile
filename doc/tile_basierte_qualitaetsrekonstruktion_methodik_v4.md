# Tile‑basierte Qualitätsrekonstruktion für DSO – Methodik v4

**Status:** Normative Referenz (experimentelle Referenzanwendung)\
**Ersetzt:** Methodik v2, v3\
**Gültig für:** `tile_compile` (lokale, nicht‑globale Rekonstruktion)

---

## 0. Leitprinzip (neu in v4)

> Es existiert **kein global konsistentes Koordinatensystem**. Jede Rekonstruktion ist **lokal, zeitlich konsistent und selbstkorrigierend**.

Diese Methodik beschreibt keinen klassischen Bildstacker mehr, sondern einen **tile‑lokalen Mehrframe‑Rekonstruktionsoperator**.

---

## 1. Grundannahmen (verbindlich)

- Rohdaten sind **linear**
- keine globale Registrierung
- keine globale Referenzgeometrie
- Bewegung kann **orts‑ und zeitabhängig** sein
- alle geometrischen Korrekturen sind **Tile‑lokal**
- jede Entscheidung muss lokal validierbar sein

Ein Verstoß führt zum **Abbruch des Laufs**.

---

## 2. Gesamtpipeline (v4, normativ)

```
Phase 0:  SCAN_INPUT         – Frames laden, Metadaten extrahieren
Phase 1:  CHANNEL_SPLIT      – OSC → R/G/B Kanaltrennung (CFA)
Phase 2:  NORMALIZATION      – globale Grobnormierung
Phase 3:  GLOBAL_METRICS     – Frame-Qualität berechnen
Phase 4:  TILE_GRID          – Tile-Geometrie festlegen
Phase 5:  LOCAL_METRICS      – lokale Tile-Qualität (vor Warp)
Phase 6:  TILE_RECONSTRUCTION_TLR – lokale Registrierung + Rekonstruktion
Phase 7:  STATE_CLUSTERING   – Zustandsbasierte Clusterung
Phase 8:  SYNTHETIC_FRAMES   – synthetische Qualitätsframes
Phase 9:  STACKING           – finales lineares Stacking
Phase 10: DEBAYER            – Farbrekonstruktion (OSC)
Phase 11: DONE               – Abschluss + Validierung
```

**Wichtig:** Registrierung ist **kein separater Schritt** mehr, sondern integriert in Phase 6 (TILE_RECONSTRUCTION_TLR).

---

## 3. Globale Grobnormierung (Pflicht)

Zweck: Entkopplung photometrischer Transparenz.

Für jedes Frame *f*:

```
I'_f = I_f / B_f
```

- B\_f: robuster globaler Hintergrund
- nur einmal
- keine lokale Anpassung

---

## 4. Tile‑Geometrie (adaptiv)

Initiale Tile‑Größe:

```
T_0 = clip(32 · FWHM, 64, max_tile_size)
```

- `max_tile_size`: Konfigurierbar (default: 128)
- Überlappung ≥ 25 %

**Rekursive Verfeinerung (implementiert):**
- Tiles werden bei hoher Warp-Varianz automatisch verfeinert
- Verfeinerungskriterium:
  - `warp_variance > refinement_variance_threshold`
  - `mean_correlation < 0.5`
- Konfigurierbar: `enable_recursive_refinement`, `refinement_max_depth`

---

## 5. Lokale Registrierung (Kern von v4)

### 5.1 Bewegungsmodell

Minimalmodell:

```
p' = p + (dx, dy)
```

Optional (experimentell, regularisiert):

```
p' = p + v + J(p − c)
```

mit ||J|| ≪ 1.

---

### 5.2 Iterative Referenzbildung (neu, zwingend)

Für jedes Tile *t*:

1. Initiale Referenz R\_t⁽⁰⁾ aus Median‑Frame
2. Lokale Registrierung aller Frames
3. Rekonstruktion I\_t⁽¹⁾
4. Neue Referenz R\_t := I\_t⁽¹⁾
5. Wiederhole bis Konvergenz (typ. 2–3 Iterationen)

---

### 5.3 Zeitliche Glättung der Warps (zwingend)

Für jedes Tile *t*:

```
Â_{f,t} = smooth_time(A_{f−k…f+k,t})
```

Empfohlen:

- Savitzky–Golay
- robuster Medianfilter

---

## 6. Lokale Qualitätsmetriken

**Implementierung:**
- Phase 5 (LOCAL_METRICS): Metriken **vor** der Registrierung
- Phase 6 (TLR): Post-Warp Metriken via `compute_post_warp_metrics()`
  - Kontrast (Laplacian-Varianz)
  - Hintergrund (robuster Median)
  - SNR-Proxy

### 6.1 Stern‑Tiles

- log(FWHM)
- Rundheit
- lokaler Kontrast

### 6.2 Struktur‑Tiles

- E / σ
- lokaler Hintergrund

Alle Metriken:

- robust normalisiert (Median + MAD)
- begrenzt auf [−3, +3]

---

## 7. Gewichte (erweitert)

Globales Gewicht:

```
G_f = exp(Q_f)
```

Lokales Gewicht:

```
L_{f,t} = exp(Q_{local})
```

Registrierungsgüte:

```
R_{f,t} = exp(β · (cc_{f,t} − 1))
```

Effektives Gewicht:

```
W_{f,t} = G_f · L_{f,t} · R_{f,t}
```

---

## 8. Tile‑Rekonstruktion

Für jedes Tile:

```
I_t(p) = Σ_f W_{f,t} · I_f(Â_{f,t}(p)) / Σ_f W_{f,t}
```

Stabilitätsregeln:

- ΣW < ε → Tile invalid
- < N\_min gültige Frames → Tile invalid

---

## 9. Overlap‑Add

**Implementierung (vollständig):**

```
w_t(p) = hann(p) · ψ(var(Â_{f,t}))
```

mit:

```
ψ(v) = exp(-v / (2·σ²))
```

- Hohe Warp-Varianz → reduziertes Fenstergewicht
- Konfigurierbar: `variance_window_sigma` (default: 2.0)

---

## 10. Zustandsbasierte Clusterung

**Implementierung (erweitert):**

Zustandsvektor pro Frame:

```
v_f = (G_f, ⟨Q_{tile}⟩, Var(Q_{tile}), ⟨cc⟩, Var(Â), invalid_tile_fraction)
```

Erweiterte Metadaten aus TLR:
- `mean_correlation`: ⟨cc⟩ über alle Tiles
- `warp_variance`: Var(Â) der Translationen
- `invalid_tile_fraction`: Anteil ungültiger Tiles

Clusterung:

- k = 15–30 (konfigurierbar)
- pro Cluster ein synthetisches Frame

---

## 11. Finales Stacking

- lineares Stacking der synthetischen Frames
- keine zusätzliche Gewichtung
- keine geometrische Transformation

---

## 12. Validierung (v4)

### Lokal (pflicht)

- FWHM‑Heatmaps
- Warp‑Vektorfelder
- Tile‑Invalid‑Karten

### Global (sekundär)

- SNR‑Verteilung
- Hintergrund‑RMS

Abbruch:

- < 30 % gültige Tiles
- großskalige systematische Warp‑Muster

---

## 13. Konfigurationsparameter (v4)

```yaml
registration:
  local_tiles:
    model: translation                    # nur Translation (Pflicht)
    ecc_cc_min: 0.2                      # min. ECC-Korrelation
    min_valid_frames: 10                 # min. gültige Frames pro Tile
    reference_method: median_time        # median_time | min_gradient
    max_tile_size: 128                   # max. Tile-Größe
    registration_quality_beta: 5.0       # β für R_{f,t}
    max_iterations: 3                    # iterative Referenzbildung
    temporal_smoothing_window: 11        # Savitzky-Golay Fenster (ungerade)
    temporal_smoothing_polyorder: 3      # Polynom-Ordnung
```

---

## 14. Kernaussage v4

> Methodik v4 ersetzt globale Geometrie durch **lokal konsistente, zeitlich geglättete Rekonstruktion**.

Sie ist:

- korrekt für Alt/Az & EQ
- robust gegen Feldrotation
- experimentell maximal flexibel
- wissenschaftlich sauber begründbar

---

**Ende der normativen Spezifikation v4**

