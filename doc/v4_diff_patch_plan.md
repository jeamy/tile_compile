# v4 Diff-Patch-Plan (verbindlich)

Dieses Dokument beschreibt exakt, welche Codepfade zu entfernen,
welche umzubauen und welche neu zu implementieren sind, um von einem
bestehenden v3/v3.5-ähnlichen Stand auf eine v4-konforme Pipeline zu kommen.

---

## 1. Sofort löschen

### 1.1 Implizite oder verteilte Normalisierung

Entfernen Sie vollständig:
- jede Skalierung innerhalb von GlobalMetrics-Berechnungen
- jede lokale Hintergrundsubtraktion vor Metriken
- jede automatische RMS-/Median-Normierung

Nach Stage 1 dürfen Pixelwerte nicht mehr verändert werden.

---

### 1.2 Vermischte Gewichtungen

Entfernen Sie alle Konstrukte der Form:

    weight = f(global_metric, local_metric)

oder

    Q = normalize(global + local)

Erlaubt ist ausschließlich:

    W_f_t = G_f * L_f_t

---

### 1.3 Direkte Endrekonstruktion

Löschen Sie jede Pipeline, die Tiles direkt zu einem Endbild kombiniert.
Tiles dürfen nur zu synthetischen Frames beitragen.

---

## 2. Umbau bestehender Komponenten

### 2.1 GlobalMetricsComputer

Alt:
    compute_global_metrics(frame, mask)

Neu:
    compute(normalized_frame)

- keine Masken
- keine Skalierung
- robuste Median/MAD-Normierung intern

---

### 2.2 Tile-Geometrie

Alt:
- fixe oder YAML-getriebene Tile-Größe

Neu:
- Tile-Größe ausschließlich aus median_FWHM
- YAML nur für Clip-Grenzen

---

## 3. Neu zu implementieren (Pflicht)

### 3.1 GlobalNormalizer

- eigene Klasse
- genau einmal pro Frame
- bei Fehlen: Pipeline-Abbruch

---

### 3.2 FrameStateVector + Clusterer

- Zustandsvektor:
  (G, mean(Q_tile), var(Q_tile), B, sigma)
- Clusterung auf Frames
- k = 15–30

Ohne Clusterung ist die Implementierung nicht v4-konform.

---

### 3.3 SyntheticFrameBuilder

- ein synthetisches Frame pro Cluster
- lineares gewichtetes Mittel
- keine Tile-Gewichte mehr

---

## 4. Harte Pipeline-Guards

Implementieren Sie zwingend:

    if (!global_normalized)
        abort;

    if (synthetic_frames.empty())
        abort;

Diese Guards verhindern stille Regressionen.

---

## 5. Ergebnis

Nach Umsetzung dieses Plans:
- ist die Pipeline strukturell v4-konform
- sind v3-Abkürzungen nicht mehr möglich
- ist die Semantik eindeutig und testbar
