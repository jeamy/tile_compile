# GitHub-Issue-Zerlegung – v4 ↔ C++-Port

**Quelle:** Abgleichbericht v4 ↔ C++  
**Ziel:** Direkte Übernahme in GitHub Issues (ein Issue = eine methodische Verpflichtung)

---

## EPIC: v4-Konformität herstellen

> Sammel-Epic zur vollständigen Umsetzung der Methodik v4 gemäß Spezifikation.

---

## Issue 1 – Globale lineare Normalisierung fehlt vollständig

**Typ:** Bug / Methodik-Bruch  
**Priorität:** Blocker

### Beschreibung
Die v4-Spezifikation fordert eine **exakt einmalige, globale, lineare Normalisierung pro Frame vor jeder Metrikberechnung**. Im aktuellen C++-Port existiert keine entsprechende Funktion oder Pipeline-Stufe.

### Akzeptanzkriterien
- [ ] Funktion `normalize_global_frame()` implementiert
- [ ] Trennung roh / normalisiert erzwungen
- [ ] Normalisierung erfolgt vor *allen* Metriken
- [ ] getrennt pro Farbkanal

---

## Issue 2 – Robuste globale Hintergrundschätzung (B_f)

**Typ:** Feature

### Beschreibung
Globale Hintergrundwerte werden aktuell nur trivial oder implizit bestimmt. v4 verlangt eine **robuste, maskierte Hintergrundschätzung**.

### Akzeptanzkriterien
- [ ] Objektmaske implementiert
- [ ] Hintergrund aus maskierten Pixeln
- [ ] robuste Statistik (Median / Biweight)

---

## Issue 3 – Globale Rauschschätzung (σ_f) fehlt

**Typ:** Feature

### Beschreibung
Die globale Rauschschätzung ist Pflichtbestandteil des globalen Qualitätsindex Q_f, fehlt jedoch vollständig.

### Akzeptanzkriterien
- [ ] robuste σ_f-Schätzung implementiert
- [ ] konsistent mit lokaler σ-Schätzung

---

## Issue 4 – Gradientenergie E_f nicht v4-konform

**Typ:** Bug / Enhancement

### Beschreibung
Gradientenergie wird nicht gemäß v4 (definiert, robust, normiert) berechnet.

### Akzeptanzkriterien
- [ ] Sobel/Scharr-basierte Ableitung
- [ ] konsistente Vor-Glättung
- [ ] robuste Aggregation

---

## Issue 5 – MAD-Normalisierung globaler Metriken fehlt

**Typ:** Bug

### Beschreibung
Alle globalen Metriken müssen per Median + MAD über alle Frames normiert werden. Dies fehlt vollständig.

### Akzeptanzkriterien
- [ ] MAD-Normalisierung implementiert
- [ ] getrennt pro Metrik

---

## Issue 6 – Globaler Qualitätsindex Q_f und Gewicht G_f fehlen

**Typ:** Feature

### Beschreibung
Der globale Qualitätsindex Q_f und das Gewicht G_f = exp(Q_f) sind Kernbestandteile der Methodik, aber nicht vorhanden.

### Akzeptanzkriterien
- [ ] Q_f gemäß v4 berechnet
- [ ] Clipping auf [-3, +3]
- [ ] G_f = exp(Q_f)

---

## Issue 7 – Seeing-adaptive Tile-Geometrie fehlt

**Typ:** Feature

### Beschreibung
Tile-Größe und Overlap müssen seeing-adaptiv aus der FWHM-Verteilung abgeleitet werden.

### Akzeptanzkriterien
- [ ] Sternselektion
- [ ] FWHM-Verteilung
- [ ] adaptive Tile-Größe
- [ ] Overlap-Definition

---

## Issue 8 – Stern- vs. Struktur-Tiles nicht unterschieden

**Typ:** Bug / Feature

### Beschreibung
Lokale Qualitätsmetriken müssen zwischen Stern- und Struktur-Tiles unterscheiden.

### Akzeptanzkriterien
- [ ] Stern-Tile-Erkennung
- [ ] Struktur-Tile-Fallback

---

## Issue 9 – Lokale Qualitätsindizes Q_local fehlen

**Typ:** Feature

### Beschreibung
Die Berechnung von Q_star bzw. Q_struct gemäß v4 fehlt vollständig.

### Akzeptanzkriterien
- [ ] Stern-Q_local implementiert
- [ ] Struktur-Q_local implementiert
- [ ] Clipping [-3, +3]

---

## Issue 10 – Orthogonale Gewichtung W_f,t = G_f · L_f,t fehlt

**Typ:** Bug / Methodik-Bruch

### Beschreibung
Globale und lokale Qualität müssen orthogonal kombiniert werden. Aktuell existiert keine solche Gewichtung.

### Akzeptanzkriterien
- [ ] effektive Gewichtung implementiert
- [ ] numerische Stabilisierung

---

## Issue 11 – Tile-Rekonstruktion ohne Stabilitätsregeln

**Typ:** Bug

### Beschreibung
Tile-Rekonstruktion erfolgt ohne ε-Guards oder Fallback-Regeln.

### Akzeptanzkriterien
- [ ] ε im Nenner
- [ ] Fallback bei ΣW→0

---

## Issue 12 – Zustandsvektor v_f fehlt vollständig

**Typ:** Feature

### Beschreibung
Die Methodik v4 basiert auf zustandsbasierter Clusterung. Der Zustandsvektor wird nicht berechnet.

### Akzeptanzkriterien
- [ ] v_f = (G_f, ⟨Q_tile⟩, Var(Q_tile), B_f, σ_f)

---

## Issue 13 – Zustands-Clusterung nicht implementiert

**Typ:** Feature

### Beschreibung
Frames müssen nach Beobachtungszustand (nicht Zeit) geclustert werden.

### Akzeptanzkriterien
- [ ] k-means oder GMM
- [ ] Feature-Standardisierung

---

## Issue 14 – Synthetische Qualitätsframes fehlen

**Typ:** Feature

### Beschreibung
Pro Cluster muss ein synthetisches Qualitätsframe rekonstruiert werden.

### Akzeptanzkriterien
- [ ] Rekonstruktion pro Cluster
- [ ] Speicherung der Frames

---

## Issue 15 – Validierung & Abbruchlogik fehlt

**Typ:** Bug / Feature

### Beschreibung
Erfolg oder Abbruch eines Laufs wird nicht validiert.

### Akzeptanzkriterien
- [ ] FWHM-Verbesserung prüfen
- [ ] Tile-Artefakte erkennen
- [ ] kontrollierter Abbruch

---

## Abschluss

Diese Issues bilden **eine 1:1-Abbildung der v4-Spezifikation**. Ihre Abarbeitung ist notwendig und hinreichend für v4-Konformität.

---

