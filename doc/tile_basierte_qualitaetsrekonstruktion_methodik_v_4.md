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
Frames laden
→ globale Grobnormierung
→ Tile‑Geometrie
→ lokale Registrierung + lokale Metriken (iterativ)
→ Tile‑Rekonstruktion (Overlap‑Add)
→ Zustandsbasierte Clusterung
→ synthetische Qualitätsframes
→ finales lineares Stacking
→ lokale & globale Validierung
```

**Wichtig:** Registrierung ist **kein separater Schritt** mehr, sondern Teil der Tile‑Rekonstruktion.

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
T_0 = clip(32 · FWHM, 64, 128)
```

- Überlappung ≥ 25 %
- Tiles dürfen **rekursiv verfeinert** werden
- Verfeinerungskriterium:
  - hohe Warp‑Varianz
  - hohe PSF‑Inhomogenität

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

## 6. Lokale Qualitätsmetriken (bewegungskorrigiert)

Alle Metriken werden **nach Anwendung des lokalen Warps** berechnet.

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

## 9. Overlap‑Add (erweitert)

Fensterfunktion:

```
w_t(p) = hann(p) · ψ(var(Â_{f,t}))
```

→ geringes Vertrauen = geringes Randgewicht

---

## 10. Zustandsbasierte Clusterung (erweitert)

Zustandsvektor pro Frame:

```
v_f = (G_f,
       ⟨Q_{tile}⟩,
       Var(Q_{tile}),
       ⟨cc⟩,
       Var(Â),
       invalid_tile_fraction)
```

Clusterung:

- k = 15–30
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

## 13. Kernaussage v4

> Methodik v4 ersetzt globale Geometrie durch **lokal konsistente, zeitlich geglättete Rekonstruktion**.

Sie ist:

- korrekt für Alt/Az & EQ
- robust gegen Feldrotation
- experimentell maximal flexibel
- wissenschaftlich sauber begründbar

---

**Ende der normativen Spezifikation v4**

