# Tile‑basierte Qualitätsrekonstruktion für DSO – Methodik v2

**Status:** Referenzspezifikation (Single Source of Truth)
**Gültig für:** `tile_compile.proc` (Clean Break) + `tile_compile.yaml`

---

## 1. Zielsetzung

Ziel ist die Rekonstruktion eines **orts‑ und zeitabhängig optimal gewichteten Signals** aus vollständig registrierten, linearen Kurzbelichtungen astronomischer Deep‑Sky‑Objekte.

Die Methode modelliert explizit zwei orthogonale Einflussgrößen:

* **globale atmosphärische Qualität** (Transparenz, Schleier, Hintergrunddrift)
* **lokale seeing‑ und strukturbedingte Qualität** (Schärfe, Detailtragfähigkeit)

Es findet **keine Frame‑Selektion** statt. Jeder Frame trägt entsprechend seiner physikalischen Aussagekraft zum Endergebnis bei.

---

## 2. Grundannahmen (verbindlich)

* Daten sind **linear** (kein Stretch, keine nichtlinearen Operatoren)
* OSC‑Daten, Verarbeitung getrennt pro Farbkanal
* vollständige Registrierung (Translation + Rotation, keine Residuen)
* große Anzahl Kurzbelichtungen (typ. ≥ 800 Frames)
* keine feste Pixel‑ oder Auflösungsannahme

Ein Verstoß gegen diese Annahmen führt zum **Abbruch des Laufs**.

---

## 3. Gesamtpipeline (normativ)

1. Registrierung der Rohframes
2. **Globale lineare Normalisierung (Pflicht, einmalig)**
3. Berechnung globaler Frame‑Metriken
4. seeing‑adaptive Tile‑Geometrie
5. Lokale Tile‑Metriken und Gewichtung
6. Tile‑weise Rekonstruktion (Overlap‑Add)
7. **Zustandsbasierte Clusterung der Frames**
8. Rekonstruktion synthetischer Qualitätsframes
9. Finales lineares Stacking
10. Validierung und Abbruchentscheidung

Die Pipeline ist **streng linear**. Es existieren keine Rückkopplungen.

---

## 4. Globale Normalisierung (Pflicht)

### Zweck

Entkopplung photometrischer Transparenzschwankungen von Qualitätsmetriken.

### Anforderungen

* global
* linear
* exakt einmal
* **vor jeder Metrikberechnung**
* getrennt pro Farbkanal

### Zulässige Verfahren

* Hintergrundbasierte Skalierung (maskiert, robust)
* Fallback: Skalierung auf globalen Median

Formal:

```
I'_f = I_f / B_f
```

### Verboten

* Histogram‑Stretch
* Asinh / Log
* lokale oder adaptive Normalisierung vor Tile‑Analyse

---

## 5. Globale Frame‑Metriken

Für jeden registrierten, normalisierten Frame *f* werden bestimmt:

* **B_f** – globales Hintergrundniveau (robust, maskiert)
* **σ_f** – globales Rauschen
* **E_f** – Gradientenergie (großskalige Struktur)

### Normalisierung

Alle Metriken werden mittels **Median + MAD** robust skaliert.

### Globaler Qualitätsindex

[
Q_f = \alpha(-\tilde B_f) + \beta(-\tilde\sigma_f) + \gamma\tilde E_f
]

mit:

* α + β + γ = 1
* Default: α = 0.4, β = 0.3, γ = 0.3

Q_f wird auf **[−3, +3]** begrenzt.

### Globales Gewicht

[
G_f = \exp(Q_f)
]

**Semantik:** G_f kodiert ausschließlich globale atmosphärische Qualität.

---

## 6. Tile‑Geometrie (seeing‑adaptiv)

### FWHM‑Schätzung

* aus **registrierten Frames**
* robuste Stichprobe über viele Sterne und Frames
* Ausreißer (hohe Elliptizität, geringe SNR) werden verworfen

### Tile‑Größe

[
T = \mathrm{clip}(32 \cdot F,; 64,; \min(W,H)/6)
]

* Überlappung: O = 0.25 · T
* Schrittweite: S = T − O

Diese Definition ist **auflösungs‑ und seeing‑invariant**.

---

## 7. Lokale Tile‑Metriken

Für jedes Tile *t* und jeden Frame *f*:

### Fall A – Sterne im Tile

**Messgrößen:**

* FWHM_f,t
* Rundheit R_f,t
* lokaler Kontrast C_f,t

**Qualitätsindex (Standard):**

[
Q_{star} = 0.6,(−\widetilde{\log(\mathrm{FWHM})}) + 0.2,\tilde R + 0.2,\tilde C
]

### Fall B – Keine Sterne im Tile

**Messgrößen:**

* Gradientenergie E_f,t
* lokale Standardabweichung σ_f,t
* lokales Hintergrundniveau B_f,t

**Qualitätsindex (Default):**

[
Q_{struct} = 0.7,\widetilde{(E/\sigma)} − 0.3,\tilde B
]

Diese Metrik ist weitgehend helligkeitsinvariant.

### Lokales Gewicht

Alle lokalen Qualitätsindizes werden auf **[−3, +3]** begrenzt.

[
L_{f,t} = \exp(Q_{local})
]

---

## 8. Effektives Gewicht

[
W_{f,t} = G_f \cdot L_{f,t}
]

G_f und L_f,t repräsentieren orthogonale Informationsachsen.

---

## 9. Tile‑Rekonstruktion

Für jedes Pixel *p* im Tile *t*:

[
I_t(p) = \frac{\sum_f W_{f,t} I_f(p)}{\sum_f W_{f,t}}
]

### Stabilitätsregeln

* Falls Σ W_f,t < ε:

  * Tile invalidieren **oder**
  * Fallback: ungewichtetes Mittel

### Randbehandlung

* Cosine‑ oder Hanning‑Fenster
* Overlap‑Add (keine harten Kanten)

### Tile‑Normalisierung

* linearer Rescale auf gemeinsamen Tile‑Median
* **nach** Hintergrundsubtraktion
* **keine** Rückwirkung auf Qualitätsmetriken

---

## 10. Synthetische Qualitätsframes

### Grundsatz

Ein synthetisches Frame repräsentiert einen **physikalisch kohärenten Beobachtungszustand**, nicht einen Zeitabschnitt.

### Zustandsvektor

Für jeden Frame *f*:

[
v_f = (G_f, \langle Q_{tile} \rangle, \mathrm{Var}(Q_{tile}), B_f, \sigma_f)
]

### Clusterung

* Clusterung der **Frames**, nicht der Tiles
* k = 15–30 Cluster
* pro Cluster genau ein synthetisches Frame

Fallbacks (nur bei explizit stabilen Bedingungen):

* Quantile nach G_f
* Zeitbuckets

---

## 11. Finales Stacking

* lineares Stacking der synthetischen Frames
* keine zusätzliche Gewichtung
* kein Drizzle

---

## 12. Validierung und Abbruch

### Erfolgsbedingungen

* Median‑FWHM ↓ ≥ 5 – 10 %
* Feldhomogenität ↑
* Hintergrund‑RMS ≤ klassisches Stacking
* keine systematischen Tile‑Artefakte

### Abbruchkriterien

* < 30 % der **signalführenden Tiles** verwertbar
* kaum Streuung der Tile‑Gewichte
* sichtbare Kachel‑ oder Übergangsartefakte
* Verletzung der Normalisierungsregeln

---

## 13. Kernaussage

Die Methode ersetzt die Suche nach „besten Frames“ durch eine **räumlich‑zeitliche Qualitätskarte**, die jede Bildinformation genau dort nutzt, wo sie physikalisch valide ist.

Diese Spezifikation ist **normativ**. Abweichungen erfordern explizite Versionierung.

---

## Appendix A – Implementationshinweise (nicht-normativ, aber verbindlich empfohlen)

Dieser Appendix konkretisiert rechnerische und algorithmische Details, um **reproduzierbare, robuste Implementierungen** zu gewährleisten. Er erweitert die Methodik, ohne sie semantisch zu verändern.

### A.1 Hintergrundschätzung (global und lokal)

**Ziel:** robuste Trennung von Signal und atmosphärischem Schleier.

Empfohlenes Verfahren:

* Grobe Objektmaske (z. B. Sigma-Clip + Dilatation)
* Berechnung des Hintergrunds aus verbleibenden Pixeln
* Robuststatistik (Median oder biweight location)

Hinweis:

> Der Hintergrund darf **keine strukturellen Gradienten** enthalten, die später in E oder E/σ eingehen.

---

### A.2 Rauschschätzung σ

**Global:**

* robuste Standardabweichung aus hintergrundmaskierten Pixeln
* keine Glättung vor der Schätzung

**Lokal (Tile):**

* identisches Verfahren, aber nur innerhalb des Tiles
* σ dient explizit als **Normierung** für Strukturmetriken

---

### A.3 Gradientenergie E

**Definition (empfohlen):**

E = mean(|∇I|²)

Alternative (robuster gegen Ausreißer):

E = median(|∇I|²)

Implementationshinweise:

* Sobel- oder Scharr-Operator
* optionale leichte Vor-Glättung (σ ≤ 1 px), **aber konsistent global & lokal**
* Randpixel verwerfen

Wichtig:

> Unterschiedliche Gradientdefinitionen verändern die Skala, **nicht** die Methodik – Skalierung wird durch MAD-Normalisierung aufgefangen.

---

### A.4 Sternselektion für FWHM

Empfohlene Kriterien:

* SNR > definierter Schwelle
* Elliptizität < 0.4
* keine Sättigung

FWHM:

* gemessen über PSF-Fit oder radiales Profil
* **log(FWHM)** erst nach der robusten Aggregation anwenden

---

### A.5 Normalisierung (Median + MAD)

Für jede Metrik x:

x̃ = (x − median(x)) / (1.4826 · MAD(x))

Hinweise:

* getrennt für jede Metrik
* getrennt für global vs. lokal
* keine Vermischung der Skalen

---

### A.6 Tile-Normalisierung vor Overlap-Add

Ablauf:

1. lokalen Hintergrund schätzen und subtrahieren
2. Tile auf gemeinsamen Median skalieren
3. Fensterfunktion anwenden
4. Overlap-Add

Guard:

* Wenn |median(tile_bgfree)| < ε_median, dann **keine** Skalierung durchführen (Scale = 1.0)

Ziel:

* Vermeidung von Patchwork-Helligkeit
* keine Beeinflussung der Qualitätsmetriken

---

### A.7 Clustering

Empfehlungen:

* Standard: k-means oder GMM
* Feature-Vektor vorher standardisieren
* mehrere Initialisierungen, bestes Inertia-/LLH-Ergebnis wählen

Warnung:

> Zeitbasierte Clusterung ist **kein Ersatz** für Zustandsclusterung.

---

### A.8 Numerische Stabilität

* ε im Nenner der Tile-Rekonstruktion explizit setzen
* exp(Q) ggf. begrenzen (z. B. Q ∈ [−3, 3])
* Double Precision bevorzugen

Empfohlene Defaults:

* ε = 1e−6
* ε_median = 1e−6

---

### A.9 Debug- und Diagnoseartefakte (empfohlen)

Während der Entwicklung speichern:

* Histogramme von Q_f und Q_local
* 2D-Karten der Tile-Gewichte
* Differenzbild rekonstruiert − klassisch

Diese Artefakte sind **kein Teil der Produktion**, aber essentiell für Verifikation.

---

## Appendix B – Validierungsplots (formal spezifiziert)

Dieser Appendix definiert **verbindliche Validierungsartefakte**, anhand derer entschieden wird, ob ein Lauf als **erfolgreich**, **grenzwertig** oder **fehlgeschlagen** gilt. Alle Plots sind aus **produktionsrelevanten Daten** zu erzeugen.

### B.1 FWHM-Verteilung (vor / nach)

**Typ:** Histogramm + Boxplot

**Eingangsdaten:**

* klassische Referenz (Stack oder Einzel-Frames)
* synthetische Qualitätsframes

**Metriken:**

* Median-FWHM
* Interquartilsabstand

**Akzeptanz:**

* Median-FWHM-Reduktion ≥ `validation.min_fwhm_improvement_percent`

---

### B.2 FWHM-Feldkarte (2D)

**Typ:** Heatmap über Bildkoordinaten

**Eingangsdaten:**

* lokale FWHM-Messungen aus Stern-Tiles

**Ziel:**

* Homogenisierung des Feldes
* Reduktion randnaher Seeing-/Rotationsartefakte

**Warnsignal:**

* harte Übergänge entlang Tile-Grenzen

---

### B.3 Globaler Hintergrund vs. Zeit

**Typ:** Liniendiagramm

**Eingangsdaten:**

* B_f (roh) = vor globaler Normalisierung (registrierte, aber noch unskalierte Frames)
* effektiver Beitrag nach Gewichtung

**Ziel:**

* korrekte Abwertung wolkiger Phasen

---

### B.4 Globale und lokale Gewichte über Zeit

**Typ:** Scatter / Linie

**Eingangsdaten:**

* G_f
* ⟨L_f,t⟩ pro Frame

**Ziel:**

* klare Trennung unterschiedlicher Zustände

---

### B.5 Tile-Gewichtsverteilung

**Typ:** Histogramm

**Eingangsdaten:**

* W_f,t aller Tiles

**Akzeptanz:**

* Varianz ≥ `validation.min_tile_weight_variance`

---

### B.6 Differenzbild

**Typ:** Bild + Histogramm

**Definition:**

Differenz = Rekonstruktion − klassisches Stacking

**Ziel:**

* Detailgewinn sichtbar
* keine großskaligen systematischen Muster

**Abbruch:**

* periodische Tile-Strukturen

---

### B.7 SNR vs. Auflösung

**Typ:** Scatter

**Eingangsdaten:**

* lokale SNR
* lokale FWHM

**Ziel:**

* physikalisch plausibler Trade-off
* keine künstliche Überschärfung

---

## Appendix C – Komplexitäts- und Performance-Budget

Dieser Appendix dient der **Planung und Skalierung** produktiver Läufe.

### C.1 Rechenkomplexität (grobe Ordnung)

Sei:

* F = Anzahl Frames
* T = Anzahl Tiles
* P = Pixel pro Tile

**Globale Metriken:**  O(F · N_pixels)

**Tile-Analyse:**      O(F · T · P)

**Rekonstruktion:**   O(T · F · P)

Die Tile-Analyse dominiert die Laufzeit.

---

### C.2 Speicherbedarf

* Ein Frame im RAM (float32): ~4 · W · H Bytes
* Tile-Puffer: ~T · P · sizeof(float)

**Empfehlung:**

* Streaming pro Tile
* keine vollständige Frame-Matrix im RAM

---

### C.3 I/O-Strategie

* Registrierung: einmaliges Lesen/Schreiben
* Tile-Analyse: bevorzugt sequentiell
* synthetische Frames: explizit zwischenspeichern

Vermeiden:

* zufällige Tile-Zugriffe auf rotierenden Medien

---

### C.4 Parallelisierung

Geeignete Ebenen:

* Tiles (embarassingly parallel)
* Frames innerhalb eines Tiles (optional)

Hinweise:

* Globale Normalisierung ist **pro Frame unabhängig** und kann parallelisiert werden (I/O kann limitieren).
* Zustandsclusterung ist typischerweise nicht der Bottleneck; parallelisieren ist optional und implementationsabhängig.

Option: RabbitMQ‑basierte Parallelisierung

Diese Option ist für eine spätere Implementierung vorgesehen und dient der horizontalen Skalierung über mehrere Worker.

* Task‑Queue: RabbitMQ
* Granularität:
  * bevorzugt: **Tile‑Tasks** (ein Task = Tile t über alle Frames f)
  * optional: Frame‑Tasks innerhalb eines Tiles (nur wenn I/O lokal schnell ist)
* Payload (minimal): tile_id, tile_bbox, frame_index_range, benötigte Metrik‑Konfig
* Ergebnisse:
  * rekonstruierter Tile‑Block + Summenstatistiken (z. B. ΣW, Tile‑Median nach bg‑sub)
  * separater Kanal/Queue für Diagnoseartefakte (Histogramme, QA‑Maps)
* Aggregation:
  * Master sammelt Tile‑Ergebnisse und führt Overlap‑Add deterministisch aus
  * deterministische Seeds/Sortierung zur Reproduzierbarkeit
* Fehlertoleranz:
  * idempotente Tasks (Tile kann neu gerechnet werden)
  * Dead‑Letter‑Queue für fehlerhafte Tiles

---

### C.5 Laufzeitabschätzung

Für typische Werte:

* F ≈ 1000
* T ≈ 200–400
* P ≈ (64–256)²

Erwartung:

* CPU (8–16 Kerne): Stundenbereich
* GPU-Beschleunigung: optional, nicht zwingend

---

### C.6 Abbruch bei Laufzeitüberschreitung

Die folgenden Limits sind **verbindlich**:

* `runtime_limits.tile_analysis_max_factor_vs_stack`
* `runtime_limits.hard_abort_hours`

Bei Überschreitung erfolgt ein **kontrollierter Abbruch**.
