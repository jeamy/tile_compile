# Tile-basierte Qualitätsrekonstruktion für DSO (Siril ≥ 1.4)

## Ziel

Kombination aus **globaler Frame-Bewertung** und **lokaler Tile-Selektion**, um bei **wechselnder Transparenz**, **Alt-Az-Rotation** und **inhomogenem Seeing** die bestmögliche Information zu rekonstruieren. Es werden **keine Frames verworfen**, sondern **gewichtet**.

---

## Grundannahmen

* OSC-Daten, linear (kein Stretch)
* vollständige Registrierung (Translation + Rotation)
* viele Kurzbelichtungen (≈ 1–2 s)
* Auflösung variabel

---

## Gesamtpipeline (Übersicht)

1. **Globale Vorbereitung (Siril)**

   * Kalibration (optional)
   * vollständige Registrierung
   * optionale *lineare* Normalisierung (Median/Hintergrund)

2. **Globale Frame-Klassifikation (Python)**

   * Ziel: Transparenz & Grundqualität erfassen

3. **Lokale Tile-Analyse (Python)**

   * Ziel: lokale Schärfe/Struktur bewerten

4. **Rekonstruktion synthetischer Qualitätsbilder**

   * gewichtetes Overlap-Add

5. **Finales Stacking (Siril)**

   * klassisches lineares Stacking

---

## 1. Globale Frame-Gewichtung

### Messgrößen (pro Frame)

* Hintergrundmittelwert B_f (Transparenz)
* Standardabweichung σ_f (Rauschen/Schleier)
* Gradientenergie E_f (Struktur)

### Normalisierung

Robust: Median + MAD

### Globaler Qualitätsindex

Q_f = 0.4·(−B̃_f) + 0.3·(−σ̃_f) + 0.3·Ẽ_f

### Globales Gewicht

G_f = exp(Q_f)

*Hinweis:* Wolkige Frames werden stark abgewichtet, aber nicht verworfen.

---

## 2. Lokale Tile-Gewichtung

### Tile-Definition (auflösungsunabhängig)

* Mediane FWHM: F (px)
* Tile-Größe: T = clip(32·F, 64, min(W,H)/6)
* Überlappung: O = 0.25·T
* Schritt: S = T − O

### Fall A: Sterne im Tile vorhanden

Messgrößen:

* FWHM_f,t
* Rundheit R_f,t
* lokaler Kontrast C_f,t

Qualitätsindex:
Q_star = 0.6·(1/FWHM²) + 0.2·R + 0.2·C

### Fall B: Keine Sterne im Tile

Messgrößen:

* lokale Gradientenergie E_f,t
* lokale Standardabweichung σ_f,t
* lokaler Hintergrund B_f,t

Qualitätsindex:
Q_struct = 0.5·E − 0.25·σ − 0.25·B

### Lokales Gewicht

L_f,t = exp(Q_star) **oder** exp(Q_struct)

---

## 3. Effektives Gewicht

W_f,t = G_f · L_f,t

---

## 4. Rekonstruktion (Tile-Ebene)

Für jedes Pixel p im Tile:

I(p) = (Σ_f W_f,t · I_f(p)) / (Σ_f W_f,t)

Randbehandlung:

* Cosine/Hanning Window
* Overlap-Add (keine harten Kanten)

Ergebnis:

* 10–30 **synthetische Qualitätsbilder**

---

## 5. Finales Stacking (Siril)

* Stack der synthetischen Bilder
* keine zusätzliche Gewichtung
* kein Drizzle
* lineare Weiterverarbeitung

---

## Abbruchkriterien

* < 30 % der Tiles zeigen verwertbare Struktur
* Qualitätsgewichte streuen kaum
* Kachelmuster oder Übergangsartefakte sichtbar
* Verbesserung < 5 % (z. B. Stern-FWHM)

---

## Kernaussage

Es werden **keine „besten Frames“ gesucht**, sondern eine **räumlich-zeitliche Qualitätskarte** aufgebaut, **kombiniert mit globaler Frame-Stabilität**. So wirkt jede Bildinformation genau dort, wo sie physikalisch valide ist.

---

## Ergänzung A – Kommentierung und Feinjustierung

### Gewichtsfaktoren anpassen

* Bei starker Bewölkung: α (Hintergrund) erhöhen
* Bei starkem Seeing: a (1/FWHM²) erhöhen
* Bei nebulösen Feldern: Struktur-Gewichte (d, e, f) priorisieren

Faustregel: **Gewichte nie hart ändern, sondern in ±0.1-Schritten**.

---

## Ergänzung B – Entscheidungs-Flowchart (textuell)

1. Sind Daten linear?

   * Nein → abbrechen
   * Ja → weiter

2. Anzahl Frames ≥ 800?

   * Nein → klassisches Stacking
   * Ja → weiter

3. Globale Qualitätsstreuung > 15 %?

   * Nein → klassisches Stacking
   * Ja → weiter

4. Tile-Analyse starten

---

## Ergänzung C – Experimentelle Test-Checkliste

Vor dem Lauf:

* [ ] Registrierung vollständig (inkl. Rotation)
* [ ] Keine Dithering-Versätze aktiv
* [ ] Daten linear

Während des Laufs:

* [ ] Tile-Gewichte prüfen (Histogramm)
* [ ] Anteil verwertbarer Tiles > 30 %

Nach dem Lauf:

* [ ] Vergleich FWHM vorher/nachher
* [ ] Prüfung auf Kachelmuster
* [ ] Hintergrund-RMS vergleichen

---

## Ergänzung D – Minimal-Python-Design (ohne Code)

Module:

1. IO-Modul: FITS lesen/schreiben
2. Statistik-Modul: globale Metriken
3. Tile-Modul: Zerlegung + Overlap
4. Qualitäts-Modul: Gewichte berechnen
5. Rekonstruktions-Modul: Overlap-Add
6. Export-Modul: synthetische Frames

Reihenfolge strikt einhalten, keine Rückkopplung zwischen Modulen.

---

## Validierungsplots (Pflichtbestandteil)

Ziel der Validierungsplots ist eine **objektive, visuelle und quantitative Überprüfung**, ob die tile-basierte Rekonstruktion dem klassischen Stacking **tatsächlich überlegen** ist.

---

### Plot 1 – FWHM-Verteilung (vor / nach)

**Typ:** Histogramm + Boxplot

**Daten:**

* klassische Stack-Frames oder Einzel-Frames
* synthetische Qualitätsbilder

**Aussage:**

* Verschiebung der Median-FWHM nach links
* Reduktion der Varianz

**Abbruchregel:**

* Verbesserung < 5 % → abbrechen

---

### Plot 2 – FWHM über Bildfeld (2D-Map)

**Typ:** Heatmap (x/y)

**Daten:**

* lokale FWHM-Werte (Sterne)

**Aussage:**

* Homogenisierung des Feldes
* Reduktion randnaher Seeing-/Rotationsartefakte

**Warnsignal:**

* scharfe Kanten → Tile-Artefakte

---

### Plot 3 – Hintergrundniveau pro Frame

**Typ:** Liniendiagramm (Frame-Index)

**Daten:**

* globaler Hintergrund vor Gewichtung
* effektiver Beitrag nach Gewichtung

**Aussage:**

* Wolkige Frames korrekt abgewichtet

---

### Plot 4 – Qualitätsgewicht vs. Zeit

**Typ:** Scatter / Linie

**Daten:**

* G_f (global)
* Mittelwert von L_f,t pro Frame

**Aussage:**

* Trennung zwischen guten und schlechten Zeitabschnitten

---

### Plot 5 – Tile-Gewichtsverteilung

**Typ:** Histogramm

**Daten:**

* W_f,t für alle Tiles

**Aussage:**

* ausreichende Spreizung

**Abbruchregel:**

* nahezu uniforme Verteilung → kein Selektionsgewinn

---

### Plot 6 – Differenzbild (klassisch vs. rekonstruiert)

**Typ:** Bild + Histogramm

**Daten:**

* rekonstruiertes Bild − klassischer Stack

**Aussage:**

* Detailgewinn sichtbar
* keine großskaligen Artefakte

**Warnsignal:**

* periodische Muster → Overlap/Feathering fehlerhaft

---

### Plot 7 – SNR vs. Auflösung

**Typ:** Scatter

**Daten:**

* lokale SNR
* lokale FWHM

**Aussage:**

* erwarteter Trade-off
* keine künstliche Überschärfung

---

## Validierungsentscheidung

Die Methode gilt als **erfolgreich validiert**, wenn:

* Median-FWHM ↓ ≥ 5–10 %
* Feldhomogenität ↑
* Hintergrund-RMS ≤ klassisches Stacking
* keine systematischen Artefakte sichtbar

Andernfalls ist klassisches Stacking die bessere Wahl.

---

## Synthetische Frames – formale Definition

### Grundsatz

Ein **synthetisches Frame** repräsentiert **einen physikalisch kohärenten Beobachtungszustand**, nicht einen Zeitabschnitt.

---

### Empfohlene Standarddefinition (Produktion)

**Clustering nach Metrikvektor**

Für jedes Frame f wird ein Metrikvektor definiert:

v_f = [
G_f,                 # globales Frame-Gewicht
⟨L_f,t⟩,              # mittlere lokale Tile-Qualität
Var(L_f,t),           # Feldinhomogenität
B_f,                 # Hintergrundniveau
σ_f                  # globales Rauschen
]

Die Frames werden anhand dieses Vektors in **k = 15–30 Cluster** gruppiert.

* Pro Cluster wird **ein synthetisches Frame** rekonstruiert
* Cluster repräsentieren unterschiedliche Seeing-/Transparenzzustände

**Fallbacks:**

* Qualitätsquantile (nach G_f)
* Zeitbuckets nur bei explizit stabilen Bedingungen

---

## Normalisierung (Pflicht)

### Warum zwingend

Ohne Normalisierung werden Transparenzschwankungen fälschlich als Struktur- oder Qualitätsunterschied interpretiert.

### Zulässige Normalisierung

* **global, linear, einmalig**
* Skalierung auf:

  * globalen Median oder
  * globales Hintergrundniveau
* **getrennt pro Farbkanal (OSC)**

Formal:
I'_f = I_f / median(I_f)  oder  I_f / B_f

### Verboten

* Histogram-Stretch
* Asinh / Log
* lokale oder adaptive Normalisierung vor Tile-Analyse

---

## Lokale Helligkeitsartefakte

### Typische Ursachen

* Wolkenränder / Cirren
* Gradienten durch Alt-Az-Rotation
* Nebelstrukturen vs. atmosphärischer Schleier

### Produktionsfeste Gegenmaßnahmen

1. **Hintergrund-entkoppelte Metriken**

   * Gradientenergie nach Subtraktion eines lokalen Hintergrundmodells

2. **Helligkeitsinvariante Gewichtung**

   * Qualitätsmetriken normiert auf lokale σ
   * keine absoluten Intensitäten

3. **Rekonstruktions-Normalisierung**

   * jedes Tile vor Overlap-Add auf gemeinsamen Median skalieren

---

## Pflicht-Validierung gegen Artefakte

* lokale Median-Map vor / nach Rekonstruktion
* Differenz darf keine Tile-Struktur zeigen

**Abbruchregel:**

* sichtbare Patchwork-Helligkeit → Lauf abbrechen

---

## Abschluss

Das Dokument definiert nun **Methodik, Gewichtung, Rekonstruktion, Clustering, Normalisierung und Validierung** vollständig und produktionsnah.
