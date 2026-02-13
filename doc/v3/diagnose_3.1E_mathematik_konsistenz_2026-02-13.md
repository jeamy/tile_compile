# Diagnose: Methodik v3.1E – Mathematik, Konsistenz, Logik

**Datum:** 2026-02-13  
**Analysiertes Dokument:** `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.1_erweitert.md`

---

## Kurzfazit

Die Methodik ist in der Grundidee mathematisch tragfähig (robuste Metrik-Normalisierung, getrennte globale/lokale Gewichte, klare Fallbacks), hat aber mehrere **innere Widersprüche** und einige **normative Unschärfen**, die bei Implementierung zu divergierendem Verhalten führen können.

**Gesamtbewertung:**
- Mathematisches Fundament: **gut**
- Interne Konsistenz: **mittel**
- Normative Eindeutigkeit: **mittel bis schwach**

---

## 1) Kritische Inkonsistenzen / logische Widersprüche

### 1.1 „Streng linear“ vs. explizit nichtlineare Verfahren

- Harte Annahme: „Pipeline ist streng linear“ in §2.1 (@doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.1_erweitert.md#45-46)
- Gleichzeitig sind normativ/optional enthalten:
  - Soft-Threshold-Denoising (§3.3.1, @...#521-530)
  - Wiener-Filter (§3.3.1, @...#532-546)
  - Sigma-Clipping (§3.6, §3.8, @...#708-743, @...#813-846)

**Problem:** Diese Schritte sind mathematisch nichtlinear. Die Forderung „streng linear“ ist so nicht haltbar.

**Empfehlung:** In §2.1 präzisieren:
- „photometrische Abbildung bleibt linear bis inkl. Rekonstruktionskern“ **oder**
- „keine nichtlinearen globalen Tonkurven; robuste/statistische Nichtlinearitäten zur Ausreißerbehandlung erlaubt“.

---

### 1.2 „Keine Frame-Selektion“ vs. Sigma-Clipping-Formulierung

- Invariante Regel §2.1/§6.7: keine Frame-Selektion, jede Formel nutzt alle Frames (@...#43-44, @...#1006-1008)
- Gleichzeitig werden Pixelwerte iterativ verworfen (Sigma-Clipping, @...#715-724, @...#820-829)

**Problem:** Semantischer Konflikt: Pixel-basierte Rejektion ist keine Frame-Selektion, klingt aber im Text teilweise wie Verstoß.

**Empfehlung:** Begriff strikt trennen:
1. **Frame-Selektion** (ganze Frames) verboten
2. **Pixel-Ausreißerrejektion** erlaubt (mit dokumentiertem Fallback)

---

### 1.3 Reduced-Mode-Grenzen widersprüchlich

- Minimum laut Tabelle: ≥ 50 Frames (§2.2, @...#52-53)
- Reduced Mode definiert für 50–199 (§2.4, @...#62-69)
- Beispiel in Degradierungstabelle: „< 50 Frames → Reduced Mode“ (§2.4, @...#91-92)

**Problem:** `<50` kann nicht gleichzeitig „Minimum verletzt“ und „Reduced Mode“ sein.

**Empfehlung:** Eindeutig festlegen:
- `<50`: kritisch/Abbruch oder separater „Emergency-Mode“
- `50–199`: Reduced Mode

---

### 1.4 Phase-7-Ergebnis vs. spätere Pipeline-Nutzung

- Phase 7 rekonstruiert tile-basiert (§3.6)
- §3.8 erzeugt synthetische Frames standardmäßig aus **Original-Frames** mit globalen Gewichten (@...#791-797)

**Problem:** Im Default (`synthetic.weighting: global`) kann die aufwendig rekonstruktive Information aus §3.6 im finalen Produkt nur indirekt wirken. Das ist logisch möglich, aber konzeptionell kontraintuitiv.

**Empfehlung:** Im Text klarstellen, ob §3.6 primär
1. Endprodukt für Reduced Mode,
2. Diagnostik/Qualitätsmodell,
3. oder Pflichtvorstufe für `tile_weighted` Synthese
ist.

---

## 2) Formale / mathematische Unschärfen

### 2.1 Kanalindex uneinheitlich

- Globales Gewicht als `G_{f,c}` (§3.2, @...#400-402)
- Effektives Gewicht dann ohne Kanalindex `W_{f,t} = G_f · L_{f,t}` (§3.5, @...#647-649)
- Später wieder kanalindiziert `W_{f,t,c}` (§3.8, @...#801-803)

**Problem:** Notation springt; Implementierende können unterschiedliche Semantik wählen.

**Empfehlung:** Einheitlich überall `G_{f,c}`, `L_{f,t,c}`, `W_{f,t,c}`.

---

### 2.2 Doppelter/inkompletter Fallback-Block in §3.6

- Stabilitätsregeln erscheinen doppelt (§3.6, @...#663-682 und @...#695-706)
- Zweiter Block enthält „Definiere den Nenner“ ohne Formel (@...#697-700)

**Problem:** Normative Redundanz + unvollständige Definition.

**Empfehlung:** Einen konsolidierten Block behalten, den anderen entfernen.

---

### 2.3 Hann-Funktion nicht diskret spezifiziert

- `hann(t) = 0.5*(1-cos(2πt))` (§3.6, @...#685-687)

**Problem:** Ohne Diskretisierung (z. B. `i/(N-1)`) ist die Implementierung nicht eindeutig.

**Empfehlung:** Diskrete Form normativ festlegen:
- `hann(i,N)=0.5*(1-cos(2π*i/(N-1)))`, `i=0..N-1`, Sonderfall `N=1`.

---

### 2.4 Tile-Normalisierung kann degenerieren

- `T'' = T'/median(|T'|)` (§3.6, @...#692-694)

**Problem:** Bei sehr schwachem Signal kann Nenner ~0 sein.

**Positiv:** Guard ist in Anhang A.6 erwähnt (@...#1148-1151), aber nicht im normativen Haupttext.

**Empfehlung:** Guard in §3.6 selbst normativ aufnehmen (`if med_abs < eps -> scale=1`).

---

### 2.5 Adaptive Gewichte (Varianz-basiert) nicht eindeutig auf welcher Skala

- §3.2 beschreibt `Var(B), Var(σ), Var(E)` (@...#410-418)
- Unklar, ob auf Rohmetrik oder robust-normalisierter Metrik.

**Problem:** Bei Rohmetriken dominieren Einheiten/Skalen, mathematisch verzerrt.

**Empfehlung:** Normativ festlegen: Varianzen auf robust-normalisierten Metriken (`B̃, σ̃, Ē`).

---

## 3) Interne Text-/Pseudocode-Probleme

### 3.1 Pseudocode A.10 mit Variablenfehler

In §A.10 wird mehrfach `return result` genutzt, obwohl `result` dort nicht gebunden wird (@...#1219-1233).

**Empfehlung:** Rückgabewert der jeweiligen Methode explizit einer Variable zuweisen.

### 3.2 Pseudocode A.11 mit undefinierten Symbolen

`g1_row/g1_col` werden verwendet, aber nicht definiert (@...#1285-1287).

**Empfehlung:** G1/G2-Koordinaten vor Nutzung explizit definieren.

---

## 4) Sachlich fragwürdige/zu starke Normen

### 4.1 PCC-Test „keine negativen Matrixelemente“

- Testfall 12 fordert: Determinante > 0 und kein Matrixelement negativ (@...#1025-1028)

**Problem:** Für allgemeine 3x3-Farbmatrizen ist „keine negativen Elemente“ zu restriktiv und nicht allgemein physikalisch zwingend.

**Empfehlung:** Stattdessen numerische Stabilität + farbmetrische Güte normieren (z. B. Konditionszahl, Residuen, Kanal-Gain-Bounds).

---

## 5) Optimierungsmöglichkeiten (fachlich + rechnerisch)

1. **Normatives Kernmodell verschlanken**
   - Pflichtkern (global/local weights, reconstruction, fallbacks) strikt von optionalen Enhancements (Wiener/Sigma-Clip/PCC) trennen.

2. **Eindeutige mathematische Konventionen zentralisieren**
   - Eine Tabelle „Symbol, Indexraum, Einheit, Berechnungsebene“ ergänzt §3.2–§3.6.

3. **Stabilitätskonstanten global vereinheitlichen**
   - `eps_weight`, `eps_median`, `eps_var` zentral definieren statt verteilt.

4. **Proof-of-Consistency Annex ergänzen**
   - Kurzbeweis: keine Bias-Einführung durch Clamping + exp + Fallback-Mittelwert.

5. **Default-Pfad semantisch präzisieren**
   - Wenn `synthetic.weighting=global`, explizit benennen, welche Information aus §3.6 ins Endbild eingeht (oder nicht).

---

## 6) Gesamturteil

Die Spezifikation ist methodisch stark, aber nicht vollständig „widerspruchsfrei normativ“. Mit den oben genannten Korrekturen (insb. §2.1 Linearität, §2.4 Reduced-Mode-Grenze, §3.5/§3.8 Notation, §3.6 Konsolidierung) wird sie deutlich robuster und implementierungsstabiler.

**Priorität für Korrektur:**
1. Linearitäts- und Selektionssemantik klären
2. Reduced-Mode-Logik reparieren
3. §3.6 Redundanz/Unvollständigkeit bereinigen
4. Notation kanalweise konsistent machen
