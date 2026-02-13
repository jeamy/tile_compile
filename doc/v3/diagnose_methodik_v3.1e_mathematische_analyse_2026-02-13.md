# Diagnose: Methodik v3.1 Erweitert

**Analyst:** Cascade  
**Datum:** 2026-02-13  
**Datei:** `tile_basierte_qualitatsrekonstruktion_methodik_v_3.1_erweitert.md`  
**Vergleichsbasis:** `tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.md`

---

## Zusammenfassung

Die Methodik v3.1E ist **grundsätzlich mathematisch korrekt**, enthält jedoch **mehrere Inkonsistenzen, Redundanzen und einen kritischen Widerspruch** im Pfad-A/Pfad-B-Design. Die spätere v3.2 korrigiert diese Probleme durch Konsolidierung auf den CFA-basierten Pfad als einzigen normativen Pfad.

**Gesamteinschätzung:** Die Methodik ist implementierbar, erfordert aber präzisere Spezifikationen für den Produktionseinsatz.

---

## 1. Mathematische Korrektheit

### 1.1 Gewichtungsformeln (Korrekt)

Die Kernformeln sind mathematisch solide:

- Globale Metrik-Normalisierung: `z(x) = (x - median(x)) / (1.4826 * MAD(x))` ✓
- Globaler Qualitätsindex: `Q_{f,c} = α(-z(B)) + β(-z(σ)) + γ(z(E))` mit `α+β+γ=1` ✓
- Globales Gewicht: `G_{f,c} = exp(Q_{f,c})` mit Clamping auf `[-3,+3]` vor `exp` ✓
- Lokales Gewicht: `L_{f,t,c} = exp(Q_{local})` ✓
- Effektives Gewicht: `W_{f,t,c} = G_{f,c} · L_{f,t,c}` ✓

**Beweis der Orthogonalität:** Die Zerlegung in globale (G) und lokale (L) Komponenten ist mathematisch valide, da beide Faktoren multiplikativ kombiniert werden und unabhängige Qualitätsdimensionen modellieren.

### 1.2 Tile-Rekonstruktion (Korrekt mit Einschränkung)

Die gewichtete Rekonstruktion:
```
I_t(p) = Σ_f W_{f,t,c} · I_f(p) / Σ_f W_{f,t,c}
```

ist mathematisch korrekt. Der Low-Weight-Fallback auf ungewichtetes Mittel ist konservativ und verletzt nicht das "Keine-Frame-Selektion"-Prinzip.

**Einschränkung:** Die Formel wird in §3.6 doppelt definiert (Zeilen 660-661 und 678-679), was zu Verwirrung führen kann.

### 1.3 Overlap-Add mit Hann-Fenster (Korrekt)

Die Spezifikation der Hann-Funktion als separable 2D-Fenster:
```
w(x,y) = hann(x) · hann(y)
hann(t) = 0.5 · (1 - cos(2πt))
```

ist mathematisch korrekt und garantiert konsistente Überlappungsbereiche.

---

## 2. Logische Fehler und Widersprüche

### 2.1 KRITISCH: Doppelter Pfad mit unklarer Normativität (Pfad A vs. Pfad B)

**Problem:** Die Methodik spezifiziert zwei parallele Pfade:

- **Pfad A:** Registrierung + Debayer (§A.1-A.3)
- **Pfad B:** CFA-basiert (§B.1-B.3)

Beide Pfade beanspruchen, "für Produktion empfohlen" zu sein. Dies führt zu:

1. **Mehrfacher Implementierungsaufwand**
2. **Unterschiedlicher mathematischer Semantik** (Pfad A debayert vor der Tile-Analyse, Pfad B erst danach)
3. **Inkonsistente Qualitätsmetriken** (Pfad A arbeitet auf debayerten RGB-Daten, Pfad B auf separaten Kanälen)

**Widerspruch:** In §A.2.2 wird behauptet: "Kanalübergreifendes Stacken führt zu kohärenter Addition farbabhängiger Resampling-Residuen" - aber Pfad A debayert VOR der gemeinsamen Pipeline, was genau dieses Problem impliziert.

**Korrektur (v3.2):** Pfad A wurde entfernt, CFA-basierter Pfad ist nun der einzige normative Pfad.

### 2.2 Notations-Inkonsistenz: Kanalindex c

In §3.5 (Effektives Gewicht) wird definiert:
```
W_{f,t} = G_f · L_{f,t}
```

Ab §3.6 (Tile-Rekonstruktion) erscheint plötzlich der Kanalindex:
```
D_{t,c} = Σ_f W_{f,t,c}
```

**Problem:** Die Notation wechselt unklar zwischen kanalaggregiert (`W_{f,t}`) und kanalsepariert (`W_{f,t,c}`). Für eine korrekte Implementierung muss durchgehend der Kanalindex `c` verwendet werden.

**Korrektur (v3.2):** Notation vereinheitlicht auf `f,t,c` durchgängig.

### 2.3 Doppelte Definition von D_t,c und Fallback-Regeln

In §3.6 werden die Fallback-Regeln für `D_t,c < ε` **zweimal** definiert:

1. **Erste Definition** (Zeilen 663-679):
   - Nenner `D_t = Σ_f W_{f,t}`
   - Fallback auf ungewichtetes Mittel
   - Markierung als `fallback_used`

2. **Zweite Definition** (Zeilen 695-706):
   - Fast identische Formulierung
   - Aber: `D_t,c` statt `D_t`
   - Epsilon explizit als `1e-6` genannt

**Problem:** Dies ist redundant und verwirrend. Die zweite Definition überschreibt/ergänzt die erste nicht klar.

**Empfohlene Korrektur:** Eine eindeutige Definition mit kanalweise korrekter Notation:
```
D_{t,c} = Σ_f W_{f,t,c}

if D_{t,c} ≥ eps_weight:
    R_{t,c}(p) = Σ_f W_{f,t,c} · I_{f,c}(p) / D_{t,c}
else:
    R_{t,c}(p) = (1/N) · Σ_f I_{f,c}(p)
    fallback_used[t,c] = true
```

### 2.4 Testfall 12 (PCC-Matrix) ist mathematisch fragwürdig

**Spezifikation in v3.1E (Zeile 1027):**
```
Then: Die Matrix-Determinante muss größer als 0 sein, und kein Element darf negativ sein
```

**Problem:** Die Forderung "kein Element darf negativ sein" ist zu streng. Eine Farbkorrekturmatrix kann durchaus negative Elemente haben (z.B. für Kreuzkopplungskorrektur), solange die Determinante positiv bleibt (orientierungserhaltend).

**Korrektur (v3.2):** Testfall geändert zu:
```
positive Determinante, begrenzte Konditionszahl, Residuen unter Schwellwert
```

Dies ist mathematisch robuster und praxisnah.

### 2.5 Reduced Mode Grenzen unklar

In §2.4 (Reduced Mode) wird definiert:
- Gültig für: 50–199 Frames
- Aber: "Bei Frame-Anzahl unterhalb des Minimums" → Abbruch oder Emergency Mode?

**Widerspruch:** In der Tabelle "Graduelles Degradieren" (Zeilen 88-93) wird `< 50 Frames` als "Degradiert" mit "Reduced Mode ohne Clusterung" gelistet, aber im Text §2.4 heißt es "Abbruch (oder nur mit `runtime.allow_emergency_mode`)".

**Korrektur (v3.2):** Klar definiert:
- `N < 50`: Abbruch (optional Emergency Mode)
- `50 ≤ N ≤ 199`: Reduced Mode
- `N ≥ 200`: Full Mode

---

## 3. Mathematische Inkonsistenzen

### 3.1 Adaptive Gewichtung - Berechnungsreihenfolge

In §3.2 (Adaptive Gewichtung) wird beschrieben:
```
α' = Var(B) / (Var(B) + Var(σ) + Var(E))
```

**Problem:** Es ist nicht klar definiert, ob die Varianzen auf den **rohen** oder **normalisierten** Metriken berechnet werden. Die Implementierung (v3.2) berechnet Varianzen auf den robust normalisierten Metriken `z(x)`, was korrekt ist.

**Empfohlene Korrektur:** Explizit spezifizieren:
```
Var(z(B)), Var(z(σ)), Var(z(E))
```

### 3.2 Clusterung - Kanalweise vs. Kanalaggregiert

In §3.7 wird der Zustandsvektor definiert:
```
v_f = (G_f, ⟨Q_tile⟩, Var(Q_tile), B_f, σ_f)
```

**Problem:** `G_f` und `Q_tile` sind kanalweise definiert (`G_{f,c}`, `Q_{f,t,c}`). Die Clusterung auf aggregierten Vektoren über alle Kanäle ist nicht konsistent mit der kanalgetrennten Verarbeitung.

**Korrektur (v3.2):** Explizit spezifiziert als "kanalweise oder kanalaggregiert, konfigurierbar".

---

## 4. Optimierungsmöglichkeiten

### 4.1 Redundanzreduktion

Die Methodik könnte um ~40% kompakter sein ohne Informationsverlust:

| Abschnitt | Redundanz | Empfohlene Aktion |
|-----------|-----------|-------------------|
| §3.6 (Tile-Rekonstruktion) | Fallback-Regeln doppelt | Eine klare Definition |
| §A.2.1.3 + §A.10 | Registrierungskaskade doppelt dokumentiert | Anhang A.10 entfernen oder integrieren |
| §3.3.1 + Anhang A.12 | Wiener-Filter doppelt | Nur eine Spezifikation |
| Pfad A komplett | In v3.2 entfernt | Entfernen oder als "legacy" markieren |

### 4.2 Präzisierung der Notation

**Vorschlag:** Einheitliche Notation durchgängig:

```
Indizes:  f ∈ [0, N-1]    Frame
          t ∈ [0, T-1]    Tile  
          c ∈ {R,G,B}     Kanal
          p ∈ Tile        Pixel

Variablen: I_{f,c}(p)      normalisiertes Eingangsbild
           I^{raw}_{f,c}   Rohdaten
           B_{f,c}         globaler Hintergrund
           σ_{f,c}         globales Rauschen
           E_{f,c}         globale Gradientenergie
           Q_{f,c}         globaler Qualitätsindex
           G_{f,c}         globales Gewicht
           Q_{f,t,c}       lokaler Qualitätsindex
           L_{f,t,c}       lokales Gewicht
           W_{f,t,c}       effektives Gewicht
```

### 4.3 Epsilon-Konstanten zentralisieren

In v3.1E sind Epsilon-Werte über die Methodik verteilt:
- §3.6: `ε = 1e-6` (implizit)
- Anhang A.8: `ε = 1e-6`, `ε_median = 1e-6`

**Empfohlene Optimierung:** Zentrale Definition aller numerischer Konstanten in einem "Numerik"-Abschnitt.

**Vorschlag (implementiert in v3.2):**
```yaml
# Numerische Konstanten
numerics:
  eps_bg: 1e-6
  eps_mad: 1e-6
  eps_weight: 1e-6
  eps_median: 1e-6
  delta_ncc: 0.01
  q_clamp: [-3, +3]
```

### 4.4 Semantische Klarstellung: Linearität

Die Methodik verwendet den Begriff "streng linear" ohne präzise Definition.

**Empfohlene Präzisierung (aus v3.2 übernommen):**
```
"Streng linear" bedeutet:
1. Photometrische Signalabbildung bleibt linear (keine globalen nichtlinearen Tonkurven)
2. Lineare Rekonstruktionsschritte (Skalierung, gewichtetes Mittel, Overlap-Add) sind Pflicht
3. Robuste/statistische Nichtlinearitäten (MAD, Clipping, Sigma-Clipping) sind als Hilfsschritte erlaubt
```

---

## 5. Vergleich mit v3.2

| Aspekt | v3.1E | v3.2 | Bewertung |
|--------|-------|------|-----------|
| Umfang | 1605 Zeilen | 512 Zeilen | v3.2 ist fokussierter |
| Pfade | A + B (beide normativ) | Nur CFA-Pfad (B) | v3.2 ist klarer |
| Notation | Inkonsistent | `f,t,c` durchgängig | v3.2 ist präziser |
| Reduced Mode | Grenzen unklar | `N<50`, `50-199`, `≥200` | v3.2 ist eindeutig |
| PCC-Test | "Kein negatives Element" | Determinante + Konditionszahl | v3.2 ist robuster |
| Testfälle | 12 Testfälle | 12 Testfälle (korrigiert) | v3.2 ist korrigiert |
| Numerik | Verteilt | Zentralisiert (§8) | v3.2 ist übersichtlicher |

---

## 6. Empfehlungen

### 6.1 Kurzfristig (für Implementierung basierend auf v3.1E)

1. **Pfad B (CFA-basiert) als einzigen Implementierungspfad verwenden**
2. **Notation durchgängig mit Kanalindex `c` verwenden**
3. **Tile-Rekonstruktions-Fallback einmalig in §3.6 definieren**
4. **PCC-Testfall 12 auf "positive Determinante" reduzieren**
5. **Reduced Mode Grenzen explizit: `< 50` = Abbruch, `50-199` = Reduced, `≥ 200` = Full**

### 6.2 Mittelfristig (Methodik-Update)

1. **Auf v3.2 migrieren** als normative Referenz
2. **Legacy-Dokumente (v3.1, v3.1E) als "historisch" markieren**
3. **Implementierungsnotizen (Anhang A) in separate Implementierungsdokumentation auslagern**

### 6.3 Empfohlene Numerische Defaults

```yaml
# Basierend auf v3.2 §8
numerics:
  eps_bg: 1e-6           # Hintergrund-Normalisierung
  eps_mad: 1e-6          # MAD-basierte Normalisierung  
  eps_weight: 1e-6       # Gewichts-Nenner
  eps_median: 1e-6       # Tile-Median-Normalisierung
  delta_ncc: 0.01        # Registrierungs-Validierung
  q_clamp_min: -3        # Globaler/lokaler Q-Clamp
  q_clamp_max: +3
  k_global: 1.0          # Exponential-Skaling für G
```

---

## 7. Schlussfolgerung

Die Methodik v3.1E ist **mathematisch fundiert** und **implementierbar**, leidet aber unter:

1. **Struktureller Redundanz** (doppelte Pfadspezifikation)
2. **Notations-Inkonsistenz** (fehlender Kanalindex)
3. **Übermäßiger Länge** (Implementierungsdetails in Methodik)

Die Konsolidierung in **v3.2** korrigiert diese Probleme systematisch und sollte als primäre Referenz verwendet werden.

**Empfohlene Vorgehensweise:**
- Falls Implementierung auf v3.1E basiert: Die 5 Kurzzeit-Empfehlungen (§6.1) anwenden
- Für neue Entwicklung: Direkt auf v3.2 referenzieren

---

**Ende der Diagnose**
