# v4 ↔ C++‑Port Abgleichbericht (normativ, zitierfähig)

**Dokumenttyp:** Review‑ & Issue‑Tracker‑Grundlage  
**Scope:** Abgleich Methodik v4 (Single Source of Truth) ↔ C++‑Port `tile_compile_cpp_snapshot_20260123`

---

## 1. Zusammenfassung

| Gesamturteil | Begründung |
|---|---|
| ❌ **nicht v4‑konform** | Methodische Kernkomponenten fehlen oder sind semantisch verletzt |

Der vorliegende C++‑Port stellt **eine strukturelle Basis**, aber **keine v4‑Implementierung** dar. Ohne substanzielle Erweiterungen ist das Ergebnis wissenschaftlich nicht gültig.

---

## 2. Tabellarischer Abgleich v4 ↔ C++

Legende:  
- ✅ korrekt implementiert  
- ⚠ teilweise / falsch  
- ❌ fehlt vollständig

### 2.1 Pipeline‑Ebene

| v4‑Anforderung | Status C++ | Kommentar |
|---|---|---|
| Registrierung vollständig | ⚠ | technisch vorhanden, Qualität ungeprüft |
| **Globale Normalisierung (Pflicht)** | ❌ | kritischer Regelbruch |
| Globale Frame‑Metriken | ⚠ | keine robuste Statistik |
| Seeing‑adaptive Tile‑Geometrie | ❌ | Tiles statisch |
| Lokale Tile‑Metriken | ⚠ | unvollständig |
| Orthogonale Gewichtung G·L | ❌ | nicht vorhanden |
| Tile‑Rekonstruktion stabil | ⚠ | keine Fallback‑Regeln |
| Zustands‑Clusterung | ❌ | fehlt vollständig |
| Synthetische Frames | ❌ | fehlt vollständig |
| Validierung & Abbruch | ❌ | nicht implementiert |

---

### 2.2 Globale Normalisierung

| Kriterium | v4 | C++ |
|---|---|---|
| exakt einmal | Pflicht | ❌ |
| linear | Pflicht | ❌ |
| vor allen Metriken | Pflicht | ❌ |
| pro Farbkanal | Pflicht | ❌ |

**Bewertung:** Alle nachfolgenden Metriken sind dadurch **physikalisch invalid**.

---

### 2.3 Globale Qualitätsmetriken

| Metrik | v4‑Definition | C++‑Status |
|---|---|---|
| Hintergrund B_f | robust, maskiert | ⚠ simpel |
| Rauschen σ_f | robust | ❌ |
| Gradientenergie E_f | definiert | ⚠ rudimentär |
| MAD‑Normierung | Pflicht | ❌ |
| Q_f‑Begrenzung | [−3,+3] | ❌ |
| G_f = exp(Q_f) | Pflicht | ❌ |

---

### 2.4 Tile‑Geometrie

| Aspekt | v4 | C++ |
|---|---|---|
| FWHM‑Schätzung | Pflicht | ❌ |
| adaptive Tile‑Größe | Pflicht | ❌ |
| Overlap‑Add | Pflicht | ⚠ |

---

### 2.5 Lokale Qualität

| Aspekt | v4 | C++ |
|---|---|---|
| Stern‑Tiles | getrennt | ❌ |
| Struktur‑Tiles | getrennt | ❌ |
| log(FWHM) nach Aggregation | Pflicht | ❌ |
| Q_local Clipping | Pflicht | ❌ |

---

### 2.6 Rekonstruktion

| Regel | v4 | C++ |
|---|---|---|
| W_f,t = G_f · L_f,t | Pflicht | ❌ |
| ε‑Stabilisierung | Pflicht | ⚠ |
| Fallback bei ΣW→0 | Pflicht | ❌ |

---

### 2.7 Zustandsmodell & synthetische Frames

| Element | v4 | C++ |
|---|---|---|
| Zustandsvektor v_f | Pflicht | ❌ |
| Clusterung nach Zustand | Pflicht | ❌ |
| synthetische Frames | Pflicht | ❌ |

---

### 2.8 Validierung & Abbruch

| Kriterium | v4 | C++ |
|---|---|---|
| FWHM‑Verbesserung | Pflicht | ❌ |
| Tile‑Artefakte | Pflicht | ❌ |
| Abbruchlogik | Pflicht | ❌ |

---

## 3. Modul‑weise To‑Do‑Liste (extrahierbar für Issues)

### normalization/
- globale lineare Normalisierung implementieren
- Trennung roh / normalisiert erzwingen

### metrics/global/
- robuste Hintergrund‑ & Rauschschätzung
- Gradientenergie definieren
- MAD‑Normalisierung über alle Frames

### tile/geometry/
- FWHM‑Schätzung
- adaptive Tile‑Größe + Overlap

### metrics/local/
- Stern‑ vs. Struktur‑Tiles
- Q_local‑Berechnung gemäß v4

### reconstruction/
- effektive Gewichte G·L
- stabile Tile‑Rekonstruktion
- Fallback‑Regeln

### state/
- Zustandsvektoren
- Clusterung (k‑means / GMM)

### synthetic/
- synthetische Qualitätsframes
- finales lineares Stacking

### validation/
- FWHM‑Vergleich
- Tile‑Artefakt‑Analyse
- Abbruchentscheidung

---

## 4. Zitierfähige Kernaussage

> *Der vorliegende C++‑Port bildet die Struktur, nicht jedoch die Methodik der v4‑Spezifikation ab. Ohne globale Normalisierung, orthogonale Gewichtung und Zustands‑Clusterung ist die Rekonstruktion physikalisch nicht valide.*

---

