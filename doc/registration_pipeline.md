# Astronomische Bildregistrierung - Analyse und Optimierungspotenziale

> **Dateistand:** April 2026 (Kombinierte Fassung)
> **Analysierte Quellen:**
> - `tile_compile_cpp/src/registration/registration.cpp`
> - `tile_compile_cpp/src/registration/global_registration.cpp`
> - `tile_compile_cpp/apps/runner_phase_registration.cpp`
> - `tile_compile_cpp/include/tile_compile/registration/*.hpp`
> - `tile_compile_cpp/include/tile_compile/config/configuration.hpp`
>
> **Dokumente integriert:**
> - `doc/registration_pipeline.md` (Hauptanalyse mit Algorithmen, Konfigurationen, Code-Beispielen)
> - `doc/registration_optimierung_tile_compile_cpp.md` (Fokus auf Gate-Konsistenz, Outlier-Kontext, Szenenklassifikation)
>
> Diese kombinierte Fassung enthält alle Erkenntnisse aus beiden Analysen mit priorisierten Umsetzungsempfehlungen.

---

## 1. Überblick über die Pipeline

Die Registrierungsphase besteht aus zwei Haupt-Schichten:

| Schicht | Datei | Verantwortung |
|---|---|---|
| **Algorithmen** | `registration.cpp` + `global_registration.cpp` | Einzelbild-Ausrichtung (ECC, Sterne, AKAZE) |
| **Orchestrierung** | `runner_phase_registration.cpp` | Alle Frames seriell + Rettungsstrategien |

Danach folgt direkt der **PREWARP**-Schritt: alle Frames werden mit dem berechneten Warp-Matrix auf ein erweitertes Canvas (inkl. Felddrehung) vorverarbeitet und disk-gecacht.

### Was bereits gut gelöst ist

- **Mehrstufige Rescue-Kaskade** (Sequential → Temporal → Seeded-ECC → Local-Reference) reduziert harte Frame-Verluste stark
- **`auto_engine`** erkennt Alt/Az-Rotations- und ECC-Qualitätsprobleme automatisch und wechselt die Engine
- **Polynomial-Vorhersage** für abgelehnte/fehlgeschlagene Frames sichert das Prinzip "kein Frame wird komplett verworfen"
- **Maskierte NCC-Validierung** verhindert großte ils schlechte Warps trotz formalen RANSAC-Erfolgs
- **CFA/Bayer-Alignment** beim Canvas-Offset (gerade Pixel) schützt das Bayer-Muster bei OSC-Kameras
- **Chain-Validation-Schutz**: sequentiell gerettete Frames sind gegen den CC-Outlier-Filter immun

---

## 2. Registrierungs-Algorithmen im Einzelnen

### 2.1 `register_single_frame` - Kaskade mit NCC-Validierung

Dies ist die kanonische Funktion für ein einzelnes Frame. Sie läuft folgende Kaskade ab:

```
Eingabe: mov (Moving), ref (Referenz), RegistrationConfig

1. NCC-Baseline (Identity) messen → ncc_identity
2. Falls ncc_identity ≥ 1 - min_ncc_improvement → Identity direkt akzeptieren
3. Primär-Engine (konfigurierbar):
   - "triangle_star_matching"  (Standard)
   - "star_similarity"
   - "opencv_feature"          (AKAZE)
   - "robust_phase_ecc"
4. Fallback 1: AKAZE feature_registration_similarity
5. Fallback 2: robust_phase_ecc
6. Letzter Ausweg: Identity (cc=0, success=false)

Jeder Kandidat wird mit NCC geprüft:
  NCC(warped, ref) > NCC(mov, ref) + min_ncc_improvement  →  akzeptiert
```

### 2.2 Triangle Star Matching (`triangle_star_matching`)

**Standard-Engine.** Robust gegen Rotation, Translation und gemäßigte Skalierung.

**Ablauf:**
1. **Sternerkennung** (`detect_stars_simple`):
   - Lokale Maxima mit Gauss-Centroid (5×5 Fenster)
   - Hot-Pixel-Filter: Konzentration > 80 % → verwerfen
   - Elongations-Filter: Eigenwert-Verhältnis < 15 % → verwerfen (Trails, Artefakte)
   - Adaptiver Schwellwert: erst 3,5σ, bei < `topk/2` Sternen Fallback auf 2,5σ
2. **Dreieck-Aufbau** (`build_triangles`):
   - Aus max. 60 hellsten Sternen, bis zu 600 Dreiecke
   - Invariante Seitenverhältnisse `[s0/s2, s1/s2]` (skalen- und rotationsinvariant)
   - Filter: nahezu gleichseitige und nahezu gleichschenklige Dreiecke werden aussortiert
3. **Dreiecks-Matching**:
   - Toleranz `ratio_tol = 0.03`; Eindeutigkeits-Margin `0.004`
   - Vorzeichen der Signum-Fläche muss übereinstimmen (Spiegelschutz)
   - Vertex-Zuordnung via Seiten-Sortierung → Voting über alle Übereinstimmungen
4. **RANSAC-Fit**: `estimateAffine2D` (affine) oder `estimateAffinePartial2D` (similarity)
5. **Inversion** der Forward-Matrix → Warp (R→M) für `WARP_INVERSE_MAP`

### 2.3 Star Pair Similarity (`star_registration_similarity`)

Fallback des Dreieck-Matching. Bildet Sternepaare, matcht nach Länge (Bin-Histogramm), prüft alle Kandidaten mit `similarity_from_pairs`, bewertet über NCC-Konsensus. Optional Verfeinerung mit affiner RANSAC.

### 2.4 AKAZE Feature Registration (`feature_registration_similarity`)

- AKAZE Keypoint-Detektion + BFMatcher (Hamming, cross-check)
- Top 30 % der Matches (Distanz-sortiert), mindestens 15
- RANSAC → affine oder similarity
- Stärke: dichte Texturen, Galaxien-Strukturen
- Schwäche: schlechte Performance bei schwachen/spärlichen Sternenfeldern

### 2.5 Robust Phase+ECC (`robust_phase_ecc`)

Multi-Skalen-Ansatz mit Gradienten-Vorverarbeitung:
1. **Gradient-Preprocessing**: Laplace-of-Gaussian (σ=2) → entfernt Nebel/Wolken-Gradienten
2. **Pyramide**: 3 Level (Original, 1⁄2, 1⁄4)
3. **Grobs-nach-Fein**: Phasenkorrelation + Log-Polar-Rotation auf gröbstem Level, dann ECC-Verfeinerung auf jedem Level
4. Seeded Variante (`robust_phase_ecc_seeded`): nimmt interpolierten Warp als ECC-Startwert

---

## 3. Orchestrierung in `runner_phase_registration.cpp`

### 3.1 Referenz-Frame-Auswahl

```
1. Top-K nach Qualitätsscore (K = max(16, N/5))
2. Innerhalb der Top-K: Frame mit kleinstem Abstand zum zeitlichen Zentrum
3. Fallback: bester Qualitätsscore-Frame
4. Fallback: zeitliches Zentrum
```

> **Ziel:** Balance zwischen Bildqualität und zeitlicher Stabilität für Alt/Az-Sequenzen mit Felddrehung.

### 3.2 Auto-Engine-Detektion

Wenn `auto_engine=true` und die konfigurierte Engine rotationsblind ist (`robust_phase_ecc`/`hybrid_phase_ecc`), wird mit 4 Probe-Frames geprüft:
- **Rotation**: Medianrotation pro Frame (Schwellwert: `auto_engine_rotation_threshold_deg = 0.05°`)
- **ECC-Qualität**: Erfolgsrate und mediane CC der ECC-Probe-Ergebnisse

Bei Auslösung → automatischer Wechsel auf `triangle_star_matching + affine`.

### 3.3 Globale Registrierung (Phase 1)

Alle Frames werden **parallel** gegen den Referenz-Frame registriert:
- Proxy-Bild: 2×2-Downsampling (Mono) oder CFA-Green-Extraktion (OSC) → halbe Auflösung
- Warp-Skalierung: Translation ×2 (von Proxy auf Vollauflösung)
- Ergebnis je Frame: `direct_global` oder `unresolved`

### 3.4 Sequentielle Verfeinerung (Phase 2)

Für jeden Frame: Registrierung gegen **direkten Zeitnachbarn**, Chaining des Warps mit dem Nachbar-Warp zur Referenz.

Akzeptiert wenn:
- `ncc_global > current_cc + 0.005` (deutlich besser), **oder**
- `ncc_global ≥ current_cc - 0.01` und `local_correlation ≥ max(0.12, current_cc × 0.25)` (vergleichbar gut, sequential bevorzugt)

Ergebnis: `sequential_refined` (Verbesserung) oder zählt zu `sequential_rescued` (war zuvor unresolved).

### 3.5 Blindes Chaining / Sequential Rescue (Phase 3)

Für Frames ohne Registrierung → Phasenkorrelation + Log-Polar gegen direkten Zeitnachbarn:

```
Validierung gegen NACHBARN (nicht globale Referenz):
  - overlap_px > 32
  - ncc_neighbor ≥ 0.3         → akzeptiert
  - ncc_neighbor ≥ 0.05 UND shift < 30px  → akzeptiert (Blind-Chaining)
```

> ⚠️ **Kritisch:** `kMaxBlindChainAnchorDepth = 12` - nach 12 Kettengliedern wird der Frame als Anker für weitere Chains **gesperrt**. Längere Wolkenblöcke können daran scheitern.

### 3.6 Multi-Pass Rettung (Phase 4, 4 Iterationen)

Für verbleibende nicht-registrierte Frames (maximal 4 Passes, bis kein Fortschritt mehr):

| Strategie | Beschreibung | Provenance |
|---|---|---|
| **Temporal Rescue** | Registrierung gegen nächsten validen Anker (überspringt Lücken), NCC-Verbesserung gegen Referenz erforderlich | `temporal_rescue` |
| **Seeded ECC Rescue** | Interpolierter Warp-Seed → `robust_phase_ecc_seeded` gegen Referenz | `seeded_ecc_rescue` |
| **Local Reference Rescue** | Stacking von bis zu 6 validen Nachbarframes als lokale Referenz → ECC mit interpoliertem Seed | `local_reference_rescue` |

### 3.7 Outlier-Rejektion

Vier unabhängige Filter:

| Filter | Kriterium | Konfiguration |
|---|---|---|
| **Reflektion** | det(Warp) < 0 (Spiegelung) | immer aktiv |
| **Skalierung** | scale ∉ [0.92, 1.08] | `reject_scale_min/max` |
| **Niedrige CC** | cc < `reject_cc_min_abs` UND nicht chain-validated | `reject_cc_min_abs = 0.25` |
| **Shift-Outlier** | shift > max(`reject_shift_px_min=100`, `5 × median_shift`) | `reject_shift_median_multiplier` |

Chain-validierte Frames (sequentiell gerettet) sind gegen den CC-Filter geschützt → `low_cc_protected`.

Für halbe Drehungen (~180°) wird ein separater Shift-Median berechnet.

### 3.8 Polynomial-Vorhersage für abgelehnte Frames

Wenn ≥ 3 valide Frames vorhanden sind, wird ein **Grad-2-Polynom** über (frame_index → angle, tx, ty) gefittet:
- Normierung auf [0, 1] für numerische Stabilität
- Zusätzlich: gewichtetes lokales Polynom (`tricube`-Kernel) für benachbarte Support-Frames
- Blending aus globalem Polynom, lokalem Polynom und Bridge-Interpolation
- Ergebnis: `model_blended`, `model_local_poly`, `model_interpolated`, `model_global_poly`, `model_nearest_copy`

> Alle Frames erhalten so einen Warp, auch wenn die Registrierung scheiterte. CC wird auf 1e-4 gesetzt → downstream tile-level Quality Metrics übernehmen die Gewichtung.

### 3.9 PREWARP

- Canvas-Erweiterung für Felddrehung (Bounding Box aller Warp-Ecken)
- **CFA-Alignment:** Offsets werden auf gerade Werte aufgerundet (Bayer-Muster-Schutz)
- Paralleles Warpen aller Frames mit `warp_affine_frame`
- Disk-Cache (`DiskCacheFrameStore`) für RAM-Effizienz
- Overlap-Coverage-Map für Common-Validity-Mask

---

## 4. Bekannte Schwachstellen und Optimierungspotenziale

### 4.0 🔴 Inkonsistente Akzeptanz-Gates zwischen Pfaden

**Problem:** Die verschiedenen Rescue-Pfade verwenden unterschiedliche NCC-Improvement-Schwellen, ohne dass ein einheitliches Prinzip dahintersteht:

| Pfad | Gate | Code-Stelle |
|---|---|---|
| `register_single_frame` | `+0.01` | `global_registration.cpp` |
| Sequential Refinement | `+0.005` | `runner_phase_registration.cpp` |
| Sequential Rescue (Nachbar) | `+0.0` (nur overlap > 32) | runner, blind chain |
| Temporal Rescue | `+0.01` | runner |
| Seeded-ECC Rescue | `+0.005` | runner |
| Local-Reference Rescue | `+0.003` | runner |

Ein falscher Warp kann durch den lockersten Pfad schlüpfen, obwohl ein anderer Pfad ihn abgelehnt hätte. **Besonders kritisch** bei dünnen Sternfeldern, Wolken oder starker Gradientenstruktur, wo der NCC-Raum flach ist.

**Optimierung - Gemeinsame Gate-Policy-Funktion:**

```cpp
// Neue Hilfsfunktion in runner_shared.hpp:
struct GateContext {
    int    overlap_px;          // tatsächliche Overlap-Pixel (aus valid_mask)
    float  expected_shift_px;   // abs. geschätzter Shift (aus Nachbar-Warp oder Polynom)
    RegistrationProvenance provenance;  // direct / sequential / rescue / model
    bool   against_global_ref;  // true = gegen globale Ref, false = gegen Nachbarn
};

bool registration_gate(
    float ncc_before, float ncc_after, const GateContext& ctx) {
    if (ctx.overlap_px <= 16) return false;

    // Adaptiver Margin: kleiner Shift + großer Overlap = strenger
    const float base = 0.005f;
    const float shift_factor = std::min(1.5f, ctx.expected_shift_px / 30.0f);
    const float overlap_factor = std::max(0.5f,
        1.0f - static_cast<float>(ctx.overlap_px) / 500000.0f);
    const float margin = base * (1.0f + shift_factor) * (1.0f + overlap_factor);

    return ncc_after > ncc_before + margin;
}
```

Alle Rescue-Pfade rufen diese Funktion auf - konsistentes Verhalten, keine pfad-spezifischen Magic-Numbers.

---

### 4.1 🔴 Blind-Chain-Tiefenbegrenzung zu starr

**Problem:** `kMaxBlindChainAnchorDepth = 12` blockiert Propagation bei langen Wolkenblöcken.
Ein Frame auf Tiefe 12 kann nicht mehr als Anker für Tiefe 13 dienen - unabhängig von seiner tatsächlichen Registrierungsqualität.

**Code:**
```cpp
case RegistrationProvenance::sequential_rescue:
    return (reg_chain_depth[fi] >= 0 &&
            reg_chain_depth[fi] < kMaxBlindChainAnchorDepth) ||
           global_frame_cc[fi] >= kBlindChainStrongAnchorCc;
```

**Optimierung:**

**A) Sofortmaßnahme (Konfiguration):**
- Erlaubnis für tiefere Ketten wenn `global_frame_cc ≥ kBlindChainStrongAnchorCc` (0.08) ist bereits implementiert, könnte aber auf 0.05 gesenkt werden
- Alternativ: Tiefe dynamisch anpassen basierend auf der Gesamtlänge der Session (`N/10` statt konstant 12)

**B) Code-Änderung: Blind-Chain-Parameter konfigurierbar machen:**

```cpp
// In configuration.hpp → RegistrationConfig:
int max_blind_chain_depth = 12;              // neu (war hardcodiert)
float blind_chain_strong_anchor_cc = 0.08f;  // neu (war hardcodiert)

// In runner_phase_registration.cpp:
// constexpr int kMaxBlindChainAnchorDepth = 12;  // ersetzen durch:
const int kMaxBlindChainAnchorDepth = registration_cfg.max_blind_chain_depth;
const float kBlindChainStrongAnchorCc = registration_cfg.blind_chain_strong_anchor_cc;
```

**Erwartung:** Bessere Wiederherstellung bei langen Wolkenblöcken ohne unkontrolliertes Wegdriften (in Kombination mit Drift-Check aus §4.2).

---

### 4.2 🔴 Drift-Akkumulation im Blind-Chain

**Problem:** NCC 0.05 mit kleinem Shift ist extrem permissiv. Über viele Ketten-Glieder akkumuliert sich Positionsdrift.

**Code:**
```cpp
} else if (ncc_neighbor >= 0.05f && small_shift) {
    accept = true; // Blind chaining allowed for small shifts
}
```

**Optimierung:**

**A) Kumulativer Drift-Check:**
- Kumulativen Drift anhand der Warp-Parameter tracken: wenn die chained Translation von der Polynomial-Vorhersage stark abweicht, Chain-Frame verwerfen
- Drift-Check: `|w_chained.tx - poly_predicted_tx| > threshold` → Rejektion

**B) Driftkontrolle über Trend-/Modellabgleich (für tiefe Chain-Stufen):**
- Optional: Bei tiefen Chain-Stufen (z.B. > 6) zusätzlicher Drift-Check gegen lokalen Polynom-Trend
- Wenn chained Warp stark vom lokalen Modell-Trend abweicht (`|w_chained.tx - trend_tx| > 2 * local_sigma`), Frame nicht als neuer Anker für weitere Chains verwenden
- Verhindert unkontrolliertes Wegdriften bei langen Wolkenblöcken ohne qualitativ hochwertige Anker

```cpp
// Zusätzlicher Schutz für tiefe Chains:
if (chain_depth > 6 && global_frame_cc[fi] < kBlindChainStrongAnchorCc) {
    const float drift_tx = std::abs(w_chained.tx - local_poly_tx);
    const float drift_ty = std::abs(w_chained.ty - local_poly_ty);
    const float drift_ang = std::abs(w_chained.angle - local_poly_angle);
    
    if (drift_tx > 2.0f * local_sigma_tx || drift_ty > 2.0f * local_sigma_ty) {
        // Frame akzeptieren, aber NICHT als Anker für weitere Chains erlauben
        allow_as_anchor = false;
    }
}
```

---

### 4.3 🟡 NCC-Verbesserungsschwelle: zu niedrig für schwache Felder, inkonsistent zwischen Pfaden

**Problem:** Zwei zusammenhängende Teilprobleme:
1. `min_ncc_improvement = 0.01` (Standard in `register_single_frame`) kann von falschem Warp bei flachem NCC-Raum überwunden werden
2. Verschiedene Pfade nutzen unterschiedliche Schwellen (0.003, 0.005, 0.01) ohne gemeinsames Prinzip - dasselbe Frame kann je nach Ausführungsreihenfolge angenommen oder abgelehnt werden (siehe 4.0)

**Optimierung:**
- **Kurzfristig (Konfiguration):** `min_ncc_improvement` adaptiv: `max(0.005, ncc_identity * 0.05)` - fordert 5 % relative Verbesserung
- Für hohe ncc_identity (≥ 0.9): Schwelle auf 0.003 senken (bereits fast perfekt registriert, kleine Änderungen genügen)
- **Mittelffristig (Architektur):** Gemeinsame Gate-Policy-Funktion (siehe 4.0) ersetzt alle pfad-spezifischen Konstanten

---

### 4.4 🟡 Stern-Detektion versagt bei sehr hellem Hintergrund / Mond

**Problem:** Der adaptive Fallback auf 2,5σ reicht bei hohem Mond-Hintergrundlicht oder starken Nebeln nicht aus. Die Standardnormalisierung auf `[0,1]` komprimiert schwache Sterne in ein enges Werteband.

**Optimierung:**
1. Dritte Fallback-Stufe bei sehr wenigen Sternen: 1,5σ mit strengerem Hot-Pixel-Filter (Konzentration < 0,6 statt 0,8)
2. Lokale Hintergrundsubtraktion vor Stern-Detektion (2D-Median-Filter, z.B. 31×31), um großräumige Gradienten zu entfernen
3. Mindest-Flussschwelle: Sterne unter absolutem Flux-Threshold von `5 × MAD` verwerfen

---

### 4.5 🟡 Triangle-Matching: Zu wenige Sterne bei dichten Feldern wirken kontraproduktiv

**Problem:** Mit `topk = 150` in dichten Feldern (z. B. Milchstraße) werden zu viele ähnlich helle Sterne ausgewählt. Viele nahezu-kongruente Dreiecke entstehen → Mehrdeutigkeiten.

**Optimierung:**
- Fluss-basierte Ausdünnung: nur Sterne mit Flussverhältnis > 1,5× zum nächsten Nachbarn im Kandidaten-Pool behalten
- Mindest-Sternabstand erzwingen: Sterne die sich innerhalb von `2 × FWHM` befinden → schwächeren entfernen
- `star_topk` für dichte Felder auf 80-100 reduzieren, für Weitwinkelaufnahmen erhöhen

---

### 4.6 🟡 Outlier-Filter fehlt Sequenzkontext

**Problem:** Die Shift- und CC-Filter verwenden Session-globale Mediane. An den **Rändern langer Alt/Az-Sitzungen** sind jedoch große Rotationen und Shifts physikalisch erwartet - der globale Median unterschreitet diese Werte, weshalb valide Rand-Frames als Outlier markiert werden.

Beispiel: Medianshift = 80 px, Limit = 5× = 400 px. Letzte Frames einer 4h-Session haben 450 px Shift (legitim bei starker Felddrehung) → werden abgelehnt.

**Optimierung:**
- **Sequenzrand-Schutz:** Für Frames innerhalb der ersten/letzten 10 % der Session: `reject_shift_median_multiplier` um Faktor 1.5× erhöhen
- **Konsistenz-Check:** Frames ablehnen nur wenn Shift **und** CC-Wert schlecht sind (OR → AND für diese Rand-Frames)
- Alternativ: Shift-Limit separat pro Sequenzhalfäfte berechnen (vordere Hälfte vs. hintere Hälfte) für asymmetrische Sessions

### 4.6b 🟡 Referenz-Frame-Auswahl bei langen Sequenzen

**Problem:** Die Top-K-Auswahl (`max(16, N/5)`) wählt bei 1000+ Frames bis zu 200 Kandidaten. Der zeitliche Mittelpunkt kann aber in einer Wolkenphase liegen.

**Optimierung:**
- Ausschließlich Frames aus dem "stabilen Kern" berücksichtigen: Frames bei denen `k` unmittelbare Nachbarn ebenfalls valide sind
- "Cluster-basierte" Referenzauswahl: längsten wolkenfreien Block finden, daraus das zeitlich zentralste Frame wählen

---

### 4.7 🟡 Polynomial-Prediction: Grad 2 reicht nicht für sehr lange Sitzungen

**Problem:** Bei Sessionen >4 Stunden mit Alt/Az-Montierung (starke Felddrehung) folgen Winkel und Shift einer nicht-linearen Trajektorie, die ein Grad-2-Polynom schlecht approximiert.

**Code:**
```cpp
V(i, 0) = 1.0f;
V(i, 1) = t;
V(i, 2) = t * t;
```

**Optimierung — Adaptiver Polynomial-Grad statt fix Grad 2:**

```cpp
// Statt konstant Grad 2:
const int poly_degree = std::min(4, std::max(2, static_cast<int>(std::sqrt(nv / 3.0f))));

// V-Matrix dynamisch aufbauen:
for (int d = 0; d <= poly_degree; ++d) {
    V(i, d) = std::pow(t, d);
}
// V(i, 3) = t^3, V(i, 4) = t^4 bei genügend Stützstellen
```

**Logik:**
- Wenige valide Frames (< 12) → Grad 2 (numerische Stabilität)
- Moderate Anzahl (12-27) → Grad 3
- Viele valide Frames (27+) → Grad 4
- Sicherheitsgrenze: Bei Overfit-Anzeichen (hohe Residuen trotz höherem Grad) auf niedrigeren Grad zurückfallen

**Zusätzlich für Winkel:**
- Beachte 2π-Wicklungen durch `unwrap_angle_sequence` (bereits implementiert)
- Der globale Fit sollte nach unwrapping auf Continuity prüfen
- Alternativ: B-Spline-Interpolation mit wenigen Kontrollpunkten statt globalem Polynom

**Erwartung:** Bessere Vorhersage auf langen, nichtlinearen Alt/Az-Sequenzen mit starkem Felddrehungseffekt.

---

### 4.8 🟡 Warp-Skalierung nur für Translation

**Problem:** Beim Hochskalieren von Proxy- auf Vollauflösung wird nur die Translation skaliert:

```cpp
WarpMatrix w_full = sfr.reg.warp;
w_full(0, 2) *= global_reg_scale;  // nur tx
w_full(1, 2) *= global_reg_scale;  // nur ty
// a00, a01, a10, a11 bleiben unverändert
```

Bei `similarity`-Transform mit `scale ≠ 1` (z. B. optischer Zoom-Drift) ist das korrekt nur für Translation + Rotation ohne Skalierung. Falls der Warp eine echte Skalierungskomponente enthält (scale ≠ 1 in der Ähnlichkeitstransformation), müsste auch die Rotationsmatrix angepasst werden - allerdings bei einer 2×-Downsampling-Proxies ist dieser Fehler < 0.01 px bei normaler Felddrehung, da das Skalenverhältnis konstant ist.

**Optimierung:**
- `scale_translation_warp` durch eine vollständige affine Skalierung ersetzen:
  ```
  w_full = S · w_proxy · S-1
  ```
  mit S = diag(global_reg_scale, global_reg_scale) als Skalierungsmatrix.
  Aktuelle Hilfsfunktion nur korrekt wenn Rotationsanteil ≪ Translationsanteil.

---

### 4.9 🟢 Parallele Registrierung: keine adaptive Drosselung

**Problem:** Bei sehr vielen Frames (>500) und intensivem IO (FITS lesen) kann die parallele Registrierung zu Speicherdruck führen.

**Optimierung:** Bereits vorhanden: `compute_adaptive_worker_count` mit `WorkerParallelProfile::MixedIo`. Verbesserung: Frame-Cache (`RunnerFrameCache`) stärker nutzen für normalisierte Frames die bereits für Metriken geladen wurden.

---

### 4.10 🟢 Modell-CC undifferenziert: alle Vorhersagen gleich behandelt

**Problem:** Die Polynomial-Vorhersage schreibt `cc = 1e-4` für **alle** modellierten Frames — unabhängig davon ob es eine saubere lineare Interpolation zwischen zwei guten Nachbarframes oder eine riskante Extrapolation über einen 200-Frame-Wolkenblock ist.

**Optimierung — Drei Qualitätsklassen:**

| Klasse | Kriterien | CC-Wert |
|---|---|---|
| `high_confidence` | Interpolation, span < 10 frames, res_tx < 5 px | `5e-4` |
| `medium_confidence` | Interpolation/local-poly, span 10–50 frames | `1e-4` |
| `low_confidence` | Extrapolation, span > 50 frames, res hoch | `1e-5` |

```cpp
// Aus Residuen und Span ableiten:
const float res_total = chosen.res_tx / 20.0f + chosen.res_ang_deg / 1.0f;
const float span_penalty = std::min(1.0f, chosen.span / 50.0f);
const float model_cc = std::clamp(
    5e-4f / (1.0f + res_total + 2.0f * span_penalty),
    1e-5f, 5e-4f);
set_registration_state(fi, w, model_cc, false, -1, chosen_provenance);
```

Downstream-Gewichtung profitiert: `high_confidence`-Frames (enge Interpolation) tragen mehr bei als unsichere Extrapolationen.

---

### 4.11 🟢 Lokale Referenz-Bildung fragil bei langen Ausfall-Blöcken

**Problem:** `build_local_reference` wählt Support-Frames ausschließlich nach zeitlicher Nähe und positivem CC. Bei großen Ausfall-Blöcken passiert:
- Alle 6 Support-Frames liegen **einseitig** (z. B. nur vor dem Block)
- Frames mit geometrisch inkonsistenten Warps (Chain-Drift) fließen ungeprüft ein
- Kein Mindest-Score für die resultierende Referenz-Qualität außer `min_valid_pixels`

**Optimierung:**
1. **Beidseitige Abdeckung erzwingen:** Mindestens 1 Support-Frame vor UND nach dem ausgefallenen Block
   ```cpp
   int frames_before = 0, frames_after = 0;
   for (const SupportFrame& s : support) {
       if (s.idx < fi) ++frames_before; else ++frames_after;
   }
   if (support.size() >= 2 && (frames_before == 0 || frames_after == 0))
       return false;  // keine einseitige Referenz
   ```
2. **Geometrische Konsistenz:** Support-Frames ablehnen deren Warp stark vom lokalen Polynom-Trend abweicht:
   ```cpp
   // Warp-Residuum gegen Trend der anderen Support-Frames
   // |w_support.tx - trend_tx| > 2 * local_sigma  → verwerfen
   ```
3. **Effektiver Support-Score** statt nur `valid_pixels`: Summe der gewichteten CC-Werte > Schwellwert

---

### 4.12 🟢 Keine Szenenklassifikation vor der Registrierungs-Kaskade

**Problem:** Der aktuelle `auto_engine`-Mechanismus prüft nur **Rotation** und **ECC-CC-Qualität**. Er erkennt nicht:
- Sehr diffuse Felder mit wenigen Sternen (große Nebel, Kometen)
- Dominant helle Objekte (Mars, Jupiter nahe Bildzentrum)
- Extrem schlechtes Seeing (Sterne zu stark verbreitert für normales Centroid)
- Sehr dichte Sternfelder (Milchstraße, Kugelsternhaufen)

Diese Szenarien wählen sub-optimale `star_topk`-, `inlier_tol_px`- und `transform_model`-Werte.

**Optimierung — Szenenklassifikation als explizite Phase vor der Kaskade (P1-D):**

**A) Probe-Phase (3-5 Frames):**

```cpp
// Vor der globalen Registrierung: 3-5 Probe-Frames analysieren
struct SceneProfile {
    float star_density;       // Sterne pro 100×100px bei 3.5σ
    float gradient_strength;  // RMS des Laplacian-Bildes
    float snr_estimate;       // Median(Signal) / MAD(Hintergrund)
    float fwhm_estimate;      // Median-FWHM der erkannten Sterne
};

SceneProfile probe_scene_profile(
    const std::vector<Frame>& frames,
    const std::vector<ScaleFactors>& norm_scales,
    int n_probe = 5);
```

**B) Parameter-Adaption basierend auf Szenenprofil:**

```cpp
void apply_scene_adaptations(RegistrationConfig& cfg, const SceneProfile& scene) {
    // Sternarme Felder (große Nebel, Kometen, diffuses Milchstraßengebiet)
    if (scene.star_density < 0.5f) {
        cfg.star_topk = 80;
        cfg.star_inlier_tol_px = 6.0f;
        cfg.star_min_inliers = 3;
        cfg.enable_star_pair_fallback = true;
        // Lockere Gates für Rettung bei wenigen Sternen
        cfg.gate_base_margin = 0.003f;
    }
    // Dichte Sternfelder (Milchstraße, Kugelsternhaufen)
    else if (scene.star_density > 5.0f) {
        cfg.star_topk = 80;              // weniger = weniger Mehrdeutigkeiten
        cfg.star_inlier_tol_px = 3.0f;  // strengere Toleranz
        cfg.star_min_inliers = 6;        // höhere Konsistenzanforderung
        cfg.transform_model = "similarity"; // weniger Freiheitsgrade
        // Strengere Gates bei vielen Alternativen
        cfg.gate_base_margin = 0.008f;
    }
    
    // Schlechtes Seeing (FWHM > 8px)
    if (scene.fwhm_estimate > 8.0f) {
        cfg.star_inlier_tol_px = std::max(cfg.star_inlier_tol_px, 6.0f);
        cfg.star_sigma_fallback = 2;     // aggressivere Threshold-Fallbacks
    }
    
    // Starke Gradienten (Mondlicht, Lichtverschmutzung)
    if (scene.gradient_strength > threshold) {
        cfg.enable_local_background_subtraction = true;
        cfg.star_sigma_fallback = 2;
    }
}
```

**C) Integration in Pipeline:**

```cpp
// In run_phase_registration_prewarp(), nach auto_engine-Block:
SceneProfile scene = probe_scene_profile(
    frames, norm_scales, detected_mode, detected_bayer_str,
    /*n_probe=*/5);
apply_scene_adaptations(registration_cfg, scene);
// Loggt: star_density, gradient_strength, snr_estimate, fwhm_estimate
```

**Erwartung:**
- Weniger frühe Fehlversuche durch bessere initiale Parameter
- Schnellere Konvergenz der Registrierungskaskade
- Automatisch optimale Parameter pro Sequenz ohne manuelles Tuning
- Ergänzt `auto_engine` (Engine-Wahl) um vollständige Parameter-Dimension

**Das geht über `auto_engine` hinaus:** Nicht nur die Engine, sondern auch alle sternbezogenen Parameter (`star_topk`, `inlier_tol_px`, `min_inliers`, `transform_model`, Gates) werden basierend auf dem erkannten Szenentyp adaptiert.

---

### 4.13 🟡 Astrometrische Registrierung für "unresolved" Frames

**Problem:** Frames, die alle internen Rescue-Pfade durchlaufen haben ohne Erfolg
(`sequential_rescue`, `temporal_rescue`, `seeded_ecc_rescue`, `local_reference_rescue`),
fallen aktuell direkt auf das polynomiale Modell zurück. Diese Frames haben oft
trotzdem erkennbare Sterne, die mit der bestehenden Astrometrie-Infrastruktur
(ASTAP oder interner Solver) gelöst werden könnten.

**Voraussetzung — ASTAP/Katalog verfügbar:**
Die astrometrische Rescue wird **nur** versucht, wenn:
1. `registration.use_astrometry: true` (default: true)
2. ASTAP-Binary verfügbar (`astap_bin` oder Systempfad)
3. Katalog-Daten verfügbar (`astap_data_dir`)

Die Phase `ASTROMETRY` ist **nicht** Voraussetzung — die Registrierung führt
eigenständiges Plate-Solving durch. Der explizite `use_astrometry` Parameter
erlaubt es, die Astrometrie-Rescue bei Bedarf zu deaktivieren.

```cpp
// Prüfung vor astrometrischem Rescue:
if (!registration_cfg.use_astrometry ||
    !astrometry_tooling_available()) {  // ASTAP + Katalog vorhanden?
    // Astrometrie-Rescue deaktiviert oder nicht verfügbar
    return try_model_fallback(fi);
}

// Eigenständiges Plate-Solving für diesen Frame:
AstrometricSolution astro = run_plate_solving(
    frame_img,
    astrometry_config,  // astap_bin, astap_data_dir, search_radius
    detected_stars      // aus detect_stars_simple()
);
```

**Position in Rescue-Hierarchie:**

```
Rescue-Hierarchie (erweitert):
1. Sequential Rescue (Blind Chain)
2. Temporal Rescue (Anker-Überspringung)
3. Seeded ECC Rescue (Interpolierter Seed)
4. Local Reference Rescue (Stacked Nachbarn)
5. ASTROMETRIC RESCUE (neu) ← Nur wenn use_astrometry && ASTAP+Katalog verfügbar
6. Model Fallback (Polynomial/Interpolation)
```

**Ablauf — Eigenständiges Plate-Solving pro Frame:**

```cpp
// In runner_phase_registration.cpp, nach local_reference_rescue:
if (provenance == unresolved && registration_cfg.use_astrometry) {
    // Eigenständiges Plate-Solving für diesen Frame
    AstrometricSolution astro = attempt_astrometric_rescue(
        frame_img,
        detected_stars,              // aus detect_stars_simple()
        astrometry_config            // astap_bin, astap_data_dir, search_radius
    );

    if (astro.success && astro.num_matches >= 4) {
        // Extrahiere Warp-Matrix aus astrometrischer Lösung
        // Transformation: Bildkoordinaten → RA/DEC → Referenz-Frame-Pixel
        WarpMatrix w_astro = compute_warp_from_astrometry(
            astro.wcs_solution,      // aus ASTAP/internem Solver
            ref_frame_wcs            // Referenz-Frame WCS (aus erstem Frame oder Header)
        );

        // Validierung gegen Referenz
        float ncc_astro = validate_warp_ncc(frame, ref, w_astro);

        if (ncc_astro > 0.20f) {  // niedrigere Schwelle für absolute Lösung
            set_registration_state(fi, w_astro, ncc_astro, true, -1,
                                   RegistrationProvenance::astrometric_rescue);
        }
    }
}
```

**Notwendige Ergebnisse:**

| Ergebnis | Quelle | Verwendung |
|----------|--------|------------|
| `wcs_solution` | ASTAP / interner Solver (pro Frame) | Koordinatentransformation Bild → Himmelskoordinaten |
| `ref_frame_wcs` | Erster Frame der Sequenz oder FITS-Header | Transformation Himmelskoordinaten → Referenz-Pixel |
| `star_matches` | Plate-Solving | Qualitätsmetrik (Anzahl Matches) |

**Provenance und CC-Wert:**

| Ergebnis | Provenance | CC-Wert | Begründung |
|----------|-----------|---------|------------|
| Erfolgreich | `astrometric_rescue` | `max(ncc, 0.30)` | Absolute Lösung, höher als Modell |
| Gescheitert | `model_*` | `1e-4..5e-4` | Polynomiale Vorhersage als Fallback |

**Astrometrische Frames als zuverlässige Anker:**

Frames mit `astrometric_rescue` können als **absolute Anker** für benachbarte
Frames dienen — unabhängig von `chain_depth`:

```cpp
// In is_chain_anchor_valid():
case RegistrationProvenance::astrometric_rescue:
    return true;  // Astrometrische Frames sind immer valide Anker
                  // (absolute Position, keine Driftakkumulation)
```

**Progressive Enhancement mit absolutem Anker:**

```
Frame N:   astrometric_rescue (absolut, CC = 0.45)
           └── N+1: temporal_rescue gegen N (verankert an absolut)
               └── N+2: sequential_rescue gegen N+1
               └── N+3: sequential_rescue gegen N+2
```

Dies erzeugt eine **Kette mit absolutem Ankerpunkt**, die drifttoleranter ist
als reine Blind-Chains.

**Validierung interner Warps durch Astrometrie (Diagnose-Modus):**

Optional: Astrometrie als "Ground Truth" für interne Registrierungen:

```cpp
// Vergleich interne Registrierung vs. astrometrische Lösung
if (internal_reg.provenance == direct_global &&
    astrometry_config.enabled &&
    astro.success) {

    float drift_px = compute_warp_drift(internal_reg.warp, astro.warp);
    if (drift_px > 5.0f) {
        emitter.warning(run_id, "Internal registration deviates from "
                       "astrometry by " + std::to_string(drift_px) + "px");
    }
}
```

**Konfiguration:**

```yaml
registration:
  use_astrometry: true       # default: true — Astrometrie-Rescue aktiv
                             # false → deaktiviert, auch wenn ASTAP verfügbar

# ASTAP-Config (wird für eigenständiges Plate-Solving verwendet):
astrometry:
  astap_bin: ""              # ASTAP-Binary (leer = Systempfad)
  astap_data_dir: /media/data/Astro/astap  # ASTAP-Datenverzeichnis
  search_radius: 183         # Suchradius in Grad
  # ... weitere ASTAP-Parameter
```

**Voraussetzungen:**
- `registration.use_astrometry: true` (default)
- ASTAP-Binary verfügbar (Pfad oder System-`$PATH`)
- Katalog-Daten in `astap_data_dir` vorhanden
- Optional: Referenz-Frame mit bekanntem WCS (aus Header oder erstem Frame)

**Erwartung:**
- 5-15% der `unresolved` Frames können gerettet werden (bei aktiver Astrometrie)
- Höhere geometrische Stabilität bei langen Sequenzen durch absolute Anker
- Keine Abhängigkeit von späterer ASTROMETRY-Phase — eigenständiges Plate-Solving

---

## 5. Empfohlene Konfigurationen

### 5.1 Standard-Sequenz (Montierung äquatorial, kein starker Mond)

```yaml
registration:
  engine: triangle_star_matching
  transform_model: similarity
  allow_rotation: true
  auto_engine: true
  star_topk: 150
  star_min_inliers: 4
  star_inlier_tol_px: 4.0
  reject_outliers: true
  reject_cc_min_abs: 0.25
  reject_shift_px_min: 100
  reject_shift_median_multiplier: 5.0
```

### 5.2 Alt/Az-Montierung mit starker Felddrehung

```yaml
registration:
  engine: triangle_star_matching
  transform_model: affine
  allow_rotation: true
  auto_engine: true
  auto_engine_rotation_threshold_deg: 0.05
  star_topk: 180          # Bereich: 180..260 bei sehr dichten Feldern
  star_min_inliers: 4
  star_inlier_tol_px: 4.0  # Bereich: 4.0..6.0 je nach Seeing
  reject_cc_min_abs: 0.20   # Bereich: 0.20..0.25 — nicht zu streng für Randbereiche
  reject_shift_px_min: 120
  reject_shift_median_multiplier: 5.0  # Bereich: 5.0..7.0 für lange Sessions
```

### 5.3 Wolken / schlechtes Seeing (maximale Rettungsrate)

```yaml
registration:
  engine: triangle_star_matching
  transform_model: affine
  allow_rotation: true
  enable_star_pair_fallback: true
  star_topk: 200            # möglichst viele Sterne für dünne Felder
  star_min_inliers: 3       # Minimum — darunter wird RANSAC unzuverlässig
  star_inlier_tol_px: 6.0   # großzügig wegen Seeing-Unschärfe
  reject_outliers: true
  reject_cc_min_abs: 0.15   # niedriger: mehr Frames behalten
  reject_shift_px_min: 80
  reject_shift_median_multiplier: 6.0   # großzügiger bei Shift-Outliers
```

### 5.4 Smart-Telescope / dichtes Sternfeld (Milchstraße, Kugelsternhaufen)

```yaml
registration:
  engine: triangle_star_matching
  transform_model: similarity
  allow_rotation: false   # Kompromiss: reduziert RANSAC-Suchraum im mehrdeutigen
                          # dichten Sternfeld. Bei langen Alt/Az-Sessions mit
                          # sichtbarer Felddrehung auf true setzen + star_topk: 40
  star_topk: 80              # weniger Sterne = weniger Mehrdeutigkeiten (Bereich: 60..100)
  star_min_inliers: 6        # höhere Konsistenzanforderung
  star_inlier_tol_px: 3.0    # enge Toleranz (Bereich: 2.5..4.0)
  reject_cc_min_abs: 0.30
```

### 5.5 Äquatorial gut getrackt, minimaler Mond

```yaml
registration:
  engine: triangle_star_matching
  transform_model: similarity   # Freiheitsgrade begrenzt für stabilere Schätzung
  allow_rotation: true
  star_topk: 150
  star_inlier_tol_px: 2.5       # enger: gutes Seeing erlaubt präzisere Centroide
  star_min_inliers: 5
  reject_shift_px_min: 60       # niedriger als Alt/Az: erwartete Dithering-Shifts klein
  reject_shift_median_multiplier: 5.0
  reject_cc_min_abs: 0.28
```

---

## 6. Provenance-Hierarchie und Frame-Qualität

Die `RegistrationProvenance` dokumentiert wie jeder Frame registriert wurde:

```
reference                → CC = 1.0   (Referenzframe)
direct_global            → CC = NCC   (direkt gegen Referenz)
sequential_refined       → CC = NCC   (verbessert durch frame-to-frame)
sequential_rescue        → CC = max(NCC, 0.01)  (blind-chained)
temporal_rescue          → CC = NCC   (gegen nächsten Anker)
seeded_ecc_rescue        → CC = NCC   (ECC mit interpoliertem Seed)
local_reference_rescue   → CC = NCC   (gegen lokale Referenz)
astrometric_rescue       → CC = max(NCC, 0.30)  (absolute Lösung, nur wenn Cache)
model_global_poly        → CC = 1e-4  (Grad-2-Polynom global)
model_local_poly         → CC = 1e-4  (lokales gewichtetes Polynom)
model_interpolated       → CC = 1e-4  (lineare Interpolation)
model_blended            → CC = 1e-4  (blended local+bridge)
model_nearest_copy       → CC = 1e-4  (nächster valider Warp kopiert)
unresolved               → CC = 0.0   (komplett gescheitert)
```

Frames mit CC = 1e-4 erhalten durch die downstream **Tile-Level Quality Metrics** automatisch sehr niedrige Gewichte - sie tragen zur finalen Komposition bei, aber nur minimal.

---

## 7. Diagnosemöglichkeiten

### 7.1 `global_registration.json` (Artifact)

Enthält für jeden Frame:
- `cc`: Korrelationskoeffizient
- `source`: Provenance-Name
- `chain_depth`: Tiefe der Blind-Chain
- `warps`: vollständige Warp-Matrix inkl. shift_px
- `dithering.detected_fraction`: Anteil registrierter Dithering-Verschiebungen

**Fehlende Felder (Optimierungspotenzial, siehe 4.0 / 8.F):**
- `overlap_px`: tatsächliche Overlap-Pixel der valid_mask (derzeit nur intern)
- `ncc_identity_overlap`: NCC der ungewarpten Version im Overlap-Bereich
- `ncc_warped`: NCC nach Warp im Overlap-Bereich
- `acceptance_margin_used`: tatsächlich genutzter Gate-Wert beim Akzeptieren
- `provenance_confidence`: Qualitätsklasse für `model_*`-Provenances (high/medium/low)
- `rescue_stages_attempted`: welche Rescue-Pfade versucht wurden
- `rescue_stages_succeeded`: welcher Pfad letztlich erfolgreich war

### 7.2 Log-Muster

```
[REG-DIAG#N]   → Erste 3 Frames: Sternzahl, NCC, Bild-Stats
[REG]          → Pro Frame: method_used, ncc_id, cc
[REG-SEQ-REFINE] → Anzahl sequential-refined Frames
[REG-SEQ]      → Anzahl phase-corr rescued Frames
[REG-TEMPORAL] → Temporal rescue Statistik
[REG-ECC]      → Seeded-ECC rescue Statistik
[REG-LOCAL-REF]→ Local-Reference rescue Statistik
[REG-FILTER]   → Abgelehnte Outlier-Warps
[REG-MODEL]    → Polynomial-Vorhersage-Statistik
[REG] cc>0: N  → Zusammenfassung valide / fehlgeschlagen
```

### 7.3 Typische Problemsignale

| Signal | Ursache | Maßnahme |
|---|---|---|
| Viele `too_few_stars` | Wolken, Mond, schlechtes Seeing | `star_topk` erhöhen, `star_inlier_tol_px` erhöhen |
| Viele `transform_fail` | RANSAC konvergiert nicht | `star_min_inliers` senken, `transform_model: similarity` |
| Viele `identity_fallback` | Kein NCC-Fortschritt | `reject_cc_min_abs` senken, ECC-Fallback prüfen |
| Viele `model_*` Provenances | Zu viele abgelehnte Frames | `reject_cc_min_abs` senken ODER Ursache (Wolken) prüfen |
| `astrometric_rescue` hoch | Astrometrie rettet viele Frames | Positiv — absolute Anker verfügbar |
| Viele `low_cc` rejected | Schlechte Bildqualität | `reject_cc_min_abs: 0.15` versuchen |
| `reflection` rejected | ECC/Phase gibt gespiegelte Lösung | Engine wechseln auf `triangle_star_matching` |
| `chain_depth` > 10 | Lange Wolkenblöcke | Erhöhung `max_blind_chain_depth` (derzeit hardcodiert) |
| Viele `model_blended/local_poly` | Outlier-Filter zu aggressiv am Session-Rand | `reject_shift_median_multiplier: 6-7`, Sequenzrand-Schutz (4.6) |
| `low_cc_protected` hoch | Viele chain-validierte Frames mit kleinem CC | Normal bei langen Wolkenblöcken; Polynomial-Modell übernimmt |

---

## 8. Vorgeschlagene Code-Änderungen (Priorisiert)

### Priorität 1 (Hohe Auswirkung, geringer Aufwand)

**A) Gemeinsame Gate-Policy-Funktion**

Zentrales Akzeptanz-Gate für alle Rescue-Pfade (ersetzt pfad-spezifische Magic-Numbers, siehe 4.0):

```cpp
// runner_shared.hpp — neue Funktion:
bool registration_gate(
    float ncc_before, float ncc_after,
    int overlap_px, float expected_shift_px,
    RegistrationProvenance provenance);

// Anwendung in runner_phase_registration.cpp:
// Alle Rescue-Pfade ersetzen:
//   if (ncc_warped <= ncc_identity + 0.005f)  →  
//   if (!registration_gate(ncc_identity, ncc_warped, overlap_px, shift_est, prov))
```

Konfigurierbare Basis-Parameter in `RegistrationConfig`:
```cpp
float gate_base_margin = 0.005f;       // neu
float gate_shift_sensitivity = 1.0f;   // neu: Einfluss des Shifts auf Margin
```

**B) Blind-Chain-Tiefe konfigurierbar machen**

```cpp
// In configuration.hpp → RegistrationConfig:
int max_blind_chain_depth = 12;         // neu
float blind_chain_strong_anchor_cc = 0.08f;  // neu (war hardcodiert)

// In runner_phase_registration.cpp:
// constexpr int kMaxBlindChainAnchorDepth = 12;  // ersetzen durch:
const int kMaxBlindChainAnchorDepth = registration_cfg.max_blind_chain_depth;
const float kBlindChainStrongAnchorCc = registration_cfg.blind_chain_strong_anchor_cc;
```

**C) Polynomial-Grad adaptiv**

```cpp
// Statt konstant Grad 2:
const int poly_degree = std::min(4, std::max(2, static_cast<int>(std::sqrt(nv / 3.0f))));
// V(i, 3) = t3  usw. wenn poly_degree >= 3
```

**D) Lokale Hintergrundsubtraktion in Stern-Detektion**

```cpp
// In detect_stars_simple(), vor dem Threshold:
// 2D-Median-Filter (31×31) subtrahieren → lokaler Hintergrund entfernt
cv::Mat f(h, w, CV_32F, const_cast<float*>(img.data()));
cv::Mat background;
cv::medianBlur(f, background, 31);
cv::Mat img_sub = f - background;
// img_sub für Stern-Detektion verwenden
```

### Priorität 2 (Mittlere Auswirkung)

**E) Szenenklassifikation / adaptive Parameter-Probe**

Vor der globalen Registrierung: 3–5 Probe-Frames analysieren und Parameter automatisch anpassen (siehe 4.12):

```cpp
// In run_phase_registration_prewarp(), nach auto_engine-Block:
SceneProfile scene = probe_scene_profile(
    frames, norm_scales, detected_mode, detected_bayer_str,
    /*n_probe=*/5);
apply_scene_adaptations(registration_cfg, scene);
// Loggt: star_density, gradient_strength, snr_estimate, fwhm_estimate
```

Ergänzt `auto_engine` (Engine-Wahl) um Parameter-Dimension (star_topk, inlier_tol, min_inliers).

**F) Vollständige affine Warp-Skalierung**

```cpp
// scale_translation_warp durch vollständige Similarity-Skalierung ersetzen:
WarpMatrix scale_affine_warp(const WarpMatrix& w, float scale) {
    WarpMatrix out = w;
    out(0, 2) *= scale;
    out(1, 2) *= scale;
    // Bei Skalierungsanteil im Warp (scale ≠ 1):
    // out(0,0) *= ?  - nur nötig wenn tatsächliche Skalen-DOF vorhanden
    return out;  // Aktuelles Verhalten korrekt für reine Rotation+Translation
}
```

**G) Differenzierte Modell-CC mit Qualitätsklassen**

```cpp
// Für model_* Provenances: CC aus Residuen und Span ableiten (drei Klassen)
const float res_total = chosen.res_tx / 20.0f + chosen.res_ang_deg / 1.0f;
const float span_penalty = std::min(1.0f, chosen.span / 50.0f);
const float model_cc = std::clamp(
    5e-4f / (1.0f + res_total + 2.0f * span_penalty),
    1e-5f, 5e-4f);
// high_confidence (enge Interpolation): ~5e-4
// medium_confidence (span 10-50 frames): ~1e-4
// low_confidence (Extrapolation/hohe Residuen): ~1e-5
set_registration_state(fi, w, model_cc, false, -1, chosen_provenance);
```

### Priorität 3 (Diagnostik-Verbesserungen)

**H) Erweiterte Telemetrie in `global_registration.json`**

```cpp
// Pro Frame zusätzlich in j["warps"] schreiben:
j["warps"].push_back(core::json{
    // ... bestehende Felder ...
    {"overlap_px",             overlap_pixels},
    {"ncc_identity_overlap",   ncc_identity_overlap},
    {"ncc_warped",             ncc_warped_value},
    {"acceptance_margin_used", margin_used},
    {"provenance_confidence",  model_confidence_class},  // "high"/"medium"/"low"
    {"rescue_stages_attempted", rescue_stages_tried},
    {"rescue_stage_succeeded",  rescue_stage_name},
});
```

Enabler für datengetriebenes Schwellwert-Tuning statt Trial-and-Error.

**I) Warn bei hohem chain_depth**

```cpp
if (reg_max_chain_depth > kMaxBlindChainAnchorDepth * 0.8f) {
    emitter.warning(run_id, "High blind-chain depth " +
        std::to_string(reg_max_chain_depth) +
        " — consider increasing max_blind_chain_depth", log_file);
}
```

---

## 9. Queranalyse: Konvergenz beider Analysen

Die folgende Tabelle zeigt welche Befunde unabhängig in **beiden** Analysen identifiziert wurden und wo sich die Dokumente gegenseitig ergänzen:

| Thema | Diese Analyse | `registration_optimierung_...md` | Konvergenz |
|---|---|---|---|
| Inkonsistente NCC-Gates | §4.3 (Schwelle `0.01` zu niedrig) | P0-A (Gates zwischen Pfaden verschieden) | ✅ Gleiche Wurzel, Gate-Policy-Funktion als Lösung |
| Outlier-Filter zu aggressiv | §4.6 (Konfiguration senken) | P0-B (Session-Rand: erwartet größere Shifts) | ✅ Ergänzend: Kontext-Sensitivität als strukturelle Verbesserung |
| Modell-CC = 1e-4 undifferenziert | §4.10 (Formel aus Residuen) | P0-C (drei Qualitätsklassen) | ✅ Gleich, Qualitätsklassen-Konzept aus P0-C übernommen |
| Stern-Detektion nicht adaptiv genug | §4.4 (Hintergrundsubtraktion, 3. Fallback-Stufe) | P1-H (lokaler Background-Rescue) | ✅ Ergänzend: P1-H ergänzt §4.4 um zusätzliche robuste Stufe |
| Lokale Referenz bei langen Blöcken | §4.11 (Clusterbasierte Auswahl) | P1-E (beidseitig + Konsistenz) | ✅ Ergänzend: P1-E spezifischer zu `build_local_reference` |
| Mehr Telemetrie | §7.1 (fehlende Felder identifiziert) | P2-I (konkrete Feldnamen) | ✅ P2-I Feldnamen in §7.1 und 8.H integriert |
| **Blind-Chain konfigurierbar** | §4.1, 8.B (Parameter aus Hardcode) | P1-F (steuerbar machen) | ✅ **Integriert:** Konfigurierbare Parameter + Code-Beispiele |
| **Drift-Check mit Trendabgleich** | §4.2 (Polynom-Drift-Check) | P1-F (Driftkontrolle) | ✅ **Integriert:** Zusätzlicher Schutz für tiefe Chains |
| **Adaptiver Polynomial-Grad** | §4.7, 8.C (Grad 2..4 adaptiv) | P1-G (adaptiver Grad statt fix) | ✅ **Integriert:** Formel und Schwellen dokumentiert |
| **Szenenklassifikation** | §4.12, 8.E (Probe-Phase) | P1-D (Probe + Auto-Params) | ✅ **Integriert:** Vollständige Phase mit allen Adaptionsregeln |
| **Chain-Tiefen-Warnung** | §7.3, 8.I (Frühwarnung) | P2-J (Warnung in Logs) | ✅ **Integriert:** Log-Warnung bei hoher Chain-Tiefe |
| Warp-Skalierung nur Translation | §4.8 (affine Vollskalierung) | — | 🔵 Nur in dieser Analyse |
| Session-Rand-Schutz im Filter | — | P0-B (Rand-Frames schützen) | 🔵 Nur in Querdokument → §4.6 ergänzt |
| Geometrische Konsistenz Support | — | P1-E (Warp-Residuum prüfen) | 🔵 Nur in Querdokument → §4.11 neu |

**Die 5 zentralen Ergänzungen aus dem Querdokument (jetzt integriert):**

| # | Ergänzung | Abschnitt | Kerninhalt |
|---|-----------|-----------|------------|
| 1 | **Blind-Chain konfigurierbar** | §4.1, 8.B | `max_blind_chain_depth` und `blind_chain_strong_anchor_cc` in `RegistrationConfig` |
| 2 | **Drift-Check mit Trendabgleich** | §4.2 | Lokaler Modellabgleich für tiefe Chains (`|w - trend| > 2σ` → kein Anker) |
| 3 | **Adaptiver Polynomial-Grad** | §4.7, 8.C | `degree = min(4, max(2, sqrt(nv/3)))` statt fixem Grad 2 |
| 4 | **Szenenklassifikation** | §4.12, 8.E | Probe-Phase mit `SceneProfile` → automatische Parameteradaption |
| 5 | **Chain-Tiefen-Warnung** | §7.3, 8.I | Log-Warnung wenn `chain_depth > 0.8 * max_depth` |

**Kernbotschaft beider Analysen:** Die Registrierung ist in ihrer Rescue-Tiefe bereits sehr gut. Die verbleibenden Schwachstellen liegen vor allem in der **Konsistenz der Entscheidungsregeln** zwischen den Pfaden und in der fehlenden **Szenenadaption** vor der Kaskade — nicht in den Algorithmen selbst.

---

## 10. Umsetzungsplan

### Kurzfristig — ohne Algorithmusänderung (Konfiguration + kleine Fixes)

| Maßnahme | Aufwand | Auswirkung |
|---|---|---|
| `blind_chain_strong_anchor_cc` auf 0.05 senken | 1 Zeile | Mehr Frames als Chain-Anker zulässig |
| `reject_shift_median_multiplier: 6-7` für lange Alt/Az-Sessions | Konfig | Weniger False-Rejects am Session-Rand |
| `reject_cc_min_abs: 0.20` bei Wolken/schwachem Seeing | Konfig | Mehr Frames bleiben mit echtem Warp |
| `star_topk: 200`, `star_inlier_tol_px: 5-6` bei Mondlicht | Konfig | Bessere Stern-Detektion |
| Warn-Log bei `chain_depth > 0.8 * max_depth` (Code 8.I) | ~5 Zeilen | Frühe Diagnose langer Wolkenblöcke |

### Mittelfristig — Moduländerungen (geringes Refactoring-Risiko)

| Maßnahme | Abschnitt | Aufwand |
|---|---|---|
| `max_blind_chain_depth` in `RegistrationConfig` | 8.B | Klein |
| `use_astrometry` in `RegistrationConfig` | 4.13 | Klein |
| Differenzierte Modell-CC (drei Klassen) | 8.G | Klein |
| Lokale Hintergrundsubtraktion in Stern-Detektion | 8.D | Klein |
| Adaptiver Polynomial-Grad | 8.C | Klein |
| Erweiterte JSON-Telemetrie (Felder in 7.1) | 8.H | Mittel |
| Sequenzrand-Schutz im Outlier-Filter | 4.6 | Mittel |
| Beidseitige Abdeckung in `build_local_reference` | 4.11 | Mittel |

### Langfristig — Architektur-Refactoring

| Maßnahme | Abschnitt | Aufwand | Erwarteter Nutzen |
|---|---|---|---|
| Gemeinsame Gate-Policy-Funktion | 8.A / 4.0 | Groß | Konsistente Entscheidungen, keine Magic-Numbers |
| Szenenklassifikation vor Kaskade | 8.E / 4.12 | Groß | Automatisch optimale Parameter pro Sequenz |
| Geometrische Konsistenz für Support-Frames | 4.11 | Mittel | Stabilere Rettung in langen Problemblöcken |
| Astrometrische Rescue-Phase | 4.13 | Mittel | Rettung von 5-15% zusätzlicher Frames, absolute Anker |

---

## 11. Zusammenfassung: Warum Frames scheitern und was dagegen hilft

```
Frame scheitert bei Registrierung
        │
        ├── Zu wenige Sterne erkannt
        │       → star_topk erhöhen (200+)
        │       → star_inlier_tol_px erhöhen (5-6px)
        │       → Lokale Hintergrundsubtraktion (Code-Change 8.D)
        │       → Szenenklassifikation probt dies automatisch (Code-Change 8.E)
        │
        ├── RANSAC findet keinen Konsens
        │       → star_min_inliers senken (3)
        │       → transform_model: similarity statt affine
        │       → enable_star_pair_fallback: true
        │
        ├── NCC-Verbesserung zu gering (identity_fallback)
        │       → reject_cc_min_abs senken (0.15)
        │       → min_ncc_improvement anpassen (adaptiv: 5% von ncc_identity)
        │       → Gate-Policy-Funktion (Code-Change 8.A) vereinheitlicht diese Schwelle
        │
        ├── Outlier-Filter zu aggressiv (besonders am Session-Rand)
        │       → reject_cc_min_abs senken (0.20)
        │       → reject_shift_median_multiplier erhöhen (6-7)
        │       → Sequenzrand-Schutz aktivieren (§4.6)
        │
        ├── Lokale Referenz zu einseitig bei langem Ausfall-Block
        │       → build_local_reference mit beidseitiger Abdeckung (§4.11)
        │       → Geometrische Konsistenz der Support-Frames prüfen
        │
        └── Langer Wolkenblock (Blind-Chain-Tiefe erschöpft)
                → max_blind_chain_depth erhöhen (Code-Change 8.B)
                → Temporal/Seeded-ECC-Rescue läuft automatisch (4 Passes)
                → Astrometrische Rescue (wenn Katalog-Cache verfügbar)
                   → Absolute Anker für benachbarte Frames
                → Polynomial-Prediction deckt Rest ab
                   (differenzierte CC via Code-Change 8.G)
```

Alle Frames die nach allen Rettungsstrategien noch `unresolved` bleiben, erhalten einen polynomial-vorhergesagten Warp — mit differenzierter CC je nach Vorhersage-Qualität (1e-5 bis 5e-4) — und werden mit entsprechend niedrigem Gewicht in das finale Stack eingebaut. **Kein Frame wird komplett verworfen** (methodisches Ziel v3.2.2 §1.2).

Frames mit `astrometric_rescue` erhalten einen absoluten, katalogbasierten Warp mit hohem CC (≥ 0.30) und dienen als zuverlässige Anker für benachbarte Frames — die höchste Qualitätsstufe der Registrierung.
