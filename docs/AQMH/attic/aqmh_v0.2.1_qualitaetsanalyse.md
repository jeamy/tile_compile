# AQMH v0.2.1 Qualitätsanalyse — Engstellen und Verbesserungsvorschläge

**Status:** Analyse, 2026-07-18; aktualisiert nach Code-/Run-Abgleich  
**Kontext:** AQMH v0.2.1 soll die Classic Tile Compile Methode übertreffen. Aktuell liegt die Qualität deutlich darunter.  
**Referenz-Runs:** `m42_20260703_083337` (Classic, besser), `M42-pi_20260717_190448` (AQMH, schlechter)

---

## 0. Zusammenfassung und Bewertung

Die Kernaussage ist korrekt: **Der nächste Fix muss in AQMH-Reconstruction/Gate-Logik erfolgen, nicht erneut über HMS/BGE/PCC-Parameter.** Im Run `M42-pi_20260717_190448` wird AQMH bereits in `AQMH_RECONSTRUCTION` auf `uniform_control` zurückgesetzt. Danach können BGE, PCC und HMS nur noch ein nahezu ungewichtetes Zwischenprodukt weiterverarbeiten.

Wichtig: Der Fix darf keine M42-Sonderbehandlung sein. Die Gate-Logik muss objektagnostisch funktionieren: helle Emissionsnebel, schwache Galaxienarme, staubige IFN-Felder, Sternhaufen und sternarme Galaxienfelder müssen nach denselben Regeln bewertet werden. Die Regeln dürfen also nicht an Objektname, Helligkeit oder eine feste Nebelstruktur angepasst werden, sondern nur an die **Anwendbarkeit und Stabilität der jeweiligen Metrik**.

Der Code-/Run-Abgleich zeigt aber: Die bisherige Analyse war zu stark auf "Thresholds sind zu eng" fokussiert. Das ist nur ein Teil des Problems. Der konkrete M42-Fallback wird durch mehrere Gate-Fehler zusammen ausgelöst:

- `background_rms`: `control_background_rms = 0.0`, `aqmh_background_rms = 0.2223`, daraus wird eine künstliche Regression von `2,000,000`.
- `seam_score`: `control_seam_score = null`, trotzdem wird `seam_score_ok = false`.
- `tail11_abs`: AQMH ist deutlich schlechter als Control (`+96.7%`), also scheitert `tail_ok`.
- Gleichzeitig ist AQMH bei FWHM besser (`-2.76%`) und Elongation besser (`-2.37%`).

Damit ist das Gate nicht nur "streng", sondern logisch nicht robust gegenüber degenerierten Control-Metriken und asymmetrischen Qualitäts-Tradeoffs.

---

## 1. Identifizierte Engstellen

### 1.1 Uniform-Control Gate behandelt degenerierte Metriken falsch (Hauptproblem)

**Run-Beleg `M42-pi_20260717_190448`:**

```json
{
  "fallback_to_uniform_control": true,
  "uniform_control_blend_accepted": false,
  "uniform_control_gate": {
    "aqmh_fwhm": 6.0228,
    "control_fwhm": 6.1938,
    "fwhm_regression": -0.0276,
    "aqmh_background_rms": 0.2223,
    "control_background_rms": 0.0,
    "background_rms_regression": 2000000.0,
    "control_seam_score": null,
    "seam_score_ok": false,
    "tail11_abs_regression": 0.9671,
    "tail_ok": false
  }
}
```

**Bewertung:** Das Gate verwirft einen Kandidaten, der FWHM und Elongation verbessert, weil Vergleichsmetriken des Controls numerisch degeneriert sind (`0.0`/`null`) und Tail/Seam als harte Einzelmetriken wirken. Das ist ein Code-/Gate-Fehler, nicht per HMS/BGE/PCC lösbar.

Notwendige Korrekturen:

1. Degenerierte Control-Metriken dürfen kein hartes Fail erzeugen. Bei `control_background_rms <= eps`, `control_seam_score == null` oder zu wenig Sternen muss die jeweilige Gate-Metrik als `not_applicable` markiert werden.
2. `NaN`/`null`-Regressionen müssen explizit behandelt werden; sie dürfen weder implizit failen noch im Artifact wie echte Regressionen aussehen.
3. FWHM-/Elongation-Verbesserungen müssen gegen Background-/Tail-Regressionen bewertet werden, statt jede Metrik isoliert als veto zu benutzen.

### 1.2 Validation Thresholds zu eng (sekundär, aber real)

```cpp
// configuration.hpp — AqmhValidationConfig
float max_seam_score_regression = 0.02f;
float max_fwhm_regression = 0.02f;
float max_background_rms_regression = 0.02f;  // nur 2% Toleranz!
float max_tail11_abs_regression = 0.05f;
float max_elongation_regression = 0.05f;
```

**Mechanismus:** Wenn das AQMH-Ergebnis in einer Gate-Metrik schlechter ist als der ungewichtete Mittelwert (`uniform_control`), fällt die gesamte Rekonstruktion auf `uniform_control` zurück oder wird per Binary-Search attenuiert (`runner_phase_aqmh_reconstruction.cpp`).

**Warum das fatal ist:** Qualitätsgewichtung verbessert FWHM (schärfer), kann aber `background_rms` leicht verschlechtern (weil weniger Frames effektiv mitteln -> höheres statistisches Rauschen). Die 2%-Schwelle kann dann den Fallback triggern, und das gesamte Schärfe-Improvement geht verloren.

**Classic hat keinen solchen Fallback** — es verwendet `W_f,t = G_f × L_f,t` direkt ohne nachträgliche Validierung gegen einen ungewichteten Vergleich.

### 1.3 Sigmoid-Kompression (zu wenig Gewichtsspreizung)

```cpp
// aqmh_global_quality.cpp
zs = std::clamp(zs, -5.0f, 5.0f);
zn = std::clamp(zn, -5.0f, 5.0f);
zb = std::clamp(zb, -5.0f, 5.0f);
const float score = w_sharp_norm * zs + w_snr_norm * zn - w_background_norm * zb;
const float sigmoid = 1.0f / (1.0f + std::exp(-score));
result.weights[i] = cfg.g_floor + (1.0f - cfg.g_floor) * sigmoid;
```

Mit `g_floor = 0.05` und dem score ∈ [-5, 5] (geclippt) liegt der theoretische Gewichtsbereich bei ca. `[0.056, 0.994]`. Da der score ein normalisierter z-Score ist, liegen viele Frames nahe bei score≈0 -> `G_f ≈ 0.52`.

**Run-Beleg M42:** `global_quality` liegt bei min/p10 `0.056`, median `0.526`, p90 `0.711`, max `0.810`. Das ist messbar zu mild für einen Qualitätsstack mit 610 Frames: p90/p10 ≈ `12.6:1`, max/min ≈ `14.4:1`. Es ist nicht Classic-ähnlich.

**Vergleich Classic:** `G_f = exp(k_global * Q)` — bei `k_global=1.0` und z-Scores von -3 bis +3 ergibt das eine Spreizung von `e^{-3}` bis `e^{3}` = **Faktor 400:1**. AQMH liegt im M42-Run mit max/min ≈ **14:1** deutlich darunter.

### 1.4 Low-Frequency Neutralisation ist zu früh und zu pauschal

```cpp
// runner_phase_aqmh_reconstruction.cpp
constexpr float neutralization_sigma_px = 96.0f;
Matrix2Df neutralized = low_frequency_neutralized_aqmh(
    aqmh_recon.output, aqmh_recon.uniform_control_output,
    neutralization_sigma_px);
```

**Was passiert:** Die Neutralisation berechnet `L(p) = GaussianBlur(A(p) - U(p), σ=96px)` und subtrahiert `L` vom AQMH-Output.

**Das Problem:** Wenn AQMH korrekt Frames mit schlechtem Hintergrund runtergewichtet, hat das Ergebnis einen flacheren Hintergrund als der ungewichtete Mean. Die Differenz `A(p) - U(p)` hat dann eine großflächige Komponente — das ist das Improvement. Die Neutralisation mit σ=96px glättet genau diese Differenz heraus und subtrahiert sie.

**Effekt:** Der Hintergrund-Qualitätsgewinn kann aktiv entfernt werden.

**Verschärfung:** Die aktuelle Logik neutralisiert auch wenn `raw_validation.background_rms_regression < 0` (AQMH bereits besser als Control):

```cpp
const bool neutralized_background_improved =
    low_frequency_neutralization_validation.background_rms_regression <
    raw_validation.background_rms_regression;
// → neutralisiert wird, wenn neutralized < raw, egal ob raw < 0!
```

Im M42-Run wurde zwar `low_frequency_neutralization_applied = true` gemessen, aber als Basis letztlich `raw` gewählt (`raw_background_regression=2e+06`, `neutralized_background_regression=2e+06`). Für diesen Run ist Neutralisation daher nicht der direkte Fallback-Auslöser. Sie bleibt aber ein Logikrisiko, weil sie vor dem eigentlichen Gate berechnet wird und nur anhand von `background_rms_regression` ausgewählt wird.

### 1.5 Structure Mask zu konservativ und Gate-geblockt

```cpp
constexpr float structure_low_q = 0.70f;   // 70% der Pixel = "Hintergrund"
constexpr float structure_high_q = 0.97f;  // nur top 3% = volle AQMH-Qualität
constexpr float structure_mask_blur_sigma_px = 2.0f;
```

**Effekt:** 70% der Pixel werden als "Hintergrund" klassifiziert und folgen dem uniform control. Nur die obersten 3% Gradient-Intensität bekommen das volle AQMH-Ergebnis.

**Besonders problematisch bei Nebeln:** Bei M42 liegt ein Großteil der interessanten Struktur im Bereich P30–P70 des Gradientenhistogramms. Diese Bereiche werden komplett vom uniform control dominiert — das AQMH-Improvement wirkt dort nicht.

**Run-Beleg M42:** `structure_masked_detail` verbessert FWHM gegenüber Control (`6.1169` vs. `6.1938`) und hält `background_rms = 0`, wird aber nicht angewendet, weil Tail/Seam-Gates nicht robust sind (`tail11_abs_regression = 1.0537`, `seam_score = null`).

### 1.6 Registration-Weight-Guard: in M42 nicht Hauptproblem

```cpp
// configuration.hpp — AqmhReconstructionConfig
float registration_sequential_factor = 0.85f;   // 15% Reduktion für sequentiell
float registration_predicted_factor = 0.35f;    // 65% Reduktion für predicted!
float registration_weight_floor = 0.35f;
```

**Problem:** Bei typischen Deep-Sky-Aufnahmen können viele Frames sequentiell oder per Chain-Prediction registriert werden. Ein `predicted_factor` von 0.35 reduziert deren Gewicht auf ein Drittel. Wenn viele Frames predicted/interpolated/unknown sind, verliert man massive effektive Integrationzeit.

**Paradox:** `registration_weight_floor = 0.35` ist identisch mit `predicted_factor = 0.35` — der Floor hat also keine Schutzwirkung für predicted Frames.

**Run-Beleg M42:** Hier sind `583/610` Frames `direct_global`, `24` `sequential_refined`, `2` `model_blended`, `1` `reference`. Der Guard dämpft zwar `562` Frames wegen CC-Mapping, aber mean factor `0.763` und median factor `0.94` zeigen: Das ist nicht der primäre Qualitätsverlust dieses Runs. Für M42 ist der Guard nachrangig gegenüber Uniform-Control-Gate und Gewichtsspreizung.

### 1.7 CV-Gating eliminiert nützliche Signale

```cpp
// aqmh_global_quality.cpp — effective_weight lambda
const float cv = mad / med;
if (cv < 0.01f)
  return 0.0f;  // Signal wird komplett deaktiviert
```

**Problem:** Wenn die Frame-zu-Frame-Variation eines Signals unter 1% liegt, wird das gesamte Gewicht für diese Dimension auf 0 gesetzt. Bei typischen Astrofotos ist SNR oft sehr gleichförmig über Frames → die SNR-Dimension fällt komplett weg, nur Sharpness bleibt als Qualitätssignal.

**Konsequenz:** Die Global Quality reduziert sich effektiv auf eine reine Sharpness-Gewichtung. Background-Penalty und SNR tragen nicht bei. Die mehrdimensionale Qualitätsbewertung degeneriert.

### 1.8 Objektübergreifende Risiken

Ein Gate, das nur an M42 validiert wird, kann bei Galaxien oder sternarmen Feldern falsch entscheiden:

- **Galaxien:** Die zentrale Bulge ist hell und gradientscharf, aber die wertvollen Details liegen oft in schwachen Spiralarmen und Staubbändern. Eine Structure Mask mit `low_q=0.70/high_q=0.97` kann diese mittleren Strukturen zu stark in Richtung `uniform_control` ziehen.
- **Sternarme Felder:** FWHM, Tail und Elongation sind nur belastbar, wenn genug vergleichbare, nicht gesättigte Sterne gefunden werden. Sonst müssen sie `not_applicable` sein, nicht `fail`.
- **Sternhaufen:** Sternmetriken sind sehr stabil, aber Background-/Seam-Metriken können durch viele Sternflügel verzerrt werden. Background-Gates müssen eine robuste Hintergrundmaske verwenden.
- **Faint nebula / IFN:** Der Bildwert liegt nahe am Hintergrund. Relative Background-RMS-Regressions gegen kleine Control-Werte sind numerisch gefährlich und brauchen absolute Mindestskalen.
- **Helle Nebel:** Metriken dürfen echte großflächige Signalverbesserung nicht als "low-frequency veil" wegneutralisieren.

Die Korrektur muss deshalb als allgemeine Gate-Architektur umgesetzt werden:

```text
metric_value -> applicability check -> pass/fail/not_applicable -> weighted decision
```

Nicht:

```text
metric_value -> relative regression -> hartes Veto
```

---

## 2. Vergleich: Warum Classic überlegen ist

| Aspekt | Classic Tile Compile | AQMH v0.2.1 |
|--------|---------------------|--------------|
| Gewichtsspreizung | `exp(k * Q)` → 400:1 | `sigmoid(z)` → 5:1 |
| Fallback bei Regression | **Keiner** — Gewichte wirken direkt | Uniform-Control Fallback bei 2% Regression |
| Räumliche Auflösung | Tile-basiert (48–64px Tiles) | Pixel-weise (theoretisch besser) |
| Lokale Gewichtung | `L_f,t = exp(k_local * Q_local)` | `Q_map` direkt (keine Exponentialexpansion) |
| Post-Processing Safety | Keine nachträgliche Abschwächung | Neutralisation + Structure Mask + Attenuation |
| Effektives Qualitätssignal | Sharpness + Roundness + Contrast | Oft nur Sharpness (wegen CV-Gate) |

**Kernunterschied:** Classic vertraut seinen Metriken und liefert konsequent das qualitätsgewichtete Ergebnis. AQMH misstraut seinem eigenen Output und validiert ihn gegen den ungewichteten Mean — wobei jede moderate Verschlechterung in einer einzelnen Metrik den gesamten Gewinn zunichte macht.

---

## 3. Lösungsvorschläge

### 3.1 Gate-Validierung robust machen (Code-Fix, zuerst)

**Dateien:** `tile_compile_cpp/src/reconstruction/aqmh_validation.cpp`, `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`

Pflichtänderungen:

```cpp
// Prinzip:
// - Regression nur berechnen, wenn control finite und oberhalb eps ist.
// - Nicht-anwendbare Metriken als not_applicable reporten.
// - not_applicable darf kein hartes Fail sein.
// - NaN/null darf nicht stillschweigend als Fail oder Pass durchrutschen.
```

Konkrete Anforderungen:

- `background_rms_regression`: Wenn `control_background_rms <= eps`, keine relative Regression berechnen. Stattdessen absolute RMS-Schwelle oder `not_applicable`.
- `seam_score_regression`: Wenn `control_seam_score` nicht finite ist, Gate als `not_applicable`.
- `tail11_abs_regression`: Nur hart werten, wenn beide Kandidaten genug stabile Sternsamples haben und die Detektion vergleichbar ist.
- Artifact muss pro Gate enthalten: `status: pass|fail|not_applicable`, `reason`, `value`, `control`, `threshold`.

Diese Regeln sind objektagnostisch: Bei einer sternarmen Galaxie wird ein Tail-Gate eher `not_applicable`, bei einem Sternhaufen eher `pass/fail`; bei schwachen IFN-Feldern verhindert die absolute Background-Skala künstliche relative Explosionen.

### 3.2 Asymmetrische Validation statt Veto pro Einzelmetrik

**Datei:** `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`

Aktuell kann ein Kandidat mit besserem FWHM/Elongation durch Tail/Background komplett verworfen werden. Stattdessen sollte die Gate-Entscheidung primäre und sekundäre Metriken trennen:

```cpp
const bool primary_improves =
    fwhm_regression <= -0.02f || elongation_regression <= -0.02f;

const bool secondary_regression_tolerable =
    background_gate != fail_hard &&
    tail_gate != fail_hard &&
    seam_gate != fail_hard;

const bool candidate_ok =
    primary_improves
        ? secondary_regression_tolerable
        : all_applicable_gates_pass;
```

Die exakten Schwellen müssen in Tests kalibriert werden. Wichtig ist die Richtung: Ein einzelner sekundärer Veto darf einen klaren Schärfegewinn nicht automatisch auf `uniform_control` zurücksetzen.

### 3.3 Festgelegte AQMH-Grundkonfiguration (erst nach Gate-Fix sinnvoll)

Die folgende Konfiguration ist als zukünftiger robuster Startpunkt festgelegt. Sie ist bewusst nicht M42-spezifisch, sondern konservativ für helle Nebel, schwache Galaxien, IFN, Sternhaufen und sternarme Felder. Sie ersetzt keine Code-Korrektur im Gate; ohne Gate-Fix kann weiterhin ein besserer AQMH-Kandidat auf `uniform_control` zurückfallen.

```yaml
aqmh:
  storage:
    resolution_divisor: 2                  # robuster Default; 1 fuer Cherry-pick/Diagnostik
    dtype: uint16                          # kompakt; float32 fuer exakte Diagnostik

  global_quality:
    g_floor: 0.03                          # etwas mehr Spreizung als bisher, aber nicht aggressiv
    g_w_sharp: 0.55
    g_w_snr: 0.30
    g_w_background_penalty: 0.25

  reconstruction:
    min_fraction: 0.40
    min_n_eff: 2.0
    registration_weight_guard: true
    registration_weight_floor: 0.30
    registration_sequential_factor: 0.92
    registration_predicted_factor: 0.50

  cherry_pick:
    enabled: false                         # Default bleibt aus
    k_frac: 0.50                           # nur relevant, wenn explizit aktiviert
    k_min_required: 30
    margin_min: 0.02

  validation:
    max_fwhm_regression: 0.02              # Schärfe bleibt streng
    max_background_rms_regression: 0.05
    max_seam_score_regression: 0.05
    max_tail11_abs_regression: 0.10
    max_elongation_regression: 0.08
```

Diese Werte sind auch in `tile_compile_cpp/examples/aqmh_tuning.example.yaml` gesetzt. Sie sind als Startkonfiguration gedacht, nicht als Ersatz für HMS/BGE/PCC-Tuning und nicht als harte Zielwerte für jede einzelne Objektklasse.

### 3.4 Exponential- statt Sigmoid-Gewichtung

**Datei:** `tile_compile_cpp/src/metrics/aqmh_global_quality.cpp`

Ersetze die Sigmoid-Funktion durch eine Exponentialfunktion analog zu Classic:

```cpp
// ALT:
const float sigmoid = 1.0f / (1.0f + std::exp(-score));
result.weights[i] = cfg.g_floor + (1.0f - cfg.g_floor) * sigmoid;

// NEU:
const float clamped_score = std::clamp(score, -3.0f, 3.0f);
const float raw_weight = std::exp(cfg.g_k_scale * clamped_score);
result.weights[i] = std::max(cfg.g_floor, raw_weight);
```

Neuer Config-Parameter in `AqmhGlobalQualityConfig`:

```cpp
float g_k_scale = 1.5f;  // Steuerung der Gewichtsspreizung
```

Effekt bei `g_k_scale = 1.5`:
- score = 0 → weight = 1.0 (Median-Frame)
- score = +2 → weight = e^3 ≈ 20× (exzellenter Frame)
- score = -2 → weight = e^{-3} ≈ 0.05× (schlechter Frame)

Gewichtsverhältnis bester/schlechtester: **400:1** statt 5:1.

### 3.5 Neutralisation nur bei tatsächlichem Gradient-Problem

**Datei:** `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`

Die Neutralisation darf nur greifen, wenn AQMH einen **schlechteren** Hintergrund hat als Uniform Control:

```cpp
// NEUER GUARD vor der Neutralisation:
const bool aqmh_background_worse_than_control =
    raw_validation.background_rms_regression > 0.0f;

const Matrix2Df &neutralization_base =
    (neutralized_background_improved && aqmh_background_worse_than_control)
        ? neutralized
        : aqmh_recon.output;
```

Wenn `raw_validation.background_rms_regression <= 0` (AQMH hat bereits besseren Hintergrund), wird die Neutralisation übersprungen.

### 3.6 Structure Mask öffnen

**Datei:** `tile_compile_cpp/apps/runner_phase_aqmh_reconstruction.cpp`

```cpp
// ALT:
constexpr float structure_low_q = 0.70f;
constexpr float structure_high_q = 0.97f;
constexpr float structure_mask_blur_sigma_px = 2.0f;

// NEU:
constexpr float structure_low_q = 0.40f;   // Nebel ab P40 einbeziehen
constexpr float structure_high_q = 0.90f;  // breitere Übergangszone
constexpr float structure_mask_blur_sigma_px = 4.0f;  // weicherer Übergang
```

Idealerweise als konfigurierbar in `AqmhReconstructionConfig` exponieren:

```cpp
struct AqmhReconstructionConfig {
  // ... bestehende Felder ...
  float structure_mask_low_q = 0.40f;
  float structure_mask_high_q = 0.90f;
  float structure_mask_blur_sigma_px = 4.0f;
};
```

Für Galaxien und IFN sollte die Maske nicht nur aus `grad(U)` und festen Quantilen bestehen. Besser ist eine mehrskalige Structure Confidence:

```text
M_s = max(
  normalized_gradient(U),
  normalized_laplacian_or_DoG(U),
  faint_structure_confidence(U, background_mask)
)
```

Damit werden nicht nur helle Kanten, sondern auch schwache Spiralarm-/Staubband-Strukturen berücksichtigt. Die Default-Quantile müssen konservativ, aber nicht M42-spezifisch sein.

### 3.7 Cherry-Pick nur optional aktivieren (für Datasets mit hoher Frame-Zahl)

Bei 610 Frames (wie im M42-Dataset) kann Cherry-Pick ein starkes Werkzeug sein, ist aber kein sicherer allgemeiner Default. Für die Grundkonfiguration bleibt es deaktiviert:

```yaml
aqmh:
  cherry_pick:
    enabled: false
    k_frac: 0.50     # beste 50% der Frames pro Pixel
    k_min_required: 30
```

Wenn Cherry-Pick für viele Frames explizit aktiviert wird, sollte `aqmh.storage.resolution_divisor: 1` verwendet werden, damit die Selektionskarte volle räumliche Details hat. Cherry-Pick umgeht teilweise die Sigmoid-Kompression, weil es die schlechtesten Frames per Pixel komplett ausschließt statt nur mild runterzugewichten; dieser Modus muss deshalb separat gegen Galaxien, IFN und sternarme Felder geprüft werden.

---

## 4. Priorisierung

| Prio | Maßnahme | Erwarteter Effekt | Risiko | Aufwand |
|------|----------|-------------------|--------|---------|
| 1 | Degenerierte Gate-Metriken reparieren (§3.1) | verhindert künstliche `2e6`/`null`-Fails | niedrig-mittel | Code |
| 2 | Asymmetrische Validation (§3.2) | FWHM-/Elongation-Gewinn wird nicht durch sekundäre Veto-Metrik weggeworfen | mittel | Code |
| 3 | Exponential- statt Sigmoid (§3.4) | deutlich stärkere Qualitätsselektion | mittel | Code |
| 4 | Structure Mask öffnen und Gate-fähig machen (§3.6) | Nebel profitiert von AQMH | niedrig-mittel | Code |
| 5 | Neutralisation-Guard (§3.5) | verhindert Entfernen echter Low-Frequency-Verbesserung | niedrig | Code |
| 6 | Cherry-Pick nach Gate-Fix testen (§3.7) | schlechteste Frames per Pixel eliminieren | mittel | Config |
| 7 | Registration Guard run-spezifisch entschärfen (§3.3) | mehr effektive Frames bei vielen predicted/interpolated Frames | niedrig | Config |

---

## 5. Empfohlene Test-Strategie

1. **Schritt 1 (Code):** Gate-Metriken robust machen (`not_applicable` statt Fail bei degeneriertem Control; keine `2e6`-Regression als harte Entscheidung).
2. **Schritt 2 (Code):** Asymmetrisches Gate implementieren und alle Kandidaten vollständig loggen: Raw AQMH, Neutralized, Structure-Masked, Final.
3. **Schritt 3 (Code):** Exponentialgewichtung als A/B-Schalter gegen Sigmoid testen.
4. **Schritt 4 (Config):** Erst danach Cherry-Pick, Validation-Schwellen und Registration-Guard sweeps.

Jeder Schritt muss gegen mehrere Objektklassen getestet werden, nicht nur M42:

| Objektklasse | Beispiel | Hauptgefahr | Muss geprüft werden |
|--------------|----------|-------------|---------------------|
| Heller Emissionsnebel | M42 | Low-frequency Signal wird neutralisiert | FWHM, Nebeldetail, Background-Gate |
| Große Galaxie | M31/M33 | schwache Arme werden als Hintergrund behandelt | Structure Mask, Staubbänder, Stern-FWHM |
| Kleine Galaxie | M51/M101 | wenige Sterne, starke lokale Struktur | `not_applicable` bei Stern-Gates, Detailerhalt |
| Sternhaufen | M13/M45 | Background durch Sternflügel verzerrt | Tail/Elongation, robuste Background-Maske |
| Faint nebula / IFN | schwache Staubfelder | relative RMS explodiert nahe 0 | absolute Background-Skala |
| Sternarmes Feld | High-latitude galaxy | FWHM/Tail nicht stabil | Gate-Applicability |

Jeder Schritt sollte gegen eine Classic-Referenz verglichen werden anhand:
- FWHM (Sterne, kleiner = besser)
- Background RMS (niedriger = besser)
- Visuelle Nebelstruktur-Detailschärfe (manuell)
- Sternform (Elongation, Tails)
- Galaxienarme/Staubbänder bzw. schwache diffuse Struktur

---

## 6. Methodische Anmerkung

Die v0.2.1-Invariante §1.3.8 ("Neutralisation must be validation-gated") ist korrekt im Prinzip, aber die Gate-Definition ist unvollständig. Die Invariante sollte lauten:

> Das finale AQMH-Ergebnis muss gegenüber dem uniform control mit robusten,
> anwendbaren Metriken validiert werden. Eine Metrik ist nur gate-fähig, wenn
> Control und Kandidat finite, stabile Vergleichswerte liefern. Nicht
> anwendbare Metriken werden reported, aber nicht als Fail gewertet.
> Schwellwerte dürfen asymmetrisch sein; eine nachgewiesene Verbesserung in
> primären Metriken (FWHM, Elongation) darf sekundäre Regressionen
> (Background, Tail, Seam) innerhalb definierter Grenzen tolerieren.

Dies bleibt konform mit dem Nicht-Halluzinations-Prinzip und der Determinismus-Invariante, gibt aber dem Qualitätsmodell den Raum, seinen Mehrwert tatsächlich zu liefern.

---

## 7. Korrektur nach `M42-default-tuning-5_20260719_115158`

Der Run belegt einen weiteren konkreten Logikfehler. Raw AQMH wurde wegen
`background_rms_regression = 0.512099` durch eine binär gesuchte Mischung mit
Uniform Control auf `alpha = 0.105469` abgeschwächt. Das ausgegebene Luminanzbild
enthielt damit nur rund 10,5 % des AQMH-Kandidaten. Dieser Entscheidungsweg wurde
erst nach dem guten Vergleichsrun `m42_20260703_083337` eingeführt.

Die bisherige Metrik maß `background_rms` als `1,4826 * MAD` aller Bildpixel.
Das ist die globale Bildspreizung und kein Hintergrundrauschen: diffuse Nebel,
Galaxienarme und IFN erhöhen den Wert absichtlich. Der Gate-Entscheid konnte
dadurch reales Signal als Rauschregression klassifizieren. Gleichzeitig war der
Vergleich gegen Raw AQMH einseitig; Signalverlust senkte die Metrik und erschien
daher als Verbesserung.

Die verbindliche Korrektur lautet:

1. `background_rms` wird aus robusten lokalen Pixel-Differenzen bestimmt, sodass
   langsam veränderliche astronomische Struktur nicht als Rauschen zählt.
2. Uniform Control ist ausschließlich eine diagnostische Referenz und niemals
   ein finaler Ersatz- oder Attenuierungskandidat.
3. Besteht ein AQMH-Nachbearbeitungskandidat das Gate nicht, wird die
   unveränderliche Raw-AQMH-Baseline samt ihrer Gewichtssumme ausgegeben.

Damit kann ein Gate weiterhin eine riskante Nachbearbeitung verwerfen, aber
nicht mehr die eigentliche AQMH-Rekonstruktion durch ein signalärmeres
Uniform-Control-Ergebnis ersetzen.
