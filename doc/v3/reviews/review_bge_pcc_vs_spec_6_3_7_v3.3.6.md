# Review: BGE + PCC Implementation vs. Spec v3.3.6 (ab §6.3.7)
Datum: 2026-02-27
Scope: `tile_compile_cpp` Implementierung (BGE + PCC) gegen
`doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.6_en.md` ab §6.3.7 (inkl. §6.3.8/6.3.9, §6.4 sowie relevante Mindesttests aus §7.3).

Dieses Dokument ist ein **Code-Review + Korrekturvorschlagskatalog**. Es enthält bewusst konkrete, umsetzbare Änderungen (inkl. Dateipfade/Zeilenreferenzen) und Hinweise auf Spec-Abweichungen.

## 0) Relevante Implementationsstellen
BGE:
- `tile_compile_cpp/include/tile_compile/image/background_extraction.hpp`
- `tile_compile_cpp/src/image/background_extraction.cpp`
- `tile_compile_cpp/apps/runner_pipeline.cpp` (Pipeline-Integration + `bge.json`)

PCC:
- `tile_compile_cpp/include/tile_compile/astrometry/photometric_color_cal.hpp`
- `tile_compile_cpp/src/astrometry/photometric_color_cal.cpp`
- `tile_compile_cpp/apps/runner_pipeline.cpp` (Auto-FWHM Radii)

Spec-Referenz:
- `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.6_en.md`:
  - §6.3.7 (Autotuned BGE) inkl. Objective, Holdout-Split, Candidate Search, Diagnostics, Fallback
  - §6.3.8/§6.3.9 (Surface Model / Adaptive Grid)
  - §6.4 (PCC: lokales Background-Modell im Annulus, FWHM-adaptive Radii)
  - §7.3 (Minimum Tests, u.a. PCC stability)

## 1) Executive Summary (kurz)
**Hauptprobleme (Spec/Robustheit):**
1. **BGE Grid Spacing** kann aufgrund Clamp-Reihenfolge **gegen §6.3.9 `G >= 2*T` und `G >= G_min` verstoßen**.
2. **BGE Autotune Objective** weicht von §6.3.7.1 ab (zusätzliche Normalisierung durch Median), dadurch ist `objective` nicht Spec-`J`.
3. **BGE Autotune Candidate Families** entsprechen nicht den konservativen Spec-Vorgaben (§6.3.7.3).
4. **BGE Autotune Diagnostics** sind **nicht pro Kanal** ("last channel wins") → `bge.json` ist nicht Spec-konform (§6.3.7.4).
5. **PCC `plane` Background Model** wird nur als **Konstante** abgezogen (am Sternzentrum), nicht als Plane über die Apertur → systematischer Flux-Bias unter Gradienten.
6. **§7.3 Mindesttests** (PCC stability: det/cond/residual thresholds) werden nicht explizit geprüft.

## 2) BGE Autotune (Spec §6.3.7) – Findings & Korrekturvorschläge

### 2.1 Adaptive Grid Definition: Clamp-Reihenfolge kann Spec-Invariante brechen (Binding §6.3.9)
**Spec:** §6.3.9 fordert `G = clip(max(2*T, min(W,H)/N_g), G_min, G_max)` und zusätzlich: `G >= 2*T`.

**Ist-Stand:** `compute_grid_spacing()` klemmt **am Ende** auf `G_max`, ohne sicherzustellen, dass `G_max >= max(G_min, 2*T)`.
- Datei: `tile_compile_cpp/src/image/background_extraction.cpp:279-293`

**Konkretes Risiko:**
- Bei kleinen Bildern oder kleiner `G_max_fraction` kann `G_max` < `G_min_px` oder < `2*T` werden.
- Ergebnis: `G` wird zu fein → BGE kann hochfrequente Strukturen lernen (Overfit), instabiler werden und gegen den "low-frequency only"-Intent aus §6.3.8/§6.3.9 arbeiten.

**Korrekturvorschlag (präzise):**
1. Berechne `G_min_effective = max(2*T, G_min_px)`.
2. Berechne `G_max_px = floor(min_dim * G_max_fraction)`.
3. Falls `G_max_px < G_min_effective`, setze `G_max_px = G_min_effective` (oder wähle deterministisch `G = G_min_effective`).
4. Dann `G = clamp(max(2*T, min_dim/N_g), G_min_effective, G_max_px)`.

**Zusätzliche Empfehlung:**
- Falls `G_max_fraction` so klein ist, dass Clamp kollabiert, logge eine Warnung + schreibe die Entscheidung in Diagnostics.

---

### 2.2 Objective J entspricht nicht der Spec-Definition (Binding §6.3.7.1)
**Spec (Binding):** `J = E_cv + alpha * E_flat + beta * E_rough`.

**Ist-Stand:** Implementation normalisiert alle Terme zusätzlich durch `bmed` (Median der Stützwerte).
- Datei: `tile_compile_cpp/src/image/background_extraction.cpp:1274-1285`

**Schwachstelle:**
- Das ist nicht mehr das Spec-`J`. Dadurch sind Objective-Komponenten nicht direkt mit Spec/anderen Implementationen vergleichbar.
- Folgeeffekt: Dokumentation/Reports können "objective" falsch interpretieren.

**Korrekturvorschlag (2 Optionen, deterministisch):**
- **Option A (Spec-konform bevorzugt):** Verwende unnormalisierte Werte:
  - `J = cv_rms + alpha * flatness + beta * roughness`
  - Dokumentiere Einheiten/Skalierung in `bge.json`.
- **Option B (falls Normalisierung gewünscht):**
  - Behalte Normalisierung, aber:
    1) benenne Feld klar um (`objective_normalized`),
    2) exportiere zusätzlich Spec-`J` roh (`objective_raw`).

---

### 2.3 Conservative Candidate Families weichen von §6.3.7.3 ab (Binding When Enabled)
**Spec (konservativ):**
- `sample_quantile`: `{q0, 0.35, 0.50}`
- `structure_thresh_percentile`: `{p0, 0.85}`
- `rbf_mu_factor`: `{m0, 1.4}`

**Ist-Stand:**
- `sample_quantile`: `{q0, 0.20, 0.30}` (und `0.35` nur in `extended`) – **kein 0.50**.
- Datei: `tile_compile_cpp/src/image/background_extraction.cpp:1296-1316`

**Schwachstelle:**
- Harte Spec-Abweichung, wenn Autotune aktiviert ist.

**Korrekturvorschlag:**
- Implementiere exakt die konservative Family gemäß Spec.
- Falls median (`0.50`) aus praktischen Gründen riskant ist (Objektkontamination), dann:
  - entweder nur im `extended` (aber dann **Spec muss angepasst** werden),
  - oder füge einen Kandidaten-"Reject"-Test hinzu (z.B. zu hohe Roughness/Flatness bzw. Residual-Limit) – deterministisch.

---

### 2.4 Autotune Diagnostics nicht pro Kanal (Binding §6.3.7.4)
**Spec (Binding):** Wenn `enabled=true`, muss das gewählte Parameter-Set + Objective-Komponenten in Diagnostics enthalten sein.
Die Formulierung ist pro Kanal ("For a given channel … objective").

**Ist-Stand:**
- `BGEDiagnostics` hat nur **einen** Satz Autotune-Felder (kein pro-channel Container).
- In `apply_background_extraction()` werden diese Felder **bei jedem Kanal überschrieben**.
- Datei: `tile_compile_cpp/src/image/background_extraction.cpp:1431-1454`

**Konsequenz:**
- `bge.json` exportiert `autotune.best.*` nur für den zuletzt getunten Kanal (typisch B).
- Pipeline-JSON Writer: `tile_compile_cpp/apps/runner_pipeline.cpp:2987-3031`

**Korrekturvorschlag (robust + auditierbar):**
1. Erweiterung `BGEChannelDiagnostics` um:
   - `autotune` Sub-Struct (enabled/strategy/max_evals/evals_performed/fallback_used + best-params + objective components).
2. In `apply_background_extraction()` Autotune-Ergebnis in `ch_diag.autotune` schreiben (statt global).
3. `BGEDiagnostics` kann weiterhin globale Summen enthalten (z.B. total evals), aber "best" muss pro Channel existieren.
4. `apps/runner_pipeline.cpp` soll `bge.json` entsprechend serialisieren:
   - `channels[i].autotune.*`.

**Zusatz:**
- Derzeit wird `autotune_fallback_used` global gesetzt, sobald ein Kanal scheitert, und bleibt true, selbst wenn andere Kanäle erfolgreich sind. Das ist OK als "any-channel fallback", aber sollte explizit so benannt werden (`fallback_used_any_channel`) oder pro Channel geführt werden.

---

### 2.5 Autotune: Evals-Hardcap / deterministische Tie-Breaks (größtenteils OK, aber dokumentieren)
**Spec (Binding §6.3.7.3/6.3.7.5):**
- bounded candidate set mit hard cap `max_evals`
- deterministisches tie-break (lower roughness, then coarser effective model)
- fallback wenn keine Candidate succeeds

**Ist-Stand (positiv):**
- `max_evals` wird als harter Abbruch benutzt.
- Tie-break deterministisch: roughness, dann `rbf_mu_factor`.
- fallback: wenn kein best.success → best.cfg = base.
- Datei: `tile_compile_cpp/src/image/background_extraction.cpp:1318-1359`

**Empfehlung:**
- Stelle sicher, dass "coarser effective model" nicht nur `mu_factor` meint, sondern ggf. auch method/order (falls später erweitert).

---

### 2.6 Autotune Laufzeit: `max_evals` deckelt nur Outer-Candidates, nicht interne Fits (operativ kritisch)
**Spec-Kontext:** §6.3.7.3 verlangt einen Hardcap `max_evals` für Candidate-Evaluierungen.

**Ist-Stand / Schwachstelle:**
- Pro Outer-Candidate wird in `try_bge_candidate()` via `fit_background_surface()` ein vollständiges Hintergrundmodell auf **Image-Auflösung** berechnet.
- Bei `fit.method=rbf` werden intern zusätzlich mehrere `lambda_trials` getestet (bis zu 6), ohne dass dies in `max_evals` sichtbar wird.
- Das führt zu stark schwankender, potentiell sehr hoher Laufzeit – besonders weil Autotune pro Kanal läuft.

**Korrekturvorschläge (deterministisch, ohne Spec-Änderung):**
1. **Objective-Berechnung ohne Full-Resolution Surface:**
   - Für `E_cv` genügt das Modell an den Val-Cells (Stützstellen).
   - Für `E_flat`/`E_rough` genügt eine **coarse** Modellrepräsentation (z.B. Evaluierung auf einem Raster mit Schritt `grid_spacing/2` oder direkt auf dem Grid selbst) – deterministisch.
   - Full-res Surface (`Matrix2Df model`) erst **einmal** für den final gewählten Kandidaten erzeugen.
2. **Interne RBF-Lambda-Search begrenzen/telemetrieren:**
   - feste Anzahl Trials (bereits 6) ist ok, aber:
     - in Diagnostics exportieren (`rbf.lambda_selected`, `rbf.lambda_trials`) oder zumindest loggen,
     - optional konfigurierbar machen (`bge.fit.rbf_lambda_trials`).
3. Optional: konservativer Default `fit.method=poly` (order 2–3) für Autotune-Evaluierung, und RBF nur als finaler Fit (falls erforderlich).

---

### 2.7 BGE Subtraction implementiert zusätzliches "Pedestal" (Spec §6.3.5 / §6.3.7 Intent)
**Spec:** §6.3.5 definiert `I' = I - B` (und "No multiplicative correction permitted").

**Ist-Stand:**
- Es wird `I' = I - B + median(B)` gerechnet.
- Datei: `tile_compile_cpp/src/image/background_extraction.cpp:1516-1531`

**Schwachstelle:**
- Streng genommen nicht die Spec-Formel.
- Nebenwirkung: Kanalweise unterschiedliche Pedestals können Farb-/PCC-Bias erzeugen.

**Warum es vermutlich eingebaut wurde:**
- PCC/Downstream arbeitet stark mit der Annahme "valid pixel => > 0".
- Ohne Offset würden mehr Werte negativ/0 werden und in PCC verworfen.

**Korrekturvorschläge:**
1. Mache Pedestal explizit **konfigurierbar** und dokumentiere es:
   - z.B. `bge.subtraction_mode: strict|preserve_pedestal` (oder `bge.pedestal: none|model_median|sample_median`).
2. Schreibe den verwendeten Pedestal in Diagnostics (`channels[i].pedestal_value`, `channels[i].subtraction_mode`).
3. Default sollte Spec-konform sein (`strict`), wenn absolute Spec-Treue Priorität hat.

---

### 2.8 Fehlende "Validation Requirements" aus §6.3.6 im Code (wichtig für robuste Aktivierung)
Auch wenn Scope ab §6.3.7 ist: Autotune sollte in der Praxis immer in ein akzeptanzkriterium eingebettet sein.
Spec §6.3.6 fordert u.a.:
- Background RMS muss besser/gleich werden,
- keine künstliche Krümmung über Tile-Boundaries,
- star flux ratios stabil,
- PCC residuals besser/gleich.

**Ist-Stand:**
- Es existiert ein "flatness guard" über `spatial_background_spread()` (pre vs post) in
  `tile_compile_cpp/src/image/background_extraction.cpp:1533-1546`.
- Es gibt aber keine expliziten Tests für die übrigen Punkte.

**Korrekturvorschläge:**
- Erweiterung der BGE-Akzeptanzlogik (deterministisch) um:
  1) RMS/Spread in klar definierten Background-Regions (analog Spec)
  2) optional star-flux stability check (nutzt PCC Photometrie-Primitive oder Tile-Metriken)
  3) optionaler PCC-residual Vergleich: run PCC photometry on fixed small subset (nur wenn PCC aktiv)

## 3) PCC (Spec §6.4 + relevante §7.3 Mindesttests) – Findings & Korrekturvorschläge

### 3.1 Background Model `plane`: Plane wird nur als Konstante am Sternzentrum abgezogen (Intent-Abweichung §6.4.1)
**Spec (Binding):**
- `plane`: robust plane fit `bg(dx,dy)=a + b*dx + c*dy` über Annulus-Pixel.
- Fallback deterministisch auf `median`.

**Ist-Stand:**
- Plane wird robust gefittet (Huber), aber dann wird nur ein **single `sky_bg`** am (cx,cy) verwendet.
- Datei: `tile_compile_cpp/src/astrometry/photometric_color_cal.cpp:335-414` (insb. 378-386 und 396-413)

**Schwachstelle:**
- Unter Gradienten ist das nicht das volle Potential des Plane-Modells.
- Systematischer Flux-Bias bleibt möglich, weil innerhalb der Apertur weiterhin ein Gradient existiert.

**Korrekturvorschlag:**
- Wenn `plane` erfolgreich, ziehe pro Aperturpixel `bg(x,y)` ab:
  - `bg(x,y) = a + b*(x - mx) + c*(y - my)`
- Alternativ: ziehe den Mittelwert der Plane über der Apertur ab (weniger Rechenlast, aber besser als konstante Center-Auswertung).
- Fallback auf median bleibt unverändert deterministisch.

---

### 3.2 Spec §6.4.2 Auto-FWHM Radii: Implementierung ist korrekt (positiv)
- Datei: `tile_compile_cpp/apps/runner_pipeline.cpp:3417-3429`
- Formeln entsprechen Spec exakt.

Empfehlung:
- Die final verwendeten Radii sollten (optional) in PCC Diagnostics/Artifacts geloggt werden.

---

### 3.3 Mindesttests §7.3: PCC stability (determinant/condition/residual threshold) wird nicht explizit geprüft
**Spec §7.3 Punkt 13 (Normative Minimum Test):**
- "PCC stability: positive determinant, bounded condition number, residuals below threshold".

**Ist-Stand:**
- PCC Fit liefert eine **diagonale Matrix** mit Guardrails (`k_max` etc.), dadurch ist die Matrix in der Praxis stabil.
- Es gibt aber **keinen** expliziten Check + kein klarer threshold/abort.

**Korrekturvorschlag:**
1. Berechne und logge:
   - `det = kw_r * kw_g * kw_b`
   - `cond = max(kw)/min(kw)` (für diagonal exakt)
2. Führe deterministische Limits ein (aus Spec/Config), z.B.:
   - `det > 0` (muss)
   - `cond <= pcc.max_condition_number` (neu, Default z.B. 2.0)
   - `residual_rms <= pcc.max_residual_rms` (neu, Default z.B. 0.35)
3. Wenn verletzt: setze `result.success=false` und liefere klaren Fehlertext.

---

### 3.4 Downstream PCC Apply: Hintergrund-Neutralisierung & Highlight/Shadow Attenuation (Extension – dokumentieren)
In `apply_color_matrix()` wird:
- pro Kanal ein "Background" geschätzt,
- dann ein gemeinsames `bg_out = mean(bg_r,bg_g,bg_b)` addiert,
- zusätzlich wird die Korrektur in Shadows/Highlights attenuiert.
- Datei: `tile_compile_cpp/src/astrometry/photometric_color_cal.cpp:794-970`

**Anmerkung:**
- Spec §6.4 sagt "Permissible downstream step, applied to linear data" – das kann passen.
- Allerdings ist das nicht mehr eine reine lineare Matrixanwendung auf das gesamte Bildsignal.

**Korrektur-/Dokuvorschlag:**
- Diese Logik als explizite, dokumentierte Extension markieren (z.B. `pcc.apply_mode: strict|background_aware`), plus Diagnostics.
- `strict`: reine Matrix ohne BG-Normalisierung und ohne attenuations.

## 4) Konkrete Patch-Liste (Vorschlag, priorisiert)
Priorität A (Spec-binding / correctness):
1. **Fix `compute_grid_spacing()`**: clamp so, dass `G >= max(2*T, G_min)` garantiert ist.
2. **Autotune Candidate Families** in conservative mode Spec-konform machen.
3. **Autotune Diagnostics pro Kanal** (Struct + JSON + Report).
4. **Objective J** Spec-konform machen oder raw+normalized exportieren.

Priorität B (photometry correctness / bias reduction):
5. PCC `plane`-Background pro Aperturpixel (oder apertur-mean plane) abziehen.

Priorität C (Validation & operations):
6. Implementiere §7.3 Mindesttests für PCC stability (det/cond/residual thresholds) + export in artifacts.
7. Dokumentiere/konfigurierbar machen:
   - BGE pedestal strategy
   - PCC apply-mode (strict vs background-aware)

## 5) Test-/Verifikationsvorschläge (deterministisch)
- Unit/Regression für Grid-Spacing:
  - kleine Bilder + extreme `G_max_fraction` → assert `G >= 2*T`.
- Autotune determinism:
  - identische Inputs → identisches best-Param-Set und identische Tie-Break Ergebnisse.
- PCC plane photometry:
  - synthetisches Bild mit linearem Gradient + künstlichem Stern → plane-Modus sollte Flux näher am ground truth liegen als constant-center.
- PCC stability:
  - erzwinge extreme d_rg/d_bg → prüfe, dass Guardrails + cond-limit greift.

## 6) Offene Fragen / Klärungspunkte
1. Soll Spec-Treue ("strict") Default sein oder operative Robustheit (pedestal/background-aware) Default?
2. Sollen Autotune und BGE per Channel unabhängig sein (aktuell ja), oder sollen Parameter kanalübergreifend konsistent gehalten werden (einheitliche Parameter)?
3. Wie sollen negative Pixelwerte downstream behandelt werden, falls BGE strict subtraction genutzt wird (PCC currently skips `<=0`)?
