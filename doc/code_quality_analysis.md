# Code-Quality-Analyse: `tile_compile_cpp`

Stand: 2026-04-23
Scope: `apps/` und `src/` (≈ 35 kLOC C++)
Methode: Grep-basierte statische Analyse + manuelle Prüfung der gefundenen Stellen.

Diese Analyse listet **toten Code**, **doppelten Code** und **Logik-Defekte**, die beim systematischen Durchgehen der Runner-/Pipeline-Pfade gefunden wurden. Die Funde sind nach Schweregrad sortiert und zeigen Fundstellen mit Zeilennummern, sodass sie punktuell behoben werden können.

## Zusammenfassung

| Kategorie | Schwer | Mittel | Kosmetisch |
|-----------|-------:|-------:|-----------:|
| Duplicate | 3 | 2 | 0 |
| Dead code | 2 | 6 | 3 |
| Logic bugs | 2 | 3 | 1 |

---

## 1 · Duplicate Code (Copy-Paste zwischen Übersetzungseinheiten)

### 1.1 `invert_affine_warp` — zwei identische Definitionen **[schwer]**
- `apps/runner_phase_registration.cpp:329-348`
- `apps/runner_resume.cpp:115-135`

Beide Definitionen sind **byte-identisch** (nur die Namespace-Wrapping unterscheidet sich). In `runner_resume.cpp` steht sie in einem anonymen Namespace, daher keine ODR-Verletzung zur Linkzeit. Trotzdem doppelte Wartungslast: Bugfixes müssen an beiden Stellen nachgezogen werden.

**Fix:** In `runner_shared.hpp/.cpp` als `tile_compile::runner::invert_affine_warp` extrahieren und beide Call-Sites umstellen. Alternativ gibt es in `include/tile_compile/core/acceleration.hpp` bzw. `src/registration/registration.cpp` bereits Inversions-Helper — prüfen ob eine davon wiederverwendet werden kann.

### 1.2 `compute_warps_bounds` + `WarpBounds`-Struct — zwei identische Definitionen **[schwer]**
- `apps/runner_phase_registration.cpp:350-405` und `apps/runner_phase_registration.cpp:204-212` (Struct)
- `apps/runner_resume.cpp:137-190` und `apps/runner_resume.cpp:105-113` (Struct)

Ebenfalls **byte-identisch**. Der gleiche 4-Ecken-Projektionscode existiert doppelt. Selbst die Kommentare sind abweichend kopiert.

**Fix:** In `runner_shared.hpp` als `struct WarpBounds` + `WarpBounds compute_warps_bounds(...)` definieren, beide CPPs importieren.

### 1.3 `compute_ncc_local_masked` / `warp_valid_mask_local` dupliziert library **[schwer]**
- `apps/runner_phase_registration.cpp:226, 255, 270`

Re-Implementierung von Bibliotheksfunktionen, die bereits als Teil des öffentlichen Headers `include/tile_compile/registration/global_registration.hpp` exportiert werden:
- `tile_compile::registration::compute_ncc_masked` (in `src/registration/global_registration.cpp:1319`)
- `tile_compile::registration::warp_valid_mask` (in `src/registration/global_registration.cpp:1304`)

Die Runner-Lokalversionen sind algorithmisch äquivalent, nur leicht in Signatur und Präzision angepasst (verwenden `double` als Akkumulator). Zwei unterschiedliche Implementierungen derselben Funktion bedeuten, dass Fixes wie der kürzliche Clamp-vor-NCC-Blur (siehe Commit-Historie `global_registration.cpp`) nicht automatisch auch in der Runner-Variante ankommen.

**Fix:** In `runner_phase_registration.cpp` die drei Helfer (`compute_ncc_local`, `compute_ncc_local_masked`, `warp_valid_mask_local`) durch die Library-Aufrufe ersetzen. Die unmasked-Variante `compute_ncc_local` ist ohnehin toter Code (siehe 2.1).

### 1.4 Mehrfache Kanvas-Bounding-Box-Berechnung für Registrierungsartefakte **[mittel]**
- `apps/runner_phase_registration.cpp:2959-2974` (frisch refactored)
- `apps/runner_resume.cpp:229` (Aufruf aus `load_registration_canvas_offsets`)

Beide berechnen die gleichen Offsets aus dem JSON-Artefakt bzw. den In-Memory-Warps. Der Resume-Pfad liest das Artefakt neu und reproduziert `offset_x/y`, während der reguläre Pipeline-Pfad die Werte aus dem Vollregister-Lauf beibehält. Der Resume-Pfad berücksichtigt aber **nicht** das neue Kriterium „Unresolved-Frames von der BBox ausschließen" (Fix aus 2026-04-23).

**Fix:** Resume-Logik ebenfalls auf die neue Semantik anheben (wenn `source == "unresolved"` oder `cc == 0`, Warp ignorieren). Vorschlag: gemeinsamer Helfer `filter_resolved_warps(const core::json &artifact)`.

### 1.5 Mehrere identische Zähler-Blöcke in `reject_outliers` **[mittel]**
- `apps/runner_phase_registration.cpp:2765-2777` schreibt `reg_reject_*_outliers` in `global_reg_extra`.
- Vergleichbare Struktur in `apps/runner_phase_registration.cpp:2010-2028` für die Rescue-Counter.

Beide Blöcke folgen dem gleichen „declare counter, increment, dump-to-JSON"-Muster, das sich mit einem kleinen `Counter`-Helfer (z.B. `struct NamedCounter { const char *key; int *value; };`) deutlich straffen ließe. Kein funktionales Problem, aber 100+ Zeilen redundanter Boilerplate.

---

## 2 · Dead Code

### 2.1 `compute_ncc_local` — definiert, nie aufgerufen **[mittel]**
`apps/runner_phase_registration.cpp:226-253`

Die unmasked Version wird nirgendwo referenziert (verifiziert durch `grep "compute_ncc_local\b"`). Es gibt nur Aufrufe des _masked_-Zwillings. Löschbar.

### 2.2 `ScalarPolyFit::rms_residual` — immer gesetzt, nie gelesen **[mittel]**
- Feld deklariert: `apps/runner_phase_registration.cpp:88`
- Wert befüllt: `apps/runner_phase_registration.cpp:199`
- Reads: **keine**

Der teure `sum_sq`-Loop wird jedes Mal in `fit_weighted_poly` (≤ 40 Aufrufe pro Session) ausgeführt, obwohl das Ergebnis niemals betrachtet wird. Entweder Feld + Berechnung löschen oder in Logging/Diagnose einbinden.

### 2.3 `WarpPredictionCandidate::{res_ang_deg, res_tx, res_ty, support, span}` — nur Write-Through **[mittel]**
- `apps/runner_phase_registration.cpp:98-102` (Definition)
- Gesetzt in `build_local_candidate` (2588–2592), `build_bridge_candidate` (2620–2622, 2638–2640, 2656–2658, 2674–2676)
- In `chosen` durchgereicht (2763–2767), aber **der `chosen` wird danach nur für `chosen.ang`, `chosen.tx`, `chosen.ty` und `chosen.ok`** ausgelesen.

Die `res_*`-Felder fließen indirekt über das Score-Feld in die Wahl zwischen Kandidaten ein; `support`/`span`/`res_*` werden **auf `chosen` kopiert, aber dort nie wieder gelesen**. Lines 2763–2767 sind tote Store-Operationen.

**Fix:** Entweder das Diagnostik-Logging um `support`/`span`/`res_*` erweitern (siehe das bereits vorhandene `[REG-MODEL]`-Log auf 2802ff), oder die fünf Zeilen streichen.

### 2.4 `use_shared_rgb_sigma_clip` — toter Boolean **[leicht]**
`apps/runner_pipeline.cpp:2330`
```cpp
const bool use_shared_rgb_sigma_clip = false;
```
Wird nirgendwo gelesen. Überbleibsel einer früheren A/B-Implementierung.

### 2.5 `normalize_tile_for_ola` — No-Op-Lambda **[leicht]**
`apps/runner_pipeline.cpp:3825-3831` definiert ein Lambda, das ausdrücklich beide Parameter verwirft (`(void)t_img; (void)tmp;`) und nur einen Kommentar enthält. Das Lambda wird auf `:3914` einmal aufgerufen — macht dort nichts.

Entweder wirklich entfernen (samt Aufrufstelle + `norm_tmp`-Vector) oder durch Implementierung ersetzen, falls die OLA-Normalisierung noch gebraucht wird.

### 2.6 `RegistrationProvenance::model_global_poly` — fast unreachable **[leicht]**
`apps/runner_phase_registration.cpp:115`

Im Prediction-Block (2744–2778) ist `chosen_provenance = model_global_poly` der Default, wird aber in praktisch allen Fällen vom If-else überschrieben:
- `best_local.ok && bridge_candidate.ok` → `model_blended`
- `best_local.ok` → `model_local_poly`
- `bridge_candidate.ok` → `model_interpolated`

Nur wenn `outside_valid_span == true` **UND** `bridge_candidate.ok == false` bleibt `model_global_poly` aktiv. `bridge_candidate.ok` ist bei `nv >= 2` aber **immer** erfüllt (siehe die 4 geschachtelten Fälle in `build_bridge_candidate`, 2602–2680). Damit ist `model_global_poly` nur erreichbar, wenn `nv == 1`, was aber den äußeren Branch `nv >= 3` gar nicht nimmt.

**Fix:** Entweder den Defaultwert entfernen und ein `chosen.ok = false` am Ende absichern, oder den Enum-Wert als explizites Signal stehen lassen und Kommentar „keeper, not currently reached — used as fallback marker".

### 2.7 `check_params.py`, `check_params_safe.py`, `fast_check.py` **[leicht]**
`tile_compile_cpp/check_params.py`, `check_params_safe.py`, `fast_check.py`

Keine Referenzen in CMake, Doku, Shell-Skripten oder anderen Python-Dateien. Datumsstempel lassen auf Ad-hoc-Tooling schließen. Kandidat zum Löschen oder Verschieben nach `scripts/` mit kurzer README.

### 2.8 Ungenutzte Parameter per `(void)param` **[kosmetisch]**
Stellen mit expliziten Parameter-Casts (= „API-Breaking-Protection"):
- `apps/runner_phase_local_metrics.cpp:42-43` (`tile_offset_x/y`)
- `apps/runner_phase_metrics.cpp:146` (`cache_naxis`)
- `apps/runner_phase_registration.cpp:1288` (`score`)
- `apps/runner_pipeline.cpp:1082` (`frame_naxis`)
- `apps/runner_pipeline.cpp:3827-3828` (No-Op-Lambda, siehe 2.5)
- `src/core/acceleration.cpp:56, 2417-2420, 2482-2483` (GPU-Backend-Stubs)

Die meisten sind gerechtfertigt (Stubs, Future-API). Der Block in `acceleration.cpp:2417-2483` deutet allerdings darauf hin, dass die CUDA-Pfade aktuell No-Ops sind — als eigenes TODO erfassen.

### 2.9 `RegistrationProvenance::sequential_rescue` **[kosmetisch]**
Wird als Enum-Value und im `registration_provenance_name`-Switch geführt, aber nur an zwei Stellen in `can_anchor_blind_chain` (660–670) UND an einer einzigen Set-Stelle (`RegistrationProvenance::sequential_rescue` auf Zeile 1581). Verifizieren, ob der Rescue-Pfad noch gebraucht wird oder das Ergebnis gleich als `sequential_refined` eingetragen werden kann.

---

## 3 · Logikfehler / Verdachtsfälle

### 3.1 `compute_warps_bounds` ignoriert individuelle Warp-Fehler still **[schwer]**
`apps/runner_phase_registration.cpp:369-391` (und duplizierter Copy in `runner_resume.cpp`)

```cpp
for (const auto &w : warps) {
  WarpMatrix fwd;
  if (!invert_affine_warp(w, fwd)) {
    continue;   // skip silently
  }
  ...
}
```

Wenn **alle** Warps invers-singulär sind, fällt die Funktion auf die Default-Box `[0, width] × [0, height]` zurück — ohne Warnung. Bei nur einem invertierbaren Warp wird das Canvas allein an diesem Frame ausgerichtet. Beide Szenarien treten in Praxis bei Bug-Runs auf (cc=0-Cluster ergibt Identity, alle invertierbar, aber wertlos).

**Fix:** `bool any_invalid` tracken und via `emitter.warning` loggen, wenn > 5% der Warps übersprungen wurden. Nach dem Fix aus 2026-04-23 (Unresolved-Frames aus `bbox_warps` filtern) ist das weniger kritisch, aber der stille Fallback bleibt gefährlich.

### 3.2 `rejected_mask` vs. `reg_provenance` gehen auseinander **[schwer]**
`apps/runner_phase_registration.cpp`

Der Outlier-Reject-Block (2386-2412) setzt für abgelehnte Frames:
```cpp
reg_rejected_mask[fi] = 1;
set_registration_state(fi, identity_warp(), 0.0f, false, -1,
                       RegistrationProvenance::unresolved);
```

Der Polynomial-Prediction-Block verwendet aber als Kriterium:
```cpp
const bool is_rejected = reg_rejected_mask[fi] != 0;
const bool is_missing_registration = global_frame_cc[fi] <= 0.0f;
if (!is_rejected && !is_missing_registration) continue;
```

Beide Bedingungen sind nach dem Reject redundant (cc wurde gerade auf 0 gesetzt). Kein Bug, aber die Verknüpfung verschleiert, dass `reg_provenance == unresolved` die eigentliche Wahrheit ist.

**Real-Bug-Risiko:** In `runner_resume.cpp` wird beim Laden aus dem Artefakt **weder** `reg_rejected_mask` **noch** `reg_provenance` rekonstruiert — nur die Warp-Matrix + `cc`. Der Resume-Pfad kann also Frames, die im Original-Lauf als `unresolved` markiert wurden, mit ihrer Identity-Warp + cc=0 wieder in die Pipeline einspeisen (cc=0 → `frame_has_data` bleibt ggf. korrekt aus, aber das ist vom Prewarp-Verhalten abhängig, nicht vom Resume-Zustand).

**Fix:** In `runner_resume.cpp` auch die `source`-Spalte aus `global_registration.json` laden und den Unresolved-Skip-Pfad im Prewarp aktivieren. Heute wird `source` von Resume ignoriert (siehe `runner_resume.cpp:211-225`).

### 3.3 `can_anchor_blind_chain` behandelt `sequential_rescue` asymmetrisch **[mittel]**
`apps/runner_phase_registration.cpp:667-670`

```cpp
case RegistrationProvenance::sequential_rescue:
  return (reg_chain_depth[fi] >= 0 &&
          reg_chain_depth[fi] < kMaxBlindChainAnchorDepth) ||
         global_frame_cc[fi] >= kBlindChainStrongAnchorCc;
```

Die Bedingung erlaubt einen Rescue-Frame als Anchor, wenn `chain_depth == 0` ist. `chain_depth` wird für Rescues aber häufig als 0 oder 1 gesetzt (siehe Rescue-Lambdas auf 1613ff / 1753ff / 1882ff), wo `set_registration_state(..., 0 oder 1, ...)` verwendet wird. Damit ist quasi jeder `sequential_rescue` ein zulässiger Anchor — das war vermutlich nicht beabsichtigt (sonst hätte der Autor `direct_global` direkt als gleichwertig markiert).

**Check:** Soll `sequential_rescue` nur bei **starken** Rescues (cc ≥ 0.7) Anchor sein? Die aktuelle OR-Bedingung wirkt geschrieben als AND (d.h. tief-verschachtelte Kette Rescue-Chains akzeptieren).

### 3.4 Gleitkomma-Vergleich ohne Epsilon **[mittel]**
`apps/runner_phase_registration.cpp:1489-1492`:
```cpp
const bool near_identity = (shift_total < rcfg.star_inlier_tol_px) &&
                           (angle_abs   < 0.1f) &&
                           (ncc >= ncc_identity_overlap - 0.02f) &&
                           (out.ncc_identity > 0.7f);
```

`ncc >= ncc_identity_overlap - 0.02f` ist absolut — aber `ncc_identity_overlap` kann theoretisch in einer zahlenmäßig instabilen Zone (= sehr kleine Überlappung) liegen. Bei `overlap_pixels <= 16` (auf Zeile 1493 geprüft) würde `ncc_identity_overlap` zurück auf 0 gesetzt → `ncc >= -0.02` trivially true, was einen bogus Warp durchlassen könnte. Die `overlap_pixels`-Guard ist vorhanden, aber die Reihenfolge (erst `near_identity` berechnen, dann Guard anwenden) lässt sich eleganter refactoren.

Kein Akutbug, Hinweis zur zukünftigen Überarbeitung.

### 3.5 Doppelte Auswertung von `robust_median` bei gleicher Eingabe **[mittel]**
`apps/runner_phase_registration.cpp:2265-2269`

`normal_shift_median` und `half_turn_shift_median` werden für zwei disjunkte Vectorpools berechnet. Das ist korrekt. Aber: Die `nth_element`-Calls mutieren ihre Eingabe (copy-by-value), und beide Vectoren werden vor der Berechnung nicht weitergenutzt — der Copy ist unnötig. Kleiner Performance-Hinweis, kein Bug.

### 3.6 Meldung nur bei `nv >= 3`, nicht bei `nv >= 1 && nv < 3` **[leicht]**
`apps/runner_phase_registration.cpp:2815-2827` (nearest-copy-Fallback-Branch) gibt einen Log aus, aber das **out-of-bounds-Counter** `reg_model_predicted_out_of_bounds` wird nur im `nv >= 3`-Zweig gemeldet. Wenn `nv == 2`, fällt der Code durch in den `else if (nv >= 1)`-Block → out-of-bounds-Schutz fehlt dort komplett.

**Fix:** Für den `nv >= 1`-Pfad analogen Hull-Check anwenden (Hull ist dann `[tx_of_sole_valid, tx_of_sole_valid]`, Margin ≥ 50 px). In der Praxis fast nie relevant (wenn `nv == 1 oder 2` sind meist ohnehin alle Frames schlecht), aber sauberer.

---

## 4 · Zusätzliche Beobachtungen (keine Bugs, aber Technical Debt)

### 4.1 `runner_phase_registration.cpp` wuchs auf 3251 Zeilen, `runner_pipeline.cpp` auf 5640 Zeilen
Beide Dateien tragen Dutzende Unterphasen (Blind-Chain, Sequential-Rescue, Temporal-Rescue, Astrometric-Rescue, Poly-Prediction, etc.). Eine Aufspaltung in `src/registration/blind_chain.cpp` / `sequential_rescue.cpp` / `warp_prediction.cpp` würde Navigations- und Reviewkosten deutlich reduzieren.

### 4.2 `global_reg_extra`-JSON: 40+ Keys, einige überlappend
- `reg_source_counts` (dict) dupliziert die Information aus `source` (Array).
- `reg_reject_*_outliers` (5 Einzelzähler) + `reg_rejected_frames` (Liste mit reasons).
- `reg_model_predicted_{rejected,missing,out_of_bounds,...}` (neuerdings 6 Varianten).

Die Downstream-Reporter (siehe `scripts/generate_report.py`) konsumieren nur einen Bruchteil. Empfehlung: Felder, die nirgendwo gelesen werden, kennzeichnen (Suffix `_diag`) oder hinter einem `cfg.diagnostics.verbose`-Flag gaten.

### 4.3 `tile_compile::runner::*` vs. `namespace { ... }` Inkonsistenz
`runner_phase_registration.cpp` ist komplett in `namespace tile_compile::runner { namespace { ... } }` eingebettet. `runner_resume.cpp` mischt Top-Level + anonymer Namespace, was zu subtilen Name-Lookup-Unterschieden führt (z.B. erklärt das die byte-identische Dupe in 1.1/1.2).

### 4.4 Eigen-Typ-Mixing (Matrix2Df vs. MatrixXf)
Mehrere Stellen in `runner_phase_registration.cpp` (2478, 2501-2507) verwenden parallel `Eigen::MatrixXf` und `Matrix2Df` (typedef aus `core/types.hpp`). Kein Bug, aber Code-Review-Hürde.

---

## 5 · Empfehlungsreihenfolge (Priorität)

1. **[schwer]** 1.1 + 1.2 + 1.3 zusammen auflösen: Helper-Modul `runner_shared_geometry.{hpp,cpp}` anlegen.
2. **[schwer]** 3.2 fixen: Resume-Pfad muss `source`/`cc=0`-Semantik respektieren (sonst wird der heutige Registrierungs-Fix bei Resume-Runs unterlaufen).
3. **[schwer]** 3.1 fixen: Warn-Event bei `compute_warps_bounds`-Fallbacks.
4. **[mittel]** 2.3 aufräumen (oder als Diagnose-Logging sinnvoll machen).
5. **[mittel]** 2.2 + 2.4 + 2.5 streichen (triviale Löschungen, CI-Grün sollte erhalten bleiben).
6. **[leicht]** 2.7 — Python-Skripte nach `scripts/` verschieben oder löschen.
7. **[leicht]** Die ungenutzten `res_*`/`support`/`span`-Writes von 2.3 entweder loggen oder entfernen.
8. **[kosmetisch]** `global_reg_extra` ausdünnen, `RegistrationProvenance` dokumentieren.

---

## Anhang A · Reproduktion der Funde

Alle Greps wurden aus `tile_compile_cpp/` ausgeführt; Ergebnisse in `/tmp/out_*.txt` gecacht.

```
grep -rn "^WarpBounds compute_warps_bounds" apps/ src/ include/
grep -rn "compute_ncc_local\b\|compute_ncc_local_masked\|warp_valid_mask_local" apps/ src/ include/
grep -rn "rms_residual\|res_ang_deg\|res_tx\|res_ty" apps/runner_phase_registration.cpp
grep -rn "use_shared_rgb_sigma_clip\|normalize_tile_for_ola" apps/
grep -rn "model_global_poly\|sequential_rescue" apps/runner_phase_registration.cpp
```

Alle Fundstellen wurden in diesem Dokument mit Datei und Zeile dokumentiert.
