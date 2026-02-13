# Diagnose: tile_compile_cpp Konsistenz zu Methodik v3.2

**Datum:** 2026-02-13  
**Soll:** `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.2.md`  
**Ist:** `tile_compile_cpp/` (Schwerpunkt: `apps/runner_main.cpp`, `src/io/config.cpp`, `src/metrics/metrics.cpp`, `src/reconstruction/reconstruction.cpp`, `src/registration/global_registration.cpp`)

---

## Executive Summary

Die C++-Implementierung ist in vielen Kernpunkten bereits nah an v3.2 (globale/lokale Gewichtung, Registrierungskaskade, robuste Metriken), hat aber **mehrere normative Abweichungen**.

**Gesamtstatus:** **teilweise konform**

### Kritische Abweichungen (hoch)

1. `synthetic.weighting=tile_weighted` ist konfigurierbar, aber im Runner nicht umgesetzt.  
2. `<50 Frames`-Regel aus v3.2 ist nicht durchgesetzt (kein harter Abbruch, kein `allow_emergency_mode`).  
3. Im Kernpfad laufen nichtlineare Post-Schritte (cosmetic correction, optionaler Output-Stretch), die in v3.2 klar als ausserhalb des Pflichtkerns betrachtet werden.

### Mittlere Abweichungen

4. Zustandsvektor fuer Clusterung weicht von v3.2-Definition ab.  
5. Tile-Rekonstruktions-Fallbacks sind nicht exakt nach v3.2 (kein explizites `D_t,c < eps_weight` + kein `fallback_used`-Artefakt).  
6. Numerische Defaults fuer Epsilons differieren von v3.2-Defaults.

---

## 1) Konsistenzmatrix (Soll/Ist)

| Bereich | Soll v3.2 | Ist C++ | Status |
|---|---|---|---|
| Registrierungskaskade | Primaer + feste Fallback-Kaskade mit NCC-Delta | Entspricht weitgehend (`register_single_frame`) | OK |
| Keine Frame-Selektion | Keine qualitaetsbasierte Frame-Selektion | Eingehalten; nur fehlende/unlesbare Daten via `frame_has_data` | OK |
| Globale Normalisierung | B aus Rohdaten, sigma/E nach Norm | Im Wesentlichen konform | OK |
| Globales Gewicht | `G=exp(k*clip(Q))`, adaptive Varianz optional | Implementiert inkl. `adaptive_weights`, `weight_exponent_scale` | OK |
| Lokale Gewichte | STAR/STRUCTURE mit Clamp->exp | Implementiert | OK |
| Tile-Rekonstruktion Fallback | explizit bei `D_t,c<eps_weight`, ungewichtetes Mittel + Marker | Nur implizite/partielle Fallbacks | TEILWEISE |
| Reduced Mode | `50<=N<=199`, `<50` separat | Schwelle via `frames_reduced_threshold`, kein `<50`-Abbruch | ABWEICHUNG |
| Synthetic weighting | `global` oder `tile_weighted` | nur global in Runner | ABWEICHUNG |
| Finaler Kernstack | linearer Stack, optionale Rejektion | plus kosmetische Korrektur + optional Stretch im Kernpfad | ABWEICHUNG |
| Notation/Kanaltrennung | kanalweise konsistent | kanalweise weitgehend eingehalten | OK |

---

## 2) Detailbefunde mit Belegen

## 2.1 Registrierung: weitgehend konform

- Kaskade + NCC-Validierung vorhanden: `register_single_frame()` in `src/registration/global_registration.cpp` (NCC-Threshold `+0.01`, feste Fallback-Reihenfolge).  
- Entspricht dem Geist von v3.2 Abschnitt Registrierung.

**Bewertung:** konform.

---

## 2.2 Global/Lokal Gewichte: konform

- Globale Gewichte inkl. adaptive Varianzgewichte und `k_global`-Aehnlichkeit (`weight_exponent_scale`) in `src/metrics/metrics.cpp`.  
- Lokale Gewichte aus STAR/STRUCTURE-Metriken und `exp(clip(q))` in `apps/runner_main.cpp`.

**Bewertung:** konform.

---

## 2.3 `synthetic.weighting` nicht bis in Runner durchgezogen (kritisch)

- Config liest und validiert `synthetic.weighting` (`global|tile_weighted`) in `src/io/config.cpp` und `include/.../configuration.hpp`.
- In `apps/runner_main.cpp` wird bei synthetischen Frames nur mit globalem Gewicht rekonstruiert:
  - `w = global_weights[fi]` in `reconstruct_subset`.
- Kein Pfad, der fuer synthetische Frames `W_{f,t,c}=G*L` nutzt.

**Soll v3.2:** Abschnitt 5.10.2 fordert optionalen `tile_weighted`-Pfad.  
**Ist:** Option existiert nur auf Config-Ebene.

**Bewertung:** **Abweichung (hoch)**.

---

## 2.4 Reduced Mode / Unterminimum nicht v3.2-konform (kritisch)

- Reduced Mode wird an `frames_reduced_threshold` gebunden (`runner_main.cpp`), praktisch "unter Schwellwert -> Clustering skip".
- Es gibt keine harte Behandlung fuer `N < 50` nach v3.2-Logik.
- `allow_emergency_mode` existiert nicht in Config/Code.

**Soll v3.2:**
- `50<=N<=199` Reduced Mode,
- `<50` kontrollierter Abbruch (oder expliziter Emergency-Mode).

**Bewertung:** **Abweichung (hoch)**.

---

## 2.5 Kernpfad enthaelt nichtlineare Post-Operationen (kritisch/mittel)

In `runner_main.cpp` nach Stacking:

1. `image::cosmetic_correction(...)` auf `recon`/`recon_R/G/B`
2. optionaler lineare Output-Stretch auf `[0..65535]` (`cfg.stacking.output_stretch`, default `true` in Config)

Das ist operational nuetzlich, sollte aber in v3.2-Semantik **nicht Teil des Pflichtkerns 1-10** sein.

**Bewertung:** **Abweichung (hoch)** fuer strenge Methodiktreue.

---

## 2.6 Tile-Rekonstruktions-Fallbacks nur teilweise deckungsgleich (mittel)

- v3.2 fordert explizit:
  - `D_t,c = sum_f W_{f,t,c}`,
  - bei `D_t,c < eps_weight`: ungewichtetes Mittel ueber alle Frames,
  - Markierung `fallback_used=true`.
- Ist-Stand:
  - gewichtetes Sigma-Clipping/Stacking in Tilephase,
  - kein explizites pro-Tile `D_t,c`-Kriterium,
  - kein `fallback_used`-Artefakt auf Tile-Ebene.

**Bewertung:** **Abweichung (mittel)**.

---

## 2.7 Cluster-Zustandsvektor weicht inhaltlich ab (mittel)

- v3.2 beschreibt als Kernvektor: `G`, `mean/var(local Q)`, `B`, `sigma`.
- Code verwendet einen 6D-Vektor mit `mean_cc_tiles`, `mean_warp_var_tiles`, `invalid_tile_fraction`; `bg/noise` werden zwar geladen, aber der kommentierte/benutzte Vektor ist anders.

**Bewertung:** **Abweichung (mittel)**.

---

## 2.8 Epsilon-Defaults differieren (niedrig/mittel)

- v3.2 empfiehlt konsistente Defaults `1e-6` (`eps_bg`, `eps_mad`, `eps_weight`, `eps_median`).
- Code nutzt gemischt z. B. `1e-10`, `1e-12` in OLA-Normalisierung/Fallback.

**Bewertung:** **Abweichung (niedrig/mittel)**, aber fuer Nachvollziehbarkeit relevant.

---

## 2.9 Hinweis auf parallele Pipeline-Implementierung

`src/pipeline/pipeline.cpp` bildet nur einen Teilfluss ab (stark verkuerzt gegen `runner_main.cpp`). Das ist nicht automatisch falsch, aber ein **Drift-Risiko**.

**Empfehlung:** eine kanonische Laufzeitpipeline definieren oder alte Pipeline klar als deprecated markieren.

---

## 3) Priorisierter Schritt-fuer-Schritt-Plan auf v3.2

## Phase A - Normative Gates zuerst (hoechste Prioritaet)

1. **Reduced-Mode-Gate korrigieren**
   - In `runner_main.cpp` explizit drei Bereiche unterscheiden:
     1) `N < 50` -> Abbruch (oder nur mit `runtime.allow_emergency_mode`),
     2) `50..199` -> Reduced Mode,
     3) `>=200` -> Full Mode.
   - `N` als klar definierte Groesse dokumentieren (empfohlen: nutzbare Frames).

2. **Config um `runtime.allow_emergency_mode` erweitern**
   - `configuration.hpp`, `config.cpp` parse/save/validate/schema.
   - Default `false`.

3. **Kern vs Post trennen**
   - `cosmetic_correction` und `output_stretch` aus Kernphase auslagern (separate Post-Phase/Output-Profil),
   - oder standardmaessig fuer methodische Runs deaktivieren.

## Phase B - Mathematische Solltreue fuer Synthese/Rekonstruktion

4. **`synthetic.weighting=tile_weighted` implementieren**
   - In SYNTHETIC_FRAMES Pfad zwei Modi:
     - `global`: bestehend,
     - `tile_weighted`: pro Cluster Tile-Rekonstruktion mit `W=G*L` + OLA.
   - Artifact ausgeben, welcher Modus aktiv war.

5. **Expliziten `D_t,c`-Fallback in Tile-Rekonstruktion einbauen**
   - pro Tile/Kanal `D_t,c` berechnen,
   - bei `D_t,c < eps_weight` auf ungewichtetes Mittel,
   - `fallback_used` je Tile im Artifact speichern.

6. **Epsilon-Konstanten zentralisieren**
   - einheitliche Defaults (`1e-6`) in Config oder zentrale Konstanten,
   - Runner/Reconstruction entsprechend angleichen.

## Phase C - Clusterung und Spezifikationsharmonisierung

7. **Zustandsvektor an v3.2 ausrichten (oder v3.2 explizit erweitern)**
   - Entweder Code auf `G, meanQ, varQ, B, sigma` umstellen,
   - oder dokumentierte Erweiterung in Methodik v3.2.1 festschreiben.

8. **Cluster-Artefakte erweitern**
   - gespeicherte Features pro Frame,
   - Standardisierungsschritte,
   - final verwendete Featureliste.

## Phase D - Konsolidierung/Regression

9. **Konfigurationsschema und Beispiele auf v3.2 aktualisieren**
   - Beispiel-YAML fuer
     - Full Mode,
     - Reduced Mode,
     - Emergency-Mode.

10. **Regressionstests einfuehren**
   - Solltests aus v3.2 Abschnitt 7.3 automatisieren:
     - Reduced-Mode-Grenzen,
     - `tile_weighted`-Pfad,
     - `fallback_used` bei low-weight Tile,
     - keine Frame-Selektion,
     - deterministische Outputs.

11. **Pipeline-Doppelstruktur aufraeumen**
   - `src/pipeline/pipeline.cpp` klar als deprecated markieren oder auf Runner-Logik anheben,
   - um zukuenftigen Spezifikationsdrift zu verhindern.

---

## 4) Empfohlene Umsetzungsreihenfolge (kurz)

1. Reduced-Mode + Emergency-Mode Gate  
2. Trennung Kern/Post (cosmetic/stretch)  
3. `synthetic.weighting=tile_weighted`  
4. expliziter Tile-Fallback + `fallback_used`  
5. Cluster-Feature-Harmonisierung  
6. Tests + Schema + Doku

---

## 5) Fazit

Der Code ist methodisch weit fortgeschritten, aber fuer "v3.2-konform" fehlen vor allem drei Punkte: **Unterminimum-Regel**, **echte `tile_weighted`-Synthese**, und **saubere Trennung von linearem Kern gegen optionale Nachbearbeitung**.

Mit den Schritten oben ist eine robuste Konvergenz auf den v3.2-Standard realistisch in einem fokussierten Refactoring-Zyklus.
