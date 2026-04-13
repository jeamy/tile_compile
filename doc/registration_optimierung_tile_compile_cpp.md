# Analyse & Optimierung der Frame-Registrierung in `tile_compile_cpp`
## Ziel
Diese Analyse fokussiert die Registrierungs-Pipeline in `tile_compile_cpp` mit dem Ziel, **alle Frames möglichst korrekt zu registrieren und im Stack zu verwenden**, statt sie durch zu harte Gates oder instabile Schätzungen zu verlieren.

## Relevante Codebereiche
- Kern-Registrierung:
  - `tile_compile_cpp/src/registration/global_registration.cpp`
  - `tile_compile_cpp/src/registration/registration.cpp`
- Runner-Integration, Rescue, Outlier-Filter, Modell-Fallback:
  - `tile_compile_cpp/apps/runner_phase_registration.cpp`
- Parameter-Defaults und Konfig:
  - `tile_compile_cpp/include/tile_compile/config/configuration.hpp`
  - `tile_compile_cpp/src/io/config.cpp`
  - `tile_compile_cpp/tile_compile.schema.yaml`

## Aktueller Ablauf (Kurzfassung)
1. Direkte Registrierung jedes Frames auf Referenzframe (parallel) über `register_single_frame(...)`.
2. Danach mehrstufige Rescue-Pfade:
   - Sequential refine/rescue (Nachbar-Kette),
   - Temporal rescue (Anker),
   - Seeded ECC rescue,
   - Local-reference rescue.
3. Outlier-Rejection (Reflection, Scale, CC, Shift).
4. Für verworfene oder nicht registrierte Frames: polynomiales Warp-Modell (oder nearest-copy), dadurch behalten Frames einen Warp und fallen nicht komplett aus.
5. Prewarp + Overlap-Maske für nachfolgende Tile-Rekonstruktion.

## Was bereits gut gelöst ist
- Mehrstufige Fallback-/Rescue-Kaskade reduziert harte Frame-Verluste deutlich.
- `auto_engine` schaltet bei Rotations-/ECC-Problemen sinnvoll auf sternbasiertes Matching.
- Für problematische Blöcke existiert ein physikalisch plausibler Modell-Fallback statt harter Frame-Exklusion.
- Validierung mit maskierter NCC verhindert viele schlechte Warps.

## Hauptursachen, warum Frames trotzdem „falsch“ oder suboptimal genutzt werden
## 1) Inkonsistente Akzeptanz-Schwellen zwischen Pfaden
Die Pipeline verwendet je nach Pfad unterschiedliche NCC-Gates (`+0.003`, `+0.005`, `+0.01`, teils sogar negatives Improvement in Nachbar-Registrierung). Das erhöht die Chance, dass Frames je nach Reihenfolge/Pfad unterschiedlich behandelt werden.

**Effekt:** Instabile Entscheidungen, besonders bei dünnen Sternfeldern, Wolken oder starker Gradientenstruktur.

## 2) Globales Outlier-Filtering kann bei langen Alt/Az-Sequenzen übergreifen
Obwohl die Filter bereits moderat sind, können Shift/CC-Filter in Randbereichen einer Session valide Frames als Outlier markieren, die physikalisch erwartbar sind (große Rotation/geringer Overlap).

**Effekt:** Unnötiges Umschalten auf Modell-Warp; reale Information wird weniger genutzt.

## 3) Modell-Fallback setzt pauschal sehr kleines positives CC (`1e-4`)
Das stellt „Nutzbarkeit“ sicher, differenziert aber nicht zwischen guter und unsicherer Modell-Prognose.

**Effekt:** Downstream-Gewichtung kann zu grob sein; sehr unsichere Modell-Warps werden ähnlich behandelt wie brauchbare Schätzungen.

## 4) Sternselektion ist robust, aber nicht adaptiv genug auf Szenenebene
`detect_stars_simple(...)` hat adaptive Schwelle (3.5σ → 2.5σ), dennoch fehlt ein explizites „Szenenprofil“ für sehr diffuse Felder, Sternarmut oder dominante helle Objekte, bevor die eigentliche Kaskade startet.

**Effekt:** Frühe Verfahren können unnötig scheitern; Rescue greift erst später.

## 5) Lokale Referenzbildung nutzt nur positiv-CC-Support
Bei großen Ausfallblöcken wird die lokale Referenz ggf. aus zu wenigen oder einseitig verteilten Frames gebaut.

**Effekt:** Rescue in problematischen Sequenzteilen wird fragiler.

## 6) Blind-Chain-Rettung: starre Tiefe und potenzielle Driftakkumulation
In der Runner-Logik sind `kMaxBlindChainAnchorDepth = 12` und permissive Blind-Chain-Akzeptanzpfade harte Steuerpunkte.

**Effekt:** Lange Ausfallblöcke können an der Tiefe scheitern; bei langen Ketten kann sich geometrische Drift aufsummieren.

## 7) Polynomial-Fallback derzeit Grad-2 fix
Der Modell-Fallback arbeitet global mit quadratischem Fit (plus lokalem Fit/Interpolation).

**Effekt:** Für sehr lange Sessions mit stärker nichtlinearer Bewegung/Felddrehung kann Grad 2 zu unflexibel sein.

## 8) Stern-Detektion: sinnvoller zusätzlicher Rescue bei extremen Hintergründen
Die adaptive Schwelle (3.5σ → 2.5σ) ist gut, aber bei Mondlicht/starken Gradienten kann eine zusätzliche lokale Hintergrundunterdrückung die Stabilität weiter erhöhen.

**Effekt:** Weniger `too_few_stars` in schwierigen Sequenzen.

## Konkrete Optimierungen (priorisiert)
## P0 – Hoher Nutzen, geringes Risiko
### A) Akzeptanz-Gates vereinheitlichen (zentrale Policy)
- Eine gemeinsame Gate-Funktion für alle Pfade einführen, z. B. abhängig von:
  - Overlap-Pixel,
  - erwarteter Bewegungsgröße,
  - Provenance (direct/sequential/model),
  - Referenztyp (global vs. neighbor).
- Statt fixer Offsets (`0.003/0.005/0.01`) ein adaptiver Margin:
  - kleiner Margin bei kleinem Shift + hohem Overlap,
  - größerer Margin bei großem Shift + kleinem Overlap.

**Erwartung:** Weniger widersprüchliche Entscheidungen, stabilere Frame-Annahme.

### B) Outlier-Filter kontextsensitiver machen
- `reject_shift_median_multiplier` adaptiv an Sequenzspanne koppeln (z. B. größer an Session-Rändern oder bei starker Rotationsdrift).
- `reject_cc_min_abs` für chain-validierte Frames bereits geschützt; zusätzlich Schutz für Frames mit konsistentem Nachbar-Warp (auch wenn CC niedrig).

**Erwartung:** Weniger False-Rejects in realen Alt/Az-Datensätzen.

### C) Modell-Fallback mit Qualitätsklassen statt konstantem `1e-4`
- Für modellierte Warps Qualitätsstufen speichern (z. B. aus Residuen/Support/Span):
  - high-confidence model,
  - medium,
  - low.
- Daraus differenzierte Start-CC oder separaten Confidence-Score für Downstream-Gewichtung ableiten.

**Erwartung:** Modellierte Frames bleiben nutzbar, aber mit realistischerem Einfluss.

## P1 – Mittleres Risiko, hoher Nutzen bei schwierigen Datensätzen
### D) Szenenklassifikation vor der Kaskade
- Frühphase pro Sequenz: Stern-Dichte, Gradient-Stärke, SNR-Hinweise.
- Darauf basierend automatisch:
  - `engine`,
  - `transform_model`,
  - `star_topk`,
  - `star_inlier_tol_px`,
  - ggf. strengere/lockerere Rescue-Gates.

**Erwartung:** Weniger frühe Fehlversuche, schnellere Konvergenz.

### E) Lokale Referenz robuster bauen
- Support-Auswahl nicht nur nach zeitlicher Nähe und CC, sondern auch:
  - geometrische Konsistenz (Warp-Residuum zum lokalen Trend),
  - beidseitige zeitliche Abdeckung erzwingen.
- Mindestanforderung für Referenzqualität (z. B. effektiver Support-Score statt nur Pixelanzahl).

**Erwartung:** Stabilere Rettung in langen Problemblöcken.

### F) Blind-Chain steuerbar und driftrobust machen
- `max_blind_chain_depth` und `blind_chain_strong_anchor_cc` aus Hardcode in `RegistrationConfig` verschieben.
- Optional Drift-Check gegen lokalen Modelltrend für tiefere Chain-Stufen:
  - Bei starker Abweichung von Trend (`tx/ty/angle`) Frame nicht als neuer Anker verwenden.

**Erwartung:** Bessere Wiederherstellung bei langen Wolkenblöcken ohne unkontrolliertes Wegdriften.

### G) Adaptiven Modellgrad für Warp-Vorhersage prüfen
- Statt global immer Grad 2: Grad abhängig von Anzahl valider Stützstellen und Residuen wählen (z. B. 2..4).
- Sicherheitsgrenze: bei Overfit-Anzeichen auf niedrigeren Grad zurückfallen.

**Erwartung:** Bessere Vorhersage auf langen, nichtlinearen Sequenzen.

### H) Stern-Detektion bei schwierigen Hintergründen erweitern
- Vor Sternsuche optionale lokale Hintergrundsubtraktion (z. B. Median- oder großräumiger Lowpass-Background).
- Zusätzliche Notfallstufe unter 2.5σ nur mit strengerem Hot-Pixel-/Artefaktfilter.

**Erwartung:** Höhere Trefferquote bei Mondlicht, Nebelgradienten und schwachem Kontrast.

## P2 – Optional, aber wertvoll für Debugbarkeit/Feintuning
### I) Mehr Telemetrie in `global_registration.json`
- Pro Frame zusätzlich:
  - `acceptance_margin_used`,
  - `overlap_px`,
  - `ncc_identity_overlap`,
  - `ncc_warped`,
  - `provenance_confidence`,
  - `rescue_stage_attempted/succeeded`.

**Erwartung:** Schnelleres, datengetriebenes Tuning statt Trial-and-Error.

### J) Blind-Chain-Frühwarnung in Logs
- Warnung, wenn `reg_max_chain_depth` nahe der maximal erlaubten Tiefe liegt.

**Erwartung:** Frühzeitiges Erkennen von Sequenzen, die tieferes Rescue-Tuning brauchen.

## Empfohlene Konfigurations-Tuning-Startpunkte (ohne Codeänderung)
Für schwierige Alt/Az-Sequenzen:
- `registration.transform_model: affine`
- `registration.star_topk: 180..260` (bei dichten Sternfeldern)
- `registration.star_inlier_tol_px: 4.0..6.0`
- `registration.reject_cc_min_abs: 0.20..0.25` (statt zu streng)
- `registration.reject_shift_median_multiplier: 5.0..7.0`
- `registration.auto_engine: true`

Für EQ/gut getrackte Sequenzen:
- `transform_model: similarity`
- `star_inlier_tol_px` eher kleiner (`2.5..4.0`)
- `reject_shift_px_min` niedriger als Alt/Az.

## Konkreter Umsetzungsplan (kurz)
1. Gemeinsame Gate-Policy-Funktion extrahieren und in allen Rescue-/Direct-Pfaden verwenden.
2. Outlier-Filter um sequenzkontextabhängige Limits erweitern.
3. Blind-Chain-Parameter konfigurierbar machen und optional Drift-Check ergänzen.
4. Modell-Fallback um adaptiven Grad + Confidence-Score ergänzen.
5. Stern-Detektion um lokalen Background-Rescue erweitern.
6. Telemetrie/Logs erweitern und anhand realer Runs Schwellwerte feinjustieren.

## Abgleich mit `doc/registration_pipeline.md` (Diskrepanzen & Übernahmen)
### Inhaltlich übereinstimmend
- Inkonsistente Gate-Logik über mehrere Pfade.
- Outlier-Filter braucht mehr Sequenzkontext (v. a. Alt/Az-Ränder).
- Modellierte Frames sollten nicht alle mit identischem Mini-CC behandelt werden.
- Szenenadaptive Parametrisierung vor/innerhalb der Kaskade ist sinnvoll.
- Lokale Referenzbildung muss robuster gegen einseitige/instabile Support-Sets werden.
- Mehr Telemetrie ist zentral für reproduzierbares Tuning.

### Bisher in dieser Datei nicht explizit erfasst, jetzt ergänzt
- Blind-Chain: harte Tiefenlimitierung als konfigurierbarer Parameter.
- Blind-Chain: optionale Driftkontrolle über Trend-/Modellabgleich.
- Polynomial-Fallback: adaptiver Grad statt fix Grad 2.
- Stern-Detektion: lokaler Background-Rescue als zusätzliche robuste Stufe.
- Diagnosewarnung bei hoher Chain-Tiefe.

## Erwartetes Ergebnis nach Umsetzung
- Mehr Frames bleiben mit **real registrierten** Warps erhalten.
- Weniger unnötige Outlier-Rejects.
- Modellierte Fallback-Frames werden weiterhin genutzt, aber mit kontrollierterem Einfluss.
- Insgesamt robustere Registrierung bei Wolken, Gradient, Rotationsfeldern und kleinen Problemblöcken.
