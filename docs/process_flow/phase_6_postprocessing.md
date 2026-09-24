# Phase 6 — Downstream: ASTROMETRY, BGE, PCC, HYPERMETRIC_STRETCH

> **C++-Implementierung:** `runner_downstream.cpp`

Alle Downstream-Phasen sind **optional** und arbeiten auf dem exportierten
Rekonstruktionsbild (`outputs/`). Die Pipeline bleibt bis einschließlich
PCC **linear**; HYPERMETRIC_STRETCH ist die explizite nichtlineare
Endstufe.

## ASTROMETRY

- Plate Solving des Rekonstruktionsergebnisses (ASTAP-Binary, lokaler
  Katalog-Fallback)
- `astrometry.enabled` steuert die Ausführung; ohne Lösung endet die Phase
  `skipped`, nicht `error`
- Ergebnis: WCS-Header in den Ausgabe-FITS, `artifacts/astrometry*.json`

## BGE (Background Gradient Extraction)

- Modusbasiert über `bge.method` (`none` deaktiviert)
- Entfernt großräumige Hintergrundgradienten vor der Farbkalibrierung
- `artifacts/bge.json` mit Modell-/Residual-Diagnostik

## PCC (Photometric Color Calibration)

- `pcc.enabled`; kalibriert die Kanalflüsse photometrisch (Katalog-basiert,
  Siril-Katalogverzeichnis konfigurierbar)
- Läuft nach BGE, damit Gradienten die Photometrie nicht verfälschen

## HYPERMETRIC_STRETCH

- `hypermetric_stretch.enabled`; VeraLux HyperMetric Stretch als letzte
  Phase — bewusst nach PCC platziert, weil sie den linearen Zustand
  verlässt

## Reihenfolge- und Resume-Eigenschaften

- Feste Reihenfolge ASTROMETRY → BGE → PCC → HMS; jede Phase kann
  einzeln `skipped` enden, ohne den Lauf zu gefährden
- Bei `resume-reconstruction` laufen alle aktivierten Downstream-Phasen
  **immer** erneut — sie sind billig gegenüber dem Gather und hängen vom
  frisch fusionierten MULTIBAND-Ergebnis ab
