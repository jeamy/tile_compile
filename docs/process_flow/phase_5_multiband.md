# Phase 5 — MULTIBAND

> **C++-Implementierung:** `runner_forward_drizzle.cpp` (`Phase::MULTIBAND`),
> `src/reconstruction/` (Band-Planner/Fusion)

## Zweck

Die Multiband-Phase zerlegt die Rekonstruktion in **Frequenzbänder** und
führt sie wieder zusammen. Motivation: Der Uniform-Forward-Drizzle liefert
eine optimale punktweise Schätzung, aber großräumige Hintergrundvariationen
und ungleichmäßige Coverage lassen sich auf einem separaten
Niederfrequenz-Band robuster behandeln als auf Pixelebene.

## Funktionsweise

- Der Band-Planner teilt das Ausgaberaster gemäß
  `reconstruction.multiband.*` in Bänder auf (Anzahl/Grenzen über die
  `reconstruction.multiband.*`-Konfiguration)
- Band-Akkumulatoren entstehen aus demselben Gather-Lauf — Samples werden
  je nach räumlicher Frequenz-Zugehörigkeit in die Band-Kanäle des
  v2-Stores geschrieben
- Fusion: Niederfrequenz- und Hochfrequenzanteile werden bandweise
  rekombiniert; Support-Masken und `n_eff` entscheiden pro Region, wie die
  Bänder gewichtet werden
- Kandidaten-Spooling über `artifacts/multiband_candidate_spool/` hält den
  Speicherbedarf begrenzt

## Artefakte

| Datei | Inhalt |
|-------|--------|
| `artifacts/reconstruction_multiband.fits` | Fusioniertes Multiband-Ergebnis |
| `artifacts/multiband_candidate_spool/` | Zwischengespeicherte Kandidaten |
| `outputs/` | Finale exportierte FITS (Mono bzw. R/G/B) |

## Resume-Verhalten

MULTIBAND ist **kein** Resume-Einstiegspunkt: sie läuft bei jedem
`resume-reconstruction` erneut (sowohl ab `GLOBAL_QUALITY` als auch ab
`FORWARD_DRIZZLE`), ebenso alle Downstream-Phasen. Nur der aufwendige
Gather kann über den FORWARD_DRIZZLE-Checkpoint übersprungen werden.
