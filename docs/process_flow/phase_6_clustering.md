# STATE_CLUSTERING — Classic-only Frame-Clusterung

> **Aktueller Code:** `runner_pipeline.cpp`
> **Phase-Enum:** `Phase::STATE_CLUSTERING` (Enum 10)

## AQMH-Verhalten

AQMH rekonstruiert direkt pixelweise. Deshalb werden `STATE_CLUSTERING` und
`SYNTHETIC_FRAMES` im AQMH-Zweig nicht ausgeführt. `STATE_CLUSTERING` wird
dort nicht als normale aktive Phase gestartet; der anschließende
`SYNTHETIC_FRAMES`-Schritt wird mit
`reason: aqmh_independent_reconstruction` als `skipped` beendet.

```text
AQMH:
  AQMH_RECONSTRUCTION -> AQMH_DIAGNOSTICS -> STACKING
  STATE_CLUSTERING    -> nicht ausgeführt
  SYNTHETIC_FRAMES    -> skipped
```

## Classic-Verhalten

Bei `method: classic_tile_compile` wird der Zustandsvektor pro Frame aus den
globalen und lokalen Qualitätswerten sowie Rekonstruktionsdiagnostik gebildet.
Die Clusterung ist nur aktiv, wenn kein Reduced-/Emergency-Mode sie sperrt.

Typische Komponenten des Zustandsvektors sind:

```text
[G_f, mean_local_quality, var_local_quality, B_f, sigma_f, ...]
```

Die Dimensionen werden deterministisch z-normalisiert. Die Clusteranzahl wird
aus `synthetic.clustering.cluster_count_range` und der Frame-Anzahl bestimmt.
Bei leeren oder degenerierten Clustern greift der dokumentierte Quantil-
Fallback. Das Ergebnis wird später von `SYNTHETIC_FRAMES` verwendet.

## Skip-Bedingungen im Classic-Zweig

| Bedingung | Verhalten |
|---|---|
| AQMH aktiviert | Phase nicht starten; synthetische Frames werden übersprungen |
| Reduced-/Emergency-Mode | `STATE_CLUSTERING` wird `skipped` beendet |
| zu wenige Frames | Reduced-/Emergency-Entscheidung des Runtime-Gates |
| leerer K-Means-Cluster | deterministischer Quantil-Fallback |

## Übergang

Nur Classic:

```text
STATE_CLUSTERING -> SYNTHETIC_FRAMES -> STACKING
```

AQMH:

```text
AQMH_DIAGNOSTICS -> STACKING
```
