# IC434 AQMH v1 vs. aqhm-v0.2.0 Run-Vergleich

Datum: 2026-07-10

Verglichene Runs:

- `runs/ic434-aqhm-v1_20260710_054122`
- `runs/ic434-aqhm-v2-gpu_20260711_183712`

## Kurzfazit

Der Run `ic434-aqhm-v1_20260710_054122` ist schneller und in den wichtigsten Endmetriken besser, aber nicht nur wegen GPU. Die Runs unterscheiden sich in mehreren relevanten Punkten:

1. v1 nutzt die alte GPU-faehige AQMH-Map- und Tile-Reconstruction-Strecke (`opencv_cuda`).
2. v0.2.0 nutzt eine neue AQMH-v2-Reconstruction-Architektur mit anderer Gewichtung, Clipping-Logik und Uniform-Control-Validierung.
3. v0.2.0 liest die AQMH-Map-Caches in der Reconstruction sehr viel haeufiger (`read_count=13568`) als v1 (`read_count=718`).
4. v0.2.0 hat eine strengere Mindestdeckung und einen anderen Clipping-/Selektionsalgorithmus (`min_fraction=0.5`, `clip_iterations=3`, `min_n_eff=2`).

Die BGE-Konfiguration unterscheidet sich ebenfalls, ist aber aus diesen Artefakten nicht als Hauptursache belastbar. Insbesondere zeigt v0.2.0 bereits im eigenen AQMH-Uniform-Control-Vergleich eine Background-RMS-Regression (`aqmh_background_rms=2.1134517` gegen `control_background_rms=1.0517298`). Der primaere Qualitaetsansatzpunkt liegt deshalb in AQMH-Reconstruction/Gewichtung, nicht in BGE.

Ein reiner Schalter "GPU wieder an" waere deshalb kein sauberer Fix. Die v2-GPU-Probleme sind nicht dadurch erklaert, dass v0.2.0 in diesem Run CPU nutzt. Der CPU-Pfad wurde eingefuehrt, weil der vorhandene v2-GPU-Pfad selbst ein Flaschenhals war. Die Ursache liegt in der Arbeitsstruktur: zu kleine GPU-Einheiten, Host-Device-Pingpong, OpenCV-CUDA-Teiloperationen statt eines nativen AQMH-v2-Kernels und I/O-lastiges Cache-Streaming.

## Laufzeitvergleich

| Bereich | v1/master | aqhm-v0.2.0 | Bewertung |
|---|---:|---:|---|
| Registration | 336.820 s | 141.268 s | v0.2.0 deutlich schneller |
| Prewarp | 111.217 s | 113.736 s | praktisch gleich |
| AQMH Maps / Local Metrics | ca. 433 s, GPU `opencv_cuda` | 746.144 s, CPU | v1 deutlich schneller |
| AQMH Reconstruction / Tile Reconstruction | 212.036 s, GPU `opencv_cuda` | 261.898 s, CPU exact | v1 schneller |
| AQMH Kern gesamt | ca. 645 s | ca. 1008 s | v1 ca. 1.56x schneller |
| BGE | 92.779 s | 41.875 s | v0.2.0 schneller, aber schlechtere Endwirkung |
| PCC | 62.620 s | 58.999 s | praktisch gleich |

Hinweis: v1 fuehrt AQMH Maps in `LOCAL_METRICS` und Reconstruction in `TILE_RECONSTRUCTION`; v0.2.0 hat eigene Phasen `AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS`. Die Phasennamen sind also nicht 1:1 identisch.

## Beschleunigungsstatus

### v1/master

- `AQMH_MAPS`: GPU aktiv, Backend `opencv_cuda`
- `AQMH_RECONSTRUCTION`: GPU aktiv, Backend `opencv_cuda`
- Cache-Reconstruction: `read_count=718`, `write_count=359`, `max_resident_maps_observed=2`
- Canvas: `4308 x 3180`
- Frames: `359`
- Rekonstruktionsparameter aus Artefakt: `sigma_low=2`, `sigma_high=1.5`, `min_fraction=0.4`

### aqhm-v0.2.0

- `AQMH_MAPS`: CPU
- `AQMH_RECONSTRUCTION`: CPU, `execution_backend=cpu_exact_v0_2`
- Cache-Reconstruction: `read_count=13568`, `write_count=359`, `max_resident_maps_observed=2`
- Canvas: `4310 x 3180`
- Frames: `359`
- Rekonstruktionsparameter: `clip_sigma=3`, `clip_iterations=3`, `min_fraction=0.5`, `min_n_eff=2`
- Layout/Algorithmus: `pixel_major_soa`, deterministic weighted selection, uniform control in same pass

Der auffaelligste technische Regressionspunkt ist `read_count=13568` in v0.2.0 gegenueber `718` in v1. Das ist ein echter I/O- und Cache-Workload-Unterschied, nicht nur fehlende CUDA-Beschleunigung.

Wichtig: Die CPU-Ausfuehrung ist nicht die Grundursache des v2-GPU-Problems. Im aktuellen Code zeigt `AccelerationOps::reconstruct_aqmh()` fuer AQMH-v2 auf den CPU-Algorithmus `reconstruct_aqmh_weighted()` und markiert bei GPU-Auswahl einen Fallback. Der Kommentar im Code sagt explizit, dass der alte v0.1-CUDA-Kernel keine v2-Merkmale wie `M_f`-Validierung, weighted-MAD-Clipping oder v0.2-Cherry-Pick-Gate implementiert. Damit ist OpenCV-CUDA kein echter AQMH-v2-Reconstruction-Kernel.

Zusaetzlich enthalten die OpenCV-CUDA-Hilfspfade Host-Uploads/Downloads fuer kleine Blöcke:

- `gpu_mat.upload(host_view)` fuer Akkumulator-/Koeffizientenzustaende,
- `weighted_tile_gpu.upload(weighted_tile_host)` pro Tile/ROI,
- `gpu_mat.download(host_view)` beim Flush.

Dieses Muster macht GPU zum Flaschenhals, wenn die eigentliche v2-Logik auf CPU/Host bleibt und nur kleine Nebenoperationen auf die GPU verschoben werden.

## Qualitaetsvergleich

| Metrik | v1/master | aqhm-v0.2.0 | Besser |
|---|---:|---:|---|
| Output FWHM Median | 6.071785 px | 6.102444 px | v1 leicht |
| FWHM Improvement | 53.663 % | 53.429 % | v1 leicht |
| Output Background RMS | 0.183704 | 0.474526 | v1 deutlich |
| AQMH Map Mean Avg | 0.496445 | 0.487871 | v1 leicht |
| AQMH Artifact Fraction Avg | 0.172954 | 0.176026 | v1 leicht |
| PCC Stars Used | 503 | 678 | v0.2.0 |
| PCC Residual RMS | 0.325835 | 0.293732 | v0.2.0 |

Die Endvalidierung bevorzugt v1 wegen deutlich niedrigerem Background-RMS und minimal besserem FWHM. PCC isoliert betrachtet sieht v0.2.0 besser aus: mehr Sterne und niedrigerer Fit-Residual. Das widerspricht sich nicht, weil PCC-Fit-Residual nicht automatisch bessere lokale Rekonstruktion oder bessere finale Flaechenruhe bedeutet.

Wichtiger als die BGE-Differenz ist der v0.2.0-interne Uniform-Control-Befund:

| v0.2.0 Uniform-Control | AQMH | Control | Bewertung |
|---|---:|---:|---|
| Background RMS | 2.113452 | 1.051730 | AQMH regressiert deutlich |
| FWHM | 6.102444 | 6.138064 | AQMH minimal besser |
| Seam Score | 0.680222 | 1.075719 | AQMH besser |

Damit verbessert v0.2.0 FWHM und Seam Score leicht/deutlich, verschlechtert aber die Hintergrundruhe massiv. Genau dieser Tradeoff verletzt die Qualitaetsvorgabe.

## BGE-Unterschied

### v1/master

- `bge.method: autobge`
- Fit-Methode: `rbf`
- Patch-Estimator: `sigma_clipped_median`
- `bge_grid_tiles=300`
- `grid_cells_valid=1725`
- `tile_samples_valid=1725`
- Alle drei Kanaele angewendet
- Guard-Flatness verbessert:
  - R: `4.225067 -> 1.752747`
  - G: `3.480225 -> 1.459930`
  - B: `3.174255 -> 1.478577`

### aqhm-v0.2.0

- `bge.method: classic`
- Konfigurierter Autobge-Block ist vorhanden, aber nicht aktiv
- Fit-Methode im BGE-Block: `poly`
- `bge_grid_tiles=266`
- `grid_cells_valid=105`
- `tile_samples_valid=651`
- Alle drei Kanaele angewendet
- Guard-Flatness verbessert, aber weniger stark:
  - R: `5.043915 -> 3.349426`
  - G: `4.542206 -> 3.291412`
  - B: `3.691101 -> 2.663635`

Dieser Unterschied ist dokumentiert, wird aber nicht als Hauptursache gewertet. Die AQMH-Uniform-Control-Metriken in v0.2.0 zeigen bereits vor einer BGE-zentrierten Interpretation den kritischen Qualitaetsverlust: AQMH senkt Seam Score und FWHM leicht, verdoppelt aber den Background-RMS gegenueber der Control-Rekonstruktion.

## Warum v1 schneller ist

1. Die AQMH-Maps laufen auf GPU. In diesem Run ist das schneller als die CPU-Variante.
2. Die v1-Reconstruction ist eine einfachere GPU-faehige Tile-Reconstruction-Strecke. Sie liest die Map-Caches viel seltener.
3. v0.2.0 macht robustere Rekonstruktion mit mehreren Clipping-Iterationen, strengerer Mindestdeckung und Uniform-Control. Das kostet Rechenzeit und I/O.
4. v0.2.0 trennt AQMH in mehr Phasen und erzeugt mehr Diagnostik/Validierung. Das ist fachlich besser nachvollziehbar, aber nicht kostenlos.

## Warum GPU in v2 ein Flaschenhals ist

Die langsameren v2-GPU-Runs werden nicht dadurch erklaert, dass v0.2.0 hier CPU nutzt. Der CPU-Exact-Pfad ist eher eine Reaktion auf den unguenstigen v2-GPU-Pfad. Die eigentliche Ursache ist, dass v2 keine durchgaengig GPU-native AQMH-Pipeline hat.

1. **Kein nativer AQMH-v2-CUDA-Kernel:** Die v2-Reconstruction-Logik laeuft ueber `reconstruct_aqmh_weighted()`. Der alte CUDA-Kernel deckt die v2-Anforderungen nicht ab und darf nicht als erfolgreicher v2-GPU-Pfad reported werden.
2. **Host-Device-Pingpong:** Kleine Tiles/ROIs werden auf die GPU hochgeladen, dort mit OpenCV-CUDA addiert/dividiert/verglichen und spaeter wieder auf Host-Seite gebraucht. Das erzeugt Transfer- und Synchronisationskosten.
3. **Zu kleine GPU-Arbeitseinheiten:** AQMH-v2 arbeitet stark tile-/chunk-/frameweise. Wenn pro Kernel nur kleine Matrizen laufen, dominieren Kernel-Launches und Speichertransfers.
4. **Branchy/reduktionslastiger Algorithmus:** Weighted selection, robustes Clipping, Valid-Mask-Gates, Mindestdeckung, Median/MAD-Logik und Uniform-Control sind keine guten OpenCV-CUDA-Standardoperationen. Sie brauchen eigene batched Kernels oder bleiben besser CPU-optimiert.
5. **I/O-getriebene Pipeline:** `read_count=13568` zeigt, dass v2 viel Zeit mit Cache-Streaming und Wiederholungslesungen verbringt. Eine GPU wartet in so einem Aufbau auf Daten, statt Rechenzeit zu sparen.

Damit ist "GPU einschalten" kein Performance-Fix. Ein sinnvoller v2-GPU-Fix muesste die Daten pro Chunk einmal auf die GPU bringen, dort die gesamte gewichtete Auswahl/Clipping-Logik ausfuehren und erst das fertige Ergebnis zurueckkopieren.

## Warum v1 besser aussieht

1. Die v1-Reconstruction ist weniger streng (`min_fraction=0.4` statt `0.5`) und nutzt nicht den neuen exakten Clipping-Pfad. Das kann bei diesem Datensatz mehr Signal erhalten.
2. v1 hat keine v0.2.0-typische Uniform-Control-Regression dokumentiert. v0.2.0 zeigt dagegen selbst, dass AQMH den Background-RMS gegenueber Control deutlich verschlechtert.
3. v1 liest die AQMH-Caches viel effizienter und bleibt naeher an der alten GPU-/Streaming-Strategie.
4. Die AQMH-Karten selbst sind zwischen den Runs nicht dramatisch unterschiedlich. Die relevante Differenz liegt in der Reconstruction-Strategie und deren Gewichtungs-/Clipping-Folgen.

## Fix-Vorschlag

### Kurzfristig: Qualitaet

1. Den v1-kompatiblen schnellen Reconstruction-Pfad als explizites Profil wieder anbieten, z. B.:
   - `aqmh.reconstruction.mode: exact_v0_2`
   - `aqmh.reconstruction.mode: fast_v1`
2. v0.2.0-Quality-Gates schaerfen: Wenn `aqmh_uniform_control.background_rms_regression` positiv und deutlich ueber Limit ist, darf AQMH nicht als Qualitaetsverbesserung gelten.
3. Die v0.2.0-Reconstruction-Parameter gegen IC434 sweepen:
   - `min_fraction: 0.4` vs. `0.5`
   - `clip_iterations: 1..3`
   - `clip_sigma: 2.0..3.5`
   - `min_n_eff: 1..3`
4. Einen Quality-Selection-Guard einfuehren:
   - AQMH-v2 wird nur akzeptiert, wenn Background-RMS nicht regressiert.
   - Wenn v2 nur Seam Score/FWHM verbessert, aber Background-RMS stark verschlechtert, faellt der Run automatisch auf `fast_v1` oder Control zurueck.
5. Die Optimierung nicht auf PCC-Residual ausrichten. PCC kann besser aussehen, obwohl die Rekonstruktion mehr Hintergrundrauschen erzeugt.

### Kurzfristig: Performance

1. v2-GPU standardmaessig deaktiviert lassen, solange kein nativer v2-Kernel existiert. OpenCV-CUDA-Teiloperationen sind hier kein echter AQMH-v2-GPU-Pfad.
2. Den CPU-v2-Pfad I/O-seitig optimieren. `read_count=13568` ist gegenueber v1 `718` zu hoch.
3. Map-Cache-Streaming chunkweise so umbauen, dass pro Chunk weniger Wiederholungslesungen entstehen.
4. Diagnostik und Uniform-Control vom Produktionspfad trennen oder strikt budgetieren. Full Diagnostics duerfen nicht den Standardlauf dominieren.
5. Einen Report-Vergleich fuer AQMH-Profile einfuehren: Laufzeit, Cache-Reads, FWHM, Background-RMS, Artifact-Fraction und PCC-Metriken nebeneinander.

### Mittelfristig: echter v2-GPU-Fix

1. Einen nativen batched AQMH-v2-CUDA-Kernel bauen, statt v2 aus OpenCV-CUDA-Teiloperationen zusammenzusetzen.
2. Datenlayout auf chunk-/pixel-major SoA fuer GPU ausrichten:
   - Frames und Q-Maps eines Chunks zusammenhaengend,
   - Masken kompakt,
   - Gewichte coalesced lesbar.
3. Pro Chunk nur einmal Host->Device kopieren, dann gewichtete Auswahl, Clipping, Mindestdeckung und Output-Erzeugung komplett auf der GPU ausfuehren.
4. Erst das fertige Chunk-Ergebnis Device->Host kopieren.
5. GPU-Auto nur ueber echten Probe-Benchmark erlauben:
   - gleiche Frames,
   - gleiche Masken,
   - gleiche v2-Qualitaetslogik,
   - Vergleich CPU-v2 vs. CUDA-v2.
6. Wenn der native CUDA-v2-Kernel nicht gebaut wird, GPU fuer AQMH-v2 nicht anbieten. Dann ist CPU-Optimierung ehrlicher und schneller als ein halber GPU-Pfad.

### Priorisierte Umsetzung

1. **Quality Gate fixen:** Background-RMS-Regression aus `aqmh_uniform_control` als harte Ablehnung verwenden.
2. **Profile einfuehren:** `fast_v1`, `balanced_v2`, `exact_v2`.
3. **Cache-Read-Reduktion:** Reconstruction so umbauen, dass Q-Maps pro Chunk/Frame nicht wiederholt gelesen werden.
4. **GPU-v2 deaktivieren oder verstecken:** Nur v1/legacy darf OpenCV-CUDA nutzen, v2 erst nach nativer Kernel-Implementierung.
5. **Parameter-Sweep automatisieren:** v2 muss seine Parameter gegen Control und v1-kompatiblen Pfad beweisen.

## Implementierter nativer CUDA-Test

Nach dem Report wurde der vorhandene native CUDA-Reconstruction-Kernel in den Produktionspfad eingebunden und mit IC434 quergetestet.

Geaenderter Pfad:

- `AccelerationPhase::aqmh_reconstruction` darf jetzt Backend `cuda` auswaehlen.
- `AccelerationOps::reconstruct_aqmh()` ruft bei `selected_backend=cuda` `reconstruct_aqmh_weighted_cuda()` auf.
- `runner_phase_aqmh_reconstruction.cpp` erzwingt nicht mehr CPU, sondern reportet `execution_backend`, `acceleration_used`, `acceleration_fallback` und `selected_backend`.
- `runner_resume.cpp` kann einen AQMH-Reconstruction-Resume aus einem fertig gecroppten Run starten, indem bei abweichender `outputs/canvas_mask.fits` die volle Common-Mask aus `cache/aqmh_masks` rekonstruiert wird.

Validierung:

- Build `tile_compile_cpp`: erfolgreich.
- Focused Tests: `ctest --output-on-failure -R aqmh_reconstruction` -> 7/7 bestanden.
- Native CUDA Unit-Test `aqmh_native_cuda_reconstruction_matches_cpu_reference`: bestanden, wenn ausserhalb der Sandbox mit sichtbarer GPU ausgefuehrt.

IC434-Quer-Test:

- Test-Run: `runs/ic434-aqmh-cuda-v2test_20260710`
- Basis: Kopie von `runs/ic434-aqhm_20260709_213951`
- Ausgefuehrt ab `PREWARP`, weil der fertig abgeschlossene Originalrun keine `.prewarped_cache` mehr enthielt.
- `AQMH_MAPS`: weiterhin CPU, 359 Maps, ca. 710 s (`06:17:08` bis `06:28:58`).
- `AQMH_RECONSTRUCTION`: `selected=cuda`, `execution_backend=cuda_native_v0_2`, `acceleration_used=true`, `acceleration_fallback=false`.
- Dauer native CUDA-Reconstruction: `611.559 s`.
- Vergleich v0.2.0 CPU-Reconstruction: `261.898 s`.
- Vergleich v1 OpenCV-CUDA Tile-Reconstruction: `212.036 s`.

Artefaktwerte des nativen CUDA-Tests:

- Canvas: `4310 x 3180`
- Frames: `359`
- `chunk_rows=183`, `chunk_count=18`
- `cuda_free_bytes=4286251008`, `cuda_device_budget_bytes=2571750604`
- `cache_stats.bytes_read=4945143150`
- `cache_stats.bytes_written=2460191100`
- `cache_stats.read_count=6785`
- `cache_stats.write_count=359`
- `unsupported_pixels=0`, `zero_veto_pixels=0`, `missing_map_samples=0`

Ergebnis: Der native CUDA-Pfad ist funktional korrekt verdrahtet, aber auf IC434 nicht performant. Er ist ca. `2.34x` langsamer als der v0.2.0-CPU-Pfad und ca. `2.88x` langsamer als v1. Das widerlegt die reine Annahme, dass ein nativer Kernel automatisch die v2-Performance loest.

Der harte Befund ist: Die aktuelle CUDA-Implementierung bleibt durch Chunk-/Region-Streaming, Cache-I/O und Datenbewegung dominiert. Sie reduziert zwar die Cache-Reads gegenueber CPU (`6785` statt `13568`), aber die GPU-Arbeit pro Chunk ist offenbar nicht gross/zusammenhaengend genug, um die Transfers, Synchronisation und den robusten Selektions-/Clipping-Overhead zu amortisieren.

Qualitaet des CUDA-Testlaufs:

| Metrik | v1/master | v0.2.0 CPU | nativer CUDA-Test |
|---|---:|---:|---:|
| Output FWHM Median | 6.071785 | 6.102444 | 6.103886 |
| Output Background RMS | 0.183704 | 0.474526 | 2.099946 |
| AQMH Uniform-Control RMS | nicht vorhanden | 2.113452 | 2.583124 |
| Control Background RMS | nicht vorhanden | 1.051730 | 1.050085 |
| Seam Score AQMH | nicht vorhanden | 0.680222 | 0.597395 |
| PCC Stars Used | 503 | 678 | 687 |
| PCC Residual RMS | 0.325835 | 0.293732 | 0.292179 |

Der native CUDA-Test verbessert die Performance nicht und verbessert die Qualitaet nicht. Die FWHM bleibt praktisch gleich zu v0.2.0 CPU, die finale Background-RMS ist schlechter, und die interne Uniform-Control-RMS-Regression bleibt bestehen. PCC sieht isoliert leicht besser aus, ist aber nicht das massgebliche Qualitaetskriterium fuer die AQMH-Reconstruction.

## Aktualisierter Fix nach CUDA-Test

## Quality-First Nachtest: Adaptive AQMH-Control-Blend

Nach der Vorgabe "Qualitaet zuerst" wurde ein weiterer IC434-Echtdatenlauf mit hartem Quality-Gate und adaptiver AQMH-Abschwaechung ausgefuehrt.

Test-Run:

- `runs/ic434-aqhm-v2-adaptiveblend_20260710_223500`

Implementierter Testansatz:

1. Die rohe AQMH-v2-Reconstruction wird wie bisher berechnet.
2. Parallel wird die Uniform-Control-Reconstruction berechnet.
3. Wenn AQMH gegenueber Control Background-RMS, FWHM oder Seam Score regressiert, wird nicht sofort nur hart verworfen.
4. Stattdessen wird per binaerer Suche die staerkste zulaessige Mischung gesucht:
   - `output = control + alpha * (aqmh - control)`
5. Diese Mischung wird nur akzeptiert, wenn alle Gates bestehen und mindestens FWHM oder Seam Score gegenueber Control verbessert werden.

Ergebnis:

- `execution_backend=cuda_native_v0_2`
- `acceleration_used=true`
- `fallback_to_uniform_control=true`
- `uniform_control_blend_accepted=false`
- `uniform_control_blend_alpha=0.0`

Gemessene Gate-Metriken vor Fallback:

| Metrik | AQMH roh | Uniform Control | Regression |
|---|---:|---:|---:|
| Background RMS | 2.603758 | 1.050085 | +147.957 % |
| FWHM | 6.103702 | 6.138064 | -0.560 % |
| Seam Score | 0.592877 | 1.072627 | -44.727 % |

Interpretation:

AQMH-v2 verbessert in diesem Lauf Schaerfe/FWHM leicht und Seam Score deutlich, erzeugt aber gleichzeitig eine massive Hintergrundverschlechterung. Das ist kein reines Threshold-Problem. Eine einfache Abschwaechung des AQMH-Ergebnisses konnte keinen nichttrivialen Anteil finden, der zugleich Hintergrundruhe, Schaerfe und Seam Score verbessert. Der akzeptierte Anteil ist deshalb `alpha=0`, also reine Uniform-Control.

Damit ist die aktuelle v2-Gewichtung fuer die Zielvorgabe "besser in allen Belangen" nicht geeignet. Sie optimiert lokal auf Schaerfe/Seams, koppelt diese Verbesserung aber an grossflaechige Hintergrundartefakte beziehungsweise Background-RMS-Regression. Ein weiterer Parameter-Feinschliff am finalen Gate reicht dafuer nicht aus.

## Konsequenz fuer die Qualitaetsverbesserung

Die naechste sinnvolle Umsetzung muss die AQMH-Qualitaetsentscheidung frueher und getrennt nach Bildinhalt treffen:

1. **Hintergrundbereiche schuetzen:** AQMH-Gewichte duerfen im glatten Hintergrund nicht durch lokale Schaerfe-/Seam-Vorteile dominieren. Fuer low-structure Pixel braucht es background-neutrale oder Control-nahe Gewichte.
2. **Signal- und Hintergrundmaske trennen:** Sterne/Nebelstruktur, Kanten/Seams und glatter Hintergrund muessen getrennte Gewichtungsregeln bekommen. Eine globale per-pixel Gewichtung erzeugt aktuell den falschen Tradeoff.
3. **Low-frequency Neutrality erzwingen:** Vor dem finalen Blend muss pro Chunk/Region sichergestellt werden, dass AQMH keine grossflaechigen Offsets/Verlaeufe gegenueber Control einfuehrt.
4. **v1/classic als Qualitaetskandidat statt nur Control:** Wenn AQMH-v2 scheitert, soll nicht nur Uniform-Control uebrig bleiben. Der v1/classic-kompatible Pfad muss als weiterer Kandidat laufen oder zumindest als explizites Qualitaetsprofil verfuegbar sein.
5. **Akzeptanz nur per Pareto-Gate:** Ein Kandidat darf nur gewinnen, wenn Background-RMS, FWHM und Seam Score gemeinsam nicht schlechter sind. Einzelne bessere PCC- oder Seam-Werte duerfen keine schlechtere Flaechenruhe ueberstimmen.

Kurz: Fuer echte Qualitaetsverbesserung muss AQMH-v2 nicht schneller, sondern selektiver werden. Der Algorithmus muss lernen, wo AQMH ueberhaupt eingreifen darf. Auf IC434 zeigt der aktuelle Pfad klar: rohe AQMH-v2-Reconstruction ist fuer Hintergrundruhe schaedlich, auch wenn sie Schaerfe und Seams verbessert.

### Performance

1. Native CUDA nicht als Standard fuer AQMH-v2 aktivieren, solange der IC434-Probe-Benchmark langsamer als CPU ist.
2. Auto-Auswahl fuer `aqmh_reconstruction` benchmarkbasiert machen:
   - kleiner repräsentativer Chunk,
   - gleiche `clip_iterations`, Masken und Q-Maps,
   - CPU vs. CUDA messen,
   - nur bei realem Speedup CUDA verwenden.
3. Den CUDA-Pfad von Region-Streaming auf groessere persistente Device-Batches umbauen:
   - Q-Maps und Frame-Regionen pro Chunk/Frame einmal laden,
   - mehrere Row-Chunks in einem Device-Batch abarbeiten,
   - pinned Host Memory fuer Frame-/Map-Transfers,
   - CUDA Streams fuer Copy/Compute-Overlap.
4. Den robusten Auswahl-/Clipping-Schritt auf GPU reduzieren:
   - nicht pro Pixel mehrfach globale Sorts/Selections mit Host-seitiger Steuerung,
   - stattdessen blockweise Top-K/Quickselect oder histogramm-/bucketbasierte Approximation mit validiertem Fehlerbudget.
5. AQMH_MAPS separat optimieren. In v2 ist MAPS weiterhin CPU und dauert in IC434 ca. 12 Minuten; selbst eine schnelle Reconstruction wuerde die Gesamtzeit sonst nicht ausreichend verbessern.

### Qualitaet

1. Background-RMS-Regression gegen Uniform-Control als harte Ablehnung behandeln. Der aktuelle v2- und CUDA-Test zeigen beide denselben Fehler: AQMH verbessert FWHM/Seam, verschlechtert aber die Flaechenruhe.
2. `min_fraction`, `min_n_eff`, `clip_iterations` und `clip_sigma` nicht global fixieren, sondern gegen Control validieren. IC434 braucht offenbar weniger aggressive Selektion oder ein anderes Hintergrundrausch-Gate.
3. Den v1-kompatiblen Fast-Profile-Pfad wieder anbieten, aber nicht als Ersatz fuer v2 deklarieren:
   - `fast_v1`: schnell und bewiesen besser auf IC434,
   - `exact_v2`: reproduzierbar und streng,
   - `adaptive_v2`: nur akzeptiert, wenn Uniform-Control-RMS nicht regressiert.
4. Der Akzeptanzentscheid muss auf FWHM, Background-RMS, Seam Score und Artifact-Fraction basieren. PCC-Residual allein ist kein ausreichender Proxy.

## Entscheidung

GPU ist fuer IC434 in v1 praktisch nutzbar, aber das ist kein Beweis, dass der aktuelle v0.2.0-Algorithmus sinnvoll ueber OpenCV-CUDA laufen sollte. Der v0.2.0-Pfad macht andere Arbeit, und der alte GPU-Pfad bildet diese Arbeit nicht nativ ab. Der richtige Fix ist deshalb:

- schnellen v1-Pfad kontrolliert wieder verfuegbar machen,
- exakten v0.2.0-Pfad behalten,
- v0.2.0-Parameter so abstimmen, dass Background-RMS nicht gegen Uniform-Control regressiert,
- v2-GPU erst nach nativem AQMH-v2-Kernel wieder anbieten,
- GPU-Auswahl messen statt raten,
- v0.2.0-I/O reduzieren.

## Implementierter Quality-Gate-Fix und Echtdatenlauf

Nach dem CUDA-Test wurde die harte Uniform-Control-Ablehnung umgesetzt und mit IC434 erneut getestet.

### Code-Aenderung

- `AQMH_RECONSTRUCTION` berechnet weiterhin den AQMH-v2-Output und den Uniform-Control-Output in einem Pass.
- Wenn `aqmh_background_rms / control_background_rms - 1` ueber `aqmh.validation.max_background_rms_regression` liegt, wird der AQMH-Output verworfen.
- In diesem Fall wird das finale Reconstruction-Output auf Uniform-Control gesetzt.
- Die Validation wird danach gegen den tatsaechlich verwendeten Output neu berechnet, damit der Run nicht mehr die verworfene AQMH-Variante bewertet.
- Das Artefakt `aqmh_reconstruction.json` dokumentiert `fallback_to_uniform_control` und die urspruenglichen Gate-Werte.

### Verifikation

Build und fokussierte Tests:

- `cmake --build build`: bestanden.
- `ctest --output-on-failure -R "aqmh_reconstruction|bge_autobge"`: 9/9 bestanden.

Echtdatenlauf:

- Run: `runs/ic434-aqhm-v2-qgate-autobge_20260710_221500`
- Config: v2-GPU-Config mit `bge.method: autobge`
- Ergebnis: Pipeline `ok`

### Messwerte

| Metrik | v1/master | v2 GPU vorher | v2 Gate+AutoBGE |
|---|---:|---:|---:|
| Run | `ic434-aqhm-v1_20260710_054122` | `ic434-aqhm-v2-gpu_20260710_192007` | `ic434-aqhm-v2-qgate-autobge_20260710_221500` |
| AQMH_MAPS Backend | GPU/OpenCV-CUDA in v1-Strecke | CPU | CPU |
| AQMH_RECONSTRUCTION Backend | GPU/OpenCV-CUDA in v1-Strecke | `cuda_native_v0_2` | `cuda_native_v0_2` |
| AQMH_RECONSTRUCTION Dauer | 212.036 s | 611.559 s | 638.580 s |
| Fallback auf Uniform-Control | nicht dokumentiert | nein | ja |
| Pre-Fallback AQMH Background RMS | nicht dokumentiert | 2.583124 | 2.601221 |
| Control Background RMS | nicht dokumentiert | 1.050085 | 1.050085 |
| Pre-Fallback RMS Regression | nicht dokumentiert | 1.459920 | 1.477153 |
| Final Output FWHM Median | 6.071785 | 6.103886 | 6.138064 |
| Final Output Background RMS | 0.183704 | 1.966955 | 0.000000 |
| Validation `aqmh_uniform_control.background_rms` | nicht dokumentiert | 2.497659 | 1.050085 |
| Validation `aqmh_uniform_control.control_background_rms` | nicht dokumentiert | 1.050085 | 1.050085 |

Hinweis zum finalen `output_background_rms=0.000000`: Diese Endvalidierungsmetrik ist nach dem Fallback auffaellig degeneriert und sollte nicht allein als "perfekt rauschfrei" interpretiert werden. Die belastbare Gate-Metrik ist `aqmh_uniform_control.background_rms=1.050085`, weil AQMH nach dem Fallback identisch zum Control-Output ist.

### BGE-Vergleich

| Metrik | v1/master AutoBGE | v2 vorher Classic/Poly | v2 Gate+AutoBGE |
|---|---:|---:|---:|
| R fit RMS residual | 0.233579 | 1.360030 | 0.291981 |
| G fit RMS residual | 0.226581 | 1.212207 | 0.298802 |
| B fit RMS residual | 0.274932 | 1.173084 | 0.315570 |
| R guard flat post | 1.752747 | 2.703461 | 2.011932 |
| G guard flat post | 1.459930 | 2.800781 | 2.444550 |
| B guard flat post | 1.478577 | 2.318420 | 1.725800 |

AutoBGE verbessert v2 gegen den vorherigen Classic/Poly-Lauf deutlich, erreicht aber nicht ganz die v1-Werte.

### Bewertung

Der Quality-Gate-Fix erzeugt eine sichtbare und messbare Verbesserung gegen den vorherigen v2-GPU-Lauf: die starke Background-RMS-Regression und die kuenstlichen Verlaeufe der AQMH-Reconstruction werden nicht mehr in das finale Bild uebernommen. Das ist ein echter Fix fuer die Qualitaetsregression.

Er loest aber nicht den Performance-Punkt und verbessert die Schaerfe nicht:

- Die native CUDA-Reconstruction bleibt mit 638.580 s langsamer als v1 und langsamer als der fruehere v0.2.0-CPU-Pfad.
- Der finale FWHM ist mit 6.138064 schlechter als v1 6.071785 und schlechter als der vorherige v2-GPU-Wert 6.103886.
- Da der Fallback auf Uniform-Control geht, gewinnt v2 hier Flaechenruhe, aber verliert die kleine AQMH-FWHM/Seam-Verbesserung.

Damit ist der aktuelle Stand: Qualitaetsregression abgefangen, aber das Ziel "v2 ist zugleich besser und schneller als v1" ist weiterhin nicht erreicht. Dafuer braucht es entweder einen wirklich schnelleren, benchmark-gesteuerten v2-CPU/Cache-Pfad oder einen groesser gebatchten nativen CUDA-Pfad; der jetzige CUDA-Kernel reicht dafuer auf IC434 nicht.

## Implementierter Quality-Detail-Fix: strukturmaskierte AQMH-Uebernahme

Nach dem reinen Gate-Fallback wurde ein zweiter Quality-First-Fix implementiert und mit IC434 getestet. Ziel war nicht mehr, rohe AQMH-v2 global abzuschwaechen, sondern AQMH-Detail nur dort zu uebernehmen, wo im Control-Bild echte Struktur vorhanden ist. Glatter Hintergrund bleibt dadurch weitgehend Control-nah.

### Code-Aenderung

- `AQMH_RECONSTRUCTION` erzeugt aus dem Uniform-Control-Output eine Strukturmaske per Sobel-Gradient.
- AQMH-v2 wird nur in strukturierten Bereichen auf Control aufaddiert:
  - `candidate = control + structure_mask * (aqmh - control)`
- Wenn dieser Kandidat noch knapp zu viel Background-RMS erzeugt, wird sein Detailanteil per binaerer Suche abgeschwaecht:
  - `output = control + alpha * (candidate - control)`
- Der Kandidat wird nur akzeptiert, wenn alle Uniform-Control-Gates bestehen:
  - Background-RMS-Regression <= 2 %
  - FWHM-Regression <= 2 %
  - Seam-Score-Regression <= 2 %
  - zusaetzlich muss FWHM oder Seam Score gegenueber Control besser sein.
- Bei Scheitern bleibt der harte Fallback auf Uniform-Control aktiv.

### Verifikation

Build und fokussierte Tests:

- `cmake --build build`: bestanden.
- `ctest --output-on-failure -R "aqmh_reconstruction|aqmh_validation|bge_autobge"`: 22/22 bestanden.

Echtdatenlauf:

- Run: `runs/ic434-aqhm-v2-structalpha_20260710_232700`
- Config: v2-GPU-Config mit `bge.method: autobge`
- Ergebnis: Pipeline `ok`
- `AQMH_RECONSTRUCTION`: `execution_backend=cuda_native_v0_2`, `acceleration_used=true`
- `fallback_to_uniform_control=false`
- `structure_masked_detail.applied=true`
- `structure_masked_detail.alpha=0.55029296875`

### Gate-Messwerte

| Metrik | Uniform Control | Strukturmaskiertes AQMH | Bewertung |
|---|---:|---:|---|
| Background RMS | 1.050085 | 1.071083 | +1.9996 %, innerhalb Gate |
| FWHM | 6.138064 | 6.109544 | -0.4646 %, besser |
| Seam Score | 1.072627 | 1.061615 | -1.0267 %, besser |

Damit ist dieser Lauf der erste getestete v2-Quality-Fix, der nicht auf reine Control zurueckfaellt und trotzdem die Hintergrund-Regression begrenzt. Gegenueber dem vorherigen rohen v2-GPU-Lauf wird die starke Background-RMS-Regression von ca. +146 % auf knapp +2 % reduziert, waehrend FWHM und Seam Score gegenueber Control verbessert bleiben.

### Vergleich gegen v1

| Metrik | v1/master | v2 structalpha | Bewertung |
|---|---:|---:|---|
| Final FWHM Median | 6.071785 | 6.109544 | v1 bleibt schaerfer |
| AQMH/Control Background-RMS-Gate | nicht dokumentiert | 1.071083 gegen 1.050085 | v2 jetzt kontrolliert |
| Fallback auf Control | nicht dokumentiert | nein | v2 uebernimmt echten AQMH-Anteil |
| BGE Guard Flat Post R | 1.752747 | 1.943298 | v1 besser |
| BGE Guard Flat Post G | 1.459930 | 2.373718 | v1 deutlich besser |
| BGE Guard Flat Post B | 1.478577 | 1.760132 | v1 besser |

Der Fix loest die v2-Qualitaetsregression gegenueber dem eigenen Control-Pfad, aber er schlaegt v1 auf IC434 noch nicht in allen Endmetriken. Besonders die finale BGE-Flatness und die absolute Schaerfe bleiben bei v1 besser. Die naechsten Qualitaetsschritte muessen deshalb vor allem die Strukturmaske und die Map-Gewichtung verbessern, nicht nur das finale Gate:

1. Strukturmaske kanal-/skalenabhaengig machen, damit Nebelstruktur staerker und glatter Hintergrund schwächer gewichtet wird.
2. AQMH-Maps in low-structure Bereichen background-neutralisieren, bevor Reconstruction startet.
3. FWHM/Seam-Vorteile nicht global suchen, sondern getrennt fuer Sterne, Nebelkanten und glatten Hintergrund bewerten.
4. BGE-Qualitaetsmetriken nach dem AQMH-Kandidaten als weiteres Akzeptanzsignal einbeziehen.

Status: Der unmittelbare Fehler "AQMH-v2 uebernimmt Hintergrundartefakte" ist behoben. Das Produktziel "v2 besser als v1 in allen Belangen" ist damit noch nicht bewiesen.

## Neuer Befund: v2-gpu1 Output-Support/Crop-Regression

Nach dem Lauf `ic434-aqhm-v2-gpu1_20260711_235442` ist die visuelle Regression weiterhin vorhanden. Die wichtigsten Messwerte gegen `ic434-aqhm-v1_20260710_054122`:

| Metrik | v1 | v2-gpu1 | Bewertung |
|---|---:|---:|---|
| Final FWHM Median | 6.071785 | 6.095237 | v1 bleibt schaerfer |
| Final Output-Bild | 4297x3149 | 3877x2189 | v2 verliert deutlich Feld |
| Final Canvas-Mask-Pixel | 10247564 | 8483073 | v2 verliert gueltige Output-Flaeche |
| BGE Grid Tiles | 300 | 180 | v2 BGE arbeitet auf kleinerer Basis |
| BGE R Fit RMS | 0.233579 | 0.291381 | v1 besser |
| BGE G Fit RMS | 0.226581 | 0.292872 | v1 besser |
| BGE B Fit RMS | 0.274932 | 0.307799 | v1 besser |

Die Common-Overlap-Artefakte beider Runs haben identische Canvas-Zahlen (`4308x3180`, `common_pixels=10247564`, `reconstruction_pixels=10421463`). Der kleinere v2-Output entsteht deshalb nicht durch eine schlechtere Registration-Abdeckung, sondern durch die AQMH-Verwendung der Masken vor der Rekonstruktion und den anschliessenden Nonzero-Crop.

### Methodik-Abgleich

`aqmh_methodik_en_v0.2.0.md` trennt die Masken explizit:

- `C(p)` ist die globale Output-Canvas, also die Rekonstruktionsdomain.
- `M_f(p)` ist die frame-spezifische Gueltigkeit innerhalb dieser Domain.
- AQMH-Reconstruction iteriert ueber `C`; einzelne Frames werden nur ueber `M_f` ausgeschlossen.
- BGE fuer AQMH soll aus AQMH-Output und `C` abgeleitet werden, nicht aus Classic Tile Metrics.

Der v2-gpu1-Codepfad verwendete fuer `AQMH_MAPS`, `AQMH_RECONSTRUCTION` und `AQMH_DIAGNOSTICS` den strikten `common_valid_mask`-Schnitt als AQMH-Canvas. Das ist fuer stark geditherte oder rotierte Daten zu restriktiv: gueltige Output-Randbereiche werden bereits vor AQMH aus der Domain entfernt, obwohl dort einzelne Frames Daten liefern koennen.

### Implementierter Fix

`runner_pipeline.cpp` leitet nun fuer AQMH eine eigene Canvas-Maske ab:

- Wenn `overlap_coverage_count` verfuegbar ist, wird `reconstruction_valid_mask = overlap_coverage_count > 0` als AQMH-Canvas `C` verwendet.
- `common_valid_mask` bleibt als Common-Overlap-Analysemaske erhalten.
- `AQMH_MAPS`, `AQMH_RECONSTRUCTION` und `AQMH_DIAGNOSTICS` erhalten die AQMH-Canvas statt des strikten Common-Overlap-Schnitts.
- `common_overlap.json` dokumentiert jetzt `aqmh_canvas_mask` und `aqmh_canvas_pixels`.

Damit entspricht der AQMH-Pfad wieder dem v0.2.0-Vertrag `C + M_f`: Die globale Output-Domain wird nicht kuenstlich auf den Schnitt aller Frames verengt, waehrend frame-spezifische Ausfaelle weiterhin ueber die gespeicherten AQMH-Masken behandelt werden.

### Erwartung fuer den naechsten IC434-Run

Der naechste Run muss zeigen:

1. Finaler Output-Crop wieder nahe v1 (`4297x3149`) statt `3877x2189`.
2. BGE Grid Tiles wieder nahe 300 statt 180.
3. BGE Fit RMS und Guard-Flatness verbessern sich durch die groessere, methodikkonforme Output-Canvas.
4. FWHM bleibt mindestens auf v2-gpu1-Niveau oder verbessert sich gegen v1.

Wenn diese vier Punkte nicht eintreten, ist der naechste Ansatz nicht mehr Canvas-Support, sondern die im vorherigen Abschnitt genannten AQMH-Map-Gewichte in low-structure Bereichen und ein BGE-Akzeptanzsignal nach dem AQMH-Kandidaten.
