# Phase 6 – TILE_RECONSTRUCTION_TLR
## Produktionsreife Parallelisierung (Methodik v4)

Dieses Dokument ist eine **verbindliche Referenz** für die **Parallelisierung von Phase 6 (`TILE_RECONSTRUCTION_TLR`)** in `tile_compile`.

Es beschreibt **was erlaubt ist**, **was verboten ist** und liefert **direkt einsetzbaren Referenzcode** sowie die **erforderlichen YAML-Erweiterungen**.

---

## 1. Grundsatz (nicht verhandelbar)

> **Parallelisiert wird ausschließlich über Tiles.**

Alle anderen Parallelisierungsachsen sind entweder:
- methodisch falsch (Frames)
- ineffektiv oder instabil (Iterationen)

Diese Entscheidung ist **direkte Konsequenz von Methodik v4**.

---

## 2. Warum Tile-Parallelität korrekt ist

Eigenschaften eines Tiles in v4:

- vollständig lokal
- keine Abhängigkeit zu anderen Tiles
- eigene Referenz
- eigener Warp-Verlauf
- eigener Akkumulator

➡️ Ein Tile ist ein **perfekt isolierter Task**.

---

## 3. Verbotene Parallelisierungen (explizit)

❌ Parallelisierung über Frames  
→ zerstört Streaming-I/O, führt zu I/O-Sättigung

❌ Parallelisierung über Iterationen  
→ bricht Referenzabhängigkeit

❌ Paralleles Overlap-Add  
→ nicht-deterministische Floating-Point-Ergebnisse

---

## 4. Empfohlenes Ausführungsmodell

```
Main Process
 ├─ Tile-Liste erzeugen
 ├─ Tile-Jobs an Worker übergeben
 ├─ Ergebnisse einsammeln
 └─ Overlap-Add (seriell, deterministisch)
```

---

## 5. Produktionsreifer Referenzcode (Python)

### 5.1 Worker-Funktion

```python
# runner/v4_parallel.py
from runner.tile_processor_v4 import StreamingTileProcessor


def process_tile_job(args):
    tile_id, bbox, frame_paths, global_weights, cfg = args
    tp = StreamingTileProcessor(
        tile_id=tile_id,
        bbox=bbox,
        frame_paths=frame_paths,
        global_weights=global_weights,
        cfg=cfg,
    )
    result, warps = tp.run()
    return tile_id, bbox, result, warps
```

---

### 5.2 Parallel-Dispatch im Runner

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from runner.v4_parallel import process_tile_job

max_workers = cfg.v4.parallel_tiles

jobs = []
for tid, bbox in enumerate(tiles):
    jobs.append((tid, bbox, frame_paths, global_weights, cfg))

results = []
with ProcessPoolExecutor(max_workers=max_workers) as exe:
    futures = [exe.submit(process_tile_job, j) for j in jobs]
    for f in as_completed(futures):
        tile_id, bbox, tile_img, warps = f.result()
        if tile_img is not None:
            results.append((bbox, tile_img, warps))
```

➡️ **Deterministisch, speichersicher, v4-konform.**

---

## 6. Overlap-Add (bewusst seriell)

```python
final = np.zeros(image_shape, dtype=np.float32)
weights = np.zeros(image_shape, dtype=np.float32)

for bbox, tile_img, _ in results:
    x0, y0, w, h = bbox
    window = hann_window(w, h)
    final[y0:y0+h, x0:x0+w] += tile_img * window
    weights[y0:y0+h, x0:x0+w] += window

final /= np.maximum(weights, 1e-6)
```

➡️ **Kein Race Condition möglich.**

---

## 7. YAML-Erweiterung (verbindlich)

```yaml
v4:
  # Anzahl paralleler Tile-Worker
  # Empfehlung:
  # 4  – HDD / SATA / kleine Sensoren
  # 8  – Default (NVMe, >= 8 Cores)
  # 16 – Nur High-End-Systeme
  parallel_tiles: 8
```

### Harte Regeln

- `parallel_tiles <= Anzahl physischer CPU-Kerne`
- Default = 8
- CI darf fehlschlagen, wenn >16

---

## 8. CI-Guard (empfohlen)

```python
import os
cores = os.cpu_count()
if cfg.v4.parallel_tiles > cores:
    raise RuntimeError("parallel_tiles exceeds CPU core count")
```

---

## 9. Performance-Erwartung (realistisch)

| Worker | Speed-up | Kommentar |
|------|---------|----------|
| 1 | 1× | Referenz |
| 4 | ~3× | I/O beginnt zu limitieren |
| 8 | ~5–6× | Sweet Spot |
| 16 | ~6–7× | I/O-gesättigt |

---

## 10. Zusammenfassung (klar)

- Phase 6 ist **exzellent parallelisierbar**
- **Nur Tile-Parallelität ist erlaubt**
- Process-Pool ist Produktionsstandard
- Overlap-Add bleibt seriell
- Ergebnis ist **deterministisch & reproduzierbar**

---

**Dieses Dokument ist die Produktionsreferenz für TILE_RECONSTRUCTION_TLR-Parallelisierung (v4).**

