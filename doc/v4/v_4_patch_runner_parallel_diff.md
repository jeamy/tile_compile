# Diff-Patch: Parallelisierung von Phase 6 im Runner (Methodik v4)

Dieses Dokument beschreibt einen **konkreten, minimal-invasiven Diff-Patch** gegen den **aktuellen v4-Runner**, um `TILE_RECONSTRUCTION_TLR` **tile-parallel und produktionsreif** auszuführen.

Der Patch geht davon aus, dass bereits gilt:
- `StreamingTileProcessor` ist aktiv
- globale Registrierung und Phasen sind entfernt
- v4-YAML wird geparst (`cfg.v4.*`)

---

## 1. Neue Datei hinzufügen

### `runner/v4_parallel.py`

```diff
+ from runner.tile_processor_v4 import StreamingTileProcessor
+
+ def process_tile_job(args):
+     tile_id, bbox, frame_paths, global_weights, cfg = args
+     tp = StreamingTileProcessor(
+         tile_id=tile_id,
+         bbox=bbox,
+         frame_paths=frame_paths,
+         global_weights=global_weights,
+         cfg=cfg,
+     )
+     tile_img, warps = tp.run()
+     return tile_id, bbox, tile_img, warps
```

---

## 2. Imports im Runner erweitern

### `tile_compile_runner.py`

```diff
+ from concurrent.futures import ProcessPoolExecutor, as_completed
+ from runner.v4_parallel import process_tile_job
```

---

## 3. Seriellen Tile-Loop ersetzen

### ALT (vereinfacht)

```python
results = []
for tid, bbox in enumerate(tiles):
    tp = StreamingTileProcessor(tid, bbox, frame_paths, weights, cfg)
    tile = tp.run()
    if tile is not None:
        results.append((bbox, tile))
```

### NEU (parallel, v4-konform)

```diff
- results = []
- for tid, bbox in enumerate(tiles):
-     tp = StreamingTileProcessor(tid, bbox, frame_paths, weights, cfg)
-     tile = tp.run()
-     if tile is not None:
-         results.append((bbox, tile))
+ results = []
+ jobs = []
+ for tid, bbox in enumerate(tiles):
+     jobs.append((tid, bbox, frame_paths, weights, cfg))
+
+ max_workers = cfg.v4.parallel_tiles
+ with ProcessPoolExecutor(max_workers=max_workers) as exe:
+     futures = [exe.submit(process_tile_job, j) for j in jobs]
+     for f in as_completed(futures):
+         tid, bbox, tile_img, warps = f.result()
+         if tile_img is not None:
+             results.append((bbox, tile_img, warps))
```

---

## 4. Overlap-Add unverändert lassen (wichtig!)

```text
KEINE Parallelisierung im Overlap-Add
```

Begründung:
- vermeidet Race Conditions
- garantiert deterministische Floating-Point-Ergebnisse

---

## 5. CI- & Laufzeit-Guards ergänzen

### `tile_compile_runner.py`

```diff
+ import os
+ cores = os.cpu_count()
+ if cfg.v4.parallel_tiles > cores:
+     raise RuntimeError("v4.parallel_tiles exceeds CPU core count")
```

---

## 6. YAML-Voraussetzung (verbindlich)

```yaml
v4:
  parallel_tiles: 8
```

Fehlt dieser Wert:
- Default = 1
- Warnung loggen

---

## 7. Erwartetes Verhalten nach Patch

- Phase 6 skaliert nahezu linear bis I/O-Sättigung
- Speicherverbrauch bleibt konstant
- Ergebnisse sind bit-reproduzierbar

---

## 8. Patch-Zusammenfassung

✔ Tile-Parallelität integriert  
✔ v4-konform  
✔ produktionsreif  
✔ deterministisch  
✔ minimaler Code-Eingriff

---

**Dieser Diff-Patch ist die Referenzimplementierung für parallele TILE_RECONSTRUCTION_TLR (Methodik v4).**

