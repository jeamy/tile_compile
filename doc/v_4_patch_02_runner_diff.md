# PATCH 02 – Umbau tile_compile_runner.py (v4-only)

## Entfernen (vollständig)

```python
from runner.phases import ...
from runner.phases_impl import ...
from runner.opencv_registration import ...
```

Alle Phasen-Dispatches und globale Registrierungslogik löschen.

---

## Neuer Kernablauf (ersetzen)

```python
from runner.tile_processor_v4 import TileProcessor

frames = load_frames(cfg)
frames = global_coarse_normalize(frames, cfg)

tiles = build_initial_tile_grid(frames[0].shape, cfg)

results = []
valid_tiles = 0

for tid, bbox in enumerate(tiles):
    tp = TileProcessor(
        tile_id=tid,
        bbox=bbox,
        frames=frames,
        global_weights=global_weights,
        cfg=cfg,
    )
    tile = tp.run()
    if tile is None:
        continue
    valid_tiles += 1
    results.append((bbox, tile))

if valid_tiles < 0.3 * len(tiles):
    raise RuntimeError("Too few valid tiles – aborting (v4)")

final = overlap_add(results)
```

---

## Konsequenzen

- Keine registrierten Frames mehr
- Registrierung ist implizit im TileProcessor
- Pipeline ist Tile-zentriert

