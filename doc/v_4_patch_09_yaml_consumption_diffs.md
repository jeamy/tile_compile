# PATCH 09 – Code-Diffs: Konsum der v4-YAML-Parameter

Dieses Dokument zeigt **konkrete, umsetzbare Änderungen**, damit der Code exakt die v4-YAML-Parameter nutzt.

---

## 1. `StreamingTileProcessor` – Konvergenzabbruch

**Datei:** `runner/tile_processor_v4.py`

### Neu in `run()`:

```python
prev_ref = None

for it in range(self.cfg.v4.iterations):
    ...
    ref = new_ref

    if self.cfg.v4.convergence.enabled and prev_ref is not None:
        diff = np.linalg.norm(ref - prev_ref)
        norm = np.linalg.norm(prev_ref) + 1e-12
        if diff / norm < self.cfg.v4.convergence.epsilon_rel:
            break

    prev_ref = ref
```

---

## 2. Adaptive Tiles – Nutzung der YAML-Werte

**Datei:** `v4_patch_07_adaptive_tiles_runner.py`

Ersetzen:
```python
max_refine_passes = cfg.v4.max_refine_passes
```

durch:
```python
max_refine_passes = cfg.v4.adaptive_tiles.max_refine_passes
threshold = cfg.v4.adaptive_tiles.refine_variance_threshold
```

und beim Aufruf:
```python
tiles = refine_tiles(tiles, warp_variances, threshold)
```

---

## 3. Speicherlimits – Laufzeitüberwachung

**Datei:** `tile_compile_runner.py`

```python
import psutil, os
proc = psutil.Process(os.getpid())

rss_mb = proc.memory_info().rss / 1e6
if rss_mb > cfg.v4.memory_limits.rss_abort_mb:
    raise RuntimeError("RSS limit exceeded (v4)")
elif rss_mb > cfg.v4.memory_limits.rss_warn_mb:
    log.warning(f"High RSS usage: {rss_mb:.0f} MB")
```

---

## 4. Diagnose-Artefakte – YAML-gesteuert

**Ort:** nach Runner-Hauptloop

```python
if cfg.v4.diagnostics.enabled:
    if cfg.v4.diagnostics.warp_field:
        save_warp_field(...)
    if cfg.v4.diagnostics.tile_invalid_map:
        save_tile_invalid_map(...)
```

---

## Ergebnis

Nach diesen Diffs:
- ist jede v4-YAML-Option **wirksam**
- existieren keine toten Konfigurationsparameter
- ist Verhalten reproduzierbar und dokumentiert

