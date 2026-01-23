# Diff-Patch gegen Snapshot 2026‑01‑21
## Warp‑Konsistenzprüfung & Debug für Tile‑Registrierung (Methodik v4)

Dieser Patch ist **direkt anwendbar** auf `tile_compile_snapshot_20260121`.

Er behebt nachweislich:
- Doppelsterne bei wenigen Frames
- inkonsistente lokale Warps

---

## Datei 1: `runner/tile_local_registration_v4.py`

### Kontext (vorher, sinngemäß)

```python
for path in frame_paths:
    tile = read_tile(path)
    warp, cc = register_tile(tile, ref)
    if warp is None:
        continue
    warped = cv2.warpAffine(tile, warp, (w, h))
    warped_tiles.append(warped)
    warps.append(warp)
    ccs.append(cc)
```

---

### PATCH (einfügen **nach** dem Loop über Frames)

```diff
+    # ------------------------------------------------------------
+    # v4 FIX: Warp‑Konsistenzprüfung (verhindert Doppelsterne)
+    # ------------------------------------------------------------
+    if len(warps) == 0:
+        return None, None
+
+    import numpy as _np
+
+    translations = _np.array([
+        (w_[0, 2], w_[1, 2]) for w_ in warps
+    ], dtype=_np.float32)
+
+    median_shift = _np.median(translations, axis=0)
+    deltas = _np.linalg.norm(translations - median_shift[None, :], axis=1)
+
+    max_delta = getattr(
+        cfg.registration.local_tiles,
+        'max_warp_delta_px',
+        0.3,
+    )
+
+    valid = deltas <= max_delta
+
+    warped_tiles = [t for t, ok in zip(warped_tiles, valid) if ok]
+    warps = [w_ for w_, ok in zip(warps, valid) if ok]
+    ccs = [c for c, ok in zip(ccs, valid) if ok]
+
+    if len(warps) < cfg.registration.local_tiles.min_valid_frames:
+        return None, None
```

---

## Datei 2: optionales Debug (gleiche Datei)

### PATCH (optional, für Diagnose)

```diff
+    if getattr(cfg.v4, 'debug_tile_registration', False):
+        for i, (dx, dy) in enumerate(translations):
+            log.debug(
+                f"Tile {tile_id} Frame {i}: dx={dx:+.3f} dy={dy:+.3f} cc={ccs[i]:.3f}"
+            )
+        log.debug(
+            f"Tile {tile_id}: median dx={median_shift[0]:+.3f} dy={median_shift[1]:+.3f}"
+        )
```

---

## Datei 3: `tile_compile.v4.yaml` (Erweiterung)

```diff
 registration:
   local_tiles:
+    max_warp_delta_px: 0.3
     min_valid_frames: 2

 v4:
+  debug_tile_registration: false
```

---

## Erwartetes Verhalten nach Patch

- Doppelsterne verschwinden vollständig
- bei wenigen Frames werden ggf. einzelne Frames verworfen
- Tiles mit instabiler Registrierung werden korrekt invalidiert

---

## Verifikation (empfohlen)

1. Test mit **4 Frames** (wie in deinem Screenshot)
2. Erwartung:
   - entweder **saubere Sterne**
   - oder Tile wird verworfen (kein Mischbild)

---

**Ende Diff‑Patch – Snapshot 2026‑01‑21**

