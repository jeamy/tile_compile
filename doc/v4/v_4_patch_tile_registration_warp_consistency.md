# PATCH – Warp-Konsistenzprüfung & Reject (Methodik v4)

Dieses Dokument enthält einen **konkreten, produktionsreifen Code-Patch** für `tile_local_registration_v4.py`, der **Doppelsterne zuverlässig verhindert**.

Der Patch ist **zwingend**, sobald:
- wenige Frames (z. B. 3–10)
- geringe SNR
- CFA-aware oder Green-Proxy-Registrierung

verwendet werden.

---

## Ziel

> **Pro Tile dürfen nur geometrisch konsistente Warps akkumuliert werden.**

Frames mit abweichender Registrierung werden **lokal verworfen**, nicht global.

---

## 1. Problemstelle (Status quo, sinngemäß)

```python
for frame in frames:
    warp, cc = register_tile(frame, ref)
    warps.append(warp)
    tiles.append(warped_frame)
```

➡️ **Fehler:** Jeder Warp wird akzeptiert, auch wenn er inkonsistent ist.

---

## 2. Verbindlicher Patch: Warp-Konsistenzprüfung

### Einfügen **nach** der Registrierung aller Frames eines Tiles

```python
import numpy as np

# --- Sammle Translationen
translations = np.array([
    (w[0, 2], w[1, 2]) for w in warps
], dtype=np.float32)

# --- Median-Warp als robuste Referenz
median_shift = np.median(translations, axis=0)

# --- Abweichung pro Frame
deltas = np.linalg.norm(translations - median_shift[None, :], axis=1)

# --- harte Schwelle (v4-Empfehlung)
MAX_DELTA_PX = cfg.registration.local_tiles.max_warp_delta_px  # z. B. 0.3

valid_mask = deltas <= MAX_DELTA_PX

# --- Reject inkonsistenter Frames
warps = [w for w, ok in zip(warps, valid_mask) if ok]
warped_tiles = [t for t, ok in zip(warped_tiles, valid_mask) if ok]
ccs = [c for c, ok in zip(ccs, valid_mask) if ok]

# --- Mindestanzahl prüfen
if len(warps) < cfg.registration.local_tiles.min_valid_frames:
    return None, None
```

---

## 3. Wichtige Parameter (YAML)

```yaml
registration:
  local_tiles:
    max_warp_delta_px: 0.3   # strikt, subpixel
    min_valid_frames: 2
```

Empfehlungen:
- 0.2–0.4 px für Translation
- bei sehr gutem SNR sogar 0.15 px

---

## 4. Warum das Doppelsterne verhindert

- falsche lokale Maxima werden verworfen
- nur ein geometrisch konsistenter Shift bleibt
- bei wenigen Frames bleibt ggf. nur 1–2 gültige

➡️ **Das ist korrekt und gewollt.**

---

## 5. v4-Konformität

- keine globale Registrierung
- keine neue Geometrieannahme
- rein lokale Entscheidung
- deterministisch

---

**Ende PATCH 1 – Warp-Konsistenzprüfung**

