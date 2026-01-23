# tile_compile – v4-spezifische Tests & Diagnose-Artefakte

**Status:** verbindlich für Methodik v4  
**Ziel:** Ersetzt alle früheren Registrierungstests und globalen Diagnoseplots

---

## Teil A – v4-spezifische Tests

### A.1 Grundsatz

Alle Tests prüfen **lokale Konsistenz**, nicht globale Geometrie.

> Ein Test ist bestanden, wenn **lokale Rekonstruktion stabil**, auch wenn globale Registrierung unmöglich ist.

---

### A.2 Zu entfernende / zu deaktivierende Tests

Diese Tests sind **methodisch falsch** unter v4:

```
 tests/test_registration.py
 tests/test_methodik_v3_conformance.py
 validation/methodik_v3_compliance.py
```

➡️ Markieren als `xfail` oder verschieben nach `legacy_tests/`.

---

### A.3 Neuer Test: Lokale Warp-Stabilität

**Datei:** `tests/test_v4_tile_warp_stability.py`

```python
import numpy as np
from runner.tile_processor_v4 import smooth_warps_translation


def test_warp_temporal_smoothing():
    # synthetic jitter
    warps = []
    for i in range(20):
        w = np.array([[1,0,i%3],[0,1,(i%3)-1]],dtype=np.float32)
        warps.append(w)

    smoothed = smooth_warps_translation(warps, window=5)

    xs_raw = [w[0,2] for w in warps]
    xs_smooth = [w[0,2] for w in smoothed]

    assert np.var(xs_smooth) < np.var(xs_raw)
```

---

### A.4 Neuer Test: Tile-Validität

**Datei:** `tests/test_v4_tile_validity.py`

```python
import numpy as np
from runner.tile_processor_v4 import TileProcessor

class DummyCfg:
    class v4:
        iterations = 1
        beta = 5.0
    class registration:
        class local_tiles:
            ecc_cc_min = 0.9
            min_valid_frames = 5


def test_tile_invalid_when_too_few_frames():
    frames = [np.zeros((32,32)) for _ in range(3)]
    weights = [1.0]*3
    tp = TileProcessor(0,(0,0,32,32),frames,weights,DummyCfg())
    out = tp.run()
    assert out is None
```

---

### A.5 Neuer Test: Iterative Referenzkonvergenz

**Datei:** `tests/test_v4_reference_convergence.py`

```python
import numpy as np
from runner.tile_processor_v4 import TileProcessor

class DummyCfg:
    class v4:
        iterations = 3
        beta = 1.0
    class registration:
        class local_tiles:
            ecc_cc_min = 0.1
            min_valid_frames = 5


def test_reference_stabilizes():
    base = np.random.randn(32,32)
    frames = [base + 0.01*np.random.randn(32,32) for _ in range(20)]
    weights = [1.0]*20

    tp = TileProcessor(0,(0,0,32,32),frames,weights,DummyCfg())
    ref = tp.run()
    assert ref is not None
    assert np.std(ref - base) < np.std(frames[0] - base)
```

---

## Teil B – Diagnose-Artefakte (v4)

### B.1 Warp-Feld-Visualisierung

**Ziel:** Sichtbarmachen lokaler Bewegungsmodelle

**Artefakt:** `warp_field_<channel>.png`

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_warp_field(tiles, warps, shape, out):
    h, w = shape
    xs, ys, us, vs = [], [], [], []
    for (x0,y0,tw,th), wlist in zip(tiles, warps):
        if not wlist:
            continue
        dx = np.median([w[0,2] for w in wlist])
        dy = np.median([w[1,2] for w in wlist])
        xs.append(x0+tw/2)
        ys.append(y0+th/2)
        us.append(dx)
        vs.append(dy)

    plt.figure(figsize=(8,6))
    plt.quiver(xs, ys, us, vs, angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.title('Local Warp Field')
    plt.savefig(out)
    plt.close()
```

---

### B.2 Tile-Invalid-Map

**Artefakt:** `tile_invalid_map.png`

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_tile_invalid_map(tiles, valid_flags, shape, out):
    img = np.zeros(shape)
    for (x0,y0,w,h), ok in zip(tiles, valid_flags):
        if not ok:
            img[y0:y0+h, x0:x0+w] = 1

    plt.imshow(img, cmap='hot')
    plt.title('Invalid Tile Map')
    plt.colorbar()
    plt.savefig(out)
    plt.close()
```

---

### B.3 Pflicht-Artefakte pro Run

| Artefakt | Zweck |
|--------|------|
| `warp_field.png` | lokale Bewegungskohärenz |
| `tile_invalid_map.png` | Abbruch- & Qualitätsdiagnose |
| `warp_variance_hist.png` | Modellstabilität |

---

## Kernaussage

Tests und Diagnose sind jetzt **lokal, v4-konform und experimentell aussagekräftig**.

Globale Registrierungstests sind bewusst entfernt – sie testen ein Modell, das nicht mehr existiert.

---

**Ende der v4-Test- & Diagnose-Spezifikation**

