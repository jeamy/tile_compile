# PATCH 06 – Produktionsreife Speicher- & I/O-Architektur (Methodik v4)

## Ziel

Garantierte OOM-Sicherheit bei:

- großen Sensoren
- tausenden Frames
- experimenteller Tile-Iteration

---

## Grundprinzip

> **Frames werden niemals vollständig in den RAM geladen.**

Alles erfolgt:

- tile-weise
- iterativ
- stream-basiert

---

## Architektur (konkret)

### Datenfluss pro Tile

```
Disk (FITS, memmap)
 → TileWindow (read-only)
 → lokale Registrierung
 → Warp
 → Akkumulation
 → Referenz
```

Maximaler Speicherbedarf:

```
O(tile_width × tile_height × (iterations + buffers))
```

Unabhängig von Frame-Anzahl.

---

## Empfohlene Implementierung (verbindlich)

### 1. FITS-Streaming

- `astropy.io.fits.open(..., memmap=True)`
- niemals `hdul[0].data.copy()`

### 2. TileProcessor-Signatur

```python
StreamingTileProcessor(
    tile_id: int,
    bbox: (x0,y0,w,h),
    frame_paths: list[str],
    global_weights: list[float],
    cfg
)
```

### 3. Overlap-Add (Streaming)

- Zielbild vorab allozieren
- pro Tile:
  - Ergebnis direkt addieren
  - Gewichtsmatrix parallel pflegen

Kein temporäres Full-Frame-Array.

---

## Typische Fehler (verboten)

❌ `frames = load_all_frames()` ❌ `np.stack(all_frames)` ❌ globale Normalisierung nach Tile-Schritt

---

## Validierung

Während Lauf speichern:

- Peak-RAM-Nutzung
- Anzahl gleichzeitig offener FITS

Abbruch, wenn:

- RAM > konfiguriertes Limit

---

## Ergebnis

Mit dieser Architektur ist `tile_compile`:

- deterministisch im Speicherverbrauch
- skalierbar
- produktionsreif für Methodik v4

---

**Ende PATCH 06**

