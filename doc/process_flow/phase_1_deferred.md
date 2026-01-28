# Phase 1–2: Deferred Channel Split & Normalization (C++)

## Überblick

In v4 werden **Channel‑Split** und **Normalisierung** nicht als separate Phasen ausgeführt, sondern **on‑the‑fly** beim Laden der Tiles. Das reduziert Speicherbedarf und vermeidet globale Vorverarbeitung.

## Ablauf (C++)

**Referenzen:**
- `tile_compile_cpp/apps/runner_main.cpp`
- `tile_compile_cpp/src/io/*`
- `tile_compile_cpp/src/image/*`

1. **Tile‑Load**
   - Für jedes Tile wird die Region aus dem Frame gelesen.
   - Bei Bedarf `ROI` oder `LRU`‑Cache (Phase‑6‑IO‑Modus).

2. **Normalization (background/median)**
   - Skalenfaktoren werden pro Frame berechnet.
   - Normalisierung wird **direkt** auf das Tile angewandt (`apply_normalization_inplace`).

3. **Channel‑Split / CFA‑Handling**
   - OSC‑Frames bleiben **CFA‑Mosaik** bis zum Warp.
   - Für Registration wird ein **Green‑Proxy** verwendet (CFA‑Downsample).

## C++‑Skizze

```cpp
Matrix2Df tile = read_tile(frame, bbox);
apply_normalization_inplace(tile, norm_scales[fi], detected_mode, bayer, rx0, ry0);
Matrix2Df reg = (OSC) ? cfa_green_proxy_downsample2x2(tile, bayer) : tile;
```

## Wichtige Parameter

- `normalization.enabled`, `normalization.mode`, `normalization.per_channel`
- `registration.mode = local_tiles`
- `data.color_mode`, `data.bayer_pattern`
- `v4.phase6_io.mode` (`roi`/`lru`/`full`)

## Konsequenzen

- Kein globaler Normalisierungs‑Schritt mehr.
- Keine globale Debayer‑Phase.
- Weniger RAM, bessere Skalierbarkeit.

