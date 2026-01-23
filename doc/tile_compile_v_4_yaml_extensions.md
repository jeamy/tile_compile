# tile\_compile.yaml – Erweiterungen für Methodik v4 (verbindlich)

# Diese Sektion ist ausschließlich für den v4-Pfad gültig.

# Alle anderen Registrierungsoptionen werden ignoriert.

v4:

# ---------------------------------------------------------------------------

# Iterative lokale Rekonstruktion

# ---------------------------------------------------------------------------

# Anzahl der Referenz-Iterationen pro Tile

# Typisch: 2–4, Default = 3

iterations: 3

# Gewichtung der Registrierungsgüte (ECC cc)

# R\_{f,t} = exp(beta \* (cc - 1))

# Höher = stärkere Abwertung schlechter lokaler Registrierungen

beta: 5.0

# Optional: Abbruch, wenn sich die Referenz kaum noch ändert

# (L2-Norm der Differenz relativ zur Referenz)

convergence: enabled: true epsilon\_rel: 1.0e-3

# ---------------------------------------------------------------------------

# Adaptive Tile-Verfeinerung (zwingend für v4)

# ---------------------------------------------------------------------------

adaptive\_tiles: enabled: true

```
# Maximale Anzahl von Refinement-Pässen
# 0 = keine Verfeinerung
max_refine_passes: 2

# Schwelle für Warp-Varianz, ab der ein Tile gesplittet wird
# Var(dx) + Var(dy)
refine_variance_threshold: 0.25

# Minimale Tile-Größe (px), unterhalb wird nicht weiter gesplittet
min_tile_size_px: 64
```

# ---------------------------------------------------------------------------

# Speicher- & Laufzeitschutz (v4-spezifisch)

# ---------------------------------------------------------------------------

memory\_limits: # Weiches Limit für RSS (MB), nur Diagnose rss\_warn\_mb: 4096

```
# Hartes Limit (MB), Lauf wird abgebrochen
rss_abort_mb: 8192
```

# ---------------------------------------------------------------------------

# Diagnose-Artefakte (v4-Pflicht)

# ---------------------------------------------------------------------------

diagnostics: enabled: true

```
# Warp-Vektorfeld pro Kanal speichern
warp_field: true

# Karte ungültiger Tiles
tile_invalid_map: true

# Histogramm der Warp-Varianzen
warp_variance_hist: true
```

# -----------------------------------------------------------------------------

# Registrierung – v4 erzwingt tile-lokal

# -----------------------------------------------------------------------------

registration: mode: local\_tiles

local\_tiles: # minimale ECC-Korrelation für gültige lokale Registrierung ecc\_cc\_min: 0.2

```
# Mindestanzahl gültiger Frames pro Tile
min_valid_frames: 10
```

# -----------------------------------------------------------------------------

# Hinweis:

# - Globale Registrierung wird vollständig ignoriert

# - Alle v4-Parameter werden ausschließlich vom v4-Runner ausgewertet

# - Fehlt die v4-Sektion, darf der Lauf nicht starten

