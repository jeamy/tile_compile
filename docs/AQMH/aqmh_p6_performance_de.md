# P6 — Leistungsengpässe: Analyse, Lösungen, Umsetzung

Status: **in Umsetzung.** Ziel: gesamte `reconstruct`-Kette **< 30 min** bei
600 Frames, 3840×2160 OSC, `internal_scale=2` / `output_scale=1`. Alles darüber
gilt als Fehlschlag (Benutzervorgabe 2026-09-09).

Erstellt 2026-09-09. Protokollverweis: **§30.72**. Setzt auf `aqmh_p6_runbook_de.md`
(§30.70) auf.

---

## 1. Gemessener Ausgangszustand (realer 600-Frame-Lauf, M31 affin)

| Phase | Gemessen | Anmerkung |
|---|--:|---|
| SAMPLING_GEOMETRY | **2 h 21 min** (~8460 s) | seriell |
| SOURCE_QUALITY_MAPS | **48 min 35 s** (~2915 s) | seriell |
| GLOBAL_QUALITY | 4 min 26 s (~266 s) | |
| FORWARD_DRIZZLE | läuft (>47 min) | wiederholtes Voll-Bild-Lesen + Re-Hash |
| MULTIBAND / Ausgabe / BGE / PCC / HMS | nicht erreicht | — |

### 1.1 SAMPLING_GEOMETRY — Attribution durch Messung

Hidden-Benchmark `tests/test_coverage_hotspot_profile.cpp` (`[.][coverage-profile]`),
affine OSC-Szene, `geomstats`-`ScopedGeometryTimer` auf beiden `rasterize_drizzle_stripe`-Aufrufen
gegen die Gesamt-Wanduhr:

```
scene: src 1280x720  scale 2  internal 2688x1568  frames 16  chunk 256  stripes 7
TOTAL coverage wall_s             : 25.410
  geometry (2x rasterize)        : 24.216  (95.3 %)
  everything else                :  1.194  ( 4.7 %)
cfa  rasterize : 694 ns / source-sample
foot rasterize : 915 ns / source-sample   (pixfrac 1.0 → größere Tropfen)
```

- **95 % ist reine Polygon-Geometrie** in `rasterize_drizzle_stripe`:
  `sample_leaves` → `build_affine_leaf` → pro Zelle `polygon_rectangle_intersection_area`
  (4 Halbebenen-Clips + Shoelace). Zwei Durchläufe pro Frame (CFA-Tropfen
  pixfrac 0,8 + dichter Footprint pixfrac 1,0), alle 600 Frames, **einthreadig**.
- **Kein K-Faktor.** Der affine Pfad begrenzt den Quell-Scan pro Streifen auf das
  rückprojizierte Zeilenband (`forward_drizzle.cpp:492-512`);
  `source_samples_visited` ≈ 1,02 × Quellpixel über alle Streifen, nicht 19 ×.
  Die frühere Cache-Bau-Zuordnung war falsch — M31 hat 0 lokale Modelle, der
  parallele lokale Geometrie-Cache wird komplett übersprungen.
- **Extrapolation:** 8,29 Mpx × 600 f × 1,02 × (694 + 915) ns ≈ **8100 s** — deckt
  sich mit den gemessenen 8460 s.
- `(void)num_workers;` in `compute_geometric_coverage` (`sampling_geometry.cpp:275`)
  — Parallelität war **nie verdrahtet**.

### 1.2 SOURCE_QUALITY_MAPS — seriell

`build_source_quality_map_cache` (`source_quality_map_cache.cpp:535`):
`for (const auto &f : plan.frames)` mit `cache.load()` + `compute_source_quality_proxy_v1`
+ `compute_source_quality_maps` + `writer.put(...)`. **Kein OpenMP, kein Thread.**
2915 s / 600 f ≈ 4,86 s/f. Der Per-Frame-Rumpf ist unabhängig; nur `writer.put`
braucht Serialisierung.

### 1.3 FORWARD_DRIZZLE — wiederholtes Voll-Bild-Lesen + Re-Hash

`VerifiedNormalizedSourceCache::load` (`normalized_source_cache.cpp:112`) liest bei
**jedem** Aufruf die vollständige `.raw` (33 MB bei 3840×2160 float) **und** rechnet
SHA-256 neu. Der Streifenpfad ruft das pro (Streifen, Frame) auf:
19 × 600 = 11 400 Loads ≈ **376 GB** gelesen + 11 400 × SHA-256 über 33 MB.
GDB-Stackprobe des laufenden Prozesses: Hauptthread wartend in `load`, aufgerufen
aus dem CUDA-Streifenpfad; 236 GB physisch gelesen. SQM trifft dieselbe Funktion,
dort 600 Loads ≈ 20 GB.

---

## 2. Lösungen

Reihenfolge nach Hebel × (1 / Risiko). Jede Stufe einzeln getestet und gemessen.

### O1 — Streifen-parallele `compute_geometric_coverage`  · größter Hebel

Der Streifen-Rumpf (`sampling_geometry.cpp:347-421`) ist bereits fast vollständig
streifen-lokal (`B/w/w2/count/support/footprint_count/touched` sind Streifenpuffer,
`rasterize_drizzle_stripe` ist read-only auf `plan`/`frames`, Quell-Scan streifen-
begrenzt, Maskenschreibzugriffe disjunkt über `offset = y*width`).

- **Achse:** ganze Streifen an Worker (`#pragma omp parallel for schedule(dynamic,1) if(workers>1)`
  über den Streifenindex), Muster wie §30.65/§30.67. Jeder Worker: eigene
  `B/w/w2/count/support/footprint_count/touched` (Allokation in die Schleife
  ziehen), eigene `DiskQuantile` je Kanal.
- **`w[c][i] += B[c][i]`** bleibt pro `i` in Frame-Reihenfolge (ein Worker pro
  Streifen) → **bit-identisch zu `workers==1`**.
- **`DiskQuantile`** je Worker; Zusammenführung durch **Spool-Konkatenation** vor
  `p10()`. Radix-Select ist multiset- und damit reihenfolgeunabhängig → bit-exakt.
- **`StripeHoles`** ist ein sequentieller Connected-Components-Akkumulator über die
  volle Höhe → **aus der Streifenschleife herausnehmen**. Die drei
  Kanal-`support`-Ebenen kanvasweit persistieren (disjunkt im Parallelbereich
  füllen), danach ein serieller `holes[c]->row()`-Durchlauf über den fertigen
  Kanvas. Bit-identisch (gleiche Zeilenreihenfolge, gleiche Daten).
- **`gate.analysis_pixels` / `supported[c]`** je Streifen sammeln, nach dem
  Parallelbereich in Streifenreihenfolge reduzieren (Integer → reihenfolgefrei).
- **`geomstats`** im Parallelbereich deaktivieren wenn `workers>1` (RAII-Restore),
  wie §30.67. Bei `workers==1` (Default) unverändert.
- **Ausnahmen:** per-Worker `try` + `std::exception_ptr` + `#pragma omp critical`.
- **Budget (kritisch):** `memory.rows` **weiterhin aus dem `workers==1`-Budget**
  auflösen — Worker-Zahl darf die Chunk-Höhe **nicht** verändern
  (`resolved_chunk_rows` geht in den `geomstats`-Kontext und in
  `sampling_geometry.json`; die P0-Compute-Invarianz-Tests prüfen
  Chunk-Höhen-Unabhängigkeit). Separat prüfen, dass `W × per_stripe_bytes` +
  die drei persistenten `support`-Ebenen ins Budget passen; wenn nicht, **W
  reduzieren** (nicht `rows`). Die persistenten `support`-Ebenen in
  `retained_bytes` des `plan_drizzle_memory`-Aufrufs aufnehmen.
- **Worker-Zahl:** vom Runner durchgereicht (`parallel_workers ∩ hardware_concurrency`),
  `TC_SAMPLING_GEOMETRY_WORKERS` überschreibt (Spiegel von
  `TC_GEOMETRY_CACHE_WORKERS` / `TC_FORWARD_DRIZZLE_WORKERS`). `gate.workers_used`
  füllen (bereits im Struct, wird in `sampling_geometry.json:504` serialisiert).
- **Erwartung:** 19 Streifen, 16 Worker, dynamisch → ~13–16 ×. 8460 s → **~550–650 s**.
- **Gate:** bestehende Compute-Invarianz-Tests + neuer `[geometry-parallel]`-Fall
  W ∈ {1,2,4}: Masken, `gate`-Felder, `n_eff`-p10, `hole_area` byte-/bit-identisch
  zu W=1.

### O2 — `VerifiedNormalizedSourceCache`: LRU + Hash nur beim ersten Zugriff

- Statt eines einzigen `image_` eine **LRU-Map** `source_index → {Matrix2Df, verifiziert}`,
  Kapazität aus `memory_budget_mb` (16 GB / 33 MB ≈ ~480 von 600 Frames).
- Treffer: gecachte Matrix zurückgeben, **kein Read, kein SHA-256**.
- Fehltreffer: lesen + hashen + verifizieren + einfügen (LRU-Verdrängung).
  Den gerade zurückgegebenen Eintrag **nie** verdrängen.
- Alle `load()`-Aufrufstellen in `forward_drizzle.cpp` prüfen: kein Aufrufer hält
  eine Referenz über einen späteren `load()` eines anderen Index hinweg (§30.67
  hat streifenweite Parallelität genau wegen dieses Aliasing verworfen — mit
  Per-Index-Einträgen entfällt der Hazard, aber die Aufrufer-Prüfung bleibt Pflicht).
- **Wirkung:** Drizzle-Lesevolumen 376 GB → ~16 GB (einmal je Frame), SQM 20 GB →
  20 GB einmalig; 11 400 → 600 SHA-256-Läufe. Entkoppelt außerdem den
  Aliasing-Block für spätere Frame-Parallelität im Drizzle-Streifenpfad.
- **Gate:** bestehende `[forward-runner]` / `[cuda-parity]` Byte-Identität;
  neuer Fall: wiederholtes `load()` desselben Index gibt bit-gleiche Daten,
  Zähler „SHA-256-Aufrufe" steigt nur beim ersten Zugriff; manipulierte Datei
  nach Cache-Eintrag wird beim ersten (verifizierenden) Zugriff erkannt.

### O3 — Paralleler SQM-Bau

- `build_source_quality_map_cache`: Frame-Schleife auf `#pragma omp parallel for
  schedule(dynamic,1) if(workers>1)`. Per-Frame: `cache.load` (nach O2 threadsicher,
  da Per-Index-Einträge — sonst je Worker eine eigene `VerifiedNormalizedSourceCache`),
  `compute_source_quality_proxy_v1`, `compute_source_quality_maps` unabhängig.
- `writer.put(stream, source_index, matrix)` — Serialisierung prüfen: entweder
  `#pragma omp critical` um die `put`-Gruppe je Frame, oder je Frame in einen
  lokalen Puffer und nach dem Parallelbereich in `source_index`-Reihenfolge
  schreiben (deterministischer Store-Inhalt). `max_scales`-Reduktion.
- `writer.commit()` unverändert seriell.
- **Erwartung:** 2915 s / 16 ≈ **~200–300 s**.
- **Gate:** `source_quality_cache_hash` byte-identisch W ∈ {1,2,4}; bestehende
  `[source-quality]`-Tests.

### O4 (Reserve) — CFA- und Footprint-Durchlauf fusionieren

Ein Quell-Scan statt zwei. Beide Durchläufe scannen dieselben Quellzeilen, rufen
`sample_leaves`. Für affine Frames unterscheiden sich der pixfrac-0,8- und der
pixfrac-1,0-Tropfen nur im Halbmaß `h`. Ein gemeinsamer Scan + gemeinsame
CFA-Kanalbestimmung + gemeinsame Bbox-Iterationsstruktur; die Polygon-Flächen-
aufrufe bleiben getrennt (verschiedene Geometrie). **Erwartung ~1,3–1,5 ×.**
Höheres Bit-Exaktheits-Risiko → separates Gate, erst nach O1–O3 + Messung.

### O5 (Reserve, gesperrt) — Affine Flächen-Schablone

Der affine Tropfen ist eine formfeste Parallelogramm; nur die Translation variiert
mit `(sx,sy)`, der Schritt pro Quellpixel ist konstant (`A·[1,0]`). Eine exakt
vorberechnete Clip-Schablone gegen ein Subpixel-Offset-Gitter würde
`polygon_rectangle_intersection_area` durch Tabellen-Lookup ersetzen — **5–20 ×**
auf der Geometrie selbst. **Aber:** die Flächen speisen `w`/`w2` → `n_eff`-p10 →
Gate-Schwellen, und ein quantisiertes Gitter reproduziert eine **stetige**
Bruch-Translation nicht bit-exakt. Das ist eine Änderung der Numerik-/
Modellidentitätsklasse (wie die `expf`-Revision in §30.69) und braucht ein eigenes
Gate. **Geparkt.** Nur anfassen, wenn O1+O2+O3 die 30 min verfehlen.

---

## 3. Gemessene Beschleunigung (Referenzbox: AMD Ryzen 7 3700X, **8 Kerne** / 16 Threads)

### O1 — Streifen-parallele Coverage (Hidden-Benchmark, affine OSC 3200×1800×2, 20 Frames)

| Worker | Wanduhr | Speedup |
|--:|--:|--:|
| 1 | 205,5 s | 1,0 × |
| 4 | 56,2 s | 3,66 × |
| 8 | 34,4 s | **5,97 ×** |
| 16 (SMT) | 29,9 s | **6,87 ×** |

~6 × auf 8 physischen Kernen, ~6,9 × mit SMT. **Bit-identisch** zu W=1
(`[drizzle-audit]` „coverage audit: stripes preserve exact geometry",
W=128 vs W=1 — Maskenbytes, `n_eff`-p10, Lochflächen alle gleich).

### O2 — Source-Cache LRU + Hash-once

Kein Rechen-Speedup, aber Drizzle-Streifenpfad-Lesevolumen **376 GB → ~16 GB**
(einmal je Frame statt 19 ×), SHA-256-Läufe **11 400 → 600**. LRU-Treffer eines
auf Platte unveränderten Frames: kein Read, kein Hash (Stat auf Größe+mtime);
jede Änderung erzwingt volle Neuverifikation → Trunkierung/Rewrite scheitern
weiterhin geschlossen. Tests `[cache-lru]`.

### O3 — Paralleler SQM-Bau

Gleiches OpenMP-`schedule(dynamic,1)`-Muster wie O1 → **~6 × @ 8 Worker**
erwartet. **Byte-identisch**: `source_quality_cache_hash` + alle Datei-SHAs
gleich für Worker ∈ {1,2,3,6} (`[source-quality-parallel]`).

## 4. Erreichbarkeit der 30-min-Grenze — aktualisiert

| Phase | vorher | nach O1–O3 (8-Kern-Projektion) |
|---|--:|--:|
| SAMPLING_GEOMETRY | 8460 s | ~1410 s (O1, ~6 ×) |
| SOURCE_QUALITY_MAPS | 2915 s | ~490 s (O3, ~6 ×) |
| GLOBAL_QUALITY | 266 s | ~266 s (noch seriell) |
| FORWARD_DRIZZLE | läuft (>47 min) | schneller durch O2 (war I/O-gebunden), real ungemessen |
| MULTIBAND + Ausgabe + BGE/PCC/HMS | — | ungemessen, teils nicht im Pfad |
| **Kette ohne Drizzle/HMS** | ~11 900 s | **~2430 s ≈ 40 min** |

Referenzbox = **8 physische Kerne** (`parallel_workers: 8` in der M6-Config); die
`/N`-Tabellen im Runbook (§30.70) sind auf N=16 gerahmt — hier gilt N=8, gemessen
5,97 ×.

**Ehrlich:** O1–O3 auf 8 Kernen bringen SAMPLING_GEOMETRY auf ~24 min und die
Q-Phasen zusammen unter 15 min — die Kette liegt damit **noch über 30 min**,
bevor FORWARD_DRIZZLE/MULTIBAND/HMS zählen. Und die aufgezählten Resthebel
reichen **voraussichtlich nicht**:
- **O4** — bei genauer Betrachtung teilt die Fusion nur das Schleifengerüst
  (~1,1 ×); der echte Hebel wäre ein billigeres Footprint-Prädikat statt
  `polygon_rectangle_intersection_area`, das aber `k > 0` an Tangenten/
  Kollinearität **exakt** reproduzieren muss (speist `analysis_common_mask` →
  Coverage-Gate → Lauf-Pass/Fail). Eigenes Gate, keine Anschlussänderung.
  Optimistisch ~1,2–1,3 × → SAMPLING_GEOMETRY ~1100 s.
- **GLOBAL_QUALITY parallelisieren** (~6 ×, `ref_star_count`-Abhängigkeit: Frame 0
  seriell, Rest parallel) → ~45 s.
- **FORWARD_DRIZZLE nach O2 messen** (war I/O-gebunden).
- **O5** (affine Flächen-Schablone) — Numerik-/Modellidentitätsklasse, gesperrt.

Selbst mit O4 optimistisch und parallelem GLOBAL_QUALITY landet die Kette bei
~1200–1400 s **vor** FORWARD_DRIZZLE + MULTIBAND + dem noch nicht vollständig
im Pfad gemessenen BGE/PCC/HMS-Block. **Die 30-min-Grenze ist aus O1–O5 allein
sehr wahrscheinlich nicht erreichbar** — es braucht zusätzlich O5 (bit-exakt
abgesichert) oder einen grundlegend anderen Geometrie-Algorithmus für den
affinen Coverage-Pfad.

## 5. Fortschritt

- [x] **O1 — Streifen-parallele Coverage** — gemessen ~6 ×/8 Kerne, bit-identisch
- [x] **O2 — Source-Cache LRU + Hash-once** — Drizzle-Lesevolumen 376 GB → ~16 GB
- [x] **O3 — Paralleler SQM-Bau** — byte-identisch W ∈ {1,2,3,6}
- [ ] O4 — CFA+Footprint-Durchläufe fusionieren
- [ ] GLOBAL_QUALITY parallelisieren
- [ ] Realer/halbrealer Messlauf, Phasenbudget final
- [ ] O5 (gesperrt) nach Bedarf
