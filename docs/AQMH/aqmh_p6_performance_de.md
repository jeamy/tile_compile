# P6 — Leistungsengpässe: Analyse, Lösungen, Umsetzung

Status: **Analyse und Lösungsentwurf; Produktionsabnahme offen.** Aktuelles Ziel:
`reconstruct` vom Kaltstart bis zur committed Rekonstruktionsausgabe **< 30 min**,
**ohne Astrometrie, BGE, PCC und HMS**, bei
600 Frames, 3840×2160 OSC, `internal_scale=2` / `output_scale=1`. Alles darüber
gilt als Fehlschlag (Benutzervorgabe 2026-09-09).

Erstellt 2026-09-09. Protokollverweis: **§30.72**. Setzt auf `aqmh_p6_runbook_de.md`
(§30.70) auf.

**Review 2026-09-09, §30.73:** Die bisherigen O2-Durchsatzaussagen sind
korrigiert. Abschnitt 6 ist der aktuelle Lösungsentwurf: CUDA-Kandidatenbudget,
echte zweidimensionale Bereichszugriffe, exakte Clip-Wiederverwendung und
Kalibrierungs-/Qualitätsarbeit. O1–O3 allein belegen die Zielzeit nicht.
Das ursprüngliche P6-Gate bis HMS bleibt ein separates, weiter offenes Gate.

**Review 2026-09-10, §30.77 (Konsolidierung und Korrekturen):** Der
maßgebliche Prioritätenstand ist Plan §0.2; die Schnittliste in Abschnitt 6.6
ist in der Reihenfolge revidiert. Vier Review-Korrekturen sind eingearbeitet:

1. Kettenbilanzen und Realraster-Hochrechnungen sind **Projektionen, keine
   Istzustände** — insbesondere ist die FD-Angabe eine Frame-lineare
   Hochrechnung aus 20 Frames.
2. „Irreduzibler Clip" ist **zurückgenommen**. Die Verwerfung des
   k=1,0-Shortcuts bleibt als Versuchsergebnis stehen; ihre ursprüngliche
   Begründung über die Nullflächenzählung trägt jedoch nicht — Fläche == 0
   sagt nichts über Fläche == 1 aus, und die maßgebliche Häufigkeit
   vollständig überdeckter Zellen wurde nicht erhoben (die ~1,6–3
   Teilüberlappungs-Clips pro Quellpixel deuten auf deren Seltenheit,
   beweisen sie nicht). Bit-exakte Hebel (R3, No-op-Clip-Elision) bleiben
   offen.
3. Das 180-s-FD-Feld in §6.6 ist ein **Entwicklungsbudget**, kein implizit
   abgeleitetes CUDA-Budget; die Wirkung von R1/R2 muss gemessen werden.
4. Der Scope umfasst affine **und** lokale Datensätze; daraus folgt
   **keine** `expf`-Freigabe — erst getrennte Messung (Cachebau, Replay,
   Clip, I/O), dann Entscheidung.

Bench-Defekt der Pre-Repair-Fassung (`test_forward_drizzle_hotspot_
profile.cpp`): die isolierten Gather-/Reduce-Sektionen verarbeiteten keine
Kandidaten, weil die vorangehende Fill-Sektion A/B nullte (`Bv > 0` griff
nie, `cnt` blieb 0). **Repariert in §30.78** (Per-Frame-Ordnung wie
Produktion, Assertions, Outlier-belebtes Clipping, bit-exakter Referenz-
abgleich, Hoist-Timer, isolierter Sweep, echte Besuchszähler). Ergebnis:
die Raster:Reduce-Aufteilung schließt jetzt mit den
`TC_FD_PROFILE`-Produktionstimern (Modell 34,66/10,44 s vs Profil
34,76/10,18 s bei W=1, < 3 %) und ist damit belegt (~77 : 23 in der
Bench-Szene); der Clip-Anteil **am Raster** bleibt eine mit diesem Vorbehalt
gekennzeichnete **Differenzschätzung**.

---

## 1. Gemessener Ausgangszustand (realer 600-Frame-Lauf, M31 affin)

| Phase | Gemessen | Anmerkung |
|---|--:|---|
| SAMPLING_GEOMETRY | **2 h 21 min** (~8460 s) | seriell |
| SOURCE_QUALITY_MAPS | **48 min 35 s** (~2915 s) | seriell |
| GLOBAL_QUALITY | 4 min 26 s (~266 s) | |
| FORWARD_DRIZZLE | vor dem benutzerseitigen Stopp >47 min beobachtet | kein abgeschlossener Phasenwert |
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

- **95 % entfallen auf die beiden Rasterizer-Aufrufe** in `rasterize_drizzle_stripe`:
  `sample_leaves` → `build_affine_leaf` → pro Zelle `polygon_rectangle_intersection_area`
  (4 Halbebenen-Clips + Shoelace). Zwei Durchläufe pro Frame (CFA-Tropfen
  pixfrac 0,8 + dichter Footprint pixfrac 1,0), alle 600 Frames, **einthreadig**.
  Der Timer umfasst auch Leaf-Bau, Enumeration und Callbacks; er isoliert
  `polygon_rectangle_intersection_area` nicht. Die Profil-Szene verwendet nur
  Translationen, keine Rotation oder Scherung.
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
**Bei angenommenen 19 Bändern**: 19 × 600 = 11 400 Loads ≈ **378,2 GB**
logisches Lesevolumen plus Hashing. Die 19 Coverage-Streifen sind jedoch
**nicht die CUDA-Drizzle-Bandzahl**; siehe Abschnitt 6.2. Die GDB-Stackprobe
zeigte einen wartenden Hauptthread in `load` im CUDA-Pfad. Die 236 GB physischen
Reads sind ein kumulativer Prozesswert, keine isolierte Drizzle-Messung.

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
- **Überholte Idealprojektion:** 13–16 × auf 16 Workern wurde nicht erreicht;
  maßgeblich ist die Messung in Abschnitt 3 (SMT: 6,87 ×).
- **Gate:** bestehende Compute-Invarianz-Tests + neuer `[geometry-parallel]`-Fall
  W ∈ {1,2,4}: Masken, `gate`-Felder, `n_eff`-p10, `hole_area` byte-/bit-identisch
  zu W=1.

### O2 — `VerifiedNormalizedSourceCache`: LRU + Hash nur beim ersten Zugriff

- Statt eines einzigen `image_` eine **LRU-Map** `source_index → {Matrix2Df, verifiziert}`,
  Kapazität aus `memory_budget_mb` (16384 MiB: 517 von 600 Frames).
- Treffer: gecachte Matrix zurückgeben, **kein Read, kein SHA-256**.
- Fehltreffer: lesen + hashen + verifizieren + einfügen (LRU-Verdrängung).
  Den gerade zurückgegebenen Eintrag **nie** verdrängen.
- Alle `load()`-Aufrufstellen in `forward_drizzle.cpp` prüfen: kein Aufrufer hält
  eine Referenz über einen späteren `load()` eines anderen Index hinweg (§30.67
  hat streifenweite Parallelität genau wegen dieses Aliasing verworfen — mit
  Per-Index-Einträgen entfällt der Hazard, aber die Aufrufer-Prüfung bleibt Pflicht).
- **Wirkung nur bei Treffern:** Der zyklische Scan über 600 Frames mit 517
  Plätzen erzeugt ausschließlich Fehltreffer. Einmaliges Lesen/Hashen ist damit
  nicht erreicht. Ein vollständiger Source-Bestand braucht 19,91 GB = 18,54 GiB
  zusätzlich zum übrigen Arbeitsspeicher. `load()` bleibt nicht threadsicher;
  Referenzen können durch Verdrängung ungültig werden.
- **Gate:** bestehende `[forward-runner]` / `[cuda-parity]` Byte-Identität;
  neuer Fall: wiederholtes `load()` desselben Index gibt bit-gleiche Daten,
  Zähler „SHA-256-Aufrufe" steigt nur beim ersten Zugriff; manipulierte Datei
  nach Cache-Eintrag wird beim ersten (verifizierenden) Zugriff erkannt.

### O3 — Paralleler SQM-Bau

- `build_source_quality_map_cache`: Frame-Schleife auf `#pragma omp parallel for
  schedule(dynamic,1) if(workers>1)`. Per-Frame: `cache.load` mit eigener
  `VerifiedNormalizedSourceCache` je Worker (O2 ist nicht threadsicher),
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

> **Für den Footprint-Pass hinfällig geworden (§30.75):** dort speisen die
> Flächen nichts — nur „berührt ja/nein". `dense_footprint_touched_stripe`
> klassifiziert Zellen gegen das eine Frame-Parallelogramm und ist ohne
> Tabellen-Lookup byte-identisch (Randzellen exakt). 13,6× auf dem
> Footprint-Pass, 2,04× auf Coverage gesamt. Für den **CFA-Droplet-Pass**
> (Flächen → `w`/`w2` → `n_eff`) gilt die Sperre unverändert.

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

Implementiert und durch `[cache-lru]` fokussiert geprüft; **kein belegter
600-Frame-Durchsatzgewinn**. Beim wiederholten Scan 0…599 passen mit 16 GiB nur
517 Frames in den Cache: null Treffer, weiterhin Hashing auf jedem Miss.
Größe/mtime/Verifikationszeit ersetzen außerdem keinen allgemeinen
Unveränderlichkeitsnachweis (etwa bei wiederhergestellter mtime oder einer
Änderung zwischen Lesen und anschließendem Stat). Siehe Abschnitt 6.3.

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
| FORWARD_DRIZZLE | unvollständige Phase (>47 min beobachtet) | O2-Gewinn nicht belegt; CUDA-Bänder separat auflösen |
| MULTIBAND + Ausgabe + BGE/PCC/HMS | — | ungemessen, teils nicht im Pfad |
| **Nur diese drei abgeschlossenen Phasen** | ~11 641 s | **~2166 s ≈ 36 min** |

Diese Summe enthält noch keinen Scan, keine Kalibrierung, Normalisierung,
Registrierung, Zwischenzeiten und Ausgabe. Die frühere 2430-s-Summe war
rechnerisch falsch; sie war auch keine vollständige Kette.

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

Eine 30-min-Prognose ist daraus nicht ableitbar. Insbesondere ist O5 keine
zwingende Voraussetzung: Es gibt Änderungen an Speicherzugriff und exakter
Clip-Auswertung, die keine quantisierte Flächenschablone benötigen (Abschnitt 6).

## 5. Fortschritt

- [x] **O1 — Streifen-parallele Coverage** — gemessen ~6 ×/8 Kerne, bit-identisch
- [x] **O2 — Source-Cache LRU** — implementiert; Hash-Vermeidung nur bei Treffern
- [ ] O2-Produktionswirkung bei 600 Frames / begrenztem Gesamtbudget nachweisen
      (LRU-Thrash bei zyklischem 0…599, §30.73 — R1/R2 muss das ablösen)
- [x] **O3 — Paralleler SQM-Bau** — byte-identisch W ∈ {1,2,3,6}
- [x] **O3-Race-Fix** (§30.74) — `worker_error` war ungeschützt gelesen; jetzt
      `std::atomic<bool>` Fast-Path + `exception_ptr` nur unter `critical`;
      Klon-Ctor in try/catch
- [x] **R1.3 — Quellrechteck auch in X** (§30.74) — `enumerate_drizzle_stripe_leaf_cells`
      bounds Quell-X (bisher nur Y), bit-identisch, hilft Coverage + affinem Drizzle
- [x] **GLOBAL_QUALITY parallelisieren** (§30.74) — Frame 0 seriell, Rest
      `omp parallel for` auf Per-Worker-Cache-Klonen, bit-identisch
      (`[drizzle-audit][geometry-parallel]`)
- [x] **R4 — `writer.put` Datei-Write + sha256 aus dem `critical`** (§30.72
      Schritt 2) — `SourceQualityMapCacheWriter::put` intern thread-safe
      (`std::mutex files_mu_` nur um `create_directories` + `files_`;
      Downsample/Quantisierung/Write/sha256 lock-frei), `critical(sqm_writer)`
      im Build entfernt; `source_quality_cache_hash` byte-identisch
- [x] **Alternativer Coverage-Footprint-Pfad** (§30.75) —
      `dense_footprint_touched_stripe`: Zellklassifikation gegen das eine
      Parallelogramm `affine_f([0,W_src]×[0,H_src])` (innen/außen/Rand), exakter
      Per-Pixel-Fallback nur für Randzellen → **byte-identisch** zum
      Per-Pixel-Rasterize (`[drizzle-audit][footprint-fastpath]`, 500 Assert.,
      MONO+OSC × scale {1,2} × fraction {0,5;1,0} × chunk {3,16,0} × 9 Transforme).
      Gemessen: Footprint-Pass **47,6 s → 3,5 s (13,6×)**, Coverage gesamt
      **87,2 s → 42,8 s (2,04×)**; `source_samples_visited` 50,9 M → 0.
- [x] **CPU-FORWARD_DRIZZLE Puffer-Hoist** (§30.76) — `candidates`/`counts`/
      `A/B/QA*` einmal auf `memory.rows` allokiert + wiederverwendet statt pro
      Streifen neu allokiert+genullt; `candidates` nie genullt (nur `counts`-
      Präfix). alloc 10,6 → 0,4 s, 1-Worker 44 → 33 s, Band-Speedup 2,1× →
      3,7× @ 8 Kerne, bit-identisch (537/537). Bench
      `test_forward_drizzle_hotspot_profile.cpp` + `TC_FD_PROFILE`.
- [~] **Innenzellen-`k=1,0`-Shortcut für den Droplet-Rasterizer** — VERWORFEN
      mit Messung (§30.76): Droplet-Leaf ~1,6 Internal-px, 0 % Null-Flächen-
      Clips, keine Innen-/Außenzellen. Nur der Footprint-Pass (§30.75) profitiert.
- [x] **`[fd-hotspot]`-Bench repariert** (§30.78, **Priorität 1**):
      Per-Frame fill→accum→gather separat getimet, Assertions +
      Outlier-Clipping, bit-exakter Referenzabgleich gegen den
      Produktions-Volllauf (0 Mismatches über 1.015.808 Zellen × 2 Profile ×
      3 Kanäle), Hoist-Allokation separat (1,58 s), Sweep mit
      requested/budgeted/used + Chunk + Peak (used==budgeted==requested),
      echte Besuchszähler (full-cover 5,6 % der Clips — neues Datum für R3;
      Leaf-Visits/src-px ≈ Framezahl). Modell schließt mit
      `TC_FD_PROFILE` (< 3 %). **Offen aus Priorität 1: die
      167-s-Vorlauflücke** (separater Messschritt).
- [x] **`apply_robust_clipping` Per-Pixel-Heap-Allokationen ersetzt**
      (§30.79, **Priorität 2**): wiederverwendbarer budgetierter
      `DrizzleClipScratch` (`accepted`/`order`/`active`/`dev_order` +
      Alpha-`contribs`), beide Pfade (Streaming-Band-Pool + Contrib-List),
      `worker_scratch`-Formel auf tatsächlichen Bedarf umgestellt (keine
      Doppelzählung), Numerik 1:1. Messung (idle, §30.78-Szene): W=1 reduce
      10,18→8,18 s, W=8 gesamt 13,91→12,76 s, Speedup 3,44→3,53×,
      `clip_scratch_grows` = exakt Workerzahl (danach warm). Bit-identisch:
      neue `[clip-scratch]`-Parität + Bench-Referenzabgleich 0 Mismatches +
      538/538. **Erkenntnis: Allokationsanteil ≈ 20 % von reduce, ≈ 4 %
      der Phase — Deckel bleibt die Speicherbandbreite.**
- [ ] R3 — exakte X-Clip-Wiederverwendung im Rasterizer + gespiegelter
      CUDA-Device-Kernel + volle Paritätsmatrix (**separater messbarer
      Schritt nach Priorität 3/4**, nicht gebündelt)
- [x] **CUDA-Bandplanungs-Messzähler** (§30.80, Prioritäts-3-Grundlage) —
      `DrizzleCudaStoreTiming` bekommt `cand_row_bytes`/`rec_row_bytes`/
      `acc_row_bytes` (Aufschlüsselung von `bytes_per_row`) + `host_budget_bytes`
      + `band_halvings`/`min_band_rows`/`max_band_rows`;
      `run_cuda_chunked(..., CudaChunkRunStats*)` optionaler Out-Parameter;
      Runner emittiert alles in `forward_drizzle.json`. Neuer GPU-freier
      `[drizzle-store]`-Test (N-Sweep 40/100/300/600 bei Realgeometrie).
      **Befund:** `cand_row` = 93,8 %→99,6 % von `bytes_per_row`; 600 Frames
      → 7-Zeilen-Bänder, ~618 Bänder. **R1.1 allein reicht nicht** (16-GiB-
      Host-Deckel → ~19 Zeilen, ~228 Bänder). Additiv, bit-identisch, 538/538.
- [~] **2D-Kachelung + Parität + gemeinsames Host-Budget (Schritt 4) fertig;
      Q-Fold/Bereichszugriffe (3a/3b) offen** (User-Review §30.81). R1.1
      (Budget-Split) + 2D-Ziel-Kachelung
      (`W`→`tile_w`) mit **host-seitigem Record-Filtern** statt echter Geräte-
      Bereichsprovider; `[cuda-parity]` durchgehend. Bandplanung nur noch gegen
      das Geräte-Glied (`rec_row+acc_row`), das Host-`cand_row` durch
      Spaltenkacheln unter dem absoluten Deckel — 600 Frames 3840×2160×2 /
      8-GiB-Karte: **618 → ~3 Bänder** (Planungszahl; `tile_w` 10, 768 Kacheln/
      Band, Host-**Kandidaten**-Peak 1919 MiB ≤ 2-GiB-Deckel). Byte-identisch
      (`[cuda-parity]`, echtes Gerät), 542/542. **Offen:** der Deckel begrenzt
      nur den Kandidatenpuffer, nicht den Prozess-Peak; Producer + Q-Provider
      lesen weiter voll-breit; die Sortierung wiederholt sich je Kachel
      (Basis-Bench §30.81 Schritt 5: 8 Kacheln → Sort 6,95×, TOTAL 3,40×).
  - [x] **Schritt 1 (§30.81):** Ziel-Spaltenfenster `x_begin`/`cols` in
        `enumerate_drizzle_stripe_leaf_cells` + `rasterize_drizzle_stripe`
        (Default = voll-breit, byte-identisch); begrenzt Source-Scan (R2) und
        Leaf-Bbox, rebased den Sink-`index`. Bedingter `ClipCandidate`-Shrink
        als Sackgasse ausgeschlossen (`emit_fine`/`emit_alpha_confidence` im
        Produktions-`persist_forward_drizzle_multiband` fest an). Neuer
        `[fd-tile-window]`-Partitionstest, 539/539.
  - [x] **Schritt 2 (§30.81):** `target_x_begin`/`target_cols` in
        `stream_forward_drizzle_uniform_and_raw`; alle Streifen-Puffer +
        Ausgabe-`ProfilePlane` auf `win_w`, `internal_width` = `win_w`,
        `rasterize`-Call reicht das Fenster durch. `_uniform` (alter M2-Pfad)
        unberuehrt. `[fd-tile-window]`-Test: ragged Mehrband-Spaltenkacheln
        fuegen sich bit-exakt zusammen, 540/540.
  - [x] **Schritt 3 (§30.81):** `target_x_begin`/`target_cols` in
        `accumulate_pair_impl` (+ `accumulate_pair_by_frame` /
        `_cuda`): Record-Producer rastern voll-breit, Segmente mit
        `key.target_x ∉ [xb,xe)` werden host-seitig verworfen; `n` / Puffer /
        `DRIZZLE_CONTRIB_LIST_BUDGET`-Schranke auf `win_w`. Ragged-2D-
        Partitionstest, 541/541.
  - [x] **Schritt 4 (§30.81):** Kachel-Schleife in
        `persist_forward_drizzle_multiband`. `chunk_plan` nur gegen
        `device_bytes_per_row = rec_row + acc_row`; je Band
        `tile_w = host_budget / (channels·frames·64·rows)`, Kacheln in einen
        Voll-Breiten-Stripe geblittet, **ein** `sink`/`down->feed` pro Band.
        `tile_w` pro `process`-Aufruf abgeleitet → Halbierung kann es nicht
        veralten lassen, keine 2D-Leiter. Neue Timing-Felder
        `resolved_tile_w`/`min_tile_w`/`max_tiles_per_band`.
        `[drizzle-store][cuda-parity]`-Test (echtes Gerät): Multi-Kachel-Store
        byte-identisch zum Ganz-Canvas-CPU-Build.
  - [x] **Schritt 5 Basis-Bench (§30.81):** `TC_FD_CUDA_PROFILE`-gated Grob-Timer
        (`produce`/`sort`/`reduce` + `.cu` malloc/upload/kernel/download), neuer
        hidden Bench `test_forward_drizzle_cuda_tile_profile.cpp`
        (`[.][fd-cuda-tile]`). 8 Kacheln, 6 Frames: **Sort 6,95× / Geräte-Phasen
        6,4–7,8× / reduce 0,87× / TOTAL 3,40×**; Records sortiert exakt 8,00×.
        → Sortierung ist der grösste Wiederholungsfaktor.
  - [x] **Schritt 5 (B) — Per-Band-Memo (§30.81):** `accumulate_pair_impl` +
        optionaler `PairTileSink`/`tile_cols`: **ein** produce + sort je (Band,
        Frame) ins Per-Band-Memo, dann billiger Segment-Scan des Memos je
        Spaltenkachel. Store-`process` ruft einmal `accumulate_pair_by_frame_cuda
        (..., &tile_sink, tile_w)`. Bandhöhe zusätzlich durch Memo-Budget
        begrenzt. `prepare_drizzle_frames` einmal je Band statt je Kachel.
        **Bench (C-Spalte):** Sort 8,00× → **1,00×**, produce 4,05× → **0,73×**,
        Geräte-Phasen 7,88× → **1,03×**, TOTAL **0,91×** — Per-Kachel-Verstärker
        weg, Parität mit dem Ein-Aufruf-Voll-Breiten-Lauf. Bit-exakt
        (`[fd-tile-window]` GPU-frei + `[drizzle-store][cuda-parity]` echtes
        Gerät), 543/543.
  - [x] **Schritt 4 — gemeinsames Host-Budget geschlossen (§30.81, `e59ec4b4`):**
        Bandhöhe **und** Spaltenkachelbreite jetzt *gemeinsam* aus **einer**
        Decke abgeleitet: `(memo_row + stripe_row)·rows + cand/col·tile_w·rows +
        src_q_const ≤ host_budget`. Vorher teilten Memo-Deckel und
        Kachelbreiten-Deckel je die **volle** Decke, der volle Reassembly-
        `stripe` war nirgends gezählt (Summe bis ~2–3× Decke). `kMinTileW =
        min(64,W)` verwirft entartete Schmalkacheln. Nach innen gereichtes
        Budget = `host_budget − stripe − src_q` (Boden 64 MiB → sonst CPU-
        Restart). `accumulate_pair_impl`: Tiled-Zweig reserviert erst den
        Kachelanteil, begrenzt dann das Memo, Prüfung **vor** produce (laufender
        Mittelwert), Memo per `capacity()` gezählt; `reduce_window` prüft gegen
        `(budget − memo)`. **Befund:** das B-Memo ist
        `frame_count·source_width·cells` je Innenzeile und **nicht** gekachelt →
        es holt die §30.80-Kollaps-mit-N zurück: 40→600 Frames = **103→2160
        Bänder** (Peak fest an der Decke). Genau das motiviert 3a. Bit-Identität
        unberührt (nur Schwellen verschoben), 543/543, Bench C/A weiter 0,87×.
  - [ ] Schritt 3a: **Q-Fold statt Voll-Memo** — je Frame/Band Records einmal
        erzeugen + kanonisch sortieren, Q-Daten einmal bereitstellen, jedes
        Zielzellsegment in unveränderter Reihenfolge zum vollen `ClipCandidate`
        falten, Kandidaten nach Zielkachel **indizieren** + budgetiert ablegen
        (ggf. begrenzter Spool — „alle Kandidaten im RAM" = dieselbe Wand),
        dann framegeordnet ins unveränderte robuste Clipping. Keine Umordnung
        der Reduktion, nur früher berechnet. Kachel-/Segmentindex Pflicht
        (sonst wird das Memo je Kachel doch wieder ganz durchsucht). Kleiner
        Produktions-Multiband-Bench mit echten Q-Providern schon hier.
  - [ ] Schritt 3b: verbleibende Provider-/Hybrid-Arbeit — Source-/Q-Zugriffe
        auf den Band-Footprint kappen; Hybrid-Producer separat auf begrenzte
        Zwischenpuffer + wiederholte Geometriearbeit prüfen.
  - [ ] Schritt 5b: Paritätsmatrix + Bench als zusammenhängende Skalierungs-
        messung (schmale/ragged Kacheln, Rotation/Scherung, lokale Modelle,
        Modus 2/1, Budget-Retries).
- [ ] Realer/halbrealer Messlauf, Phasenbudget final

**Realraster-Hochrechnung FORWARD_DRIZZLE (affin, nach Hoist, §30.76):** 20
Frames 3840×2160×2 = 217 s / 1 Worker, 56 s / 8 Worker (3,87×) — gemessen.
Frame-linear hochgerechnet → **600 Frames ≈ 1680 s ≈ 28 min @ 8 Kerne**:
**Hochrechnung, kein 600-Frame-Istwert** — Streifenhöhe, Worker-Auslastung,
Speicherzugriffe und Clippingkosten ändern sich mit N. Der Clip
(`polygon_rectangle_intersection_area`) dominiert den Rasteranteil
(78–81 % **des Rasters** im Bench vor dem Hoist; die Raster:Reduce-Aufteilung
der Gesamtphase ist wegen des im Kopf benannten Bench-Defekts nicht belegt).
Zur Leistungsuntergrenze: der verworfene k=1,0-Shortcut bleibt als
Versuchsergebnis verworfen; seine ursprüngliche Begründung (0 % Null-Flächen)
trug nicht — Fläche == 0 sagt nichts über Fläche == 1, die Häufigkeit
vollständig überdeckter Zellen wurde nicht erhoben. Daraus folgt weiterhin
**kein** allgemeiner Irreduzibilitätsbeweis — exakte Hebel (R3,
No-op-Clip-Elision) bleiben offen. Aktuell priorisierte Schritte: Plan §0.2 / Abschnitt 6.6.
**Lokale Warps** (M42/M66) gehören zum Scope; die `expf`-Revision (§30.69)
bleibt geschlossen, bis getrennte Messung (Cachebau, Replay, Clip, I/O) einen
nachweislichen Budgetbruch zeigt.

<a id="p6-loesungsweg-30min"></a>

## 6. Code-Review: konkreter Weg zum 30-Minuten-Ziel (§30.73)

Direkte Codequellen (Stand des uncommitteten Arbeitsbaums beim Review):

| Befund | Quelle |
|---|---|
| CUDA-Budget und Bandaufrufe | [drizzle_profile_store.cpp](../../tile_compile_cpp/src/reconstruction/drizzle_profile_store.cpp) |
| CPU-Sortierung/Reduktion im CUDA-Pfad | [forward_drizzle_contrib_list.cpp](../../tile_compile_cpp/src/reconstruction/forward_drizzle_contrib_list.cpp) |
| LRU-Verdrängung und Verifikation | [normalized_source_cache.cpp](../../tile_compile_cpp/src/reconstruction/normalized_source_cache.cpp) |
| Q-Provider/Vollbildexpansion | [source_quality_artifact.cpp](../../tile_compile_cpp/src/reconstruction/source_quality_artifact.cpp) |
| Bereichslesen und SQM-Writer | [source_quality_map_cache.cpp](../../tile_compile_cpp/src/reconstruction/source_quality_map_cache.cpp) |
| Polygon-Clips und Enumeration | [forward_drizzle.cpp](../../tile_compile_cpp/src/reconstruction/forward_drizzle.cpp) |
| Coverage-Parallelität | [sampling_geometry.cpp](../../tile_compile_cpp/src/registration/sampling_geometry.cpp) |
| Kalibrierung und Vorlauf | [runner_pipeline.cpp](../../tile_compile_cpp/apps/runner_pipeline.cpp), [runner_phase_metrics.cpp](../../tile_compile_cpp/apps/runner_phase_metrics.cpp) |
| Frame-0-Abhängigkeit | [global_quality.cpp](../../tile_compile_cpp/src/reconstruction/global_quality.cpp) |

### 6.1 Messgrenze und belastbare Ausgangsbasis

Start ist der Kaltstart des Runners, Ende der Commit der Rekonstruktionsausgabe
einschließlich Mehrbandfusion, STACKING-Pass-through und erforderlicher
Photometrie-Rücknahme. Scan, vorhandene Kalibriermaster anwenden, Normalisierung,
Registrierung, alle Hashes, Transfers und Writes zählen mit. Astrometrie
(einschließlich eines gegebenenfalls aktivierten Rescue-Solvers), BGE, PCC und
HMS zählen in diesem **neuen Teilziel nicht mit** und müssen tatsächlich
deaktiviert sein. Registration ist weiterhin erforderlich. Kein Resume als
Kaltlauf ausgeben; keine Reduktion der 600 Frames oder der Auflösung.

Aus dem erhaltenen Ereignisextrakt `/tmp/out_bn_phases.txt` des inzwischen
gestoppten M31-Laufs `20260909_155821_833897f4`:

| Vorlauf | Sekunden |
|---|---:|
| SCAN_INPUT einschließlich Kalibrierung | 459,611 |
| NORMALIZATION | 375,481 |
| Abstand Normalisierung → Registrierung | 167,453 |
| REGISTRATION | 206,834 |
| NORMALIZED_CACHE | 12,813 |
| SCAN_INPUT-Start → SAMPLING_GEOMETRY-Start, einschließlich Lücken | **1222,262** |

Diese Zeiten stammen aus einem erhaltenen Diagnoseextrakt, nicht aus einem
neuen Lauf. Das ursprüngliche `p6_runs/.../logs/run_events.jsonl` ist an seinem
damaligen Pfad bei diesem Review nicht mehr vorhanden. Die Zuordnung der
167-s-Lücke zu einzelnen Metrikoperationen braucht eigene Timer. Bereits dieser
Vorlauf zeigt: Selbst perfekte Geometrie allein löst das Kaltstartziel nicht.

### 6.2 Größter zusätzlicher Befund: CUDA-Bandhöhe kollabiert mit N

Quellen: `src/reconstruction/drizzle_profile_store.cpp`,
`persist_forward_drizzle_multiband`; `forward_drizzle_contrib_list.cpp`,
`accumulate_pair_impl`; `forward_drizzle_cuda.cpp`, `plan_cuda_chunking`.

Der CUDA-Pfad reserviert **dicht für jede Kanal-Zielzelle alle Frames**:

```
candidate_bytes_per_row = channels * internal_width * N * sizeof(ClipCandidate)
```

Bei der üblichen 64-Bit-ABI (`sizeof(ClipCandidate)=64`), 7852 Pixeln Breite,
3 Kanälen und 600 Frames: **904.550.400 B = 862,65 MiB pro interner Zeile**.
Das ist Host-RAM. Der Planer addiert diesen Term jedoch zum Device-Term und
teilt den freien **VRAM** durch die Summe. Bei höchstens 6 GiB freiem VRAM und
20 % Reserve passen schon allein wegen dieses Terms höchstens **5 Zeilen**.
4620 / 5 bedeutet **mindestens 924 Bänder**, nicht die 19 Coverage-Streifen.
Weniger freier VRAM oder ein Retry verkleinern die Bänder weiter. Das ist eine
Code-/Budgetableitung, keine nachträglich gemessene Bandzahl des alten Laufs.

Jedes Band iteriert wieder über alle Frames. `cuda_pair_producer` begrenzt zwar
den Upload auf Quellzeilen, ruft **vorher aber den Vollbild-Source-Provider** auf.
Der Q-Provider in `source_quality_artifact.cpp` expandiert dabei bis zu vier
vollständige Karten (`composite`, `scale_0`, `scale_1`, `artifact`). Das bedeutet
bei vier Karten rund **79,63 GB erzeugte Float-Daten pro Vollpass**. Bei 924
Bändern wären es rund **73,6 TB allein für diese Expansionen**. Das ist ein
logischer Aufwand, keine gemessene physische Diskmenge. Ein Cache-Miss auf allen
Quellen würde zusätzlich rund 18,4 TB Source-Reads/Hashes erzeugen.

Außerdem laufen im CUDA-Zweig `std::sort` der Records, Segmentreduktion und
`reduce_pixel_profiles` auf der CPU **seriell**. Der Kommentar „CUDA ignoriert
workers“ beschreibt keine vollständig auf der GPU ausgeführte Rekonstruktion.
Ihre genaue Zeit muss getrennt gemessen werden. Ein pauschaler I/O-Befund
aus einer einzigen Stackprobe erklärt nicht die ganze Phase.

**Entscheidung R1, erste Priorität:**

1. Host- und Device-Budget getrennt planen. Größere Host-Bänder dürfen nicht
   fiktiv VRAM verbrauchen; Device-Microbatches bleiben unabhängig klein.
   Alle gleichzeitig residenten Source-, Q-, Kandidaten-, Sortier- und
   Ausgabepuffer tatsächlich vom gemeinsamen Host-Budget abziehen.
2. Den Vollbreiten-Zwang des Kandidatenpuffers durch **zweidimensionale
   Zielkacheln** auflösen. Beispiel 128×64 intern: derselbe dichte
   Kandidatenpuffer benötigt etwa 900 MiB statt 55 GiB für 64 volle Zeilen.
   Das ist eine Arbeitskachel, keine Änderung des wissenschaftlichen Bildrasters.
3. Für jede affine Zielkachel Quellrechteck in **X und Y** invers bestimmen,
   einschließlich konservativem Tropfenrand; dieselben `sample_leaves` und
   Zellclips aufrufen. Der heutige Pfad begrenzt nur Quell-Y. Für lokale Modelle
   den vorhandenen Geometrieindex verwenden; ein affines Rechteck allein wäre
   dort keine gültige Schranke.
4. Reihenfolge pro Zielzelle exakt erhalten: Frame, Quell-Y, Quell-X, Leaf.
   Unabhängige Zielzellen dürfen parallel reduziert werden. Kein ungeordnetes
   Floating-Point-Atomic-Add. `reduce_pixel_profiles` zunächst unverändert;
   pro Worker eigener Scratch und deterministische Diagnostikreduktion.
5. Kleine Kacheln nicht in Millionen synchroner CUDA-Einzelaufrufe übersetzen:
   mehrere Frame-/Kachel-Aufträge mit begrenztem Speicher bündeln, Device-Puffer
   wiederverwenden, CPU-Reduktion disjunkter Zellen parallelisieren. CPU und CUDA
   separat messen. Einfach nur mehr kleine Kernel zu starten ist kein Fix.

Getrennte Budgets allein sparen Wiederholungen, beseitigen aber nicht den
Vollbild-Provider. R1 und R2 müssen zusammen umgesetzt werden.

### 6.3 R2: echte Bereichsprovider statt zyklischem Vollbild-LRU

Quellen: `normalized_source_cache.cpp`, `VerifiedNormalizedSourceCache::load`;
`source_quality_map_cache.cpp`, `read_bin`, `read_region`, `read_full`;
`source_quality_artifact.cpp`, `persist_multiband_store_from_predecessors`.

**Nachgerechnet und als LRU-Simulation geprüft:** ein Frame hat 33.177.600 B;
600 Frames haben 19.906.560.000 B = 18,54 GiB. Mit 16384 MiB und 1 MiB Reserve
passen 517 Frames. Ein zyklischer Durchlauf 0…599 verdrängt die jeweils als
nächstes gebrauchten Frames: über 19 Pässe **0 Treffer, 11.400 Misses**.
O2 hält die Verifikation auch nicht unabhängig von den LRU-Einträgen vor.

Das Runner-Objekt erhält derzeit das volle Drizzle-Budget; SQM-Klone und
Kandidatenpuffer kommen zusätzlich dazu. 16 GiB Source-LRU plus 16 GiB
Drizzle-Arbeitsbudget sind kein gemeinsames 16-GiB-Budget. Bei 30 GiB freiem RAM
darf deshalb nicht ohne Gesamtbilanz ein größerer Source-Cache zugesagt werden.

**Umsetzung:**

- Source-Provider um Rechteck-/Zeilenansichten mit absolutem Ursprung erweitern.
  Nur die für R1 erforderlichen Rohdaten lesen; Source-Indizes im Clipping-Key
  bleiben absolut. Kein Crop mit stillschweigend verschobenem CFA-Ursprung.
- `SourceQualityMapCacheReader::read_region` muss echtes Range-I/O erhalten:
  derzeit lädt es mit `read_bin` die gesamte Datei und expandiert erst danach
  die gewünschten Zeilen. Werte- und Vetoebene liegen getrennt in der Datei;
  deren Offsets und Speicherzeilen separat lesen und validieren.
- Noch günstiger: vorhandene uint16-Werte und Veto-Bytes direkt über einen
  unveränderlichen Karten-View adressieren. `dequantize_quality`, `y/divisor`,
  `x/divisor` und Hard-Veto→NaN müssen exakt dieselben Ergebnisse liefern.
  Das vermeidet die wiederholte volle Float-Expansion, ohne neue Quantisierung.
- Hashes nicht einfach streichen. Entweder verifizierte unveränderliche
  Generationen mit klarer Besitz-/Mutationsregel, oder verifizierbare Blöcke
  mit an das Manifest gebundenen Digests. Bei Schemaänderung Version/Identität
  und Resume-Vertrag anpassen. Ein offener Dateideskriptor verhindert allein
  keine In-place-Schreibzugriffe; Größe+mtime allein garantiert keine Integrität.
- Zielzähler: physische und logische Reads getrennt, gelesene/expandierte Pixel,
  Hash-Bytes, Source-/Q-Treffer, Band-/Kachelzahl, Host-/Device-Peaks und
  `source_pixels_loaded / (N*P)`. Schranken aus tatsächlich geschnittenen
  Rechtecken plus Randüberlappung ableiten, nicht pauschal „genau einmal“ sagen.

### 6.4 R3: schnellere exakte Geometrie, ohne O5-Schablone

Quelle: `forward_drizzle.cpp`, `clip_one_plane`,
`polygon_rectangle_intersection_area`, `enumerate_drizzle_stripe_leaf_cells`.

Der vorhandene Clip führt X-min, X-max, Y-min, Y-max und Shoelace für **jede**
Zielzelle erneut aus. Die ersten beiden Ergebnisse hängen von Leaf und Ziel-X
ab, **nicht von Ziel-Y**. Sie können pro Leaf/X berechnet und über alle
betroffenen Zielzeilen wiederverwendet werden. Die Ausgabe bleibt in derselben
Y/X-Reihenfolge; lediglich ein kleiner Puffer für X-Zwischenpolygone kommt hinzu.
Wenn alle Polygonpunkte eine Clip-Halbebene bereits erfüllen, kopiert der
Referenzschritt die Punkte unverändert in identischer Reihenfolge. Dieser
Schritt kann entfallen. Schnittformeln, Vertex-Reihenfolge und Shoelace bleiben
unverändert. Keine Translation der Shoelace-Koordinaten, kein FMA/Reassociation,
kein `fast-math`, keine gerasterte Subpixel-Schablone.

**Isolierter Machbarkeitsversuch dieses Reviews:**
`/tmp/p6_clip_reuse_probe.cpp`, gebaut mit
`g++ -O3 -std=c++20 -ffp-contract=off`. 100.000 Quads mit Translationen und
kleinen Rotationen, beide Pixfracs, 784.712 Zellflächen:

| Variante | Zeit für fünf Wiederholungen | Bitabweichungen |
|---|---:|---:|
| vier Clips pro Zelle | 0,256082 s | Referenz |
| X-Clip-Wiederverwendung + wirkungslose Clips auslassen | 0,165475 s | **0** |

Das sind **1,55× in einem eigenständigen Mikroversuch**, der die Referenzformeln
nachbildet, nicht die vollständige Library aufruft. Es ist weder ein
Produktionsbenchmark noch ein Beweis für sämtliche degenerierten Polygone.
Original- und optimierte Library-Pfade müssen direkt gegeneinander getestet
werden: Rotation, Scherung, Spiegelung, Tangenten, ganzzahlige Grenzen und
`nextafter`-Nachbarn, große Koordinaten, lokale Leaves und mehrere Chunkhöhen.
Naive Variante `Rechteckfläche statt Shoelace` ist ausdrücklich ausgeschlossen:
mathematische Flächengleichheit garantiert keine gleichen Floating-Point-Bits.

**Zweiter, größerer Hebel:** Der dichte affine Footprint benötigt nur die
Information, ob mindestens ein Pixelquadrat eine Zelle mit `k>0` berührt.
Die Vereinigung aller ungeschrumpften Quellpixelquadrate ist mathematisch das
transformierte Frame-Rechteck. Daher ist ein rasterisierter Frame-Footprint
statt Milliarden Einzelpixelclips grundsätzlich möglich. Aber das heutige
numerische `k>0` ist der Referenzvertrag: Tangenten, degenerierte Transformationen
und Rundungsartefakte dürfen nicht stillschweigend anders behandelt werden.
Nur konservativ zertifizierte Innen-/Außenzellen beschleunigen, Grenzfälle mit
dem bisherigen Einzelpixelpfad prüfen. Eine numerische Fehlerschranke muss
auch die berechneten Quellpixelgrenzen abdecken; bloß ein Rand um die vier
Framekanten reicht als Beweis nicht. Der Footprint-Anteil des bisherigen Profils
liegt bei etwa 57 % der Rasterzeit: seine Beseitigung wäre ein wesentlich
größerer Hebel als die bloße Fusion zweier Schleifengerüste.

**Umgesetzt in §30.75** (`dense_footprint_touched_stripe`): genau dieser Ansatz —
Innen-/Außen-/Randklassifikation gegen das eine Parallelogramm
`affine_f([0,W_src]×[0,H_src])`. Randzellen (inkl. numerischer Randband-Breite
`2·ext+3`) laufen exakt über `sample_leaves` + `polygon_rectangle_intersection_area`
auf einem invers-gemappten, um ±1 Quellpixel erweiterten Quellpixel-Superset —
also **derselbe Referenzvertrag auf den Randzellen, byte-identisch per
Konstruktion**. Reflexionen sind upstream (`invert_affine_2x3`,
orientierungserhaltend) ausgeschlossen; lokale/singuläre Frames delegieren an den
Rasterizer. Paritäts-Gate `[drizzle-audit][footprint-fastpath]`. Gemessen:
Footprint-Pass 47,6 s → 3,5 s (13,6×), Coverage gesamt 87,2 s → 42,8 s (2,04×).

### 6.5 R4: Vorlauf und Qualität gezielt verkürzen

- `runner_pipeline.cpp`, Kalibrierungsschleife über `input_frames`: derzeit
  sequenzielles Lesen, Bias/Dark/Flat und FITS-Schreiben. Frames mit begrenzten
  I/O-Workern verarbeiten, Ausgabepfade/Ergebnisvektor nach Index festlegen,
  Masters read-only teilen. FITS-Threadfähigkeit und tatsächlichen
  Speicherdurchsatz prüfen; die 459,6 s sind nicht vollständig Rechenzeit.
- `runner_phase_metrics.cpp`: Normalisierung und frühe globale Metriken besitzen
  **bereits Worker**. Noch ein Parallel-Pragma verspricht hier keinen Faktor 8.
  Lesen, Hintergrundschätzung, Normalisierung, Cache-Schreiben und Proxybau
  einzeln messen. Doppelte Arbeit nur bei identischen Inputs/Definitionen
  wiederverwenden. Registrierungsmetriken dürfen nicht als AQMH-Gewichte dienen.
- `source_quality_map_cache.cpp`: O3 hat eine serielle `writer.put`-Sektion
  einschließlich Serialisierung/Hash/Write; ca. 6× ist bisher eine Erwartung.
  Unabhängige Dateien parallel vorbereiten, nur Metadatenaufnahme/Commit
  serialisieren. SQM-Working-Set pro Worker einschließlich `pending` budgetieren.
  Nebenbefund vor Ausbau beheben: `worker_error` wird außerhalb von `critical`
  gelesen und darin geschrieben; das ist ein ungeschützter gemeinsamer Zugriff.
  Ein atomarer Fehlerindikator bzw. ausschließlich synchronisierte Zugriffe
  sowie Exception-Weitergabe auch für die Cache-Klon-Konstruktion sind nötig.
- `global_quality.cpp`: Frame 0 setzt `ref_star_count`; zuerst diesen Frame
  berechnen, dann 1…N−1 parallel mit fixem Referenzwert. Endgültige
  Gewichtsbildung in ursprünglicher Reihenfolge. SQM und Global Quality bauen
  beide `compute_source_quality_proxy_v1`: gemeinsamen Proxy innerhalb eines
  Frame-Jobs nutzen, sofern Input und Konfiguration identisch sind. Die
  getrennten Phasen-/Resume-Artefakte dennoch vollständig veröffentlichen.

### 6.6 Budget, Reihenfolge und Entscheidung

**Es gibt keinen aus dem Code ableitbaren mathematischen Grund für mehrere
Stunden.** Der derzeitige N-abhängige Vollbreiten-Kandidatenpuffer erzwingt
kleinste Bänder, deren Provider dann erneut Vollbilder bearbeiten. Dieser
vermeidbare Aufwand ist zusätzlich zur eigentlichen Rekonstruktion vorhanden.
Seine Beseitigung ist der erste Lösungsweg; O5 ist dafür nicht erforderlich.

Ein **Entwicklungsbudget, keine Laufzeitprognose**, für das affine 600-Frame-Ziel:

| Teil | Zu erreichendes Budget |
|---|---:|
| gesamter Vorlauf bis Coverage | 650 s |
| Coverage einschließlich Gate/Masken | 300 s |
| SQM | 400 s |
| Global Quality | 45 s |
| Forward Drizzle einschließlich Clipping/Transfers | 180 s |
| Mehrband, STACKING, Ausgabe-Commit | 55 s |
| sonstige Übergänge/Hashes | 50 s |
| **Summe / Reserve bis 1800 s** | **1680 s / 120 s** |

Die anspruchsvollsten offenen Posten sind Coverage 300 s und Drizzle 180 s.
Der 1,55×-Clipversuch plus O1 allein trägt Coverage 300 s nicht; dafür muss die
Footprint-Stufe oder ein separat paritätsgeprüfter beschleunigter Coverage-Pfad
zusätzlich bestehen. Die Tabelle darf nicht als erreichte oder bereits
wahrscheinliche Endzeit kommuniziert werden. Lokale Modelle brauchen dieselbe
Abnahme separat; die affine Ableitung überträgt sich darauf nicht automatisch.

**Verbindliche nächste Schnitte (Revision 2026-09-10, §30.77 — ersetzt die
ursprüngliche Reihenfolge dieses Reviews; maßgeblich ist Plan §0.2):**

1. **Benchmarkfehler und Timerlücken beheben; tatsächliche Worker und
   Quellbesuche erfassen.** Konkret: `[fd-hotspot]`-Gather/-Reduce auf
   befülltem Zustand, Abgleich mit den `TC_FD_PROFILE`-Produktionstimern;
   die 167-s-Vorlauflücke attribuieren.
2. **Wiederverwendbaren, budgetierten Clipping-Scratch** in
   `apply_robust_clipping` implementieren (bit-identisch; Scratch in
   `retained_bytes`). Hilfreich für die spätere Reduce-Parallelisierung,
   aber keine zwingende Voraussetzung dafür.
3. **R1.1/R1.2/R2 gemeinsam:** aktuelle Bandzahl/Peaks erfassen, getrennte
   Host-/Device-Budgets, zweidimensionale Zielkacheln, echte Bereichsprovider
   und CPU-Reduktionsparallelität; Vollbildexpansion aus dem inneren
   Band-/Frame-Loop entfernen. Zunächst ohne Geometrieänderung. Welchen
   Faktor das bringt, muss gemessen werden; auch die CPU-Seite profitiert
   von den Bereichsprovidiern.
4. **Produktionsnaher Skalierungsbenchmark** mit realer Canvasgröße,
   40/100/600 synthetischen Frames, gleichem RAM-Budget und realem
   CPU-/CUDA-Pfad. Auch 600 bei kleinem Budget: dieses N löst die
   Bandkollaps-/LRU-Probleme aus. Hidden-Test misst tatsächlich aktive
   Worker; ein Requested-Wert reicht nicht.
5. **R4-Vorlauf** (Kalibrierung, SQM-Writer-Rest, Normalisierung anhand ihrer
   Subtimer — nicht ungemessen weitere Worker ergänzen); danach **R3** als
   separaten messbaren Schritt: exakte Clip-Wiederverwendung gegen den alten
   Pfad testen, dieselbe Paritätsmatrix wiederverwenden, nicht bündeln.
6. Erst wenn die Summe der Phasen auf voller Größe das Budget trägt: separat
   autorisierte reale Kaltläufe, zwei pro Klasse, identische Selektion und
   Gates; alle Laufstart→Commit-Zwischenzeiten mitzählen.

Abnahme jeder Änderung: Source-/Masken-/Coverage-Gate-Bits, Clipping-Entscheide,
Q-Artefakt-Identität und committed Profilebytes gegen Referenz; CPU-Fallback,
Korruption/Resume und harte RAM-Grenzen. Keine Toleranzlockerung als Ersatz.
Kein Produktionslauf wurde für diesen Review gestartet. Die Arbeit dieses
Abschnitts besteht aus Codeanalyse, LRU-Simulation, isoliertem Clipversuch und
Dokumentation; R1–R4 sind damit **noch nicht implementiert oder abgenommen**.
