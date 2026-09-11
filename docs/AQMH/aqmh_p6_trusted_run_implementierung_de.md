# P6 Trusted-Run und Performance — Implementierungsplan

Status: **verbindlicher Umsetzungsentwurf, noch nicht implementiert.**

(2026-09-11: der kurzlebige Begleitplan `aqmh_sha_verifikation_streichung_plan_de.md`
wurde nach Abgleich hier eingearbeitet — §2/§6a — und entfernt. Eine Datei
bleibt maßgeblich.)

Dieses Dokument konkretisiert die Leistungsarbeit aus
[aqmh_p6_performance_de.md](aqmh_p6_performance_de.md). Es gilt für den
600-Frame-P6-Lauf mit 3840×2160 OSC, `internal_scale=2`, ohne Astrometrie,
BGE, PCC und HMS.

## 1. Entscheidung und Grenze

Ein normaler AQMH-Lauf ist ein **Trusted Run**: Der Benutzer besitzt Eingaben,
normalisierten Cache und Artefaktverzeichnis und verändert sie nicht während
des Laufs. Eine Änderung ist ein Bedienfehler außerhalb des Rechenvertrags.

Folge: SHA-256 darf nicht in normalen Phasen, Readern, LRU-Treffern oder
Kachel-/Frame-Schleifen laufen. Atomisches Schreiben, erwartete Dateigrößen,
Schema, Dimensionen, Frame-Indizes, Konfigurations- und Geometrieidentität
bleiben erhalten. Sie erkennen unvollständige oder falsch zugeordnete
Artefakte ohne einen Vollscan der Nutzdaten.

Eine vollständige Inhaltsprüfung bleibt als expliziter Offline-Befehl
`verify-artifacts` erhalten. Sie ist Diagnose vor einem Resume aus fremdem
oder unklarem Verzeichnis, nicht Teil von `run` oder `resume-reconstruction`.
Tests dürfen weiterhin jede relevante Ausgabe hashen.

## 2. Gemessene Ausgangslage

Ein SHA-256 über einen normalisierten Frame mit 33,18 MB dauert auf der
Referenzmaschine 17,2 ms. Ein vollständiger 600-Frame-Durchlauf kostet somit
10,32 s reine SHA-Zeit.

| Beobachtung | Nachweis im Code | Direkte Konsequenz |
|---|---|---|
| `NORMALIZED_CACHE` dauerte 13 s | `seal_normalized_cache()` ruft `publish_normalized_source_manifest()`; dieses hasht jede `.raw` | rund 10,32 s sind vermeidbar |
| Vollscan zwischen Coverage und SQM | `runner_forward_drizzle.cpp` lädt jede Quelle vor `SOURCE_QUALITY_MAPS` | ungetimte Ganzbild-Reads plus SHA und Block-SHA |
| Jeder LRU-Miss hat zwei Digest-Durchläufe | `verify_and_insert()` hasht das volle Bild, `digest_blocks()` danach dieselben Bytes blockweise | der existierende Timer meldet nur den ersten Durchlauf |
| SQM und Global Quality verwenden Cache-Klone | jeder Klon hat einen leeren LRU und keinen Blockindex | zwei weitere vollständige Source-Pässe |
| SQM und Global Quality erzeugen beide den Vollbild-Proxy | `compute_source_quality_proxy_v1()` in beiden Phasen | eine vollständige, vermeidbare zweite Proxyrechnung |
| Coverage dauert 1513 s | `compute_geometric_coverage()` nutzt keine normalisierten Quellen | SHA-Entfernung löst diesen Hauptblock nicht |
| `SourceQualityMapCacheWriter::put()` hasht jede geschriebene `.bin` sofort nach dem Schreiben | `source_quality_map_cache.cpp:389`, `e.sha256 = core::sha256_file(target)` | bei 610 Frames × ~6 Streams ≈ 11,4 s, eingebettet in die gemessenen 522,9 s SOURCE_QUALITY_MAPS |
| `SourceQualityMapCacheReader` hasht beim Öffnen jede `.bin` einmal komplett, UND wird pro vollem Durchlauf **zwei- bis dreimal** konstruiert | `source_quality_map_cache.cpp:514-519` (Verifikationsschleife im Konstruktor); Konstruktionsstellen: `runner_forward_drizzle.cpp:390` (Resume-Validierung vor GLOBAL_QUALITY), `source_quality_artifact.cpp:176`/`284` (Single-Band- bzw. Multiband-FORWARD_DRIZZLE) | je Konstruktion ≈11,4 s bei einem 22-GB-Cache (610 Frames); ein reiner FORWARD_DRIZZLE-Resume bezahlt das **zweimal** (Validierung + eigentlicher Reader), nicht einmal |

Die gemeldeten Phasen ergeben bereits vor Forward Drizzle etwa 40,7 min.
Trusted Run entfernt unnötige I/O- und CPU-Arbeit, ist aber keine Ersatzlösung
für den Coverage-Algorithmus oder den gekachelten CUDA-Producer.

## 3. Hot-Path-Entfernung: Artefakte und Prüfungen

### 3.1 Normalisierte Quellen

Betroffene Dateien:

- `tile_compile_cpp/apps/runner_shared.cpp`
- `tile_compile_cpp/src/reconstruction/normalized_source_cache.cpp`
- `tile_compile_cpp/include/tile_compile/reconstruction/normalized_source_cache.hpp`
- `tile_compile_cpp/apps/runner_forward_drizzle.cpp`

Entfernen im Trusted-Run-Pfad:

1. `publish_normalized_source_manifest()` darf nicht mehr jede `<index>.raw`
   mit `core::sha256_file()` lesen.
2. `VerifiedNormalizedSourceCache::load()` darf weder Ganzdatei-SHA noch
   Block-SHA ausführen. **Ausnahme, bewusst (User-Entscheidung 2026-09-11):**
   der bestehende `stat()`-Restat auf dem LRU-Hit-Pfad (Vergleich von
   Dateigröße und mtime gegen die beim letzten Laden gespeicherten Werte,
   `mt < verified_at`-Wächter) bleibt erhalten. Er kostet keinen messbaren
   Betrag (ein Syscall, keine 33-MB-Operation) und fängt weiterhin eine
   externe Dateiänderung während eines laufenden Prozesses ab — kein
   Zielkonflikt mit dem Trusted-Run-Vertrag, da er kein Hashing ist.
3. `verify_and_insert()`, `digest_blocks()`, `ensure_block_index()` und
   `read_rect()` dürfen im Trusted-Run-Pfad keine Digest-Kontexte erzeugen.
4. Der Vollscan `for (const auto &f : sampling.frames) cache.load(...)` vor
   `SOURCE_QUALITY_MAPS` entfällt vollständig.

Behalten:

- atomare Publikation der normalisierten Cache-Generation;
- einmalige Prüfung beim Öffnen, dass alle erwarteten Dateien vorhanden sind
  und die erwartete Größe `width*height*sizeof(float)` besitzen;
- `source_index`, Frame-ID, CFA-Ursprung, Dimensionen, Farbmodus und
  Konfigurationsidentität;
- LRU für Vollbildzugriffe sowie Bereichslesen ohne Digest.

Neues Manifest-Schema `normalized-source-v2`:

```json
{
  "schema_version": 2,
  "integrity_mode": "trusted",
  "context": { "...": "bestehender Kontext" },
  "files": [{"source_index": 17, "bytes": 33177600}],
  "manifest_identity": "sha256-ueber-dieses-kleine-Metadatenobjekt"
}
```

`manifest_identity` hasht nur JSON-Metadaten und bleibt für Resume- und
Vorgängeridentität zulässig. Er ist kein Dateiinhaltsnachweis. Schema v1 darf
im Trusted Run gelesen werden, ohne seine Dateihashes zu prüfen; der Offline-
Prüfer versteht v1 und v2 und führt bei v1 die historische Vollprüfung aus.

### 3.2 Q-Map-Cache

Betroffene Dateien:

- `tile_compile_cpp/src/reconstruction/source_quality_map_cache.cpp`
- `tile_compile_cpp/include/tile_compile/reconstruction/source_quality_map_cache.hpp`
- `tile_compile_cpp/src/reconstruction/source_quality_artifact.cpp`

Entfernen:

1. `SourceQualityMapCacheWriter::put()` darf eine frisch geschriebene `.bin`
   nicht mit `sha256_file(target)` wieder einlesen.
2. Der `SourceQualityMapCacheReader` darf beim Öffnen nicht jede `.bin` hashen.
3. `SourceQualityCacheFileEntry::sha256` wird in Schema v2 durch `bytes`
   ersetzt; die Metadatenidentität bleibt ein Hash über kleine Metadaten.

Behalten:

- `AtomicOutput` pro Binärdatei und `metadata.json` als einziger Commitpunkt;
- Header-Magic, Version, Datentyp, Speichergeometrie, Frame-Index und
  Dateigröße in `read_bin_window()`;
- Qualitätswerte, Quantisierung, Veto-/NaN-Semantik und absolute CFA-
  Koordinaten unverändert.

### 3.3 Profilstore, Geometriecache und Ausgabe

Normale Builds und normale Resumes prüfen nur Generation, erwartete Ebenen,
Dateigröße, FITS-Geometrie und Manifest-Metadaten. Entfernt werden die
Vollfile-SHAs aus:

- `profile_store_manifest.cpp` beim Commit und beim Öffnen;
- `drizzle_geometry_cache.cpp` beim normalen Resume;
- `runner_forward_output.cpp` für Zwischenartefakte;
- Runner-Checkpointfeldern, die nur einen unmittelbar danach erneut gelesenen
  Dateihash transportieren.

Nicht entfernen: atomare Generationswechsel, fsync/rename, vollständige
Ebenenliste, Größen-/Offsetgrenzen des Geometriecaches und die fachlichen
Predecessor-Identitäten. Der Offline-Prüfer berechnet weiterhin die bisherigen
Dateihashes und vergleicht sie mit Schema-v1-Manifests.

## 4. Datenarbeit, die tatsächlich doppelt ist

### 4.1 SQM und Global Quality zusammenführen

`build_source_quality_map_cache()` besitzt pro Frame bereits den normalisierten
Source und `proxy.proxy_full`. Es erweitert seinen Frame-Job um
`calculate_frame_metrics()` und `measure_frame_stars()`.

Frame 0 wird vor den parallelen Frames verarbeitet, weil es
`ref_star_count` bestimmt. Die restlichen Frames bleiben unabhängig. Der
SQM-Commit schreibt zusätzlich ein kleines, nach `source_index` sortiertes
Artefakt `source_quality_metrics-v1.json` mit Frame-Metriken,
Sternmetriken und Referenzsternzahl. `GLOBAL_QUALITY` liest nur dieses
Artefakt und führt `calculate_global_weights_with_stars()` aus.

Damit entfallen der zweite Source-Read, die zweite Proxybildung und die zweite
Sternmessung; die Qualitätgewichte und ihre Reihenfolge bleiben gleich.
Das Artefakt enthält keine Vollbilder und verändert keine AQMH-Gewichte durch
Registrierungsmetriken.

Abnahme:

- neue und alte `QualityFrameWeightPlan`-Bytes sind gleich;
- SQM-Q-Dateien und ihre quantisierten Werte bleiben gleich;
- Frame-0-Referenz, NaN-/Veto-Fälle und Workerzahlen 1/2/4/8 werden geprüft;
- Resume von `GLOBAL_QUALITY` verlangt das neue Metrikartefakt oder erzeugt
  es deterministisch aus einem vorhandenen SQM-Cache nach.

### 4.2 Q-Metadaten indexieren

`SourceQualityMapCacheReader::has()` und `file_path()` durchsuchen heute
`meta_.files` linear. Der Multiband-Provider ruft beide mehrfach pro
`(Band, Kachel, Frame)` auf. Das ist `O(B*T*F*E)` mit `E` Cache-Dateien.

Beim Konstruktor einmal bauen:

```cpp
unordered_map<SourceQualityKey, fs::path> paths;
```

Der Schlüssel ist `(stream, source_index)`, sein Hash und Gleichheitsoperator
sind explizit zu definieren. `has()` und `file_path()` werden danach O(1).
Die Reihenfolge von `meta_.files` bleibt für die Metadatenidentität sortiert;
der Index ist nur ein abgeleiteter Reader-Cache.

Abnahme: jeder vorhandene und fehlende Stream/Frame-Fall liefert exakt
dieselbe Antwort oder Fehlerkennung; der Reader zählt zusätzlich
`metadata_lookup_calls` und `metadata_linear_scans` (letzterer muss null sein).

### 4.3 Quellzugriff im gekachelten Forward Drizzle

Der aktuelle Tiled-Zweig von `accumulate_pair_impl()` ruft pro Zielkachel
`produce_sorted()` und damit pro Frame `source_of(source_index)` auf. Im
CUDA-Producer wird die volle Sourcebreite in `src_buf` kopiert und der Kernel
produziert vollbreit; erst danach werden Records außerhalb der Zielkachel
verworfen.

Umsetzung in zwei getrennten Schritten:

1. **Schedule 3:** pro Band ein budgetiertes Framefenster `K` aufbauen.
   Jedes Sourceband wird für einen Frame genau einmal gelesen und bleibt für
   alle Kacheln dieses Fensters resident. Kandidaten bleiben framegeordnet;
   die robuste Reduktion erhält unverändert die Reihenfolge je Zielzelle.
   `K` wird aus dem gemeinsamen Hostbudget für Kandidaten, Quellbänder,
   Q-Rechtecke, Records, Stripe-Reassembly und Scratch abgeleitet.
2. **CUDA X+Y-Fenster:** aus der Zielkachel konservativ das affine
   Quellrechteck bestimmen, nur dieses in `src_buf` kopieren und an den Kernel
   übergeben. Der Kernel enumeriert nur diese Quelle und emittiert nur
   In-Fenster-Records. Ränder, Rotation, Scherung, Skalierung und lokale
   Warps erhalten eigene Paritätsfälle; lokale Warps bleiben bis zu einem
   bewiesenen Rechteckpfad beim sicheren Hybrid-Provider.

Kein Spool: Ein dichter 600-Frame-Kandidatenstore wäre mehrere Terabyte groß
und verschiebt den Engpass auf NVMe-I/O.

Abnahme: CPU- und CUDA-Profile, alle Multibandebenen und Alpha-Maps sind für
Kachelbreiten, Bandhöhen und Framefenster bitidentisch. Zähler erfassen
Sourcebytes, Q-Zellen, Producer-Aufrufe, Kernelstarts, Records vor/nach
Fensterung und reale Host-/Device-Peaks.

## 5. Eigenständige Algorithmenprobleme

### 5.1 Sampling Geometry

`compute_geometric_coverage()` ist der größte gemessene Block und enthält
keinen Source-SHA. Für jeden Frame und Band führt der CFA-Pass exakte
Polygon-Schnittflächen aus. Zusätzlich werden die CFA-Akkumulatoren pro Frame
geleert und kanvasweit reduziert; danach folgen Footprint- und Zählpässe.

Die Umsetzung darf nicht einen riesigen Leaf- oder Kandidatenspool einführen.
Die Reihenfolge ist:

1. Subtimer für CFA-Raster, Akkumulator-Reset, Akkumulator-Reduktion,
   Footprint und Loch-/Quantilnachlauf in einem realgroßen Benchmark.
2. affine GPU-Rasterisierung mit derselben Clip-/Flächenroutine und CPU-
   Referenzvergleich; keine Floating-Point-Atomics für gewichtete Summen;
3. getrennt danach CFA-/Footprint-Schleifen fusionieren oder aktive
   Zellenlisten prüfen, wenn die neuen Subtimer Speicherpässe als relevant
   belegen;
4. lokale Warps weiter über die kanonische CPU-Geometrie behandeln, bis eine
   GPU-Übertragung der vorab berechneten Leaves bytegleich nachgewiesen ist.

### 5.2 Registrierung und Normalisierung

Für die gemessenen 218 s Registrierung und 103 s Normalisierung liegt noch
keine genügende Unterteilung vor. Kein Algorithmusumbau ohne Messung.

Ein belegter Nebenpfad: Bei aktiviertem `auto_engine` werden höchstens vier
Probe-Frames registriert. Deren Proxys und Sternlisten gehen nicht in den
regulären Proxycache ein und werden später neu gebaut. Diese Daten in den
bestehenden Cache übernehmen; die Probe bleibt fachlich unverändert.

Zuerst erfassen: FITS-Read, Kalibrierung, Normalisierung, Cache-Write,
Proxybau, Sterndetektion, Paarzuordnung, ECC, Ankerwahl und lokale Verfeinerung.

## 6. Reihenfolge der Änderungen

| Paket | Änderung | Numerik | Messbare Wirkung |
|---|---|---|---|
| T0 | Referenzartefakte und Phasen-/Subtimer festschreiben | keine | Vergleichsbasis |
| T1 | Trusted-Run-Schema; Source-, Q-, Profil- und Geometrie-Hot-Path-SHAs entfernen | keine Pixeländerung | `NORMALIZED_CACHE` minus etwa 10,32 s; keine Digestbytes im Run |
| T2 | Q-Reader-Index | keine | lineare Metadatenscans = 0 |
| T3 | SQM erzeugt Global-Quality-Metriken | Gewichte bytegleich | kein zweiter Proxy-/Sourcepass |
| T4 | Schedule 3 und gemeinsames Hostbudget | Profile bytegleich | begrenzte Sourcebytes je Band/Fenster |
| T5 | CUDA-X+Y-Provider und Kernel | CPU/CUDA-Parität | Producer-, Transfer- und Kernelarbeit skaliert mit Fenster |
| T6 | Coverage-Subtimer und beschleunigter affiner Pfad | Coveragebits gleich | Coverage unter eigenes Budget bringen |
| T7 | Registrierungs-/Normalisierungsoptimierung nach Subtimern | Phasenartefakte gleich | Restbudget schließen |

T1 bis T3 sind unabhängig von CUDA und sofort prüfbar. T4 und T5 sind ein
gemeinsamer Durchsatzschnitt: ein Bereichsread ohne geänderte Lebensdauer
erzeugt nur wiederholte Reads. T6 bleibt separat, weil Coverage ein anderer
Algorithmus und ein anderer Akzeptanzvertrag ist.

## 6a. Konkrete Testfälle für T1 (Bestand, 2026-09-11)

Vor Umsetzung mit `grep -rn "hash_computation_count\|NORMALIZED_CACHE_CONTENT_MISMATCH\|SQM_CACHE_FILE_CORRUPT\|block_index_builds\|blocks_verified\b\|whole_file_sha_seconds" tests/` den vollständigen Bestand
bestätigen; bekannt sind:

- `tests/test_source_quality_artifact.cpp`: der ganze `TEST_CASE`
  `"source cache: block-check index --- ..."` (`[cache-blocks]`, §30.81
  3a-3 Teil 1) entfällt vollständig — der Blockindex wird entfernt (§8, "Block-
  SHA im normalen Kachelpfad" ist bereits als Nicht-Optimierung verworfen).
  `BlkCacheDir`-Testhelfer entfällt mit, falls sonst ungenutzt.
- Dieselbe Datei, `"source cache: content bound frame loading rejects
  replacements and truncation"`: die `REQUIRE_THROWS`-Assertions nach einer
  reinen **Inhaltsänderung** (`f.write(0, andere Pixelwerte, gleiche Größe)`)
  entfallen — das wird im Trusted Run nicht mehr erkannt. Die
  Trunkierungs-Assertion (kürzere Datei) bleibt gültig (Größenprüfung, kein
  Hash).
- `"source cache: LRU hit skips hashing, tamper still fails closed"`
  (Name danach unpassend): `hash_computation_count()`-Assertions entfallen
  (Zähler entfällt mit der Verifikation). Die Eviction-unter-engem-Budget-
  Section bleibt (reine LRU-Mechanik). Die Rewrite-mit-identischer-Größe-
  Section bleibt **inhaltlich gültig**, weil der mtime-Restat auf dem Hit-Pfad
  laut §3.1-Ausnahme erhalten bleibt — sie prüft nur nicht mehr "und wird
  dabei neu gehasht", sondern nur noch "und wird dabei neu gelesen".
- `tests/test_source_quality_map_cache.cpp`: jede Section, die eine `.bin`
  manipuliert und `SQM_CACHE_FILE_CORRUPT` erwartet, entfällt oder wird
  umgedreht (Manipulation wird nicht mehr erkannt). Die `[.]`-Section zu
  `bin_loads()`/`read_rect()` (§30.81 3a-2b, Dekodier-Optimierung) bleibt
  unverändert — sie hat keinen SHA-Bezug.
- Volle Suite muss nach der Streichung wieder grün sein; die Testzahl sinkt
  netto (mindestens der `[cache-blocks]`-Fall entfällt vollständig).

## 7. Test- und Migrationsvertrag

Vor jedem Paket werden die bisherigen committed Profile, Qualitätspläne,
Coverage-Masken und finalen Ausgaben als Referenz gespeichert. Ihre Hashes
bleiben Testwerkzeug, erscheinen aber nicht mehr im normalen Laufzeitpfad.

Für jede Schema-v2-Änderung gelten:

- neue Trusted-Run-Tests: normale v2-Artefakte werden ohne Inhalts-Hash
  verwendet;
- Offline-Tests: absichtlich geänderte Dateien werden vom Prüfbefehl erkannt;
- Migrationsfälle: v1 strikt prüfen, v2 anhand Struktur und Größe öffnen;
- Resume-Fälle: fehlende, zu kurze, falsch dimensionierte oder falsch
  zugeordnete Artefakte lehnen weiterhin ab;
- numerische Fälle: Bytegleichheit über Worker, Bandhöhe, Kachelbreite,
  Framefenster, CPU und CUDA.

## 8. Nicht als Optimierung akzeptieren

- globale Toleranzlockerung zwischen CPU und CUDA;
- ungeordnete Floating-Point-Atomics oder veränderte Clip-/Reduktionsordnung;
- ein Kandidaten- oder Leaf-Spool ohne vollständige Transferbilanz;
- Block-SHA im normalen Kachelpfad als vermeintliche Bereichsoptimierung;
- mehr Parallelität ohne gemeinsame Speicherbilanz;
- Behauptungen über Registrierung oder Normalisierung ohne deren Subtimer.

