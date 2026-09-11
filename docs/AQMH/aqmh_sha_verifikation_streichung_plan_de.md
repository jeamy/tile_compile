# SHA-Content-Verifikation streichen — Implementierungsplan (§30.82)

**Entscheidung (User, 2026-09-11):** Die kryptografische Inhaltsverifikation
(SHA-256) für große Astro-Datendateien wird gestrichen. Begründung: Single-User-
Pipeline, keine Angreifer-Annahme, niemand greift während der Verarbeitung auf
die Daten zu — Datenintegrität der Rohdaten liegt im Zuständigkeitsbereich des
Users. Gemessen kostet SHA-256 bei 33-MB-Frames **17,2 ms** (1930 MB/s) — das
ist ⅓ der kalten Plattenlesezeit derselben Datei (55 ms) und wird an mehreren
Stellen **mehrfach auf dieselben Bytes** angewendet (siehe Teil B).

**Ausdrücklich NICHT betroffen** (User-Entscheidung, kein Kostenproblem):
die kleinen Identitäts-/Konsistenz-Hashes für Resume/Checkpoint
(`config_sha256`, `sampling_plan_hash`, `geometry_hash`, `cache_manifest_hash`,
`source_quality_cache_hash`/`_identity_hash`/`_config_hash`,
`multiband_reconstruction_hash`, `profiles_current_sha256`,
`cache_manifest_hash(meta_)` in der SQM-Metadata) — die Ausnahme sind die zwei
Geometrie-Masken-FITS (siehe unten), die ebenfalls klein im Verhältnis zur
Gesamtlaufzeit sind (gemessen 124,4 ms/Datei, ≈0,01 % der Prädezessorzeit).
Diese Hashes bleiben unverändert; sie sichern Pipeline-interne Konsistenz
zwischen Prozessaufrufen (Resume mit geänderter Config/Geometrie), nicht
Datenintegrität der Rohdaten.

**Zwei getrennte, aufeinanderfolgende Umsetzungsschritte:**

1. **Teil A — SHA-Content-Verifikation entfernen** (dieser Plan, Kapitel 1–4).
2. **Teil B — die dabei sichtbar gewordene strukturelle Redundanz** (Kapitel 5):
   Teil A senkt die Kosten pro Zugriff (kein Hash mehr), beseitigt aber NICHT,
   dass dieselben 33-MB-Dateien mehrfach **gelesen** werden. Das ist der
   nächste, separate Schnitt.

---

## 1. Betroffene Komponenten — was genau entfernt wird

### 1.1 `VerifiedNormalizedSourceCache` (`.raw`-Quellframes)

**Datei:** `include/tile_compile/reconstruction/normalized_source_cache.hpp`,
`src/reconstruction/normalized_source_cache.cpp`

Entfernen:
- `verify_and_insert`: die `SHA256(...)`-Berechnung + der Vergleich gegen
  `hashes_[source_index]` + `NORMALIZED_CACHE_CONTENT_MISMATCH`-Wurf. Übrig
  bleibt: Datei öffnen, exakt `bytes` lesen, `NORMALIZED_CACHE_READ_FAILED`
  bei Kurzlesung/EOF-Fehler (reine I/O-Fehlerbehandlung, kein Hash).
- Der komplette §30.81-3a-3-Blockprüfindex (Commits `53e07038`, `a876bf09`,
  `8cf4558c`, Teile von `0fa1c770`): `struct BlockIndex`, `block_rows_`,
  `block_index_`, `digest_blocks()`, `publish_block_index()`,
  `ensure_block_index()`, `read_rect()`, `read_region()`, `block_row_span()`,
  `blocks_verified()`, `block_index_builds()`, `expanded_floats()`,
  `block_rows_for()`, `hex_digest()`. **Begründung:** der Index existierte
  ausschließlich, um SHA-256-Verifikation billiger zu machen (nur die
  berührten Blöcke neu hashen statt die ganze Datei). Ohne Verifikation gibt
  es nichts mehr zu verbilligen — der Code wäre toter Ballast.
- `hash_computations_` / `hash_computation_count()` — es wird nichts mehr
  gehasht.
- `whole_file_sha_seconds_` / `whole_file_sha_seconds()` — die
  §30.81-3a-3-Baseline-Instrumentierung (Commit `0fa1c770`); ohne SHA-Aufruf
  gibt es nichts mehr zu stoppen.
- `hashes_` (`std::map<size_t,std::string>`, die Manifest-SHA-Strings) — wird
  zu einem reinen Indexbestand: `std::set<size_t> known_indices_` (oder die
  bestehende Map, Wert ignoriert) reicht für den
  `NORMALIZED_CACHE_UNKNOWN_FRAME`-Check.

**Bleibt (eigene, kleinere Entscheidung, siehe 3.1):**
- `require_file()` (Existenz + exakte Bytegröße) — kein Hash, ein `stat()`,
  fängt Trunkierung/Falschgröße weiterhin ab.
- `load_calls_`, `lru_hits_`, `evictions_`, `bytes_read_` — reine
  Zugriffszähler, jetzt für Teil B (Kapitel 5) relevant, nicht für Teil A.
- LRU-Mechanik selbst (Kapazität aus `memory_budget_mb`, Verdrängung).

**Zu entscheiden (3.1):** der mtime/`verified_at`-Staleness-Recheck auf dem
Hit-Pfad in `load()` — der existierte, um zu entscheiden, ob ein residenter
Eintrag ohne Re-Read/Re-Hash vertraut werden darf. Ohne Hashing ist die Frage
selbst hinfällig (siehe 3.1).

### 1.2 `publish_normalized_source_manifest` (Schreibseite, NORMALIZATION-Phase)

**Datei:** `src/reconstruction/normalized_source_cache.cpp`

Entfernen: `core::sha256_file(file)` je Frame beim Publizieren des Manifests
(eine SHA-256-Berechnung über 33 MB **pro Frame**, einmal während
NORMALIZATION — bisher nicht separat gemessen, aber bei 610 Frames ≈ 610 ×
17,2 ms ≈ **10,5 s**, eingebettet in die gemessenen 102,9 s der
NORMALIZATION-Phase, ≈10 %). Das Manifest-Feld `sha256` je Datei entfällt
(siehe 3.2, Schema-Verträglichkeit).

### 1.3 `SourceQualityMapCacheWriter::put` (Schreibseite, SOURCE_QUALITY_MAPS)

**Datei:** `src/reconstruction/source_quality_map_cache.cpp:389`

Entfernen: `e.sha256 = core::sha256_file(target);` — läuft **pro geschriebenem
Stream-File** (composite + bis zu 4 Skalen + artifact ≈ 6 Streams je Frame).
Bei 610 Frames und dem gemessenen Cache-Gesamtvolumen (22 GB) ≈ 22 GB /
1930 MB/s ≈ **11,4 s**, eingebettet in die gemessenen 522,9 s der
SOURCE_QUALITY_MAPS-Phase (≈2,2 %).

### 1.4 `SourceQualityMapCacheReader`-Konstruktor (Leseseite)

**Datei:** `src/reconstruction/source_quality_map_cache.cpp:514-519`

Entfernen: die Schleife `for (const auto &fe : meta_.files) { ... if
(core::sha256_file(p) != fe.sha256) { error_="SQM_CACHE_FILE_CORRUPT..."; } }`
— verifiziert beim Öffnen **jede** `.bin`-Datei des Caches komplett, einmal
pro Reader-Konstruktion. Kostet beim gemessenen 22-GB-Cache ≈ **11,4 s pro
Konstruktion**. Der Reader wird **zweimal pro vollem Durchlauf** konstruiert
(siehe Teil B, 5.4): einmal in `persist_forward_drizzle_from_predecessors`
bzw. `persist_multiband_store_from_predecessors`
(`source_quality_artifact.cpp:176`/`284`), und bei einem **Resume ab
FORWARD_DRIZZLE** zusätzlich zur Validierung in
`apps/runner_forward_drizzle.cpp:390` — dort wird die ganze Cache-Prüfung
bezahlt, bevor GLOBAL_QUALITY überhaupt erreicht ist, UND später nochmal beim
eigentlichen FORWARD_DRIZZLE-Reader. **Zwei volle 22-GB-Hash-Durchläufe pro
Resume-Versuch**, nicht einer.

**Bleibt unverändert (kein SHA-Bezug):** die §30.81-3a-2/3a-2b-Arbeit
(`read_rect`, `read_bin_header`, `read_bin_window`, `bin_loads()`,
`bin_cells_decoded()`, `expanded_floats()`) — das war Dekodier-/I/O-Optimierung
(nur die überdeckenden Speicherzellen lesen/entpacken), **nicht**
SHA-motiviert. Der Reader vertraut den Daten nach dem Öffnen ohnehin (wie im
Kommentar an `read_rect` dokumentiert); dieser Vertrauensanker verschiebt sich
jetzt einfach von "einmal beim Öffnen gehasht" auf "gar nicht mehr geprüft" —
`read_rect`/`read_bin_window` selbst ändert sich nicht.

### 1.5 Was ausdrücklich NICHT betroffen ist (Gegenkontrolle)

- `cache_manifest_hash(meta_)` (SQM-Metadata-Identität, kleine JSON-Struktur)
  — bleibt (§B).
- Die zwei Geometrie-Masken-FITS-Hashes im FORWARD_DRIZZLE-Checkpoint
  (`runner_forward_drizzle.cpp`, `core::sha256_file(artifacts/name)` für
  `geometry_files`) — **technisch dieselbe Kostenklasse** wie A (244 MB/Datei,
  124,4 ms/Datei gemessen), aber der User hat sie explizit als "B lassen"
  bestätigt (0,01 % der Prädezessorzeit, kein spürbarer Hebel). Unverändert.
- Alle Output-FITS-SHA-256 in `forward_drizzle.json` (`outputs[].sha256`,
  Provenienz-Reporting) — nicht Teil dieser Entscheidung, da nicht als Kosten
  benannt; separat zu klären, falls gewünscht.
- Store-/CUDA-Parity-Identitätshashes (`DrizzleStoreIdentity`,
  `multiband_reconstruction_hash`, `[cuda-parity]`-Vergleiche) — B-Klasse,
  bleiben; sie beweisen Bit-Exaktheit zwischen CPU/GPU, keine
  Rohdaten-Integrität.

---

## 2. Implementierungsschritte (Reihenfolge)

1. **`normalized_source_cache.{hpp,cpp}`**: `verify_and_insert` auf
   Read-only reduzieren (Umbenennung erwägen, z. B. `load_and_insert`, da
   "verify" nicht mehr zutrifft); Blockindex-Komplex komplett entfernen;
   `hashes_`→`known_indices_`; SHA-bezogene Counter entfernen; Header
   entsprechend bereinigen (Doku-Kommentare, die "verifiziert"/"SHA-256"
   versprechen, umschreiben — sie beschreiben sonst ein Verhalten, das nicht
   mehr existiert).
2. **`publish_normalized_source_manifest`**: `core::sha256_file`-Aufruf
   entfernen; Manifest-JSON ohne `sha256`-Feld schreiben (3.2 klärt
   Rückwärtskompatibilität mit bereits publizierten Manifesten).
3. **`source_quality_map_cache.cpp`**: `writer.put` — `sha256_file`-Aufruf
   entfernen, `SourceQualityCacheFileEntry.sha256` optional/leer lassen;
   Reader-Konstruktor — die Verifikationsschleife entfernen; `metadata.json`
   ohne (oder mit leerem) `sha256`-Feld je File schreiben.
4. **`SourceQualityCacheFileEntry`** (Header): `sha256`-Feld als optional
   markieren oder ganz entfernen (Entscheidung: siehe 3.2 — Schema-Frage ist
   für Quell-Manifest UND SQM-Metadata identisch, gemeinsam klären).
5. **`runner_forward_drizzle.cpp`**: `source_cache`-JSON-Block (Commit
   `0fa1c770`) auf die noch gültigen Felder reduzieren (`capacity_frames`,
   `frames`, `memory_budget_mb`, `load_calls`, `lru_hits`, `evictions`,
   `bytes_read`) — `verify_and_insert`/`whole_file_sha_seconds` raus, da diese
   Größen nicht mehr existieren.
6. **Fehlercodes**: `NORMALIZED_CACHE_CONTENT_MISMATCH`,
   `SQM_CACHE_FILE_CORRUPT` verschwinden aus dem tatsächlichen Verhalten;
   Doku-Kommentare an den verbleibenden Fehlercodes (`_READ_FAILED`,
   `_MISSING_OR_INVALID_FILE`) präzisieren, dass sie jetzt nur noch
   Größe/Lesbarkeit prüfen, keinen Inhalt.
7. **Tests** (Kapitel 4) anpassen.
8. **Dokumentation**: §30.81-3a-3-Abschnitte im Entwicklungsprotokoll und die
   P6-Checkliste als **rückgängig gemacht** kennzeichnen (nicht löschen —
   Historie bleibt nachvollziehbar, mit Verweis auf diesen Plan); M9-Status-
   Memory aktualisieren.
9. Volle Suite grün, dann committen (mehrere kleine Commits entlang 1–6
   empfehlenswert, damit jeder Schritt isoliert bit-/verhaltens-nachvollziehbar
   bleibt — kein Big-Bang-Commit).

---

## 3. Offene Entscheidungen (vor Umsetzung zu klären)

### 3.1 mtime/`verified_at`-Staleness-Recheck auf dem LRU-Hit-Pfad

`load()` prüft heute bei jedem Hit erneut `fs::file_size`/`fs::last_write_time`
gegen die beim Laden gespeicherten Werte, um zu entscheiden, ob der residente
Puffer noch vertrauenswürdig ist (kostet einen `stat()`-Syscall, keinen Hash —
Mikrosekunden, nicht die gemessene Kostenquelle). Mit der Prämisse "niemand
greift während der Verarbeitung auf die Daten zu" ist auch dieser Zweck
hinfällig. **Zwei Optionen:**
- **(a) Beibehalten** — kostet praktisch nichts, fängt weiterhin eine
  externe Dateiänderung während eines laufenden Prozesses ab (nicht
  Kern-Zweck mehr, aber ein kostenloser Rest-Schutz).
- **(b) Auch entfernen** — `load()`-Hit wird zu einem reinen
  Map-Lookup+Splice, keine Syscalls mehr. Konsequenter zur Entscheidung,
  spart aber nichts Messbares.
Empfehlung: **(a) beibehalten** — kein Aufwand, kein Zielkonflikt mit der
Entscheidung (es ist kein SHA, keine 33-MB-Operation), und Code-Diff bleibt
kleiner. Nur auf expliziten Wunsch (b).

### 3.2 Manifest-/Metadata-Schema: Feld entfernen oder nur ignorieren?

Sowohl `normalized_source_manifest.json` als auch SQM's `metadata.json`
tragen heute ein `sha256`-Feld je Datei. Zwei Wege:
- **(a) Feld weglassen, Parser tolerant machen** (bestehende Felder mit
  `sha256` werden beim Lesen ignoriert, keine neue Schreibpflicht). Bereits
  publizierte Manifeste/Metadata (inkl. der 610-Frame-Caches aus dem
  M9-Messlauf auf `/media/tc_500/m9_baseline`) bleiben **ohne Neuaufbau**
  gültig — kein `schema_version`-Bump nötig.
- **(b) `schema_version` erhöhen, Feld strikt verboten** — saubereres Schema,
  aber jeder bestehende Cache (inkl. der bereits gebauten M42-Prädezessoren)
  müsste neu publiziert/gebaut werden.
Empfehlung: **(a)** — kein Grund, den bereits investierten
40-Minuten-Prädezessor-Lauf zu entwerten; die Feldexistenz ist harmlos, wird
nur nicht mehr geschrieben noch gelesen.

### 3.3 Reihenfolge Teil A / Teil B

Dieser Plan deckt nur Teil A. Teil B (Kapitel 5) sollte **nach** Teil A
umgesetzt werden, weil die dortige Restrukturierung (gemeinsamer statt
Pro-Worker-Cache) einfacher wird, sobald `load()` keine
Hash-Vergleichs-Semantik mehr trägt (siehe 5.5).

---

## 4. Tests — was sich ändert

- **`test_source_quality_artifact.cpp`**:
  - Abschnitt `"source cache: block-check index --- ..."` (`[cache-blocks]`,
    ganzer TEST_CASE aus Commit `53e07038`/`a876bf09`) komplett entfernen.
  - `"source cache: content bound frame loading rejects replacements and
    truncation"`: die Assertions `REQUIRE_THROWS(cache.load(0))` nach
    `f.write(0, ...)` (Inhaltsänderung) entfallen — das wird nicht mehr
    erkannt, per Entscheidung. Die Trunkierungs-Assertion (`file << "short"`)
    bleibt gültig (Größenprüfung via `require_file`, kein Hash).
  - `"source cache: LRU hit skips hashing, tamper still fails closed"` (Name
    passt nicht mehr): `hash_computation_count()`-Assertions entfernen
    (Zähler entfällt); je nach 3.1-Entscheidung bleibt der
    Eviction-unter-engem-Budget-Teil gültig (reine LRU-Mechanik, kein Hash).
    Die Rewrite-Detection-Section (`"rewrite with identical size is still
    caught"`) entfällt, wenn 3.1(b) gewählt wird — bleibt (ohne Hash-Bezug,
    reiner mtime-Vergleich) bei 3.1(a).
  - `BlkCacheDir`-Helper (aus Commit `53e07038`) kann entfernt werden, wenn
    von keinem verbleibenden Test mehr gebraucht.
- **`test_source_quality_map_cache.cpp`**: die `[.]`-Section, die
  `bin_loads`/`read_rect`-Zähler testet, bleibt (kein SHA-Bezug). Jede
  Section, die absichtlich eine Datei manipuliert und
  `SQM_CACHE_FILE_CORRUPT` erwartet, entfällt oder wird umgedreht (Manipulation
  wird jetzt NICHT mehr erkannt — falls ein solcher Test existiert, prüfen).
- **`test_drizzle_profile_store.cpp`**, **`test_forward_drizzle_*`**: prüfen,
  ob irgendein Test `NORMALIZED_CACHE_CONTENT_MISMATCH` oder
  `hash_computation_count()` referenziert (Grep vor Umsetzung).
- Volle Suite (544 Tests vor diesem Schnitt) muss nach der Streichung wieder
  grün sein — erwartete Netto-Testzahl sinkt (mind. der `[cache-blocks]`-Fall
  entfällt komplett).

**Vor Implementierung:** `grep -rn "hash_computation_count\|NORMALIZED_CACHE_CONTENT_MISMATCH\|SQM_CACHE_FILE_CORRUPT\|block_index_builds\|read_rect\b" tests/` laufen lassen, um jede betroffene Stelle vollständig zu erfassen (dieser Plan listet die bekannten, nicht notwendigerweise alle).

---

## 5. Teil B — wo sich weitere Lücken finden (nächster Schnitt, NACH Teil A)

Teil A senkt die Kosten pro Dateizugriff (kein SHA mehr), ändert aber nichts
an der **Häufigkeit**, mit der dieselben 33-MB-Dateien angefasst werden. Aus
dem echten M42-610-Frame-Lauf (Predecessor-Teil vollständig gemessen, siehe
`/tmp/.../scratchpad/baseline/RESULTS.md`) und Code-Lektüre:

### 5.1 SOURCE_QUALITY_MAPS und GLOBAL_QUALITY: Kapazität-1-Worker-Klone

`source_quality_map_cache.cpp:672-676` und `source_quality_artifact.cpp:118`
konstruieren je Worker einen eigenen `VerifiedNormalizedSourceCache`-Klon mit
`worker_mb = frame_byte_size/(1024*1024) + 4` → **LRU-Kapazität exakt 1**
(nicht "~2 Frames resident", wie ein Kommentar dort behauptet — nachrechnen
ergibt 1). Jeder Frame, den ein Worker verarbeitet, ist ein Kaltstart-Read,
unabhängig davon, ob dieselben Bytes Sekunden vorher von einem anderen Klon,
einer anderen Phase oder dem Preflight bereits gelesen wurden. Nach Teil A
ist das nur noch ein Read (kein Hash), aber der Read bleibt vollständig
redundant.

### 5.2 Der unbedingte Preflight-Sweep

`runner_forward_drizzle.cpp`, `for (const auto &f:sampling.frames)
cache.load(f.source_index);` — läuft bei **jedem** Prozessstart (frisch UND
resumed), einmal komplett über alle N Frames im geteilten Cache. Sichtbar im
echten Lauf als unbeschriftete **42,7-s-Lücke** zwischen COMMON_OVERLAP-Ende
und SOURCE_QUALITY_MAPS-Start. Bei Kapazität < N (z. B. 258 von 610 bei
8192 MiB) verdrängt der Sweep selbst schon innerhalb sich selbst die frühen
Frames.

### 5.3 FORWARD_DRIZZLE: Band-Sweep unter einem geteilten Cache

Die 3a-1-Kachel-außen/Frame-innen-Schleife nutzt den geteilten `cache` direkt.
Bei Kapazität < N sweept jedes der `n_bands` Bänder erneut über (praktisch)
alle N Frames — mit `n_bands` ≈ 5 (8192 MiB) bis 20 (2048 MiB) bei
Produktionsgeometrie (siehe `scratchpad/source_access_sim.py`-Projektion,
§30.81-3a-3-Protokoll).

### 5.4 SQM-Reader wird zwei- bis dreimal pro Lauf konstruiert

Siehe 1.4 — bis Teil A entfernt wird, kostete das zusätzlich 2× volle
Cache-Hashes (22 GB) pro Resume-Versuch. Nach Teil A bleibt die
**Redundanz der Konstruktion selbst** (Metadata neu parsen, `has()`-Zustand
neu aufbauen) — kleiner, aber real, und ein Kandidat für "einen Reader pro
Lauf statt pro Validierungsschritt + Verbrauchsstelle".

### 5.5 Empfohlene Richtung für Teil B

Die vier Fundstellen teilen eine Ursache: **jede Phase/jeder Worker hält
seinen eigenen, isolierten Cache-Zustand**, obwohl dieselben Dateien
phasenübergreifend gebraucht werden. Nach Teil A (kein Hash-Vergleich mehr,
`load()` ist im Kern "lies Bytes, cache sie") wird ein **echt geteilter
Cache über Phasen- und Worker-Grenzen hinweg** technisch einfacher: die
heutige Pro-Worker-Klon-Begründung ("`load()` ist nicht thread-safe") lässt
sich durch einen Mutex um die LRU-Buchhaltung (Map-Insert/-Evict) ersetzen,
während gleichzeitiges Lesen bereits residenter, unveränderlicher `Matrix2Df`-
Puffer unproblematisch ist. Das würde die Preflight-/SQM-/GQ-/FORWARD_DRIZZLE-
Redundanz Richtung ~1× statt ~8× kollabieren — unabhängig von jeder
Band-/Kachelgeometrie-Frage. **Das ist der nächste vorzuschlagende Schnitt,
nach Abschluss und grüner Suite von Teil A.**
