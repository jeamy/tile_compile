# P6 — Runbook und Phasenbudget-Ableitung

**Aktueller Nachtrag (2026-09-09, §30.73):** Für das neue Teilziel
**<30 Minuten bis zur Rekonstruktionsausgabe ohne Astrometrie/BGE/PCC/HMS** gilt
die [aktuelle Codeanalyse und der Lösungsentwurf](aqmh_p6_performance_de.md#p6-loesungsweg-30min).
Die folgenden 2400-s-Vorläufe sind historische Diagnoseversuche. Der spätere
M31-Lauf `20260909_155821_833897f4` erreichte FORWARD_DRIZZLE und wurde auf
Benutzerwunsch gestoppt; M42 wurde danach nicht gestartet. Die alte Aussage
„keiner erreichte FORWARD_DRIZZLE“ bezieht sich nur auf die zwei unten genannten
Vorläufe. Eine vollständige Abnahme liegt weiterhin nicht vor.

Entscheidende neue Befunde: Source-LRU mit 16 GiB verdrängt bei 600 Frames
zyklisch die als Nächstes benötigten Einträge;
CUDA-Host-Kandidaten erzwingen rechnerisch hunderte kleinste Bänder;
Vollbild-Q-Expansionen und Source-Loads wiederholen sich pro Band. Daher sind
die frühere O2-Durchsatzzusage und die Zuordnung affiner Coverage-Kosten zum
lokalen Cache-Bau überholt. Das ursprüngliche P6-Gate einschließlich HMS wird
durch das neue Teilziel nicht als erfüllt markiert.

Status: **Implementierung verifiziert; die realen 600-Frame-Läufe sind bis zum
P6-Gate dokumentiert, aber keine P6-Abnahme.** Stand 2026-09-09. Beide Läufe
wurden ausdrücklich gestartet und am 2400-s-Gate kontrolliert beendet; keiner
erreichte FORWARD_DRIZZLE, STACKING oder HMS.

### Aktuelle Voraussetzungen

- P3 plant Reader-, Clipping- und Alpha-Scratch pro Worker gemeinsam mit dem
  Streifenbudget; angeforderte, budgetierte und tatsächlich verwendete
  Workerzahlen werden getrennt erfasst. Frische Läufe behalten die konfigurierte
  Workerzahl. Die reale Skalierung ist weiterhin ungemessen.
- Der neue Rekonstruktionspfad besitzt einen STACKING-Pass-through mit einmaliger
  Photometrie-Rücknahme und gemeinsamer RGB-Downstream-Funktion für Astrometrie,
  BGE, PCC und HMS. Kein erneutes Debayern; Raw und `reconstructed_*` bleiben
  im normalisierten Arbeitsraum. Reale vollständige P6-Abnahme steht aus.
- MONO erhält derzeit keinen vollständigen RGB-Downstream; dessen Phasen sind
  explizit nicht anwendbar. Das ersetzt keine allgemeine M9-MONO-Abnahme.
- **Keine belastbare 600-Frame-End-to-End-Endzeit verfügbar.** Beide realen
  Läufe stoppten in `SAMPLING_GEOMETRY`; frühere Summen und `/N`-
  Extrapolationen werden in §5 nicht als Startzusage verwendet.
- 600 Registrierungsproxies bei 3840×2160 benötigen etwa **4,97 GB = 4,63 GiB**.
  16 GiB sind ein zu prüfender Budgetkandidat, kein automatisch sicherer Wert.
  Freien RAM/cgroup-Headroom und Tempdisk unmittelbar vor dem Lauf messen.
- Beide Klassen benötigen 600 verschiedene Realframes, unveränderte Gates,
  vorhandene Kalibriermaster und ein eingefrorenes Hardware-/Configprofil.

### Verifikation dieser Änderung (2026-09-09)

- `tile_compile_runner` und `tests` erfolgreich gebaut; `git diff --check` sauber.
- Gesamtsuite: **532 Testfälle, 1.043.956 Assertions bestanden**.
- Nach der abschließenden Worker-Diagnostikkorrektur: CPU-Fokus
  `[forward-downstream],[forward-runner],[geometry-parallel],[synthetic-quality]`
  mit **15 Testfällen / 354 Assertions** bestanden.
- Native GTX 1660 Ti, Treiber 610.57.04:
  `[cuda-parity],[forward-runner],[forward-downstream]` mit
  **19 Testfällen / 387.531 Assertions** bestanden. Der CPU-Scheduler-Test
  erzwingt auf GPU-Hosts den CPU-Fallback; CUDA meldet null CPU-Reduktionsworker.
- Reale 600-Frame-Läufe sind im folgenden Laufabschnitt dokumentiert. Die Tests belegen weiterhin
  keine reale Solver-/Katalog-/HMS-Abnahme.

### Reale 600-Frame-Läufe (2026-09-09)

Die Läufe wurden mit dem migrierten M6-Basisprofil, `internal_scale=2`,
`output_scale=1`, `parallel_workers=8` und einem expliziten
`memory_budget=16384` gestartet. Das Zielverzeichnis lag auf dem großen
Arbeitsvolume unter `tile_compile/p6_runs`; das frühere `/media/tc_500`-Volume
war für den projizierten Record-Bestand nicht ausreichend.

| Klasse | Laufverzeichnis | Ergebnis bis zum Gate |
|---|---|---|
| affin (M31) | `p6_runs/20260909_140715_ad7c7da2` | 600/600 Frames; Registrierung 600 gültig, 0 lokale Modelle, affine/field-rotation; `NORMALIZED_CACHE` ok; `SAMPLING_GEOMETRY` ab 12:27:25Z; kontrollierter Stopp nach 2400 s; kein Drizzle/HMS |
| lokal verzerrt (M42) | `p6_runs/20260909_144826_7db55408` | 600/600 Frames; Registrierung 600 gültig, 4 Frames mit aktivem lokalem Modell, 596 affine-only; `NORMALIZED_CACHE` ok; `SAMPLING_GEOMETRY` ab 12:58:35Z; kontrollierter Stopp nach 2400 s; kein Drizzle/HMS |

Die vor der Geometrie erzeugten Laufdaten umfassten etwa 38 GiB (M31) bzw.
23 GiB (M42). Der M31-Lauf auf `/media/tc_500` mit demselben 16-GiB-Budget
wurde zuvor wegen des dort knappen freien Plattenplatzes kontrolliert beendet;
der erste 4-GiB-Versuch brach bereits mit
`DRIZZLE_MEMORY_BUDGET` ab. Diese Vorläufe sind Diagnoseartefakte und keine
zusätzlichen P6-Abnahmen.

Die beiden finalen Läufe liefen bereits mit `memory_budget=16384`. Das Budget
ist eine Pipeline-Admission-Grenze und keine harte Obergrenze für den gesamten
Prozess-RSS; während der M42-Registrierung wurde zeitweise ein RSS-Snapshot von
rund 27,6 GiB beobachtet. Für eine Wiederholung müssen deshalb sowohl das
Budget als auch der reale verfügbare RAM gemeinsam vorab geprüft werden.

Die Registrierung nutzte den OpenCV-CUDA-Prewarp (GPU); die eigentliche
Registrierung lief mit acht CPU-Workern. Beide Läufe meldeten
`ASTAP not available`, obwohl `/home/lux/.local/share/tile_compile/astap/astap_cli`
ausführbar vorhanden ist. Die aktuelle Verfügbarkeitsprüfung akzeptiert im
Katalogverzeichnis nur `.290`/`.291`; vorhanden sind dort `.1476`-Dateien und
`d80_star_database.zip`. Das ist eine erkannte Prüfungs-/Kompatibilitätslücke,
kein fehlendes Binary. Da HMS nicht erreicht wurde, sind Solver, BGE, PCC und
HMS aus diesen Läufen nicht bewertet.

**Abnahmeentscheidung:** Die 600-Frame-Voraussetzung und die Klassenbelege sind
erfüllt, die End-to-End-Zeitgrenze und der HMS-Commit sind nicht erfüllt bzw.
nicht gemessen. P6 bleibt daher offen; M10 bleibt blockiert. Der dominante
Restterm ist die CPU-bound `SAMPLING_GEOMETRY`-Phase. Ein nächster Lauf muss
deren Cache-/Range-Read-Implementierung und die ASTAP-Auflösung separat
beheben oder mit einer begründeten neuen Budgetierung neu messen.

---

## 1. Was §3.4 verlangt (Abnahmekriterien, wörtlich zusammengefasst)

- Zielband **1800–2400 s**, harte Obergrenze **2400 s**, gemessen vom Laufstart
  bis zum Commit der HMS-Ausgabe, **einschließlich** Scan, Kalibrierung mit
  vorhandenen Mastern, Normalisierung, Registrierung, Geometrie, Q-Maps, Drizzle,
  Mehrband, Ausgabe, BGE/PCC/HMS, Astrometrie und **allen** Transfers/Hashes/I/O.
- OSC, internes 2×-Raster, Ausgabe 1×, unveränderte Qualitäts-Gates.
- **Alle 600 angebotenen Frames** durchlaufen die reguläre Selektion. **Kein**
  `max_frames`, **kein** Downscale.
- Mindestens **ein affiner** und **ein lokal verzerrter** Datensatz, je mit
  **600 verschiedenen Realframes**. Reine Rotation ist affin und **kein** Beleg,
  dass ein lokales Modell nötig ist — die Notwendigkeit ist pro Datensatz aus dem
  Laufartefakt zu belegen (Anteil Frames mit aktivem lokalem Modell,
  Rotationswinkel).
- Pro Klasse **zwei vollständige Kaltläufe** ohne lauf­abhängige Cache-Wieder­
  verwendung, **beide ≤ 2400 s**.
- **Referenzprofil einfrieren**: CPU/GPU, RAM, SSD, Threadzahl, Treiber/Compiler/
  Binary-Hash, vollständige Config, Input-Manifest mit Datei-Hashes.
- P6-Planungsziel §11.14.7: **projizierte Kette ≤ 1920 s** (20 % Reserve zur
  2400-s-Grenze). Diese Reserve ist kein behaupteter erreichbarer Wert.
- Scheitert die Abnahme: **M10 blockiert**; den dominierenden Restterm gezielt
  neu entwerfen. Weder Minimax noch zusätzliche Threads ohne Messung als Lösung
  deklarieren.

---

## 2. Historische Datenlage — keine aktuelle Endzeitprognose

Alle drei Läufe: OSC, 3840×2160, `internal_scale=2` / `output_scale=1`,
`parallel_workers=8`, GTX 1660 Ti vorhanden. Phasendauern aus
`runs/<lauf>/logs/run_events.jsonl` (`phase_start`→`phase_end`).

| Phase | m31 40 f (M6, 06.09.) | m42 40 f (M6, 06.09.) | m66 100 f (M8 `27ba6e82`, 08.09.) |
|---|--:|--:|--:|
| SCAN_INPUT | 6,3 | 1,0 | 1,6 |
| CHANNEL_SPLIT | 0 | 0 | 0 |
| NORMALIZATION | 30,6 | 28,0 | 65,2 |
| REGISTRATION | 21,3 | 20,3 | 80,4 |
| NORMALIZED_CACHE | 0,95 | 0,87 | 2,3 |
| SAMPLING_GEOMETRY | 565,0 | 1249,0 | 2158,2 |
| COMMON_OVERLAP | 0,21 | 0,24 | 0,2 |
| SOURCE_QUALITY_MAPS | 120,1 | 120,5 | 279,0 |
| GLOBAL_QUALITY | 17,8 | 17,6 | 36,8 |
| FORWARD_DRIZZLE | 1576 / 1552 (2 Läufe) | 2930 / 2940 / 2954 / 2875 / 2843 | *durch Supervisor abgebrochen* |
| MULTIBAND | 38,7 / 33,6 | 43,3 / 37,7 / 34,5 / 32,7 / 35,0 | *nicht erreicht* |
| Ausgabe / BGE / PCC / HMS / Astrometrie | *läuft im `reconstruct`-Pfad nicht* | *dito* | *dito* |

Frame-Bezug (aus den `phase_end`-Feldern, **nicht** geschätzt):
- **NORMALIZATION / REGISTRATION**: m66 `global_registration.json num_frames=100` → alle 100.
- **SOURCE_QUALITY_MAPS**: `phase_end.frames` = m31 **40**, m42 **40**, m66 **100**
  (nicht 64 — der `linearity.max_frames=64`-Cap greift erst bei der
  Drizzle-Eingangsselektion). ⇒ **3,00 / 3,01 / 2,79 s pro Frame** — konsistent,
  ~linear.
- **SAMPLING_GEOMETRY**: m66 sah ~64 Frames (Selektions-Cap), m31/m42 40. Zahl
  ohnehin überholt (Klasse B).
- **FORWARD_DRIZZLE**: m31/m42 auf 40 Frames (`max_frames=64`, aber nur 40
  entdeckt). m66 abgebrochen.
- **MULTIBAND**: m31/m42 40 Frames; canvasgebunden, praktisch framezahl-unabhängig.

### 2.1 Zwei Provenienzklassen — pro Phase getrennt behandelt

**Klasse A — historische Messungen, für eine neue Prognose erneut zu prüfen:** SCAN, CHANNEL_SPLIT, NORMALIZATION, REGISTRATION, NORMALIZED_CACHE,
COMMON_OVERLAP, SOURCE_QUALITY_MAPS, GLOBAL_QUALITY, MULTIBAND.

**Klasse B — überholt oder nie gemessen, Zahl nicht durch Skalierung
gewinnbar:**

- **SAMPLING_GEOMETRY**: Die 565 / 1249 / 2158 s stammen vom **Pre-P1/P2-Pfad**
  (K-fache Geometrieberechnung, einthreadig). P1/P2 haben diesen Pfad ersetzt.
  Die frühere Cache-Bau-Extrapolation aus §30.69 ist kein realer
  3840×2160-Lauf und wird hier nicht als aktuelles Budget übernommen.
- **FORWARD_DRIZZLE**: m31/m42 liefen vor Cache-/Schedulerintegration.
  Die damaligen Zeiten können nicht direkt durch die heutige Workerzahl
  geteilt werden. Aktuelle Messungen fehlen (§5).
- **Ausgabe / BGE / PCC / HMS / Astrometrie**: in diesen historischen Läufen
  nicht ausgeführt. Die inzwischen integrierte gemeinsame Funktion benötigt
  noch reale End-to-End-Verifikation; es gibt keinen nachgetragenen Messwert.

---

## 3. P6-Lauf-Konfiguration

### 3.1 Kein handgeschriebenes Single-Method-Config

Die Referenzläufe nutzen das Legacy-YAML über den migrierenden Loader
(`from_yaml_text_migrated`). Was die Migration für `internal_scale`/`output_scale`
exakt erzeugt, ist hier **nicht** zu erraten. Vorgehen: den **M66-Gate-Config als
Basis** nehmen, die §3.4-Pflicht­änderungen als Delta anbringen (§3.2), und die
**tatsächlich aufgelösten Werte** über ein Trocken-Validat bzw. einen
4-Frame-Smoke-Lauf aus `run_provenance.json` / dem Config-SHA verifizieren
lassen — **bestätigen statt behaupten**.

Basis: `runs/m66_m9gate_20260908/config.yaml` (Legacy-Format,
`pipeline.mode: production`, `aqmh.reconstruction.*`).

### 3.2 Pflicht-Deltas zur Basis

| Schlüssel | Basis (m66) | P6 | Grund |
|---|---|---|---|
| `linearity.max_frames` | 64 | **entfernen** (bzw. ≥ 600) | §3.4: alle 600 Frames durch reguläre Selektion |
| Datensatz / Frame-Manifest | 100 M66-Frames | **600 verschiedene Realframes** pro Klasse | §3.4 |
| `runtime_limits.memory_budget` | 8192 | **16384 als Budgetkandidat** — nur nach RAM-/cgroup-Preflight, explizit einfrieren (§4) | §3.4 Referenzprofil |
| `runtime_limits.parallel_workers` | 8 | **explizit festlegen und messen**; Requested/Budgeted/Used getrennt protokollieren, keine lineare `/N`-Prognose | §5 |
| `aqmh.reconstruction.chunk_rows` | 0 (auto) | 0 lassen **oder** eingefroren dokumentieren | Referenzprofil |
| interne/Ausgabe-Skala | (migriert) | **verifizieren = 2 / 1** nach Migration | §3.4 |
| Kalibrierung | vorhandene Master | vorhandene Master (Darks/Flats), im Manifest gehasht | §3.4 |
| `astrometry` / `bge` / `pcc` / `hypermetric_stretch` | (siehe §7) | müssen im Ausführungspfad **aktiv und gemessen** sein | §3.4 — aktiviert und erfolgreich nachzuweisen |

### 3.3 Datensatzwahl

- **Lokal verzerrte Klasse**: DwarfII alt-az (Feldrotation), hier die
  M42-Familie aus `verify_m6m7/m42_base.yaml` mit **600** echten Frames.
  Beleg der Lokalmodell-Notwendigkeit aus dem Lauf:
  `forward_drizzle.json` / Checkpoint-Feld `hybrid_local_frames` bzw.
  `has_smooth_local_model`-Anteil und die Rotationswinkelspanne. (§30.57: M42
  40 f ergab `hybrid_local_frames=1` — bei 600 f neu zu belegen.)
- **Affine Klasse**: M31 (`verify_m6m7`) wurde mit 600 Frames verwendet. Die
  Laufartefakte zeigen 0 aktive lokale Modelle; reine Rotation zählt als affin
  und ist kein Beleg für die lokal-verzerrte Klasse.

### 3.4 Skalierungsleiter 40 / 100 / 200 / 600 (Plan §11.14.7, erster Punkt)

Vor den beiden 600-f-Kaltläufen pro Klasse eine Leiter bei **realer
3840×2160-Canvas** und **realem `internal_scale=2`**, gleicher Config bis auf die
Frameanzahl. Zweck: die Skalierungsannahmen aus §4/§5 gegen echte Punkte prüfen —
**keine O(N·P)-RAM-Zunahme bei gleichem Budget**, **keine erneute K-fache
Geometrie**, wenn Kandidaten mehr Zeilenbudget verbrauchen (Plan-Wortlaut).
Synthetische Daten dürfen die **Lastskalierung** testen; die **echte
Qualitätsabnahme** bleibt davon getrennt (§21 / M9). Pro Sprosse festhalten:
Peak-RSS, `geometry_cache_max_row_record_count`, `enumerate_call_count`,
Phasendauern, `forward_drizzle_reduction_workers`.

---

## 4. Speicher-/I-O-Referenzprofil (§3.4 „Referenzprofil einfrieren")

Aus der P4-Analyse (§30.66): der residente framezahl­abhängige Term sind die
Registrierungs-Proxies, nach der Admission-Term-Korrektur **`pixels/4`** pro
Frame. Bei 3840×2160 = 8,29 Mpx → **≈ 2,07 M Einträge/Frame**.

- 600 Frames × 2,07 M Einträge × 4 B ≈ **4,97 GB** allein für die Proxy-Residenz,
  **vor** Arbeitsmenge, Canvas-Bändern (`A/B/QA*`), Kandidatenpuffern und
  CFITSIO-I/O.
- Der eingefrorene 4-GB-Envelope (§11.13) **fasst 600 Frames bei 3840×2160
  nicht** — das ist bereits bei M9-Start festgestellt (M66 lief bei 8 GB).
- **Folge:** P6 muss sein `memory_budget` explizit deklarieren. 16384 MiB ist
  ein Kandidat; dessen Eignung wird anhand aktueller RAM-/cgroup-Grenzen,
  vorhandener Proxies, Downstream-Arbeitssätze und der Leiter geprüft. Frühere
  Angaben „32 GB frei“ sind keine aktuelle Ressourcenprüfung.
- Pro Worker entstehen neben dem Geometry-Reader-Block framezahlabhängige
  temporäre Clipping-/Alpha-Vektoren. Die gemeinsamen Bild-/Kandidatenpuffer
  werden nicht dupliziert. Der CPU-Plan budgetiert dennoch den vollständigen
  zusätzlichen Scratch einschließlich Reader-Reallokation und Reserve;
  zu viele Worker werden reduziert, passt ein Worker nicht, wird abgebrochen.
- Der Geometry-Index skaliert mit **Framezahl × Quellhöhe**. Eine Hochrechnung
  von ~3 KiB/Frame bei 96 Zeilen auf 2160 Zeilen muss auch die Höhe skalieren:
  etwa 600 × 3 KiB × 2160/96 ≈ **39,6 MiB pro solcher Indexvariante**,
  nicht 1,8 MiB. Variantenanzahl und Metadaten zusätzlich berücksichtigen.
- Record-Disk, Hash-Lesevolumen, Reader-I/O und persistente Q-/Normalized-
  Caches separat budgetieren. Kleine warme Testdateien belegen keinen
  nachhaltigen SSD-Durchsatz für mehrere hundert GiB.

Einzufrieren (Vorlage, vom Benutzer auszufüllen):

```
CPU:                 <Modell, Kernzahl, Basistakt>
GPU:                 <Modell, VRAM, Treiberversion>
RAM:                 <GB, Typ, Takt>
SSD:                 <Modell, freier Platz, fs>
parallel_workers:    <n>
memory_budget:       <MB>   (Erwartung >= 8192, Begründung siehe oben)
chunk_rows:          <auto/fix>
Compiler:            <gcc/clang Version, Flags: -ffp-contract=off-Liste aktiv>
Binary-SHA256:       <hash des runner-Binaries>
Config-SHA256:       <hash der migrierten P6-Config>
Input-Manifest:      <Datei -> SHA256, 600 Zeilen pro Klasse>
Kalibrier-Master:    <dark/flat -> SHA256>
CUDA-Backend:        <auto/on/off, tatsächlich genutzt?>
```

---

## 5. Phasenbudget: korrigierte Evidenzlage

Die historischen Messungen in §2 bleiben als Belege erhalten. Sie sind keine
Messung des aktuellen vollständigen 600-Frame-Pfads.

**Rechenkorrektur:** Die frühere Klasse-A-Tabelle summiert sich auf rund
**2850–3000 s** (Scan ~5 + Normalisierung ~390 + Registrierung ~480 +
Normalized-Cache ~15 + Overlap ~1 + Q-Maps ~1700–1800 + Global-Quality ~220–270 +
Multiband ~40). Der früher eingesetzte Sockel von 1500 s war falsch. Darauf
beruhende Gesamtsummen 4450–11200 s sind zurückgezogen.

**Methodische Korrektur:** Pre-P1/P2-Drizzlezeiten enthalten wiederholte lokale
Geometrieberechnung. Sie dürfen weder unverändert dem neuen Cachepfad
zugeordnet noch pauschal durch die CPU-Kernzahl geteilt werden. Ein separater
Cache-Bauterm plus unveränderter alter lokaler Drizzletakt kann denselben
Geometrieaufwand doppelt zählen. Der aktuelle `SOURCE_QUALITY_MAPS`-Orchestrator
arbeitet Frame für Frame; aus `parallel_workers=8` in der Config folgt keine
achtfache Parallelität dieser Phase. Vermutete I/O-Bindung ist zu messen.

### 5.1 Neu zu messende Größen

| Messung | Trennung / Vergleich |
|---|---|
| Geometrie | Wall-Zeit für Build, Records, fsync, Commit/Hashes und anschließende Coverage; Varianten getrennt |
| Drizzle CPU | Cache aktiv, 1/2/4/budgetverträgliche Worker, tatsächliche Teamgröße, gleiche Bild-/Canvasgeometrie |
| Drizzle CUDA/Hybrid | GPU-Backend separat; CPU-Workerparameter ist kein Device-Speedup |
| Q-Maps/Quellen | Rechenzeit, Dateiladungen, SHA-I/O, verifizierte Wiederverwendung |
| Downstream | STACKING, Astrometrie, BGE, PCC, HMS einschließlich Dateischreiben/Commit |
| Gesamtlauf | Gemeinsame monotone Wall-Zeit vom Start bis HMS-Commit; Summe paralleler Taskzeiten ist kein Ersatz |

**Entscheidungsregel:** Erst die aktuelle Leiter liefert eine belastbare
Projektion. Das Planungsziel bleibt ≤1920 s, die tatsächliche P6-Grenze ≤2400 s.
Bei überschrittener Projektion ist ein weiterer Diagnoselauf von einer
Erfolgserwartung zu unterscheiden; M10 bleibt ohne bestandene Abnahme blockiert.
Keine Gate-Lockerung und keine neue Numerik aus einer veralteten Projektion
ableiten.

---

## 6. Ausführungs-Checkliste (für den Benutzer, wenn autorisiert)

1. Referenzprofil (§4-Vorlage) vollständig ausfüllen und einfrieren.
2. 600-Frame-Manifest je Klasse erzeugen, Datei-Hashes ins Manifest.
3. P6-Config aus M66-Basis + §3.2-Deltas; **`max_frames` entfernt** bestätigen;
   migrierte Werte via `run_provenance.json` / 4-Frame-Smoke prüfen
   (`internal_scale=2`, `output_scale=1`, `memory_budget` wie deklariert).
4. Skalierungsleiter 40/100/200/600 pro Klasse (§3.4 hier), Kennzahlen
   protokollieren.
5. Pro Klasse **zwei Kaltläufe in verschiedenen neuen Laufverzeichnissen**
   (keine laufabhängigen Caches übernehmen; bestehende Läufe nicht löschen), Wanduhr Laufstart →
   HMS-Commit.
6. Pro Lauf sichern: `logs/run_events.jsonl`, alle `artifacts/*.json`,
   `forward_drizzle_geometry_profile.json` (enthält
   `forward_drizzle_stage_stats_suppressed_reduction_workers`,
   `geometry_cache_max_row_record_count`), angeforderte/budgetierte/tatsächliche
   Workerzahl, Peak-RSS und GPU-Auslastung. CUDA verwendet keine CPU-
   Reduktionsworker; unterdrückte Zähler sind kein Cache-Nachweis.
7. Ergebnis gegen §3.4 auswerten: beide Läufe je Klasse ≤ 2400 s? Falls nein:
   dominierenden Restterm aus den realen Phasendauern bestimmen → Neuentwurf,
   M10 bleibt blockiert.

---

## 7. Verbleibende Start- und Abnahmevoraussetzungen

1. Zwei vollständige Input-Manifeste und Kalibriermaster festlegen, je 600
   verschiedene Realframes. Affine und lokal verzerrte Klasse aus den
   Registrierungsartefakten nachweisen; reine Feldrotation ist affin.
2. Aktuellen Runner bauen und relevante CPU-/CUDA-/Downstream-Tests bestehen
   lassen. Ein funktionierender synthetischer Eingang ersetzt keine reale
   Astrometrie-/PCC-/HMS-Verifikation mit Katalog und Solver.
3. Solver/Katalog vorhanden und lesbar; `astrometry.enabled`, `pcc.enabled`
   und `hypermetric_stretch.enabled` aktiv, BGE passend konfiguriert. Ein
   fehlender HMS-Commit oder übersprungene Pflichtphase ist kein P6-Pass.
4. RAM/cgroup/SSD-Preflight für beide Klassen, migrierte effektive Config,
   Master-/Input-/Binary-Hashes und Zielverzeichnisse einfrieren.
5. Aktuelle 40/100/200/600-Leiter bei realer Geometrie messen; Index-Residenz,
   vollständiger Scratch und Cache-I/O gehören zur Ressourcenprüfung.
6. Erst vollständige Kaltläufe bis HMS belegen P6. Der Gesamtzeitnachweis,
   reales MONO/Schmalband und die übrige M9-Matrix bleiben separate Abnahmen.

---

## 8. Was P6 **nicht** ist

- Kein Arbeitspaket P0–P5 hat P6 autorisiert (Plan §11.14: „Kein Arbeitspaket
  autorisiert einen Benutzerrun").
- Dieses Runbook **bereitet vor** und **projiziert**. Es startet nichts.
- Die 20-%-Reserve (≤ 1920 s projiziert) ist Planungsreserve, **kein** behaupteter
  erreichbarer Wert; §5 verlangt eine neue, aktuelle Messgrundlage.
