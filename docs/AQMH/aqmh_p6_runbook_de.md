# P6 — Runbook und Phasenbudget-Ableitung (Vorbereitung)

Status: **Vorbereitung. Kein Lauf autorisiert.** Dieses Dokument spezifiziert den
P6-Abnahmelauf (§11.14.7 des Implementierungsplans, Abnahme gemäß §3.4) und leitet
ein Phasenbudget aus den vorhandenen realen Messungen ab. Es führt selbst nichts
aus. P6 ist der Benutzerlauf; M9-Zuständigkeit, M10-Freigabe.

Erstellt 2026-09-09, Branch `CFA-aware-Forward-Drizzle`, HEAD `b23ba727`.
Protokollverweis: **§30.70**.

### Kernbefund vorab

- **`memory_budget` ≥ 8192 MB, explizit deklarieren + einfrieren.** 600
  Registrierungs-Proxies × `pixels/4` × 4 B ≈ **4,97 GB** allein, vor
  Arbeitsmenge — der 4-GB-Envelope fasst 600 Frames nicht (§4). Referenzbox hat
  **32 GB frei** → `memory_budget` bis **16384 MB** unbedenklich; empfohlen
  **16384 MB** (Kopf­raum für Arbeitsmenge + Kandidatenpuffer + CFITSIO-I/O), im
  Referenzprofil festhalten.
- **Projizierte Gesamtkette 600 f = ~4450 s (bester gerechneter Fall) bis
  ~11200 s**, gegen die harte 2400-s-Grenze aus §3.4 → **P6 als Abnahmelauf
  verfehlt die Grenze in jedem gerechneten Szenario** (§5).
- Drei dominierende Terme, keiner durch P0–P5 behoben: **SOURCE_QUALITY_MAPS
  ~1700–1800 s**, **FORWARD_DRIZZLE lokal ~2720–5400 s**, **Geometrie-Bau
  ~1440–4300 s** (§5.2).
- **Benannte Lücke:** BGE / PCC / HMS / Astrometrie laufen im `reconstruct`-Pfad
  nicht → keine Messung, obwohl §3.4 sie innerhalb der 2400 s verlangt (§7).
- Alle 600-f-Zahlen sind **Extrapolation** — es gibt keinen realen
  3840×2160/600-f-Lauf. P6 erzeugt ihn.

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

## 2. Datenlage — die realen Messungen, auf die sich die Ableitung stützt

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

**Klasse A — gemessen, Phase seither strukturell unverändert, skaliert mit
Framezahl:** SCAN, CHANNEL_SPLIT, NORMALIZATION, REGISTRATION, NORMALIZED_CACHE,
COMMON_OVERLAP, SOURCE_QUALITY_MAPS, GLOBAL_QUALITY, MULTIBAND.

**Klasse B — überholt oder nie gemessen, Zahl nicht durch Skalierung
gewinnbar:**

- **SAMPLING_GEOMETRY**: Die 565 / 1249 / 2158 s stammen vom **Pre-P1/P2-Pfad**
  (K-fache Geometrieberechnung, einthreadig). P1/P2 haben diesen Pfad ersetzt.
  Die 600-f-Zahl ist die **Cache-Bau-Extrapolation** aus §30.69
  (~1440 s / 16 Kerne), **kein** realer 3840×2160-Lauf.
- **FORWARD_DRIZZLE**: m31/m42 liefen mit **einem** Reduktions-Worker
  (Pre-P3-Teil-2). Die 600-f-Zahl braucht die Neuableitung aus §5 — und die ist
  eine **doppelte Annahme**: extrapolierte Einthread-Rate × angenommener
  Band-Speedup `/N`, wobei der Real-Canvas-Speedup von P3 Teil 2 **noch nicht
  gemessen** ist (§30.68 nennt das als offenen P6-Punkt).
- **Ausgabe / BGE / PCC / HMS / Astrometrie**: im `reconstruct`-Ausführungs­
  bereich (`execution_scope: forward_drizzle_m1_m3`) **nicht implementiert**.
  Es gibt **keine Messung**, weder bei 40 f noch bei 100 f. §3.4 verlangt diese
  Phasen **innerhalb** der 2400 s. → **benannte Lücke**, siehe §7.

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
| `runtime_limits.memory_budget` | 8192 | **16384** (32 GB frei, Kopfraum über 4,97 GB Proxys) — explizit + eingefroren (§4) | §3.4 Referenzprofil |
| `runtime_limits.parallel_workers` | 8 | **N = Referenzhardware-Kernzahl, explizit setzen** (M66 hatte 8; die §5-Projektion rechnet mit `/N` — bei N=8 bleibt es bei 8, für N=16 muss dieser Wert **von 8 auf 16 angehoben** werden) | §5 |
| `aqmh.reconstruction.chunk_rows` | 0 (auto) | 0 lassen **oder** eingefroren dokumentieren | Referenzprofil |
| interne/Ausgabe-Skala | (migriert) | **verifizieren = 2 / 1** nach Migration | §3.4 |
| Kalibrierung | vorhandene Master | vorhandene Master (Darks/Flats), im Manifest gehasht | §3.4 |
| `astrometry` / `bge` / `pcc` / `hypermetric_stretch` | (siehe §7) | müssen im Ausführungspfad **aktiv und gemessen** sein | §3.4 — derzeit Lücke |

### 3.3 Datensatzwahl

- **Lokal verzerrte Klasse**: DwarfII alt-az (Feldrotation), z. B. die
  M42-Familie aus `verify_m6m7/m42_base.yaml` — aber mit **600** echten Frames,
  nicht 40. Beleg der Lokalmodell-Notwendigkeit aus dem Lauf:
  `forward_drizzle.json` / Checkpoint-Feld `hybrid_local_frames` bzw.
  `has_smooth_local_model`-Anteil und die Rotationswinkelspanne. (§30.57: M42
  40 f ergab `hybrid_local_frames=1` — bei 600 f neu zu belegen.)
- **Affine Klasse**: **vom Benutzer zu benennen.** M31 (`verify_m6m7`) ist mit
  40 f affin gelaufen; ein 600-Frame-Äquivalent mit vernachlässigbarer
  Feldrotation (`auto_engine_rotation_threshold_deg` unterschritten) ist
  erforderlich. Reine Rotation zählt als affin und ist kein Beleg für die
  lokal-verzerrte Klasse.

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
- **Folge:** P6 muss sein `memory_budget` **explizit deklarieren** und im
  eingefrorenen Referenzprofil festhalten. Referenzbox: **32 GB frei** →
  **16384 MB empfohlen** (deckt 4,97 GB Proxys + Arbeitsmenge + stripe-begrenzte
  Kandidatenpuffer + CFITSIO-I/O mit Kopfraum). Die Zahl mit dieser Arithmetik
  **im Runbook-Kopf**, nicht als Fußnote — damit sie nicht mitten im Lauf
  entdeckt wird.
- P3 Teil 2 (§30.67) fügt **keinen** frame­skalierten RAM pro Worker hinzu
  (Bänder teilen `A/B/QA*/candidates`); der einzige Pro-Worker-Term ist ein
  `enumerate_stripe`-`ifstream` + `block`-Puffer, beschränkt durch
  `max_row_record_count × 72 B`. Das ändert die 8-GB-Erwartung nicht.

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

## 5. Phasenbudget — Ableitung und Projektion (600 Frames)

**Alle 600-f-Zahlen sind Extrapolationen. Es existiert kein realer
3840×2160/600-Frame-Lauf. P6 erzeugt ihn.**

**N = Kernzahl der Referenzhardware** (M66-Config: `parallel_workers=8`). Die
`/N`-Terme unten sind mit **N=8** (Config unverändert) **und N=16** (Wert von 8
auf 16 angehoben, §3.2) gerechnet. Wo `/N` steht, ist der Speedup als **linear
angenommen** und **nicht real gemessen** — bei der Geometrie widerspricht dem
bereits die §30.65-Messung (5,4× bei 8 Kernen, sublinear).

**Klasse A** — gemessen, Phase unverändert, Rate pro Frame:

| Phase | Rate | Quelle | 600-f-Projektion |
|---|--:|---|--:|
| SCAN_INPUT | ~const | alle | ~5 s |
| NORMALIZATION | 0,65 s/f | m66 65,2/100 | **~390 s** |
| REGISTRATION | 0,80 s/f | m66 80,4/100 | **~480 s** (Superlinearitätsrisiko) |
| NORMALIZED_CACHE | 0,023 s/f | m66 2,3/100 | ~15 s |
| COMMON_OVERLAP | ~const | alle | ~1 s |
| SOURCE_QUALITY_MAPS | **2,79–3,01 s/f** | m31 120,1/40 · m42 120,5/40 · m66 279,0/**100** (Nenner aus `phase_end.frames`) | **~1700–1800 s** ⚠️ |
| GLOBAL_QUALITY | 0,37–0,45 s/f | m66 36,8/100 · m31 17,8/40 | **~220–270 s** |
| MULTIBAND | ~const (canvasgebunden; m31 40 f ≈ m42 40 f) | m31/m42 ~37 s | **~40 s** |

**Klasse B** — überholt oder nie gemessen:

| Phase | Ableitung | N=8 | N=16 |
|---|---|--:|--:|
| SAMPLING_GEOMETRY (= Cache-Bau) | §30.69 ~2150 ns/Sample × 600 f × 3840×2160 × 2 Varianten ≈ **23000 s einthreadig**; Speedup **angenommen linear**, §30.65 misst 5,4× @ 8 K (sublinear) | **~4300 s** (23000/5,4) | **~1440 s** (23000/16, optimistisch) |
| FORWARD_DRIZZLE — affin | m31 1560 s/40 f = 39 s/f → **23400 s einthreadig**; P3-Teil-2-Band-Speedup **angenommen linear, real-canvas nicht gemessen** (§30.68) | **~2900 s** | **~1460 s** |
| FORWARD_DRIZZLE — lokal verzerrt | m42 ~2900 s/40 f = 72,5 s/f → **43500 s einthreadig** (der §30.69 `std::exp`-Hotspot) | **~5400 s** | **~2720 s** |
| Ausgabe / BGE / PCC / HMS / Astrometrie | **keine Messung** — §7 | **unbekannt** | **unbekannt** |

### 5.1 Projizierte Gesamtkette (Klasse A ~1500 s + Klasse B)

| Klasse | N=8 | N=16 |
|---|--:|--:|
| **affin** | ~1500 + 4300 + 2900 + 40 ≈ **~8700 s + Ausgabe/HMS** | ~1500 + 1440 + 1460 + 40 ≈ **~4450 s + Ausgabe/HMS** |
| **lokal verzerrt** | ~1500 + 4300 + 5400 + 40 ≈ **~11200 s + Ausgabe/HMS** | ~1500 + 1440 + 2720 + 40 ≈ **~5700 s + Ausgabe/HMS** |

Selbst der **günstigste** dargestellte Fall (affin, N=16, linearer Speedup
angenommen, Ausgabe/HMS = 0) liegt bei **~4450 s ≈ 1,85× über der harten
2400-s-Grenze**. Der realistischere Fall (N=8 oder sublinearer Speedup, lokale
Klasse) liegt bei **~3–5×**.

### 5.2 Ehrliche Schlussfolgerung

Bei aktueller Leistung ist **zu erwarten, dass P6 als Abnahmelauf die §3.4-Grenze
verfehlt** — in **jedem** der oben gerechneten Szenarien. Die drei dominierenden
Terme (Reihenfolge je nach N):

1. **SOURCE_QUALITY_MAPS ~1700–1800 s** — allein an der 2400-s-Grenze. Nicht durch
   P0–P5 berührt. Skaliert linear mit Framezahl (2,8–3,0 s/f, Nenner verifiziert);
   nutzt `parallel_workers=8` bereits, vermutlich teils I/O-gebunden.
   **Hauptkandidat für den Neuentwurf.**
2. **FORWARD_DRIZZLE** — affin ~1460–2900 s / lokal ~2720–5400 s je nach N. P3
   Teil 2 parallelisiert die Reduktion streifenintern (Speedup real-canvas noch
   nicht gemessen); der lokale Pfad bleibt am `std::exp` / `sample_leaves`
   (§30.69) hängen → braucht die separat abgesicherte Numerikrevision
   (Vektor-/Minimax-`expf`).
3. **SAMPLING_GEOMETRY / Cache-Bau ~1440–4300 s** je nach N — Extrapolation, kein
   realer Lauf, §30.65 misst sublinearen Speedup. Hebel: mehr Kerne (P3 skaliert
   per Frame, aber sublinear) oder Numerikrevision.

Das entspricht der Plan-Vorgabe §11.14.7: „Falls sie scheitert: M10 blockiert;
den dominierenden Restterm gezielt neu entwerfen." Die drei Terme sind hiermit
**mit Zahlen benannt** — das ist der Input für diesen Neuentwurf. P6 misst, ob die
Extrapolation stimmt (insbesondere den real-canvas P3-Speedup und die
Ausgabe/HMS-Lücke), und liefert die realen Phasendauern, an denen der Neuentwurf
ansetzt.

---

## 6. Ausführungs-Checkliste (für den Benutzer, wenn autorisiert)

1. Referenzprofil (§4-Vorlage) vollständig ausfüllen und einfrieren.
2. 600-Frame-Manifest je Klasse erzeugen, Datei-Hashes ins Manifest.
3. P6-Config aus M66-Basis + §3.2-Deltas; **`max_frames` entfernt** bestätigen;
   migrierte Werte via `run_provenance.json` / 4-Frame-Smoke prüfen
   (`internal_scale=2`, `output_scale=1`, `memory_budget` wie deklariert).
4. Skalierungsleiter 40/100/200/600 pro Klasse (§3.4 hier), Kennzahlen
   protokollieren.
5. Pro Klasse **zwei Kaltläufe** (Cache-Verzeichnisse zwischen den Läufen
   entfernen / lauf­abhängige Wiederverwendung ausschließen), Wanduhr Laufstart →
   HMS-Commit.
6. Pro Lauf sichern: `logs/run_events.jsonl`, alle `artifacts/*.json`,
   `forward_drizzle_geometry_profile.json` (enthält
   `forward_drizzle_stage_stats_suppressed_reduction_workers`,
   `geometry_cache_max_row_record_count`), Peak-RSS, GPU-Auslastung.
7. Ergebnis gegen §3.4 auswerten: beide Läufe je Klasse ≤ 2400 s? Falls nein:
   dominierenden Restterm aus den realen Phasendauern bestimmen → Neuentwurf,
   M10 bleibt blockiert.

---

## 7. Benannte Lücken (blockieren die §3.4-Abnahme, unabhängig von der Laufzeit)

1. **BGE / PCC / HMS / Astrometrie laufen im `reconstruct`-Pfad nicht**
   (`execution_scope: forward_drizzle_m1_m3`). §3.4 verlangt sie **innerhalb** der
   2400 s. Für diese Phasen existiert **keine Messung** — das Budget in §5 hat
   dort ein Loch, das nicht mit einer Schätzung gefüllt wird. (M9/M10-Arbeit:
   diese Phasen in den Ausführungspfad bringen.)
2. **Kein MULTIBAND-/Ausgabe-Wert bei 100 f oder 600 f.** MULTIBAND ist bei 40 f
   canvasgebunden (~37 s) und dürfte ~konstant bleiben; die Ausgabe-Serialisierung
   (6 × 33 MB FITS bei m42, mehr bei 600 f Mehrband) ist nicht separat gemessen.
3. **Kein realer 3840×2160/600-Frame-Lauf.** Jede 600-f-Zahl hier ist
   Extrapolation aus 40-f-/100-f-/Sprossen-Daten.
4. **Kein realer MONO-/Schmalband-Datensatz.** M31 und M42 sind beide OSC. §3.4
   fordert das nicht explizit, aber die generelle Abnahme (M9) schon.
5. **P3-Teil-2-Reduktions-Speedup real-canvas nicht gemessen** (§30.68). Die
   `/N`-Terme in §5 nehmen ihn linear an; §30.65 misst für den Geometrie-Bau
   5,4× bei 8 Kernen (sublinear). Die Leiter (§3.4 hier) liefert die realen
   Speedup-Punkte.
6. **SOURCE_QUALITY_MAPS-Skalierung aus 40-f- und 100-f-Punkten.** Nenner
   verifiziert (`phase_end.frames`), Rate über drei Punkte konsistent
   (2,79–3,01 s/f); Restunsicherheit ±15 %. Die Leiter schließt das.

---

## 8. Was P6 **nicht** ist

- Kein Arbeitspaket P0–P5 hat P6 autorisiert (Plan §11.14: „Kein Arbeitspaket
  autorisiert einen Benutzerrun").
- Dieses Runbook **bereitet vor** und **projiziert**. Es startet nichts.
- Die 20-%-Reserve (≤ 1920 s projiziert) ist Planungsreserve, **kein** behaupteter
  erreichbarer Wert — und die Projektion in §5 liegt ohnehin weit darüber.
