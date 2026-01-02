# Tile Compile – Parallele Ausführung mit RabbitMQ

**Status:** Optionales Erweiterungsdokument
**Gültigkeit:** kompatibel mit *tile_basierte_qualitatsrekonstruktion_methodik.md*, PROC (Clean Break) und YAML

---

## 1. Ziel und Einordnung

Dieses Dokument beschreibt eine **optionale, nicht‑normative Parallelisierungsarchitektur** für *Tile Compile*, basierend auf **RabbitMQ**. Ziel ist die **horizontale Skalierung** der rechenintensiven Tile‑Analyse über mehrere Worker‑Prozesse oder Hosts hinweg, ohne die deterministische Semantik der Methodik zu verletzen.

Die Parallelisierung ist:

* **optional** (Single‑Node‑Ausführung bleibt Referenz)
* **deterministisch**
* **fehlertolerant**
* **reproduzierbar**

---

## 2. Designprinzipien

Die Parallelisierung folgt fünf harten Prinzipien:

1. **Keine Änderung der Methodik**
   Parallelisierung beeinflusst *nur* die Ausführung, nicht die mathematische Definition.

2. **Master‑determinierte Reduktion**
   Alle nicht‑assoziativen Operationen (Overlap‑Add, Summation) erfolgen zentral.

3. **Idempotente Tasks**
   Jeder Task kann ohne Seiteneffekte erneut ausgeführt werden.

4. **Deterministische Ordnung**
   Ergebnisaggregation erfolgt in stabil sortierter Reihenfolge.

5. **Explizite Trennung von Produktion und Diagnose**
   Diagnoseartefakte dürfen den Produktionspfad nicht blockieren.

---

## 3. Architekturübersicht

### 3.1 Rollen

* **Master**

  * liest Konfiguration
  * erzeugt Tile‑Tasks
  * aggregiert Ergebnisse
  * führt Overlap‑Add aus

* **Worker**

  * konsumieren Tile‑Tasks
  * führen vollständige Tile‑Analyse und ‑Rekonstruktion aus
  * liefern rekonstruierte Tile‑Blöcke zurück

* **RabbitMQ**

  * Task‑Verteilung
  * Retry‑ und Fehlerbehandlung

### 3.2 Betriebsmodi & Orchestrierung

Dieses Kapitel beschreibt die **organisatorische Verteilung** (Mainserver vs. Worker) in zwei Betriebsmodi. Es geht hierbei ausschließlich um die Parallelisierungsausführung (nicht um App‑Features).

#### Modus A: Local (Single‑Host)

Ziel: schnelle Entwicklung/Debugging der Parallelisierung bei minimaler Infrastruktur.

* Mainserver und Worker laufen auf demselben Host.
* RabbitMQ läuft lokal (z. B. Docker Compose Service).
* Worker skalieren horizontal über mehrere lokale Prozesse/Container.
* Diagnose‑Queues/Artefakte können aktiviert werden, dürfen aber den Produktionspfad nicht blockieren.

Empfohlene Verifikation (automatisierbar):

* RabbitMQ erreichbar (Ping)
* Queues vorhanden: `tile.compute`, `tile.results`, `tile.diagnostics`, `tile.dlq`
* Worker konsumiert Tasks (Consumer‑Count > 0)
* Dedupe funktioniert: für (`run_id`, `tile_id`) entsteht genau ein akzeptiertes Ergebnis

#### Modus B: Production (Mainserver + Remote Worker)

Ziel: horizontale Skalierung über mehrere Hosts.

* **Mainserver** betreibt die zentrale RabbitMQ‑Instanz.
* **Worker‑Hosts** verbinden sich über ein Overlay‑Netz (z. B. Tailscale/NetBird) zum Mainserver.
* RabbitMQ ist idealerweise nur im Overlay erreichbar (nicht öffentlich im Internet).
* Worker laufen als eigenständige Deployments (z. B. per Docker Compose auf den Worker‑Hosts).

Orchestrierungsprinzip (Setup‑Script‑Style):

* Mainserver‑Setup startet RabbitMQ und erzeugt/verwaltet Queues.
* Worker‑Setup verbindet sich ausschließlich als Client und skaliert über `N` Worker‑Instanzen.
* Test/Verifikation kann aus dem Mainserver heraus erfolgen (z. B. „Worker B/C erreichbar und Worker‑Container laufen“).

Empfohlene Verifikation (automatisierbar):

* Mainserver: RabbitMQ ping
* Worker‑Hosts erreichbar (Overlay‑Ping / SSH optional)
* Jeder Worker hat Zugriff auf:
  * Task‑Queue (consume)
  * Results‑Queue (publish)
  * Shared Storage/Objekt‑Storage für `tile_data_ref`
* End‑to‑End: Tasks werden abgearbeitet, Ergebnisse erscheinen in `tile.results`, und Master kann alle Tiles aggregieren.

Hinweis:

* Der Modus „Production“ ändert keine Semantik. Determinismus wird weiterhin über Master‑Aggregation erzwungen.

### 3.3 Run‑Lifecycle (Setup‑Script‑Style, methodisch)

Dieses Kapitel beschreibt den **Ablauf eines Runs** so, dass er in einem Setup‑Script (local/production) abbildbar ist.

#### Phase 0: Run‑Initialisierung

* Master erzeugt `run_id` (UUID) und schreibt ein **Frames‑Manifest** (sortierte Frame‑Liste).
* Master berechnet `frames_manifest_id` (Hash über Manifest).
* Master berechnet `config_hash` (Hash über relevante Teile von `tile_compile.yaml`).
* Master erzeugt Tile‑Grid deterministisch.

#### Phase 1: Dispatch

* Master publiziert Tile‑Tasks nach `tile.compute`.
* Jeder Task enthält mindestens: `run_id`, `correlation_id`, `frames_manifest_id`, `config_hash`, `tile_id`, `tile_bbox`.

#### Phase 2: Compute (Worker)

* Worker validiert Task‑Kompatibilität:
  * `frames_manifest_id` bekannt/abrufbar
  * `config_hash` passt zur geladenen Konfiguration
  * `input_stage` wird erfüllt (z. B. `registered_normalized`)
* Worker verarbeitet Tile deterministisch und schreibt Ergebnis unter `tile_data_ref`.
* Worker publiziert Result nach `tile.results` und ACKt erst danach den Task.

#### Phase 3: Aggregate (Master)

* Master sammelt Results bis alle Tiles vollständig sind.
* Master dedupliziert über (`run_id`, `tile_id`) und prüft `tile_data_checksum`.
* Master führt Overlap‑Add deterministisch aus (stabile Tile‑Reihenfolge).

#### Phase 4: Abschluss & Cleanup

* Diagnoseartefakte werden optional gesammelt.
* `tile.compute`/`tile.results` können pro Run über TTL/Auto‑Delete oder Routing‑Keys isoliert werden.

### 3.4 Worker‑Inventar & Registrierung (Production)

In einem Setup‑Script‑Pattern wird die Worker‑Seite oft über eine Inventarliste/Env‑Konfiguration beschrieben.

Empfohlene Mindestangaben (konzeptionell):

* Worker‑ID (Name)
* Overlay‑Adresse (Tailscale/NetBird IP oder DNS)
* Worker‑Kapazität (z. B. Anzahl paralleler Consumer/Prozesse)

Semantik:

* Worker sind **stateless** bezüglich globaler Run‑Aggregation.
* Worker dürfen dynamisch kommen/gehen (RabbitMQ Consumer Model).
* Master kann optional eine Verifikation machen, ob erwartete Worker online sind (kein Muss für Funktionalität).

### 3.5 Overlay‑Netz (Tailscale/NetBird)

Für Remote Worker ist ein Overlay‑Netz empfohlen, damit RabbitMQ nicht öffentlich exponiert werden muss.

Empfehlung:

* RabbitMQ bindet nur auf Overlay‑Interface (oder wird per Firewall nur im Overlay freigegeben).
* Management‑UI (falls aktiv) ebenfalls nur im Overlay.
* Worker verbinden sich ausschließlich über die Overlay‑Adresse.

Hinweis:

* Auch im Overlay gelten weiterhin RabbitMQ‑Permissions (VHost/Users), weil Overlay nicht automatisch „Trusted“ bedeutet.

### 3.6 Storage‑Varianten für `tile_data_ref`

Die Parallelisierung funktioniert nur, wenn Master auf die von Workern erzeugten Tile‑Daten zugreifen kann.

Optionen:

* Shared FS (NFS/SMB) im Overlay
  * Pro: einfach (Pfad als `tile_data_ref`)
  * Contra: WAN‑Performance/Fragilität
* Objekt‑Storage (S3/MinIO)
  * Pro: WAN‑tauglich, gute Skalierung
  * Contra: Credentials/Policies notwendig

Faustregel:

* RabbitMQ für Control‑Plane (JSON), Storage für Data‑Plane (Tile‑Blöcke).

---

## 4. Task‑Granularität

### 4.1 Standard: Tile‑Tasks (empfohlen)

**Ein Task entspricht genau einem Tile** *t* über **alle Frames** *f*.

```text
Task = Tile t × Frames [0…N]
```

**Vorteile:**

* maximale Parallelität
* minimale Abhängigkeiten
* cache‑freundlicher I/O
* ideale Granularität für CPU‑lastige Workloads

---

### 4.2 Option: Tile‑Batch‑Tasks (Supertiles)

Mehrere Tiles werden zu einem Task zusammengefasst.

**Sinnvoll bei:**

* sehr kleinen Tiles
* hohem RabbitMQ‑Overhead

**Trade‑off:** weniger Parallelität, bessere I/O‑Amortisierung.

---

### 4.3 Option: Frame‑Chunks innerhalb eines Tiles (experimentell)

Ein Tile wird über mehrere Tasks aufgeteilt, die jeweils nur einen Frame‑Bereich verarbeiten.

**Nur sinnvoll, wenn:**

* extrem viele Frames
* sehr schneller lokaler I/O (NVMe, RAM‑Disk)

**Nachteile:**

* komplexe Teilreduktion
* höhere numerische Sensitivität

---

## 5. Task‑Payload (Minimaldefinition)

```json
{
  "task_type": "tile_compute",
  "correlation_id": "uuid",
  "run_id": "uuid",
  "config_hash": "sha256",
  "frames_manifest_id": "sha256",
  "input_stage": "registered_normalized",
  "tile_id": 123,
  "tile_bbox": [x0, y0, w, h],
  "frame_index_range": [0, 999],
  "metrics_config": {...},
  "seed": 42
}
```

### Pflichtfelder

* `tile_id` – eindeutige Tile‑Identifikation
* `tile_bbox` – räumliche Begrenzung
* `frame_index_range` – deterministischer Frame‑Bereich
* `seed` – deterministische Initialisierung

Hinweise:

* `frames_manifest_id` referenziert eine deterministische Frame‑Liste (sortierte Pfade + optional Checksums).
* `config_hash` stellt sicher, dass Worker exakt dieselbe Konfiguration verwenden.
* `input_stage` muss festlegen, ob die Worker bereits normalisierte Frames konsumieren.

---

## 6. Ergebnis‑Payload

### 6.1 Produktionsdaten

```json
{
  "correlation_id": "uuid",
  "run_id": "uuid",
  "tile_id": 123,
  "tile_data_ref": "path-or-object-key",
  "tile_data_checksum": "sha256",
  "tile_data_dtype": "float32",
  "tile_data_shape": [h, w],
  "sum_weights": "ΣW",
  "tile_median": "after bg subtraction"
}
```

Hinweis:

* Große Binärdaten sollten **nicht** direkt in RabbitMQ transportiert werden. Stattdessen wird `tile_data_ref`
  als Referenz auf ein Filesystem/Objekt‑Storage verwendet.

### 6.2 Diagnoseartefakte (separate Queue)

* Histogramme von Q_local
* Tile‑Gewichtsverteilungen
* QA‑Maps

Diese Artefakte sind **nicht blockierend**.

---

## 7. Queues und Routing

Empfohlene Queues:

* `tile.compute` – Produktions‑Tasks
* `tile.results` – Tile‑Ergebnisse
* `tile.diagnostics` – Diagnoseartefakte
* `tile.dlq` – Dead‑Letter‑Queue

RabbitMQ‑Features:

* Prefetch‑Limit (z. B. 1–2)
* Priority Queues (optional)
* Manual ACKs

Hinweis zur Message‑Größe:

* `tile.compute` und `tile.results` enthalten nur kleine JSON‑Payloads.
* Tile‑Daten werden über `tile_data_ref` außerhalb von RabbitMQ gespeichert.

---

## 8. Ablauf (End‑to‑End)

### 8.1 Master

1. liest YAML
2. erzeugt Tile‑Grid
3. erstellt deterministische Tasks
4. publiziert Tasks in `tile.compute`

---

### 8.2 Worker

1. konsumiert Task
2. lädt benötigte Frames (lokal gecached)
3. berechnet lokale Metriken und Gewichte
4. rekonstruiert Tile
5. publiziert Ergebnis
6. ACK

---

### 8.3 Aggregation

1. Master sammelt alle Tile‑Ergebnisse
2. dedupliziert Ergebnisse über (`run_id`, `tile_id`) und validiert `tile_data_checksum`
3. sortiert nach `tile_id`
4. führt Overlap‑Add deterministisch aus
5. schreibt synthetische Frames

---

## 9. Determinismus und Reproduzierbarkeit

### Regeln

* keine Floating‑Point‑Reduktion in zufälliger Reihenfolge
* Aggregation ausschließlich im Master
* stabile Sortierung nach `tile_id`

### Seed‑Empfehlung

```text
seed = uint32(sha256(run_id + ":" + tile_id)[0:4])
```

Hinweis:

* Keine sprach-/prozessabhängigen Hashfunktionen verwenden.

---

## 10. Fehlertoleranz

### Idempotenz

* Tasks erzeugen keine globalen Seiteneffekte
* Ergebnisse werden atomar geschrieben

Praktisch:

* Ergebnis‑Schlüssel = (`run_id`, `tile_id`) ist eindeutig.
* Worker dürfen ein Ergebnis neu schreiben, solange atomar ersetzt wird.

### Retries

* begrenzte Retry‑Anzahl
* Exponential Backoff

### Dead‑Letter‑Queue

* fehlerhafte Tiles werden isoliert
* Lauf schlägt erst fehl bei Überschreiten eines Schwellenwerts

---

## 11. Performance‑Hinweise

* Worker möglichst datenlokal betreiben
* Frames im RAM oder auf NVMe cachen
* keine zufälligen Dateizugriffe

Hinweis:

* Bei Multi‑Host‑Setups ist ein shared storage (NFS/Objekt‑Storage) nötig, damit `tile_data_ref` für den Master erreichbar ist.

---

## 12 Security / Operations (Minimalanforderungen)

* RabbitMQ Authentifizierung aktivieren (User/Pass oder Zertifikate)
* TLS nutzen, wenn Worker nicht auf dem gleichen Host laufen
* pro `run_id` Routing‑Key‑Prefix oder getrennte vhosts, um Runs sauber zu isolieren
* TTL/Max‑Length Policies für Diagnose‑Queues setzen (Diagnose darf Produktion nicht verstopfen)

## 13. Abgrenzung

Diese Parallelisierung:

* ersetzt **keine** Methodik
* ändert **keine** mathematischen Definitionen
* ist **optional und konfigurationsabhängig**

Die Single‑Node‑Ausführung bleibt Referenz für Validierung.

---

## 14. Zusammenfassung

Die RabbitMQ‑basierte Parallelisierung ermöglicht skalierbare, fehlertolerante und deterministische Tile‑Analyse. Sie ist modular, optional und vollständig kompatibel mit der Methodik v2.
