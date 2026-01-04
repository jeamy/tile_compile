# Tile Compile – GUI-Spezifikation (Tauri Frontend)

**Status:** normativ für GUI-Implementierung
**Gültigkeit:** kompatibel mit `tile_basierte_qualitatsrekonstruktion_methodik.md` (v2), `tile_compile.proc` (Clean Break), `tile_compile.yaml`

---

## 1. Ziel der GUI

Die GUI dient ausschließlich als **Run-Controller und Monitor** für *Tile Compile*.
Sie ermöglicht:

* Auswahl von Eingabedaten (Frames-Manifest oder Input-Verzeichnis)
* Auswahl und **Editieren der Konfiguration vor dem Run**
* Starten deterministischer Runs
* Überwachung laufender Runs
* kontrolliertes Abbrechen von Runs
* Einsehen von Logs, Status und Validierungsartefakten

Die GUI ist **kein interaktives Bildbearbeitungs- oder Analysewerkzeug**.

---

## 2. Grundprinzipien (verbindlich)

1. **Konfiguration ist vor dem Run editierbar, danach read-only**
2. **Keine Eingriffe während der Ausführung**
3. **Jeder Run ist deterministisch und eindeutig identifiziert**
4. **Abbruch ist erlaubt, Resume nicht**
5. **GUI ist Client, nicht Rechenkern**

---

## 3. Technologiestack

### 3.1 Frontend

* **Tauri**
* UI über Web-Technologien (HTML/CSS/JS)
* UI-Framework frei wählbar (z. B. Svelte, React, Vanilla)

### 3.2 Backend (lokal)

* Python-Backend
* Aufgaben:

  * Erkennen / Erzeugen von Konfigurationsdateien
  * Starten / Abbrechen von Runs
  * Lesen von Status- und Artefaktdateien
* Kommunikation:

  * Tauri Commands oder lokale HTTP-API

---

## 4. Konfigurationsmodell

### 4.1 Konfigurationsquelle

Die GUI arbeitet immer kontextbezogen zu einem Input-Verzeichnis.

* Rohframes / registrierte Frames
* optional: `tile_compile.yaml`

---

### 4.2 Konfigurationslogik

1. Falls `tile_compile.yaml` vorhanden ist → laden und editierbar anzeigen
2. Falls nicht vorhanden → Erstellung aus Template anbieten
3. Nach Run-Start → Konfiguration einfrieren und hashen (`config_hash`)

Ein Run referenziert **immer genau eine Konfiguration**.

---

### 4.3 Konfigurationseditor

* strukturierter Editor (Schema-basiert)
* Validierung vor Run-Start
* Methodik-Verstöße blockieren Run-Start
* Raw-YAML-Ansicht optional (nur vor Run)

---

## 5. GUI-Screens

### 5.1 Project / Input Selection

* Frames-Manifest oder Input-Verzeichnis auswählen
* Anzeige erkannter Frames
* Status: Konfiguration vorhanden / fehlt

---

### 5.2 Configuration

* Editierbare YAML-Konfiguration
* Validierung (grün / rot)
* Aktionen: Save / Discard / Continue

---

### 5.3 New Run

* Read-only Zusammenfassung
* Anzeige `config_hash`
* Start Run → erzeugt `run_id`

---

### 5.4 Run Status

* Pipeline-Fortschritt gemäß PROC
* Status: RUNNING / FAILED / SUCCESS / ABORTED
* **Abort Run** möglich (mit Bestätigung)

---

### 5.5 Logs & Artefakte

* strukturierte Logs
* Validierungsplots (read-only)

---

### 5.6 Run History

* Liste vergangener Runs
* Re-Run mit gleicher Konfiguration möglich

---

## 6. Einschränkungen

Die GUI darf **keine** Parameter während eines Runs ändern und keine Bildinteraktion erlauben.

---

## 7. Zusammenfassung

Die GUI ist ein deterministischer, reproduzierbarer Run-Controller mit kontrollierter Abbruchmöglichkeit und ohne Eingriffe in die Methodik.
