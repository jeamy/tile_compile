# Backend – Siril Registration Integration

**Datei:** `backend_siril_registration.md`
**Status:** Verbindliche Implementierungsdokumentation
**Geltungsbereich:** Backend / Runner / CLI
**Bezug:** tile-compile Methodik (OSC, CFA-aware Verarbeitung)

---

## 1. Zweck

Dieses Dokument beschreibt die **optionale externe geometrische Vorverarbeitung** von OSC-Daten mittels **Siril**.

Siril wird **ausschließlich** verwendet für:

* technisches Debayering **nur zur Registrierung**
* geometrische Registrierung (Translation / Rotation / Subpixel)

Siril ist **nicht Teil** der Qualitäts-, Metrik- oder Rekonstruktionspipeline.

---

## 2. Grundsätze (normativ)

1. Siril darf **keine** Qualitätsentscheidungen treffen
2. Siril darf **keine** Normalisierung durchführen
3. Siril darf **keine** Gewichtung oder Selektion durchführen
4. Siril darf **keine** nichtlinearen Operationen ausführen
5. Alle Siril-Outputs dienen **nur** der Geometrie

Ein Verstoß gegen diese Regeln macht einen Run **nicht konform**.

---

## 3. Rolle von Siril in der Pipeline

### 3.1 Einordnung

```
Input (OSC FITS)
→ [Siril: Debayer + Registrierung]
→ registrierte Frames (linear)
→ tile-compile Pipeline (kanalweise)
```

Siril ist eine **isolierte Vorstufe**.
Alle physikalischen Entscheidungen erfolgen **nachgelagert** im Backend.

---

## 4. Arbeitsverzeichnis und Datenfluss

### 4.1 Arbeitsverzeichnis

Siril arbeitet **immer relativ zu einem Arbeitsverzeichnis**.

**Verbindliche Regel:**

> Das Siril-Arbeitsverzeichnis (`-d`) ist das Run-Lights-Verzeichnis.

Beispiel:

```
runs/<run_id>/lights/
```

Dieses Verzeichnis enthält ausschließlich:

* die materialisierten Input-Lights (Hardlink/Reflink/Copy)
* von Siril erzeugte Zwischen- und Output-Dateien

---

### 4.2 Script-Datei

* Siril wird **immer** mit Script gestartet (`-s`)
* Script-Pfad ist **explizit**
* Scripts sind **statisch**, versioniert und policy-geprüft

---

## 5. Siril-Aufruf (verbindlich)

### 5.1 Kommandozeile

```bash
siril \
  -d <run_dir>/lights \
  -s <path>/register_osc.ssf \
  -q
```

### 5.2 Bedeutung der Optionen

| Option | Bedeutung                   |
| ------ | --------------------------- |
| `-d`   | Arbeitsverzeichnis (Lights) |
| `-s`   | Siril-Script                |
| `-q`   | Headless / Quiet            |

Andere Aufrufvarianten sind **nicht zulässig**.

---

## 6. Siril-Script: Anforderungen

### 6.1 Zulässiger Inhalt

Ein Siril-Script darf ausschließlich enthalten:

* `load seq`
* `debayer` (lineares Verfahren, z. B. bilinear)
* `register`
* Verwaltungsbefehle (`setcpu`, `setmem`, `quit`)

### 6.2 Verbotener Inhalt (Hard-Fail)

Ein Script ist **nicht konform**, wenn es enthält:

* `-norm=*`
* `-rej=*`
* `-weight`
* `-drizzle`
* Stretch / Histogram / Asinh / Log
* Frame-Selektion

---

## 7. Output-Semantik

### 7.1 Erwartete Outputs

Nach erfolgreichem Siril-Lauf müssen im Arbeitsverzeichnis vorhanden sein:

* **registrierte Frames** mit Prefix `r_*.fit*`

Diese Dateien sind:

* linear
* geometrisch ausgerichtet
* nicht qualitätsbewertet

### 7.2 Ignorierte Dateien

Folgende Dateien werden **nicht** weiterverwendet:

* debayerte Zwischenprodukte
* `.seq`-Dateien
* Siril-Logs

---

## 8. Übergabe an tile-compile

Nach Siril gilt zwingend:

1. Es werden **nur** `r_*.fit*` übernommen
2. RGB-Frames werden **wieder kanalweise getrennt**
3. Alle folgenden Schritte sind strikt kanalweise:

   * Normalisierung
   * globale Metriken
   * Tile-Analyse
   * Rekonstruktion
   * Stacking

Ein gemeinsames RGB-Bild darf **erst nach Abschluss der gesamten linearen Pipeline** entstehen.

---

## 9. Backend-Referenzimplementierung

### 9.1 Siril-Aufruf

```python
def run_siril_registration(
    lights_dir: Path,
    siril_script: Path,
    siril_binary: str = "siril",
):
    cmd = [
        siril_binary,
        "-d", str(lights_dir),
        "-s", str(siril_script),
        "-q",
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
```

### 9.2 Validierung nach Lauf

* Exit-Code ≠ 0 → **FAILED**
* keine `r_*.fit` → **FAILED**

---

## 10. Fehler- und Abbruchverhalten

| Situation                  | Verhalten  |
| -------------------------- | ---------- |
| Siril Exit-Code ≠ 0        | Run FAILED |
| Script fehlt               | Run FAILED |
| keine registrierten Frames | Run FAILED |
| verbotene Script-Optionen  | Run FAILED |

Ein fehlgeschlagener Siril-Schritt **blockiert den gesamten Run**.

---

## 11. Auditierbarkeit

Folgende Informationen sind im Run zu protokollieren:

* Siril-Binary (Pfad / Version)
* Script-Name + Hash
* Aufrufparameter
* Anzahl Input-Frames
* Anzahl `r_*.fit`-Outputs

Diese Daten sind Bestandteil der Run-Metadaten.

---

## 12. Kurzform (Policy)

> **Siril = Geometrievorstufe.**
> **Kein Einfluss auf Qualität, Gewichtung oder Auswahl.**
> **Aufruf immer:**
> `siril -d <lights> -s <script> -q`

---

## 13. Abgrenzung

Diese Dokumentation beschreibt **ausschließlich**:

* externe Registrierung mit Siril

Nicht enthalten sind:

* interne tile-compile Registrierung
* Qualitätsmetriken
* Normalisierung
* Rekonstruktion
* Stacking

---

**Ende der Spezifikation**
