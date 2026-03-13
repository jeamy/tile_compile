# Dashboard Run Schritt-fuer-Schritt

Diese Anleitung beschreibt den kompakten Run-Ablauf direkt im Dashboard.
Der Fokus liegt auf dem schnellen Weg:

- Daten eingeben
- Scan ausfuehren
- Validieren
- Run starten

## Ziel des Dashboard-Runs

Der Dashboard-Run ist der schnellste Einstieg fuer einen neuen Lauf.
Er findet direkt auf der Dashboard-Seite statt und verwendet die kompakten Felder im Bereich `Guided Run`.

Typische Felder im Dashboard-Run sind:

- `Eingabeordner`
- `Ausgabeordner`
- `Farbmodus`
- `Preset`
- `Preset-Ordner`
- `Run-Name`
- `Run-Pfad`

Je nach Modus sind ausserdem sichtbar:

- `Einfach`
- `Erweitert`
- optional `MONO Queue`

---

## Schritt 1: Daten eingeben

![Step 1 - Dashboard Input](mockups/dashboard_run_step_01_input.png)

### Vorgehen

1. Oeffne das `Dashboard`.
2. Bleibe im Block `Run`.
3. Trage die Grunddaten ein:
   - `Eingabeordner`
   - `Ausgabeordner`
   - `Farbmodus` - wird normalerweise automatisch aus den gescannten Frames erkannt und muss nur gesetzt werden, wenn die Erkennung nicht greift
   - `Preset`
   - optional `Preset-Ordner`
   - `Run-Name`
4. Pruefe den angezeigten `Run-Pfad`.
5. Nutze bei Bedarf den Modus `Einfach` oder `Erweitert`.

### Hinweise

- `Einfach` zeigt nur die Kernfelder.
- `Erweitert` zeigt zusaetzlich weitere Run-Felder und bei `MONO` auch die Queue.
- Wenn ein anderes Preset-Verzeichnis gebraucht wird, nutze `Durchsuchen` und `Neu laden`.

### Ergebnis

- Die Daten fuer den Lauf sind direkt im Dashboard vorbereitet.

---

## Schritt 2: Scan ausfuehren

![Step 2 - Dashboard Scan](mockups/dashboard_run_step_02_scan.png)

### Vorgehen

1. Klicke auf `Scan`.
2. Warte, bis der Scan abgeschlossen ist.
3. Pruefe anschliessend den Scan-Status im Dashboard.
4. Bestätige besonders:
   - gefundene Eingabedaten
   - plausiblen Farbmodus
   - erkannte Bilddaten
   - Warnungen oder Fehler

### Ergebnis

- Die Eingabedaten wurden technisch geprueft.
- Der Dashboard-Run ist fuer den naechsten Schritt vorbereitet.

---

## Schritt 3: Validieren

![Step 3 - Dashboard Validate](mockups/dashboard_run_step_03_validate.png)

### Vorgehen

1. Klicke auf `Validieren`.
2. Warte auf den Status unterhalb der Buttons.
3. Pruefe, ob dort `Validierung: OK` oder ein Fehler-/Warnstatus angezeigt wird.
4. Wenn Fehler vorhanden sind, korrigiere zuerst Daten oder Preset.

### Hinweise

- Ohne erfolgreiche Validierung bleibt der Start blockiert.
- Der Dashboard-Run benutzt die aktuelle Konfiguration im Hintergrund.

### Ergebnis

- Der Lauf ist fachlich und technisch freigegeben.

---

## Schritt 4: Run starten

![Step 4 - Dashboard Start](mockups/dashboard_run_step_04_start.png)

### Vorgehen

1. Pruefe noch einmal:
   - Eingabeordner
   - Ausgabeordner
   - Preset
   - Run-Name
   - Run-Pfad
   - Validierungsstatus
2. Klicke auf `Run starten`.
3. Nach erfolgreichem Start wechselt die GUI in den `Run Monitor`.

### Wichtig

- Der Startbutton ist erst freigegeben, wenn die Validierung erfolgreich war.
- Bei `MONO` kann im erweiterten Modus auch eine Queue mit gestartet werden.

### Ergebnis

- Der Run wurde direkt aus dem Dashboard gestartet.

---

## Kurze Checkliste fuer den Dashboard-Run

- Ist der `Eingabeordner` korrekt?
- Ist der `Ausgabeordner` korrekt?
- Stimmt der `Farbmodus`?
- Wurde das richtige `Preset` geladen?
- Ist der `Run-Name` sinnvoll?
- Ist der `Run-Pfad` plausibel?
- War der `Scan` erfolgreich?
- Ist die `Validierung` erfolgreich?
- Ist `Run starten` freigegeben?

---


