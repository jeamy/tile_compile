# Gefuehrter Wizard Schritt-fuer-Schritt

Diese Anleitung beschreibt den tatsaechlichen gefuehrten Ablauf in `web_frontend/wizard.html`.
Hier wird nicht ueber die normalen Menuepunkte gearbeitet, sondern direkt ueber die Schritte innerhalb der Wizard-Seite.

## Ziel des Wizards

Der Wizard fuehrt in einer einzigen Seite durch diese Schritte:

- `Step 1: Input & Scan`
- `Step 2: Optionale MONO Filter-Queue`
- `Step 3: Kalibrierung`
- `Step 4: Preset + Situation + Validation`
- danach `Run starten`

---

## Schritt 1: Input & Scan im Wizard setzen

![Step 1 - Input & Scan](mockups/wizard_step_01_input_scan.png)

### Vorgehen

1. Oeffne `wizard.html`.
2. Trage im ersten Block ein:
   - `Eingabeordner`
   - `Runs Dir`
   - `Run Name`
   - `Dateimuster`
   - `Frames Minimum`
   - `Max. Frames`
   - `Sortierung`
   - `Farbmodus` - wird normalerweise automatisch aus den gescannten Frames erkannt und muss nur gesetzt werden, wenn die Erkennung nicht greift
   - optional `Bayer-Pattern`
   - optional `Checksummen`
3. Pruefe die `Run path`-Vorschau.

### Ergebnis

- Die Grunddaten fuer den Run sind direkt im Wizard gesetzt.

---

## Schritt 2: Optionale MONO Filter-Queue

Step 2 - MONO Queue

Dieser Schritt ist nur relevant, wenn `MONO` verwendet wird.

### Vorgehen

1. Pruefe, ob `Farbmodus` auf `MONO` steht.
2. Trage pro Filter eine Queue-Zeile ein.
3. Setze je nach Bedarf:
   - `Filter`
   - `Input Dir`
   - optional `Pattern`
   - optional `Run Label`
4. Aktiviere nur die benoetigten Filter.

### Ergebnis

- Der serielle MONO-Ablauf ist direkt im Wizard vorbereitet.

---

## Schritt 3: Kalibrierung im Wizard setzen

![Step 3 - Kalibrierung](mockups/wizard_step_03_calibration.png)

### Vorgehen

1. Aktiviere die benoetigten Kalibrierarten:
   - `Bias`
   - `Dark`
   - `Flat`
2. Waehle je Kalibrierart:
   - Ordner
   - Master-Datei
3. Trage die Pfade ein.
4. Pruefe, dass Modus und Pfad zusammenpassen.

### Ergebnis

- Die Kalibrierdaten sind direkt innerhalb des Wizards gesetzt.

---

## Schritt 4: Scan starten und Ergebnis pruefen

![Step 4 - Scan Ergebnis](mockups/wizard_step_04_scan_result.png)

### Vorgehen

1. Klicke auf `Scan starten`.
2. Warte auf das `Scan-Ergebnis`.
3. Pruefe besonders:
   - `Status`
   - `Frames entdeckt`
   - `Color Mode`
   - `Bildgroesse`
   - `Bayer Pattern`
   - `Fehler`
   - `Warnungen`
4. Korrigiere Eingaben, falls das Ergebnis unplausibel ist.

### Ergebnis

- Die Eingabedaten sind geprueft, bevor der Wizard weitergeht.

---

## Schritt 5: Weiter zu Preset, Situation und Validation

![Step 5 - Preset und Validierung](mockups/wizard_step_05_preset_validation.png)

### Vorgehen

1. Klicke auf `Weiter zu Preset/Validation`.
2. Waehle ein `Preset`.
3. Nutze bei Bedarf `Szenario anwenden`.
4. Pruefe das Feld `Validation`.
5. Gehe nur weiter, wenn keine echten Fehler mehr offen sind.

### Ergebnis

- Der Wizard-Draft ist konfiguriert und validiert.

---

## Schritt 6: Run direkt aus dem Wizard starten

![Step 6 - Run Start](mockups/wizard_step_06_run_start.png)

### Vorgehen

1. Pruefe nochmals `Run path`, Preset und Validation.
2. Klicke auf `Run starten`.
3. Nach erfolgreichem Start wechselt die GUI in den `Run Monitor`.

### Wichtig

- Der Startbutton bleibt blockiert, solange die Validierung nicht erfolgreich ist.
- Ueber `Zum Parameter Studio` kannst du den Wizard-Kontext verlassen, falls du tiefer eingreifen willst.

### Ergebnis

- Der Run wird direkt aus dem Wizard gestartet.

---

## Schritt 7: Resume aus dem Wizard-Kontext

![Step 7 - Resume Kontext](mockups/wizard_step_07_resume_context.png)

### Vorgehen

1. Oeffne spaeter den `Run Monitor`.
2. Nutze bei Bedarf eine passende Config-Revision.
3. Starte ein Resume ab der gewuenschten Phase.

### Hintergrund

- Nach Parameteraenderungen entsteht eine neue Config-Revision.
- Resume selbst passiert nicht im Wizard, sondern danach im `Run Monitor`.

### Ergebnis

- Der Wizard ist der Einstiegsfluss, Resume bleibt Teil des Betriebsflusses.

---

## Kurze Checkliste fuer den Wizard

- Sind Eingabeordner, Runs Dir und Run Name gesetzt?
- Ist die MONO-Queue nur dann gepflegt, wenn `MONO` aktiv ist?
- Sind die Kalibrierpfade korrekt?
- Ist das Scan-Ergebnis plausibel?
- Wurde ein passendes Preset gewaehlt?
- Ist die Validierung erfolgreich?
- Ist der Startbutton freigegeben?

---

## Empfohlene PNG-Reihenfolge

1. `wizard_step_01_input_scan.png`
2. `wizard_step_02_mono_queue.png`
3. `wizard_step_03_calibration.png`
4. `wizard_step_04_scan_result.png`
5. `wizard_step_05_preset_validation.png`
6. `wizard_step_06_run_start.png`
7. `wizard_step_07_resume_context.png`
