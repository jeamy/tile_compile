# PI Workflow in GUI3

Stand: 2026-07-14

## Empfehlung bis Apply

1. In GUI3 die AI-Empfehlungsseite oeffnen.
2. Scan-AI Analyse erzeugen oder vorhandene Analyse laden.
3. Empfehlungen auswaehlen.
4. `PI Preview` ausfuehren.
5. YAML-Diff und Validierung pruefen.
6. `PI anwenden` nur ausfuehren, wenn die Preview plausibel ist.

PI schreibt nicht direkt. Config-Aenderungen laufen ueber Action-Plan, Preview, `validate-config` und explizites Apply.

## Lernen mit Memories

- `Aus dieser Optimierung lernen` speichert nach erfolgreichem Apply einen Memory-Kandidaten.
- Kandidaten sind nicht automatisch vertrauenswuerdig.
- Ein Memory wird erst nach Review als `accepted` fuer spaetere Sessions relevant.
- `rejected` und `deprecated` Memories werden als Negativsignal genutzt.

## Review

In `PI Memories`:

- `Accept`: gute Optimierung fuer spaetere Sessions freigeben.
- `Reject`: falsche oder unpassende Optimierung ablehnen.
- `Deprecate`: frueher nuetzliche, inzwischen ueberholte oder verschlechternde Optimierung markieren.

Outcome-Felder zeigen, welche Pfade angewendet wurden und ob die Config-Validierung erfolgreich war.

## Audit

`PI Audit` zeigt:

- PI Action-Plan Applies
- Scan-AI Config Applies
- Memory Reviews

Die Ansicht ist read-only und dient der Nachvollziehbarkeit.

## Export, Import, Dedupe

- `Export` speichert ein `pi.memories-export.v1` JSON-Bundle mit metadata-only Memories und Reviews.
- `Import` liest ein solches Bundle wieder ein und ueberspringt Duplikate.
- `Dedupe` entfernt vorhandene doppelte Memory-Eintraege nach Signatur und legt backendseitig ein Backup an.

## Sicherheit

- Accepted Memories sind historischer Kontext, keine Regel.
- Jede neue Empfehlung muss weiterhin Schema, aktuelle Evidenz und `validate-config` bestehen.
- PI Tools sind read-only/mutation-free. Schreibende Aktionen laufen ausschliesslich ueber Action-Plan, Preview und Apply.
