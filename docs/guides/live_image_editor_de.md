# Live Image Editor

Der Live Image Editor ist ein nicht-destruktiver Editor für das aktuelle FITS-Ergebnis eines Runs. Er wird über den Button **Live Editor** im Bereich **Letztes Bild** oder durch Klick auf die Preview geöffnet.

## Arbeitsbild und Speicherung

Das Quell-FITS bleibt unverändert. Der aktuelle Arbeitsstand wird gespeichert unter:

```text
runs/<run-id>/outputs/live_edit.fits
```

Diese Datei enthält das aktuelle lineare Float-Bild. Sie wird nach jeder erfolgreichen Operation, bei Undo, Redo, Wiederholen und Reset geschrieben. Die aktive Operations-History wird getrennt von einer vollständigen Bearbeitungs-Timeline in den PI-Laufdaten gespeichert; die Timeline enthält auch explizite Undo- und Redo-Aktionen.

Reset stellt das unveränderliche Quell-FITS wieder her, ersetzt `live_edit.fits`, löscht Undo/Redo und Chat-History und aktualisiert die Run-Preview. Alte abgeleitete `live_edit`-PNG/JPEG-Dateien werden entfernt, damit sie nicht mit dem aktuellen FITS verwechselt werden.

## Preview und FITS-Werte

Die Editor-Preview wird aus dem aktuellen Float-Bild im Speicher erzeugt. Dabei wird keine zusätzliche Bildoperation ausgeführt. Lineare Werte werden für die Browserdarstellung direkt von `[0, 1]` auf 8-bit abgebildet; es gibt keinen Histogramm-Stretch und keine Gamma-Korrektur. Das FITS bleibt die maßgebliche Datenrepräsentation. Die JPEG-Kodierung kann geringfügige Kompressionsunterschiede erzeugen, aber keine zusätzliche Helligkeits- oder Kontraständerung.

Nach jeder erfolgreichen Operation bleibt die vorherige Preview erhalten. Ein Klick auf das Bild oder auf das **Vorher/Aktuell**-Badge schaltet zwischen dem Zustand vor und nach der Operation um.

## Bedienung und Parameterdialoge

Operationen mit veränderbaren Parametern öffnen einen einheitlichen Dialog über der Bildansicht. Der Dialog kann über seine Kopfzeile verschoben werden und bleibt innerhalb des sichtbaren Browserbereichs.

Änderungen an Reglern und Auswahlfeldern erzeugen nach einer kurzen Entprellzeit eine Live-Preview. Diese Preview wird auf einer Kopie des aktuellen Arbeitsbildes berechnet und verändert weder `live_edit.fits` noch History oder Undo/Redo.

Unten links steuert **Vorher/Aktuell-Ansicht** die Darstellung:

- aktiviert: die Live-Preview mit den eingestellten Parametern wird angezeigt;
- deaktiviert: der unveränderte aktuelle Arbeitsstand wird angezeigt.

**Anwenden** führt die Operation mit den sichtbaren Parametern aus und speichert sie. **Abbrechen** verwirft Timer und ausstehende Preview-Antworten, stellt den kanonischen aktuellen Bildstand wieder her und schreibt nichts in die History.

Die Reglergrenzen entsprechen der Backend-Validierung. Dazu gehören beispielsweise `0,5…5` für den Schärferadius, `0,5…10` für den Radius des lokalen Kontrasts, `0,1…5` für Levels-Gamma und `0…1` für Stärke- und Schutzparameter. Zusammenhängende Werte wie Schwarz- und Weißpunkt werden zusätzlich so begrenzt, dass der Schwarzpunkt kleiner als der Weißpunkt bleibt.

## Operationen

Unterstützt werden Helligkeit, Kontrast, Sättigung, Schärfen, Entrauschen, Bilateralfilter, Grünentfernung, CLAHE/lokale Details, Levels, Shadow Recovery, Highlight Recovery, Color Balance, Local Contrast, Chroma-Denoise, Curves, Zuschneiden, Invertieren, Zurücksetzen, Vibrance, Farbtemperatur, Entfernung lila Farbsäume, Banding-Reduktion, Sternentsättigung und Dehaze. Das Backend validiert und begrenzt Parameter vor der Anwendung.

Crop ist eine rein deterministische Operation und wird nie an die KI gesendet.

Signierte Operationen wie Helligkeit, Kontrast, Sättigung, Vibrance und Farbtemperatur können `+/-`-Regler anzeigen. Nicht-invertierbare oder einseitige Operationen verwenden keine `+/-`-Regler.

**Erneut anwenden** wiederholt die letzte nicht-stufenweise Operation mit exakt denselben Parametern. Dafür wird direkt der Repeat-Endpunkt aufgerufen; die KI wird nicht verwendet. Die Wiederholung wird in Undo/Redo und Operations-History aufgenommen.

Das Schärfen verwendet eine Unsharp-Mask-ähnliche Methode (Gaussian-Blur plus gewichtete Subtraktion). Das Entrauschen verwendet OpenCV Non-Local Means; das Float-Bild wird dabei vorübergehend in 8-bit umgewandelt und danach zurückkonvertiert. Bei identischem Eingang und identischen Parametern sind beide Operationen deterministisch.

Levels, Shadow Recovery, Highlight Recovery, Color Balance, Local Contrast und Chroma-Denoise werden lokal und deterministisch ausgeführt. Die KI schlägt nur Anfangswerte vor; der Chat-Endpunkt wendet diese Werte noch nicht auf das Arbeitsbild an. Der Parametereditor zeigt den Vorschlag als nicht-persistierende Preview und stellt Regler für die Feinabstimmung bereit. Erst **Anwenden** verändert `live_edit.fits` und die History; **Abbrechen** lässt den Zustand vor dem AI-Vorschlag vollständig unverändert. Local Contrast besitzt Stärke und Radius, Chroma-Denoise Stärke, Strukturschutz und einen Soft/Strong-Modus. Color Balance unterstützt globale RGB-Werte sowie getrennte Schatten-, Mittelton- und Highlight-Korrekturen.

Curves wird ausschließlich im grafischen Kurveneditor bearbeitet und niemals von der KI erzeugt. Die Ausgangskurve verläuft diagonal. Ein Klick in die Kurvenfläche fügt einen Kontrollpunkt hinzu, Ziehen verschiebt ihn und Doppelklick oder Rechtsklick entfernt einen inneren Punkt. Die Endpunkte bleiben erhalten. Darstellung und Bildoperation verwenden dieselbe begrenzte Catmull-Rom-Spline. Auch Curves unterstützt Live-Preview, Vorher/Aktuell-Ansicht, Anwenden und Abbrechen.

## Chat, History und Wiederholen

Chat-Einträge mit ausgeführten Operationen sind anklickbar. Nach der Bestätigung **Befehl noch einmal anwenden?** wird die gespeicherte Operation mit exakt denselben Parametern lokal erneut ausgeführt; dafür wird keine KI aufgerufen.

AI-Vorschläge, GUI-Anpassungen, Curves, wiederholte Operationen und Preset-Operationen werden im selben strukturierten Operationsformat gespeichert:

- `operation_history` enthält die aktuell wirksame Operationsfolge und dient zur Rekonstruktion des aktuellen Bildes;
- `edit_history` enthält die vollständige Timeline einschließlich Apply, Undo und Redo;
- `chat_history` enthält die sichtbaren Nachrichten und die zugehörigen Operationen.

Eine reine Live-Preview erscheint in keiner dieser Histories. Erst **Anwenden** erzeugt einen History- und Undo-Eintrag.

## Verwendung der KI

### KI und API-Key einrichten

Die optionale KI-Funktion wird einmalig unter **Tools → KI & API** eingerichtet:

1. Provider auswählen und ein Modell wählen.
2. Den API-Key dieses Providers ohne führende oder nachgestellte Leerzeichen einfügen und **Key speichern** wählen.
3. Mit **Status abrufen** die Verbindung prüfen. Ein Modell kann Bilddaten verwenden, wenn der Vision-Status als bildfähig angezeigt wird.

Der Key wird im lokalen PI-AuthStorage gespeichert, nicht in `tile_compile.yaml`, Config-Revisions, Run-Daten, Bilddateien oder Chat-History. Alternativ kann ein Key als Provider-Umgebungsvariable in einer lokalen `.env`-Datei hinterlegt werden. Die vollständige aktuelle Liste aller unterstützten Key-/Credential-Variablen steht in der [`.env.example`](../env.example). Ein `401`-Fehler wie `invalid x-api-key` bedeutet, dass der Provider den verwendeten Key ablehnt; in diesem Fall den Key beim richtigen Provider erneut speichern oder den `.env`-Eintrag prüfen.

Wenn der optionale PI-Sidecar läuft, ein API-Key für einen Provider vorhanden ist und ein Modell ausgewählt wurde, wird die Chat-Anfrage an dieses Modell gesendet. Der Sidecar erhält:

- die Anweisung des Benutzers;
- die letzten Operationen;
- regelmäßig eine JPEG-Vision-Preview des aktuellen Bildes (zur Begrenzung der Vision-API-Kosten).

Das Modell liefert eine strukturierte Operation mit Parametern. Das C++-Backend validiert diese Operation, führt sie lokal aus, speichert `live_edit.fits` und liefert die neue Preview zurück. Die KI schreibt niemals direkt in die FITS-Datei.

Wenn kein Modell oder kein API-Key verfügbar ist, verwendet das Backend für unterstützte einfache Befehle einen lokalen Parser. Der Text bestimmt die Operation; für Helligkeit, Kontrast und Sättigung werden konservative Stärken aus Bildstatistiken berechnet. Dafür ist keine Netzwerkverbindung nötig. Bei freien KI-Anweisungen ohne passende lokale Operation zeigt der Chat einen Hinweis auf **Tools → KI & API**.

API-Keys werden über die PI-/Sidecar-Authentifizierung verwaltet und nicht in Bilddateien, Operations-History oder FITS-Header geschrieben.

## Timeline-Presets

Die aktuelle Operationsfolge kann als Preset gespeichert und auf einen anderen Run angewendet werden. Im Preset-Bereich stehen **Sichern unter**, **Sichern**, eine Auswahlliste und **Anwenden** zur Verfügung.

**Sichern unter** fragt einen Namen ab und legt ein neues JSON-Preset an. **Sichern** überschreibt das ausgewählte Preset erst nach einer Bestätigung. Gespeichert werden die aktiven Operationen (also genau der aktuell wirksame Bildstand) sowie die vollständige Timeline zur Nachvollziehbarkeit. Die Dateien liegen global unter `.pi_memory/presets` und können daher für jeden Run verwendet werden. Beim Anwenden führt das Backend die Operationen sequenziell und ohne KI erneut auf dem aktuellen Bild aus; jede angewendete Operation ist anschließend über Undo rückgängig.

## Exporte

Die Aktionen **PNG exportieren** und **FITS exportieren** schreiben separate Exportdateien:

```text
runs/<run-id>/outputs/live_image_export_<session-id>.png
runs/<run-id>/outputs/live_image_export_<session-id>.fits
```

Diese Exporte sind getrennt von der kanonischen Arbeitsdatei `live_edit.fits`.
