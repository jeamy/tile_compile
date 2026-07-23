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

## Operationen

Unterstützt werden Helligkeit, Kontrast, Sättigung, Schärfen, Entrauschen, Bilateralfilter, Grünentfernung, CLAHE/lokale Details, Zuschneiden, Invertieren, Zurücksetzen, Vibrance, Farbtemperatur, Entfernung lila Farbsäume, Banding-Reduktion, Sternentsättigung und Dehaze. Das Backend validiert und begrenzt Parameter vor der Anwendung.

Crop ist über den Chat mit einer ausdrücklichen Anweisung wie „schneide 10% Rand ab“ verfügbar. Das Backend wandelt den Prozentwert in Pixelkoordinaten um und begrenzt das Rechteck auf die aktuelle Bildgröße.

Signierte Operationen wie Helligkeit, Kontrast, Sättigung, Vibrance und Farbtemperatur können `+/-`-Regler anzeigen. Nicht-invertierbare oder einseitige Operationen verwenden keine `+/-`-Regler.

**Erneut anwenden** wiederholt die letzte nicht-stufenweise Operation mit exakt denselben Parametern. Dafür wird direkt der Repeat-Endpunkt aufgerufen; die KI wird nicht verwendet. Die Wiederholung wird in Undo/Redo und Operations-History aufgenommen.

Das Schärfen verwendet eine Unsharp-Mask-ähnliche Methode (Gaussian-Blur plus gewichtete Subtraktion). Das Entrauschen verwendet OpenCV Non-Local Means; das Float-Bild wird dabei vorübergehend in 8-bit umgewandelt und danach zurückkonvertiert. Bei identischem Eingang und identischen Parametern sind beide Operationen deterministisch.

## Verwendung der KI

Wenn der optionale PI-Sidecar läuft, ein API-Key für einen Provider vorhanden ist und ein Modell ausgewählt wurde, wird die Chat-Anfrage an dieses Modell gesendet. Der Sidecar erhält:

- die Anweisung des Benutzers;
- die letzten Operationen;
- regelmäßig eine JPEG-Vision-Preview des aktuellen Bildes (zur Begrenzung der Vision-API-Kosten).

Das Modell liefert eine strukturierte Operation mit Parametern. Das C++-Backend validiert diese Operation, führt sie lokal aus, speichert `live_edit.fits` und liefert die neue Preview zurück. Die KI schreibt niemals direkt in die FITS-Datei.

Wenn kein Modell oder kein API-Key verfügbar ist, verwendet das Backend einen lokalen Parser. Der Text bestimmt die Operation; für Helligkeit, Kontrast und Sättigung werden konservative Stärken aus Bildstatistiken berechnet. Für andere Operationen werden sichere Fallback-Parameter verwendet. Dafür ist keine Netzwerkverbindung nötig.

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
