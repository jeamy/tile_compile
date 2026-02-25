# Ablaufplan – Funktionsweise des Systems

## Kurz gesagt

Das System verarbeitet viele Astro-Einzelbilder (FITS) zu **einem sauberen, scharfen und farblich konsistenten Endbild**.

Der Ablauf folgt klaren Verarbeitungsschritten: prüfen, ausrichten, bewerten, tile-basiert rekonstruieren sowie abschließend farblich und astrometrisch kalibrieren.

---

## Skizze: Gesamtfluss

```text
Eingabe (viele FITS-Frames)
        |
        v
[1] Prüfen & Vorbereiten
        |
        v
[2] Bilder exakt ausrichten
        |
        v
[3] Prewarp auf gemeinsamen Canvas
        |
        v
[4] Kanal-Info + Helligkeit vergleichbar machen
        |
        v
[5] Bild in Kacheln (Tiles) aufteilen
        |
        v
[6] Common Overlap bestimmen (nur gemeinsame Datenbereiche)
        |
        v
[7] Qualität je Tile messen + beste Infos je Tile zusammensetzen
        |
        v
[8] Optional: ähnliche Zustände clustern -> synthetische Zwischenbilder
        |
        v
[9] Finales Stacking (robustes Mitteln)
        |
        v
[10] Debayer (bei OSC), Astrometrie, Farbkalibrierung (PCC)
        |
        v
Ausgabe: finales FITS-Bild (+ RGB/PCC-Version, WCS, Artefakte, Logs)
```

---

## Warum das Ganze in Tiles?

In Astro-Serien ist die Qualität oft **nicht überall gleich**:
- Seeing ändert sich,
- Nachführfehler sind lokal verschieden,
- einzelne Bereiche können unschärfer oder verrauschter sein.

Darum schaut das System nicht nur auf das ganze Bild, sondern auf viele kleine Bereiche (Tiles).
So kann es pro Bereich entscheiden: **Welche Frames sind hier am besten?**

---

## Schritt-für-Schritt in verständlicher Form

## 0) Eingang prüfen (SCAN_INPUT)

Was passiert:
- Alle Eingabedateien werden gefunden.
- Header/Modus werden geprüft (Mono oder OSC/CFA).
- Offensichtliche Problemfälle werden aussortiert.
- Es wird geprüft, ob genug Speicherplatz da ist.

Ergebnis:
- Eine bereinigte Liste nutzbarer Frames.

---

## 1) Globale Ausrichtung (REGISTRATION)

Was passiert:
- Das System sucht ein Referenzbild.
- Alle anderen Frames werden darauf ausgerichtet.
- Falls ein Verfahren nicht gut klappt, wird auf Fallbacks gewechselt.

Ergebnis:
- Alle Frames liegen geometrisch möglichst deckungsgleich.

Skizze:
```text
vorher:    *   .*    ..*    *..
nachher:   *    *      *      *
```

---

## 2) Prewarp auf gemeinsamen Canvas (PREWARP)

Was passiert:
- Nach der Registrierung werden alle Frames auf einen gemeinsamen Zielbereich gewarpt.
- Für OSC erfolgt das CFA-sicher (Subplane-Warp), damit das Bayer-Muster konsistent bleibt.
- Bei Feldrotation wird die Canvas-Größe erweitert und Offsets (`tile_offset_x/y`) werden mitgeführt.

Ergebnis:
- Alle Folgephasen arbeiten auf derselben Geometrie und im selben Koordinatensystem.

---

## 3) Kanal-Info (CHANNEL_SPLIT)

Was passiert:
- Für OSC/Mono wird festgelegt, wie später mit Farbkanälen gearbeitet wird.
- In dieser Phase ist es vor allem Metadaten-Logik.

Ergebnis:
- Klarer Kanal-Plan für die nächsten Schritte.

---

## 4) Helligkeit angleichen (NORMALIZATION)

Was passiert:
- Frames werden auf ein gemeinsames Helligkeitsniveau gebracht.
- Hintergrund und Bildniveau werden vergleichbar gemacht.

Ergebnis:
- Unterschiedliche Aufnahmebedingungen wirken weniger störend.

---

## 5) Globale Qualitätsbewertung (GLOBAL_METRICS)

Was passiert:
- Jedes Frame bekommt globale Qualitätswerte (z. B. Schärfe/Signal).
- Daraus entsteht ein globales Gewicht pro Frame.

Ergebnis:
- Gute Frames zählen später stärker, schwache weniger.

---

## 6) Tile-Gitter bauen (TILE_GRID)

Was passiert:
- Das Bild wird in viele leicht überlappende Kacheln geteilt.
- Tile-Größe wird passend zur Datenqualität gewählt.

Ergebnis:
- Struktur, um lokal (statt nur global) zu entscheiden.

Skizze:
```text
+----+----+----+
| T1 | T2 | T3 |
+----+----+----+
| T4 | T5 | T6 |
+----+----+----+
| T7 | T8 | T9 |
+----+----+----+
```

---

## 7) Gemeinsamen Datenbereich bestimmen (COMMON_OVERLAP)

Was passiert:
- Es wird berechnet, welche Pixel auf dem Canvas in allen relevanten Frames tatsächlich Daten tragen.
- Daraus entstehen globale und tile-lokale Valid-Fraktionen.
- Rand-/Leerbereiche durch Rotation/Translation werden maskiert.

Ergebnis:
- Rekonstruktion und Stacking nutzen nur robuste gemeinsame Bildbereiche.

---

## 8) Lokale Qualitätswerte je Tile (LOCAL_METRICS)

Was passiert:
- Für jedes Frame und jedes Tile wird lokal bewertet.
- So sieht das System, welche Bildteile in welchem Frame gut sind.

Ergebnis:
- Lokale Gewichte pro Tile und Frame.

---

## 9) Tile-Rekonstruktion (TILE_RECONSTRUCTION)

Was passiert:
- Für jedes Tile werden die besten Informationen aus vielen Frames kombiniert.
- Übergänge werden weich zusammengesetzt (damit keine harten Kanten entstehen).

Ergebnis:
- Ein rekonstruiertes Gesamtbild mit lokal optimierter Qualität.

Skizze:
```text
Frame A ist in Tile links gut
Frame B ist in Tile rechts gut
=> Rekonstruktion nimmt links mehr aus A, rechts mehr aus B
```

---

## 10) Zustands-Cluster (STATE_CLUSTERING, optional)

Was passiert:
- Frames mit ähnlichem Zustand (z. B. ähnliche Qualität/Wetterlage) werden gruppiert.

Ergebnis:
- Bessere Trennung unterschiedlicher Aufnahmebedingungen.

---

## 11) Synthetische Frames (SYNTHETIC_FRAMES, optional)

Was passiert:
- Aus Clustern werden stabile Zwischenbilder erzeugt.
- Diese Zwischenbilder sind oft ruhiger und robuster.

Ergebnis:
- Bessere Grundlage für das finale Stacking.

---

## 12) Finales Stacking (STACKING)

Was passiert:
- Alles wird final zusammengeführt.
- Ausreißer werden robust behandelt (z. B. Satellitenspuren, einzelne Hotpixel-Spitzen).

Ergebnis:
- Ein lineares, sauberes Summenbild.

---

## 13) Debayer (DEBAYER, bei OSC)

Was passiert:
- Bei OSC/CFA wird aus dem Mosaik ein RGB-Bild erzeugt.
- Bei Mono ist das meist ein Durchlauf ohne Umwandlung.

Ergebnis:
- Farbbild (oder Mono-Pass-through).

---

## 14) Astrometrie (ASTROMETRY)

Was passiert:
- Das Bild wird am Himmel verortet (WCS/Plate Solving).

Ergebnis:
- Pixel haben Himmelskoordinaten-Bezug.

---

## 15) Farbkalibrierung (PCC)

Was passiert:
- Farben werden über Sternkatalog-Abgleich auf realistischere Balance gebracht.

Ergebnis:
- Natürlichere, wissenschaftlich plausiblere Farben.

---

## 16) Abschluss (DONE)

Was passiert:
- Pipeline endet mit Status (`ok` oder `validation_failed`).
- Artefakte, Logs und Ausgaben liegen im Run-Ordner.

Ergebnis:
- Nachvollziehbarer, reproduzierbarer Endstand.

---

## Typische Ausgabedateien

- `outputs/stacked.fits` (lineares Endbild)
- `outputs/stacked_rgb.fits` (RGB nach Debayer)
- `outputs/stacked_rgb_pcc.fits` (nach Farbkalibrierung)
- WCS-/Astrometrie-Artefakte
- JSON-Artefakte pro Phase (Qualität, Gewichte, Validation)
- Lauf-Logs (`run_events.jsonl`)

---

## Auswertung mit `tile_compile_cpp/generate_report.py`

Für die strukturierte Qualitätsanalyse kann aus einem Run-Verzeichnis ein HTML-Report erzeugt werden:

```text
python tile_compile_cpp/generate_report.py runs/<run_id>
```

Der Report wird unter `runs/<run_id>/artifacts/report.html` abgelegt und ergänzt durch `report.css` sowie PNG-Diagramme.

Folgende Daten und Auswertungen können daraus direkt entnommen werden:

- **Normalisierung**: Hintergrundverlauf (Mono bzw. R/G/B) und Stabilität über die Zeit.
- **Globale Metriken**: Hintergrund, Rauschen, Gradientenenergie, globales Frame-Gewicht, Qualitätsverteilungen.
- **Sternmetriken (Siril-ähnlich)**: FWHM, wFWHM, Rundheit, Sternanzahl, FWHM-vs-Rundheit-Plot.
- **Registrierung**: Translation/Drift, Rotationsverlauf, CC-Verteilung.
- **Tile-Analyse**: Tile-Grid, lokale Qualitätskarten, räumliche Heatmaps.
- **Rekonstruktion**: Tile-Rekonstruktionskennzahlen (z. B. Kontrast/Hintergrund/SNR pro Tile).
- **Clustering & Synthetic Frames**: Clustergrößen, Nutzung synthetischer Frames, Reduktionsverhalten.
- **Validation**: FWHM-Verbesserung, Tile-Pattern-Indikatoren, weitere Qualitätschecks.
- **Pipeline-Timeline**: zeitlicher Ablauf der Phasen aus `run_events.jsonl`.
- **Frame-Usage-Funnel**: Entwicklung von „discovered“ bis „stacked/synthetic“.

Zusätzlich bindet der Report die verwendete `config.yaml` ein. Dadurch ist die Ergebnisbewertung direkt mit den Laufparametern nachvollziehbar.

---

## Hinweise zur Interpretation

1. **Linear bedeutet dunkel:** Ein lineares Astrobild wirkt in Viewer oft flach/dunkel, bis gestretcht wird.
2. **Validation kann fehlschlagen, obwohl Bild brauchbar ist:** Dann sind Grenzwerte verletzt, nicht zwingend das ganze Ergebnis unbrauchbar.
3. **Tile-basierte Methode optimiert lokal:** Das ist der Hauptvorteil gegenüber rein globalem Stacking.

---

## Kurzfazit

> Viele Bilder rein -> sauber ausrichten -> lokal bewerten -> tile-weise bestes Signal kombinieren -> robust stacken -> Farbe/Himmel kalibrieren -> fertiges Astro-Endbild raus.
