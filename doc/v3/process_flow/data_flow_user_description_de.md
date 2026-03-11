# Ablaufplan – technischer Datenfluss des Systems

## Zielbild der Pipeline

Das System verarbeitet eine Menge kalibrierter astronomischer Einzelaufnahmen zu einem reproduzierbaren Endprodukt im gemeinsamen geometrischen und photometrischen Referenzrahmen.

Technisch besteht die Pipeline aus drei Hauptblöcken:

- **Vorbereitung und Vereinheitlichung**
  - Eingaben prüfen
  - Geometrie vereinheitlichen
  - Intensitäten normalisieren
- **Qualitätsmodellierung und Rekonstruktion**
  - globale und lokale Metriken berechnen
  - tile-basierte Selektion und Rekonstruktion ausführen
  - optional Zustände clustern und synthetische Frames erzeugen
- **Post-Processing und Kalibrierung**
  - Debayer
  - Astrometrie / WCS
  - optional BGE
  - PCC

Das primäre Ergebnis ist ein lineares Summenbild. Je nach Konfiguration entstehen zusätzlich debayerte, gradientenkorrigierte und photometrisch kalibrierte Ableitungen sowie strukturierte Diagnoseartefakte.

## Zentrale Begriffe

- **Run**
  - Ein vollständiger Pipeline-Durchlauf mit eigenem Run-Verzeichnis unter `runs/<run_id>/`.
- **Phase**
  - Ein klar abgegrenzter Verarbeitungsschritt wie `REGISTRATION`, `LOCAL_METRICS` oder `PCC`.
- **Artifact**
  - Persistierte Diagnose- oder Zwischeninformation, typischerweise als JSON oder Report-Datei unter `artifacts/`.
- **Event-Timeline**
  - Zeitlich geordnete Laufereignisse in `logs/run_events.jsonl`.
- **Methodik-Profil**
  - `assumptions.pipeline_profile` bestimmt, ob die Pipeline eher streng normativ (`strict`) oder praktisch robust (`practical`) läuft.
- **Resume**
  - Bestehende Run-Verzeichnisse können für Post-Run-Phasen erneut verwendet werden, aktuell insbesondere ab `ASTROMETRY`, `BGE` oder `PCC`.

---

## Gesamtfluss

```text
Input frames (FITS)
   -> SCAN_INPUT
   -> REGISTRATION
   -> PREWARP
   -> CHANNEL_SPLIT
   -> NORMALIZATION
   -> GLOBAL_METRICS
   -> TILE_GRID
   -> COMMON_OVERLAP
   -> LOCAL_METRICS
   -> TILE_RECONSTRUCTION
   -> [optional] STATE_CLUSTERING
   -> [optional] SYNTHETIC_FRAMES
   -> STACKING
   -> [optional / datenabhängig] DEBAYER
   -> ASTROMETRY
   -> [optional] BGE
   -> [optional] PCC
   -> DONE
```

---

## Warum tile-basiert gearbeitet wird

Eine globale Bewertung pro Frame ist für astrophotografische Serien oft nicht ausreichend. Die lokale Bildqualität variiert innerhalb desselben Frames unter anderem durch:

- ortsabhängige Seeing-Unterschiede
- lokale Guiding- oder Verformungseffekte
- Randartefakte nach Warp/Rotation
- ungleichmäßige Hintergrund- oder Rauschverteilungen

Deshalb modelliert das System die Daten nicht nur auf Frame-Ebene, sondern zusätzlich auf Tile-Ebene. Dadurch kann pro Raumregion entschieden werden, welche Frames oder Frame-Anteile dort die höchste nutzbare Qualität liefern.

---

## Phasen im Detail

## 0) Eingang prüfen (`SCAN_INPUT`)

**Eingabe**

- ein Eingabepfad oder mehrere Eingabeverzeichnisse
- FITS-Dateien mit Headern und Aufnahmemetadaten

**Verarbeitung**

- Dateierkennung und Enumerierung der Eingaben
- Plausibilitätsprüfung von Headern, Bit-Tiefe, Bildabmessungen und Farbmodus
- Vorabklassifikation in Mono oder OSC/CFA
- Erkennung offensichtlicher Ausschlussfälle
- Prüfung, ob ausreichend Speicherplatz und Arbeitsverzeichnis-Kapazität verfügbar sind

**Ausgabe**

- bereinigte Frame-Liste
- Scan-Zusammenfassung mit Metadaten, Warnungen und Fehlern
- Guardrails für nachgelagerte Startentscheidungen

---

## 1) Globale Registrierung (`REGISTRATION`)

**Ziel**

- alle Frames in ein gemeinsames geometrisches Bezugssystem überführen

**Verarbeitung**

- Auswahl eines Referenzframes
- Schätzung geometrischer Transformationen relativ zum Referenzframe
- Nutzung von Fallback-Strategien, falls das primäre Registrierungsverfahren unzureichend ist
- Persistenz von Registrierungsmetrik und Transformationsparametern

**Ausgabe**

- registrierte Transformationsinformationen pro Frame
- Qualitätsindikatoren wie Korrelation, Drift, Rotation oder Fehlversatz

---

## 2) Prewarp auf gemeinsamen Canvas (`PREWARP`)

**Ziel**

- alle registrierten Frames auf denselben Zielcanvas und dieselbe Pixelgeometrie bringen

**Verarbeitung**

- Anwendung der berechneten Transformationen auf einen gemeinsamen Zielbereich
- bei OSC/CFA: CFA-sicheres Warping über Subplane-Logik, damit das Bayer-Muster semantisch stabil bleibt
- Erweiterung des Canvas bei Feldrotation oder Translation außerhalb der ursprünglichen Begrenzung
- Verwaltung von Offsets wie `tile_offset_x` und `tile_offset_y`

**Ausgabe**

- prewarped Frames mit einheitlicher Geometrie
- konsistenter Koordinatenraum für alle tile-basierten Folgeschritte

---

## 3) Kanalmodell festlegen (`CHANNEL_SPLIT`)

**Ziel**

- ein konsistentes internes Kanalmodell für Mono- oder OSC-Daten definieren

**Verarbeitung**

- Festlegung, ob spätere Metriken und Rekonstruktionen auf Mono, CFA-Subplanes oder RGB-kompatiblen Repräsentationen operieren
- Ableitung kanalbezogener Metadaten für nachgelagerte Stufen

**Ausgabe**

- Kanal- und Modusbeschreibung für weitere Phasen

---

## 4) Normalisierung (`NORMALIZATION`)

**Ziel**

- Signal- und Hintergrundniveau zwischen Frames vergleichbar machen

**Verarbeitung**

- Schätzung von Hintergrund- und Intensitätsstatistik pro Frame bzw. Kanal
- Skalierung auf einen gemeinsamen Referenzzustand
- Persistenz der Normalisierungsparameter

**Ausgabe**

- normalisierte Frames oder äquivalente Normalisierungsparameter
- Diagnostik zur Stabilität von Hintergrund und Signalniveau

---

## 5) Globale Qualitätsmetriken (`GLOBAL_METRICS`)

**Ziel**

- pro Frame ein globales Qualitätsprofil ableiten

**Verarbeitung**

- Berechnung globaler Kennzahlen wie Hintergrundniveau, Rauschen, Gradientenenergie, Sternmetriken oder globale Schärfeindikatoren
- Ableitung eines globalen Frame-Gewichts
- im `strict`-Profil: vollständige Bewertung auf der vereinheitlichten Geometrie vor lokalen Schritten

**Ausgabe**

- globale Metriken je Frame
- globale Gewichte und Selektionsgrundlagen

---

## 6) Tile-Gitter erzeugen (`TILE_GRID`)

**Ziel**

- das Bildfeld in lokal auswertbare Regionen zerlegen

**Verarbeitung**

- Erzeugung eines überlappenden oder weich kombinierbaren Tile-Rasters
- Parametrisierung von Tile-Größe, Überdeckung und gültiger Nutzungsregion

**Ausgabe**

- Tile-Geometrie als Grundlage für lokale Metriken und Rekonstruktion

---

## 7) Gemeinsamen Überlappungsbereich bestimmen (`COMMON_OVERLAP`)

**Ziel**

- nur Pixelbereiche verwenden, die nach dem Warp tatsächlich belastbare Daten tragen

**Verarbeitung**

- Ermittlung globaler und tile-lokaler Valid-Masken
- Berechnung der gültigen Flächenanteile nach Warp, Translation und Rotation
- Maskierung leerer oder unzureichend überlappender Randregionen

**Ausgabe**

- globale Valid-Fraktionen
- tile-lokale Gültigkeitsmaße
- robuste Nutzungsmaske für Rekonstruktion und Stacking

---

## 8) Lokale Metriken je Tile (`LOCAL_METRICS`)

**Ziel**

- pro Tile und pro Frame die lokal beste Datenqualität modellieren

**Verarbeitung**

- lokale Schärfe-, Kontrast-, Rausch- oder Sternmetriken je Tile berechnen
- Kombination mit globalen Gewichten und Valid-Masken
- im `strict`-Profil auf prewarped Rohdaten zur geometrisch konsistenten Vergleichbarkeit

**Ausgabe**

- lokale Gewichte und lokale Qualitätsprofile für jede Tile/Frame-Kombination

---

## 9) Tile-Rekonstruktion (`TILE_RECONSTRUCTION`)

**Ziel**

- aus den lokal besten Beiträgen ein räumlich konsistentes Zwischenbild rekonstruieren

**Verarbeitung**

- Selektion oder gewichtete Fusion der besten Tile-Beiträge
- weiche Übergänge zwischen benachbarten Tiles, um Nahtartefakte zu vermeiden
- Rekonstruktion auf Basis lokaler Qualitätskarten und Nutzungsgewichte

**Ausgabe**

- rekonstruiertes Bild mit lokal optimierter Informationsnutzung
- Rekonstruktionsmetriken pro Tile

---

## 10) Zustands-Clustering (`STATE_CLUSTERING`, optional)

**Ziel**

- Frames mit ähnlichen Qualitäts- oder Beobachtungszuständen gruppieren

**Verarbeitung**

- Clustering anhand globaler und/oder lokaler Merkmalsräume
- Trennung heterogener Teilpopulationen innerhalb einer Serie

**Ausgabe**

- Clusterzuordnung der Frames
- Diagnostik zur Clusterstabilität und Clustergröße

---

## 11) Synthetische Frames (`SYNTHETIC_FRAMES`, optional)

**Ziel**

- aus Clustern robuste Zwischenrepräsentationen ableiten

**Verarbeitung**

- Aggregation von Frame-Gruppen zu synthetischen Repräsentanten
- Reduktion von Varianz innerhalb eines Zustandsclusters

**Ausgabe**

- synthetische Frames als alternative Eingänge für spätere Aggregationsstufen

---

## 12) Finales Stacking (`STACKING`)

**Ziel**

- das finale lineare Summenbild erzeugen

**Verarbeitung**

- robuste Aggregation über rekonstruierte oder synthetische Zwischenstufen
- Ausreißerunterdrückung für Hotpixel, Satellitenspuren oder sporadische Artefakte
- gewichtete Fusion unter Berücksichtigung der zuvor berechneten Qualitätsmodelle

**Ausgabe**

- lineares Endbild, typischerweise `outputs/stacked.fits`

---

## 13) Debayer (`DEBAYER`, bei OSC)

**Ziel**

- CFA-/OSC-Daten in eine RGB-Repräsentation überführen

**Verarbeitung**

- Demosaicing auf dem gestackten oder entsprechend vorbereiteten linearen Datensatz
- bei Mono: Durchreichen ohne Farbinterpolation

**Ausgabe**

- RGB-FITS, typischerweise `outputs/stacked_rgb.fits`

---

## 14) Astrometrie (`ASTROMETRY`)

**Ziel**

- WCS-Lösung für das Endbild erzeugen

**Verarbeitung**

- Plate Solving gegen Astrometrie-Werkzeuge und Kataloge
- Eintrag oder Ableitung von Himmelskoordinatenbezug und Bildskalierung

**Ausgabe**

- WCS-informiertes Bild oder zugehörige WCS-Datei
- Diagnoseartefakte zum Solve-Prozess

---

## 15) Background Gradient Extraction (`BGE`, optional)

**Ziel**

- großskalige Hintergrundgradienten vor der Farbkalibrierung reduzieren

**Verarbeitung**

- Schätzung eines Hintergrundmodells pro RGB-Kanal
- Subtraktion des Modells vom RGB-Bild
- Persistenz von Diagnosedaten, z. B. `artifacts/bge.json`

**Ausgabe**

- gradientenkorrigiertes RGB-Bild, typischerweise `outputs/stacked_rgb_bge.fits`
- BGE-Diagnostik

---

## 16) Photometrische Farbkalibrierung (`PCC`)

**Ziel**

- das RGB-Bild auf eine astrophysikalisch plausiblere Farbbalance kalibrieren

**Verarbeitung**

- Match mit Sternkatalogen unter Nutzung der WCS-Information
- Bestimmung und Anwendung von Farbskalierungs- bzw. Kalibrierfaktoren

**Ausgabe**

- photometrisch kalibriertes RGB-Bild, typischerweise `outputs/stacked_rgb_pcc.fits`
- PCC-Diagnostik und ggf. Katalog-Nebenprodukte

---

## 17) Abschluss (`DONE`)

**Ziel**

- den Run in einen konsistenten Endzustand überführen

**Verarbeitung**

- Abschlussstatus persistieren, z. B. `ok` oder `validation_failed`
- Artefakte, Logs und Konfigurationssnapshot vervollständigen

**Ausgabe**

- reproduzierbarer und auditierbarer Run-Stand

---

## Typische Run-Struktur

Ein Run erzeugt typischerweise `runs/<run_id>/` mit folgender logischer Struktur:

- `outputs/`
  - finale und abgeleitete FITS-Produkte
  - z. B. `stacked.fits`, `stacked_rgb.fits`, `stacked_rgb_bge.fits`, `stacked_rgb_pcc.fits`
- `artifacts/`
  - JSON-Diagnostik pro Phase
  - Report-Dateien und Diagramme
- `logs/`
  - `run_events.jsonl` als Event-Timeline des Laufs
- `config.yaml`
  - Snapshot der tatsächlich verwendeten Konfiguration

Wichtig ist weniger der exakte Dateiname als die Semantik: Ausgaben, Artefakte, Logs und Konfigurationssnapshot sind sauber getrennt abgelegt.

---

## Resume von Post-Run-Phasen

Wenn ein Run bereits existiert, können Post-Processing-Phasen erneut auf Basis des vorhandenen Run-Zustands ausgeführt werden:

```text
./tile_compile_runner resume --run-dir runs/<run_id> --from-phase ASTROMETRY
```

Dabei werden insbesondere verwendet:

- der Konfigurationssnapshot `config.yaml`
- vorhandene Outputs und Artefakte der früheren Phasen
- das Run-Verzeichnis als maßgeblicher Arbeitskontext

Resume ist damit kein „teilweise neuer Run“, sondern eine kontrollierte Fortsetzung auf Basis bereits persistierter Laufdaten.

---

## Auswertung mit dem integrierten Report-Generator

Für technische Auswertung und Qualitätssicherung kann aus einem Run-Verzeichnis ein HTML-Report erzeugt werden:

```text
./tile_compile_cli generate-report runs/<run_id>
```

Der Report liegt typischerweise unter `runs/<run_id>/artifacts/report.html` und korreliert Laufereignisse, Diagnoseartefakte und Konfiguration.

Typische Auswertungsblöcke sind:

- **Normalisierung**
  - Hintergrundtrends und Stabilität der Intensitätsskalierung
- **Globale Metriken**
  - Hintergrund, Rauschen, Gradientenenergie, globale Gewichte, Verteilungen
- **Sternmetriken**
  - FWHM, wFWHM, Rundheit, Sternzahl, Korrelationsplots
- **Registrierung**
  - Drift, Rotation, Matching- bzw. Korrelationsqualität
- **Tile-Analyse**
  - Tile-Grid, lokale Qualitätskarten, Heatmaps
- **Rekonstruktion**
  - tile-lokale Rekonstruktionskennzahlen und Nutzungsbilder
- **Clustering und Synthetic Frames**
  - Clustergrößen, Reduktionsverhalten, Nutzung synthetischer Repräsentanten
- **BGE / PCC**
  - Hintergrundmodell, Residuen, Kalibrierungsdiagnostik
- **Validation**
  - abgeleitete Qualitätsindikatoren und Grenzwertprüfungen
- **Timeline**
  - zeitliche Sequenz der Phasen aus `run_events.jsonl`

Der Report bindet zusätzlich die verwendete `config.yaml` ein. Damit bleibt jeder Befund direkt auf den konkreten Parametrisierungszustand zurückführbar.

---

## Hinweise zur Interpretation

1. **Lineare Bilder wirken dunkel**
   - Das ist erwartbar. Eine lineare Summenaufnahme ist nicht für sofortige visuelle Präsentation gestretcht.
2. **`validation_failed` bedeutet nicht automatisch „nutzlos“**
   - Es bedeutet zunächst, dass definierte Qualitäts- oder Guardrail-Kriterien verletzt wurden.
3. **Tile-basierte Optimierung ist das Kernprinzip**
   - Der wesentliche Mehrwert entsteht dadurch, dass lokale Qualität genutzt wird, statt globale Durchschnittsqualität blind auf alle Regionen zu übertragen.

---

## Kurzfazit

> Die Pipeline transformiert eine heterogene Serie von FITS-Frames in einen gemeinsamen geometrischen und photometrischen Referenzraum, bewertet die Daten global und lokal, rekonstruiert das Signal tile-basiert und erzeugt daraus ein reproduzierbares Endbild samt Diagnostik, WCS- und optionaler Farbkalibrierung.
