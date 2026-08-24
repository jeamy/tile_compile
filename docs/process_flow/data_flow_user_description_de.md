# Ablaufplan – technischer Datenfluss des Systems

## Zielbild der Pipeline

Das System verarbeitet eine Menge kalibrierter astronomischer Einzelaufnahmen zu einem reproduzierbaren Endprodukt im gemeinsamen geometrischen und photometrischen Referenzrahmen.

Technisch besteht die Pipeline aus drei Hauptblöcken:

- **Vorbereitung und Vereinheitlichung**
  - Eingaben prüfen
  - Geometrie vereinheitlichen
  - Intensitäten normalisieren
- **Qualitätsmodellierung und Rekonstruktion**
  - globale Metriken und dichte AQMH-Quality-Maps berechnen
  - pixelweise gewichtete AQMH-Rekonstruktion ausführen
  - optional Classic Tile-Compile mit lokalen Tile-Metriken, Clustering und synthetischen Frames verwenden
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
  - Ein klar abgegrenzter Verarbeitungsschritt wie `REGISTRATION`, `AQMH_MAPS` oder `PCC`.
- **Artifact**
  - Persistierte Diagnose- oder Zwischeninformation, typischerweise als JSON oder Report-Datei unter `artifacts/`.
- **Event-Timeline**
  - Zeitlich geordnete Laufereignisse in `logs/run_events.jsonl`.
- **Assumptions-Schwellen**
  - `assumptions.frames_min` und `assumptions.frames_reduced_threshold` bestimmen, ob der Runner abbricht, in Reduced Mode wechselt oder die volle Pipeline ausführt.
- **Resume**
  - Bestehende Run-Verzeichnisse können für unterstützte Folgephasen erneut verwendet werden, insbesondere `STACKING`, `ASTROMETRY`, `BGE`, `PCC` und `HYPERMETRIC_STRETCH`.

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
   -> TILE_GRID (Hilfsgeometrie; Rekonstruktionsraster für Classic)
   -> COMMON_OVERLAP
   -> AQMH_MAPS (Enum 19)
   -> AQMH_GLOBAL_QUALITY (Enum 20)
   -> AQMH_RECONSTRUCTION (Enum 21)
   -> AQMH_DIAGNOSTICS (Enum 22)
      oder [Classic] LOCAL_METRICS -> TILE_RECONSTRUCTION
   -> [nur Classic, optional] STATE_CLUSTERING
   -> [nur Classic, optional] SYNTHETIC_FRAMES
   -> STACKING
   -> [optional / datenabhängig] DEBAYER
   -> ASTROMETRY
   -> [optional] BGE
   -> [optional] PCC
   -> [optional] HYPERMETRIC_STRETCH
   -> DONE
```

---

## Warum AQMH der Standard ist

Eine globale Bewertung pro Frame ist für astrophotografische Serien oft nicht
ausreichend, weil die Qualität räumlich variiert. AQMH berechnet deshalb für
jeden Frame eine dichte Quality-Map und gewichtet jeden Ausgabepixel
unabhängig. Damit werden ein festes Tile-Raster und Overlap-Add-Nähte vermieden,
während weiterhin folgende Effekte berücksichtigt werden:

- ortsabhängige Seeing-Unterschiede
- lokale Guiding- oder Verformungseffekte
- Randartefakte nach Warp/Rotation
- ungleichmäßige Hintergrund- oder Rauschverteilungen

Die ursprüngliche tile-basierte Methode bleibt über
`method: classic_tile_compile` als **Classic Tile-Compile** verfügbar. Sie
approximiert lokale Qualität mit überlappenden Tiles und ist nicht mehr der
Standard.

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
- Ausführung auf CPU-Workern; diese Phase verwendet keine GPU

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
- Nutzung von CUDA oder OpenCL für Vollbild-Warps, andernfalls CPU-Fallback

**Ausgabe**

- prewarped Frames mit einheitlicher Geometrie
- konsistenter Koordinatenraum für AQMH- und Classic-Folgeschritte

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

- Hilfsgeometrie und das Rekonstruktionsraster für den Classic-Pfad bereitstellen

**Verarbeitung**

- Erzeugung eines überlappenden oder weich kombinierbaren Tile-Rasters
- Parametrisierung von Tile-Größe, Überdeckung und gültiger Nutzungsregion

**Ausgabe**

- Hilfsgeometrie; bei Classic Tile-Compile zusätzlich Raster für lokale Metriken und Rekonstruktion

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

## 8) AQMH-Quality-Maps (`AQMH_MAPS`, Enum 19)

**Ziel**

- ein dichtes pixelweises Qualitätsmodell für jeden Frame erzeugen

**Verarbeitung**

- Multi-Scale-Schärfe und SNR mit einer Laplacian-Pyramide berechnen
- artefaktdominierten Support erkennen und die gemeinsame Canvas-Maske anwenden
- eine `Q_map` pro Frame für die unabhängige Rekonstruktion cachen
- verfügbare CUDA-/OpenCL-Filter verwenden

**Ausgabe**

- gecachte AQMH-Quality-Maps und AQMH-Diagnostik

Danach folgt `AQMH_GLOBAL_QUALITY` (Enum 20) für die globalen Frame-Gewichte.
Mit `method: classic_tile_compile` wird stattdessen `LOCAL_METRICS` (Enum 8)
ausgeführt und lokale Tile-Metriken und Gewichte `L_f,t` berechnet.

---

## 9) Rekonstruktion (`AQMH_RECONSTRUCTION`, Enum 21)

**Ziel**

- das finale lineare Signal standardmäßig aus pixelweisen AQMH-Quality-Maps oder optional aus klassischen lokalen Tile-Beiträgen rekonstruieren

**Verarbeitung**

- AQMH: jeden Pixel aus globalen Frame-Gewichten und Quality-Maps kombinieren und gewichtet sigma-clippen
- Classic: `TILE_RECONSTRUCTION` (Enum 9) fusioniert gewichtete Tile-Beiträge
  und benachbarte Überlappungsbereiche.
- Streaming-CUDA für AQMH-Rekonstruktion verwenden, wenn Cherry-Pick deaktiviert ist
- CUDA/OpenCL für klassisches Sigma-Clipping und Overlap-Add verwenden; sonst CPU-Fallback

**Ausgabe**

- rekonstruiertes Bild mit qualitätsoptimierter Informationsnutzung
- AQMH- oder Tile-Rekonstruktionsdiagnostik

---

## 10) Zustands-Clustering (`STATE_CLUSTERING`, nur Classic Tile-Compile)

**Ziel**

- Frames mit ähnlichen Qualitäts- oder Beobachtungszuständen gruppieren

**Verarbeitung**

- Clustering anhand globaler und/oder lokaler Merkmalsräume
- Trennung heterogener Teilpopulationen innerhalb einer Serie

**Ausgabe**

- Clusterzuordnung der Frames
- Diagnostik zur Clusterstabilität und Clustergröße

---

## 11) Synthetische Frames (`SYNTHETIC_FRAMES`, nur Classic Tile-Compile)

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

- AQMH: finales Rekonstruktionsergebnis aus `AQMH_RECONSTRUCTION` (Phase 21)
  unverändert übernehmen
- Classic: rekonstruierte oder synthetische Zwischenstufen robust aggregieren
- Classic: Hotpixel, Satellitenspuren oder sporadische Artefakte unterdrücken
- Classic: Daten anhand der zuvor berechneten Qualitätsmodelle gewichtet fusionieren
- Classic: CUDA/OpenCL für gewichtete oder Sigma-Clip-Reduktion verwenden und OSC-RGB-Kanäle parallel verarbeiten

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

- zuerst ASTAP-Plate-Solving; liefert es keine WCS, werden erkannte Sterne ohne Siril-Start und ohne Netzabfrage gegen den lokal installierten PCC-Gaia-DR3-Katalog abgeglichen
- Eintrag oder Ableitung von Himmelskoordinatenbezug und Bildskalierung

**Ausgabe**

- WCS-informiertes Bild oder zugehörige WCS-Datei
- Diagnoseartefakte und Phasenfelder zum gewählten Löser, zu Gaia-Sternzahlen und zu einem möglichen Fallback-Fehler

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

## 17) HyperMetric Stretch (`HYPERMETRIC_STRETCH`, optional)

**Ziel**

- das PCC-kalibrierte RGB-Bild mit VeraLux HMS final und reproduzierbar stretchen

**Verarbeitung**

- liest das PCC-RGB-Ergebnis, typischerweise `outputs/stacked_rgb_pcc.fits`
- ermittelt bzw. nutzt das konfigurierte Sensorprofil, den adaptiven Anchor und Auto-LogD
- wendet die HyperMetric-Stretch-Kurve und die Farb-Erhaltung an

**Ausgabe**

- gestretchtes RGB-Bild, typischerweise `outputs/stacked_rgb_hms.fits`
- bei `write_channels: true` zusätzlich `hms_R.fit`, `hms_G.fit`, `hms_B.fit`

---

## 18) Abschluss (`DONE`)

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
  - z. B. `stacked.fits`, `stacked_rgb.fits`, `stacked_rgb_bge.fits`, `stacked_rgb_pcc.fits`, `stacked_rgb_hms.fits`
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

Die vollständige Resume-Matrix mit den tatsächlich implementierten Einstiegen
und Mindestabhängigkeiten steht in
[resume_dependencies_de.md](resume_dependencies_de.md). Die dortige
Unterscheidung zwischen direktem Resume und In-Place-Vollständigkeitslauf ist
verbindlich.

Wenn ein Run bereits existiert, können unterstützte Post-Processing-Phasen auf
Basis des vorhandenen Run-Zustands ausgeführt werden:

```text
./tile_compile_runner resume --run-dir runs/<run_id> --from-phase ASTROMETRY
./tile_compile_runner resume --run-dir runs/<run_id> --from-phase HYPERMETRIC_STRETCH
```

Dabei werden insbesondere verwendet:

- der Konfigurationssnapshot `config.yaml`
- vorhandene Outputs und Artefakte der früheren Phasen
- das Run-Verzeichnis als maßgeblicher Arbeitskontext

Für direkte Post-Processing-Resumes ist dies eine kontrollierte Fortsetzung auf
Basis persistierter Laufdaten. Die in der Resume-Matrix als In-Place-
Vollständigkeitslauf markierten frühen Phasen starten dagegen die komplette
Pipeline im selben Run-Verzeichnis.

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
  - nur Classic: Tile-Raster, lokale Metriken und räumliche Heatmaps
- **AQMH-Analyse**
  - Quality-Map-Statistiken, Artefakt-Support und Rekonstruktionsdiagnostik
- **Rekonstruktion**
  - pixelweise AQMH-Rekonstruktion oder lokale Classic-Tile-Nutzungsmetriken
- **Clustering und Synthetic Frames**
  - nur Classic: Clustergrößen, Reduktionsverhalten und Nutzung synthetischer Repräsentanten
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
3. **Pixelweise AQMH-Qualität ist das Standardprinzip**
   - Der Hauptvorteil entsteht durch dichte lokale Qualitätsgewichtung statt einer rein globalen Durchschnittsbewertung. Classic Tile-Compile bleibt verfügbar, wenn ausdrücklich tile-basierte Diagnostik oder Clustering benötigt wird.

---

## Kurzfazit

> Die Pipeline transformiert eine heterogene Serie von FITS-Frames in einen gemeinsamen geometrischen und photometrischen Referenzraum, erzeugt dichte AQMH-Quality-Maps, rekonstruiert das Signal pixelweise und liefert ein reproduzierbares Endbild samt Diagnostik, WCS-Metadaten und optionaler Farbkalibrierung. Der frühere tile-basierte Workflow bleibt als Classic Tile-Compile erhalten.
