# BGE Preview GUI – detaillierter Implementierungsplan

## 1. Ziel und Umfang

Für die Resume-Phase `BGE` wird eine interaktive Konfigurations- und
Vorschauoberfläche ergänzt, analog zum vorhandenen HMS-Dialog. Als fachliche
Referenz dient `AutoBGE.py` Version 2.0.2. Die eigentliche Berechnung erfolgt
nicht in Python und wird nicht im Web-Backend nachgebaut, sondern verwendet den
bereits vorhandenen C++-AutoBGE-Core.

Der erste Ausbau unterstützt:

- RGB- und vom Core unterstützte Mono-Eingaben,
- `bge.method: autobge`,
- Vorschau des korrigierten Bildes und des berechneten Hintergrundmodells,
- Darstellung der verwendeten Sample-Punkte,
- manuell gezeichnete Ausschlussflächen,
- Übernahme der Werte in die Resume-YAML,
- Resume ab `BGE` über den bestehenden Resume-Endpoint.

Die klassische Grid-/Tile-BGE bleibt im MVP im normalen Config-Editor
konfigurierbar. Eine interaktive Classic-BGE-Preview ist eine spätere
Ausbaustufe, weil sie zusätzlich `local_metrics.json`, `tile_grid.json` und die
Classic-BGE-Fitdiagnostik benötigt.

## 2. Bezug zu AutoBGE.py

Die Referenzimplementierung arbeitet zweistufig:

1. Bild in einen Arbeitsraum transformieren.
2. Bild und Maske verkleinern.
3. Hintergrund-Sample-Punkte bestimmen:
   - Rand- und Eckpunkte,
   - zufällige Punkte verteilt auf vier Quadranten,
   - Ausschluss heller Bereiche,
   - Verschieben der Punkte zu lokal dunkleren Positionen.
4. Pro Kanal ein Polynommodell fitten und abziehen.
5. Auf dem Residuum neue Sample-Punkte bestimmen.
6. Ein RBF-Modell fitten.
7. Polynom- und RBF-Hintergrund kombinieren.
8. Ergebnis in den ursprünglichen Arbeitsraum zurücktransformieren.
9. Zwischen korrigiertem Bild und Hintergrundmodell umschalten.

Der C++-Core bildet diesen Ablauf bereits in `autobge.cpp` ab und erweitert ihn
um deterministische Zufallsauswahl, robuste Patch-Schätzer, Varianzfilter,
Canvas-Maske, Guard-Prüfungen und Diagnostik. Die GUI bildet deshalb den
C++-Vertrag ab. Abweichende Defaults aus dem Python-Skript dürfen die aktuellen
Projekt-Defaults nicht überschreiben.

## 3. Verbindlicher Einstieg im Resume-Bereich

Wenn die Phase `BGE` ausgewählt ist, zeigt der rechte Resume-Bereich:

```text
[ BGE konfigurieren ]  [ BGE ]
                        ^ vorhandenes Phasen-Badge
```

- Der Button steht unmittelbar links neben dem BGE-Badge.
- Er wird ausschließlich bei ausgewählter Phase `BGE` angezeigt.
- Der Klick öffnet ein Modal innerhalb des Run Monitors.
- Das Modal lädt seine Startwerte aus der aktuell im Resume-YAML-Editor
  geladenen Konfiguration.
- Bei `bge.method: none` oder `classic` erklärt der Dialog, dass die interaktive
  Vorschau AutoBGE verwendet. Der Benutzer kann im Dialog ausdrücklich auf
  `autobge` wechseln; es erfolgt keine stille Methodenänderung.
- Deutsche Beschriftung: `BGE konfigurieren`.
- Englische Beschriftung: `Configure BGE`.

## 4. Benutzerablauf

1. Benutzer wählt `BGE` in der Phasenliste.
2. Der Button `BGE konfigurieren` erscheint links neben dem Phasen-Badge.
3. Klick öffnet das BGE-Modal.
4. Das Backend lädt das lineare RGB-Bild vor BGE und die Canvas-Maske.
5. Das Modal zeigt zunächst das Original-Proxybild.
6. Nach gültigen Parameteränderungen wird nach Debounce eine AutoBGE-Vorschau
   berechnet.
7. Benutzer kann zwischen `Original`, `Korrigiert` und `Hintergrundmodell`
   umschalten sowie Sample-Punkte einblenden.
8. Optional zeichnet der Benutzer Ausschlusspolygone über dem Bild.
9. `Übernehmen & Resume starten` merged die BGE-Werte in die Resume-YAML und
   ruft den bestehenden Resume-Endpoint mit `from_phase: "BGE"` auf.
10. Das Modal schließt und der Run Monitor zeigt den Resume-Job.

`Zurücksetzen` stellt Parameter und Ausschlussflächen auf den Zustand beim
Öffnen zurück. `Abbrechen` verändert weder YAML noch FITS-Dateien.

## 5. Eingangsartefakt-Vertrag

Die Preview verwendet dasselbe lineare Bild, das der BGE-Resume-Pfad vor der
Hintergrundextraktion verwendet:

1. `outputs/stacked_rgb_solve.fits`, wenn vorhanden,
2. sonst `outputs/stacked_rgb.fits`.

Bereits BGE-korrigierte Dateien wie `stacked_rgb_bge_linear.fits` oder
`stacked_rgb_bge.fits` dürfen niemals als Eingang verwendet werden, da sonst
eine doppelte Hintergrundkorrektur entstünde.

Zusätzlich erforderlich:

- `outputs/canvas_mask.fits`,
- identische Bild- und Maskengeometrie,
- drei konsistente RGB-Kanäle oder ein gültiger Mono-Cube entsprechend der
  Core-Semantik.

Die Artefaktauflösung wird aus dem Runner in eine gemeinsame Servicefunktion
extrahiert. Preview und finaler Resume dürfen keine getrennten Prioritätslisten
pflegen.

## 6. Vorarbeiten und notwendige Core-Korrekturen

### 6.1 AutoBGE von Classic-Tile-Artefakten entkoppeln

Im aktuellen Resume-Pfad wird `apply_background_extraction()` nur aufgerufen,
wenn Tile-Metriken und Tile-Grid vorhanden und konsistent sind. AutoBGE benötigt
diese Daten intern jedoch nicht; es verwendet Bild, Canvas-Maske und eigene
Sample-Punkte.

Der Runner wird so aufgeteilt:

- `method == autobge`: AutoBGE mit leerem Tile-Kontext direkt aufrufen,
- `method == classic`: bisherigen Tile-Metrik-/Grid-Vertrag beibehalten,
- `method == none`: BGE überspringen.

Dieser Fix ist Voraussetzung für eine korrekte Preview und einen zuverlässigen
Resume ab BGE.

### 6.2 Separate Sampling-Ausschlussmaske

`common_valid_mask` bestimmt den gültigen Bildbereich und wird auch auf das
Ausgabebild angewendet. Benutzerpolygone dürfen nicht darin gespeichert werden,
sonst würden ausgeschlossene Nebelbereiche im Ergebnis auf null gesetzt.

`image::BGEConfig` erhält deshalb einen separaten internen Vertrag:

```cpp
std::vector<uint8_t> sampling_valid_mask;
int sampling_mask_rows = 0;
int sampling_mask_cols = 0;
```

Semantik:

- `common_valid_mask`: gültiger Canvas; beeinflusst Analyse und Ausgabe,
- `sampling_valid_mask`: zusätzliche Zulässigkeit nur für Sample-Erzeugung,
- effektive Sampling-Maske: `common_valid_mask AND sampling_valid_mask AND
  structure_mask`,
- Ausgabe wird ausschließlich mit `common_valid_mask` begrenzt.

Die neue Maske wird in `build_autobge_models()` beim Aufbau von
`sampling_mask` berücksichtigt. Classic BGE bleibt unverändert, bis eine
separate Classic-Preview geplant wird.

### 6.3 Preview-fähiges Core-Ergebnis

Der aktuelle `AutoBGEResult` enthält Hintergrundmodelle und Kanaldignostik,
aber kein direkt serialisierbares Preview-Ergebnis. Es wird eine reine
Core-Funktion ergänzt, die keine Dateien schreibt:

```cpp
struct AutoBGEPreviewResult {
    bool success;
    Matrix2Df corrected_r, corrected_g, corrected_b;
    Matrix2Df background_r, background_g, background_b;
    std::vector<SamplePoint> stage1_points;
    std::vector<SamplePoint> stage2_points;
    BGEDiagnostics diagnostics;
};

AutoBGEPreviewResult run_autobge_preview(
    const Matrix2Df& R,
    const Matrix2Df& G,
    const Matrix2Df& B,
    const BGEConfig& config);
```

Finaler Runner und Preview verwenden dieselben Modellierungs- und
Finalisierungsfunktionen. Die Preview-Funktion ist nur eine orchestrationale
Hülle; es entsteht keine zweite AutoBGE-Implementierung.

Sample-Punkte werden getrennt für Stufe 1 und Stufe 2 zurückgegeben. Bei RGB
werden Punkte entweder pro Kanal geliefert oder als kanalweise Gruppen in der
Diagnostik gekennzeichnet.

## 7. Proxy-Strategie

Das lineare Eingangsbild wird auf maximal 1600 Pixel an der langen Kante
verkleinert. Die Canvas-Maske wird mit Nearest Neighbor auf exakt dieselbe
Geometrie gebracht.

AutoBGE besitzt zusätzlich `downsample_scale`. Deshalb gelten zwei Ebenen:

- äußerer GUI-Proxy: begrenzt Netzwerk, Speicher und Preview-Laufzeit,
- interner AutoBGE-Downsample: Bestandteil des Algorithmus.

Pixelbezogene Parameter werden für den äußeren Proxy skaliert:

- `patch_size`: proportional skalieren, danach auf eine ungerade Zahl >= 3
  begrenzen,
- `border_margin`: proportional skalieren und auf >= 0 begrenzen,
- Polynomgrad, RBF-Smoothing, Ausschlussfraktion und Seed unverändert lassen,
- `num_sample_points`: unverändert verwenden; bei `0` auf Basis der
  Proxygeometrie automatisch bestimmen und den tatsächlich verwendeten Wert in
  der Diagnostik anzeigen.

Die GUI kennzeichnet das Ergebnis ausdrücklich als Preview. Wegen der
reduzierten Geometrie können Sample-Punkte und Modell im Vollbild leicht
abweichen. Deterministischer Seed und identische Algorithmen halten die
Abweichung reproduzierbar.

## 8. API-Vertrag

### 8.1 Preview berechnen

`POST /api/runs/<run_id>/bge-preview`

Request:

```json
{
  "run_dir": "/optional/alternate/run/path",
  "params": {
    "method": "autobge",
    "num_sample_points": 0,
    "poly_degree": 2,
    "rbf_smooth": 2.0,
    "downsample_scale": 4,
    "patch_size": 35,
    "patch_estimator": "sigma_clipped_median",
    "stretch_mode": "linear",
    "stretch_target_median": 0.25,
    "border_margin": 10,
    "bright_exclusion_fraction": 0.2,
    "gradient_descent_max_iters": 100,
    "random_seed": 42,
    "normalize_between_stages": true,
    "apply_guards": true,
    "mono_mode": "rgb_duplicate"
  },
  "exclusion_polygons": [
    [[0.12, 0.18], [0.28, 0.17], [0.31, 0.35], [0.14, 0.39]]
  ]
}
```

Polygonkoordinaten sind auf `[0,1]` normalisierte Bildkoordinaten mit Ursprung
oben links. Dadurch bleiben sie bei Zoom, Pan, Proxygröße und finaler
Vollauflösung eindeutig.

Response:

```json
{
  "preview_id": "opaque-short-lived-token",
  "source": "stacked_rgb_solve.fits",
  "width": 1600,
  "height": 1173,
  "diagnostics": {
    "success": true,
    "failure_reason": "",
    "guard_rejected": false,
    "channels": []
  },
  "sample_points": {
    "stage1": [{"x": 0.2, "y": 0.3, "channel": "R"}],
    "stage2": []
  },
  "images": {
    "original": "/api/runs/<run_id>/bge-preview/<id>/original.png",
    "corrected": "/api/runs/<run_id>/bge-preview/<id>/corrected.png",
    "background": "/api/runs/<run_id>/bge-preview/<id>/background.png"
  }
}
```

PNG und umfangreiche Sample-Punkt-Daten werden nicht in Header oder Base64
transportiert. Der POST liefert JSON; die drei PNGs werden separat geladen.

### 8.2 Previewbilder abrufen

```text
GET /api/runs/<run_id>/bge-preview/<preview_id>/original.png
GET /api/runs/<run_id>/bge-preview/<preview_id>/corrected.png
GET /api/runs/<run_id>/bge-preview/<preview_id>/background.png
```

- Token ist zufällig, undurchsichtig und an Run-ID und aufgelösten Run-Pfad
  gebunden.
- `Cache-Control: no-store`.
- Ein Token läuft nach kurzer Inaktivität ab, beispielsweise nach 10 Minuten.
- Bilder werden als 8-Bit-Display-PNG geliefert; Berechnung und Apply bleiben
  floatbasiert.

### 8.3 Kein eigener Apply-Endpoint

Wie bei HMS gibt es keinen separaten BGE-Apply-/Resume-Endpunkt. Das Frontend
merged die bekannten Felder unter `bge` beziehungsweise `bge.autobge` in die
aktuell geladene YAML und ruft auf:

```json
{
  "from_phase": "BGE",
  "run_dir": "/resolved/run/dir",
  "config_yaml": "<vollständige aktualisierte YAML>",
  "filter_context": "<bestehender Kontext>"
}
```

Ziel ist `POST /api/runs/<run_id>/resume`. Snapshot, Revision, Schreiben der
Config, Event-Erzeugung und Jobstart bleiben damit zentralisiert.

Manuelle Ausschlusspolygone sind zunächst Preview-Hilfen und nicht Teil des
bestehenden YAML-Schemas. Für eine exakte Übernahme in den finalen Resume muss
vor Implementierung eine der folgenden Varianten verbindlich gewählt werden:

1. **Empfohlen:** normalisierte Polygone unter
   `bge.autobge.exclusion_polygons` in Schema und Config aufnehmen und im Runner
   in `sampling_valid_mask` rasterisieren.
2. Polygone nur für die Preview verwenden und im Dialog deutlich anzeigen,
   dass sie den finalen Resume nicht beeinflussen.

Da visuelle Ausschlüsse ein Kernelement von AutoBGE.py sind, ist Variante 1
Teil dieses Plans.

## 9. Parametervertrag und Validierung

| Feld | UI | zulässiger Bereich |
|------|----|--------------------|
| `bge.method` | Select | `autobge` im Preview-MVP |
| `num_sample_points` | Zahl | 0–3000; 0 = automatisch |
| `poly_degree` | Zahl/Slider | 1–6 |
| `rbf_smooth` | Zahl/Slider | 0–10 |
| `downsample_scale` | Zahl/Slider | 1–8 |
| `patch_size` | Zahl/Slider | ungerade, 3–101 |
| `patch_estimator` | Select | `median`, `sigma_clipped_median` |
| `stretch_mode` | Select | `none`, `linear`, `mtf` |
| `stretch_target_median` | Zahl/Slider | >0 und <1; UI 0.01–0.99 |
| `border_margin` | Zahl/Slider | 0–250 |
| `bright_exclusion_fraction` | Zahl/Slider | >0 und <1; UI 0.01–0.99 |
| `gradient_descent_max_iters` | Zahl | 1–500 |
| `random_seed` | Zahl | 32-Bit-Integer |
| `normalize_between_stages` | Checkbox | bool |
| `apply_guards` | Checkbox | bool |
| `mono_mode` | Select | `rgb_duplicate`, `disabled` |

Die Wertebereiche werden vor Implementierung mit dem Schema konsolidiert. Das
aktuelle Schema besitzt für einige Felder nur eine Untergrenze, während die
bestehenden UI-Hilfetexte engere Bereiche nennen. Schema, Backend-Validierung,
HTML-Controls, Tooltips und Dokumentation müssen denselben Vertrag verwenden.

Regeln:

- Frontend startet bei ungültigem Wert keine Preview und erlaubt kein Apply.
- Werte werden beim Verlassen des Feldes auf den zulässigen Bereich begrenzt.
- `patch_size` wird zusätzlich auf die nächste gültige ungerade Zahl gesetzt.
- Backend validiert unabhängig und nennt alle ungültigen Felder mit Wert und
  erlaubtem Bereich.
- `stretch_target_median` ist nur bei `stretch_mode == mtf` editierbar.
- Ausschlusspolygone benötigen mindestens drei unterschiedliche Punkte, dürfen
  nicht selbstschneidend sein und werden auf `[0,1]` begrenzt.

## 10. Modal und Darstellung

Desktop-Aufteilung:

```text
┌──────────────────────────────────────────────────────────────────┐
│ AutoBGE konfigurieren                                      [×]   │
├───────────────────────────────────┬──────────────────────────────┤
│ [Original|Korrigiert|Hintergrund] │ Parameter                    │
│ ┌───────────────────────────────┐ │ Sample-Punkte                │
│ │ Bild + Punkte + Polygone      │ │ Polynomgrad                  │
│ │ Zoom / Pan / Polygonmodus     │ │ RBF-Smoothing                │
│ └───────────────────────────────┘ │ ...                          │
│ RGB-Histogramm / Diagnostik       │ Guards / Stretch             │
├───────────────────────────────────┴──────────────────────────────┤
│ [Zurücksetzen] [Polygone löschen] [Abbrechen] [Übernehmen ...]  │
└──────────────────────────────────────────────────────────────────┘
```

Ansichten:

- **Original:** lineares Bild mit derselben Display-Transformation wie die
  anderen Ansichten,
- **Korrigiert:** AutoBGE-Ergebnis,
- **Hintergrund:** berechnetes additives Hintergrundmodell,
- optionaler Vorher/Nachher-Wischer in einer späteren Iteration.

Overlays:

- Stufe-1-Punkte: eine Farbe,
- Stufe-2-Punkte: zweite Farbe,
- Kanäle optional durch Form oder Farbe unterscheidbar,
- Ausschlusspolygone halbtransparent rot,
- Canvas-Grenze beziehungsweise ungültige Maskenbereiche schraffiert.

Polygonbedienung:

- `Ausschluss zeichnen` aktiviert Polygonmodus,
- Klick setzt Punkte,
- Doppelklick oder Klick auf den Startpunkt schließt das Polygon,
- Escape verwirft das aktuell gezeichnete Polygon,
- ausgewähltes Polygon kann entfernt werden,
- `Alle Ausschlüsse löschen` entfernt alle Polygone,
- Zoom und Pan bleiben außerhalb des Polygonmodus wie im HMS-Dialog verfügbar.

Alle Labels, Statusmeldungen, Fehlermeldungen, Optionen und Tooltips werden in
`de.json` und `en.json` geführt. Vor Abschluss wird automatisch geprüft, dass
beide Dateien denselben Key-Satz besitzen.

## 11. Diagnostik

Das Modal zeigt kompakt:

- tatsächlich verwendetes Eingangsartefakt,
- Proxygröße und äußeren Skalierungsfaktor,
- tatsächlich verwendete Sample-Anzahl je Stufe und Kanal,
- Fit-RMS je Kanal,
- Median/Standardabweichung von Eingabe, Ausgabe und Modell,
- Flatness und Slope vor/nach Korrektur,
- Guard-Status und Ablehnungsgrund,
- Laufzeiten für Sampling, Polynomfit, RBF-Fit und Finalisierung,
- Anzahl ausgeschlossener Pixel durch Canvas, Strukturmaske und Benutzerpolygone.

Wenn Guards die Anwendung ablehnen, bleibt die berechnete Preview sichtbar,
wird aber deutlich als `von Guards abgelehnt` markiert. Apply ist standardmäßig
deaktiviert, solange `apply_guards` aktiv ist und die Preview abgelehnt wurde.
Der Benutzer kann Guards nicht unbemerkt umgehen; ein bewusstes Deaktivieren
erzeugt eine Warnung.

Histogramme werden für Original und Korrektur mit identischer Display-Skalierung
berechnet, damit sie vergleichbar bleiben. Clipping aus der 8-Bit-Anzeige wird
nicht mit floatbasierter Core-Diagnostik verwechselt.

## 12. Cache und Nebenläufigkeit

Zwei getrennte Caches werden verwendet:

### Input-Proxy-Cache

Key:

- kanonischer Run-Pfad,
- Quell-FITS mit Größe und Änderungszeit,
- Canvas-Maske mit Größe und Änderungszeit,
- maximale Proxykante.

Der Eintrag enthält immutable Proxy-RGB- und Maskendaten.

### Preview-Ergebnis-Cache

Key:

- Input-Proxy-Signatur,
- kanonisch serialisierte AutoBGE-Parameter,
- normalisierte Polygone,
- Core-/Algorithmus-Version.

Der Eintrag enthält nur codierte PNGs, Diagnostik und normalisierte
Sample-Punkte. Speicherbudget und LRU-Verdrängung sind verbindlich; Tokens
laufen zeitbasiert ab.

Frontend:

- 300–500 ms Debounce, da AutoBGE teurer als HMS ist,
- vorherigen Fetch mit `AbortController` abbrechen,
- lokale Generation-ID verhindert Out-of-order-Updates,
- Parameteränderungen während einer laufenden Berechnung erzeugen höchstens
  einen nachfolgenden Request,
- Umschalten der Ansicht löst keine Neuberechnung aus.

## 13. YAML-Merge und Persistenz

Beim Apply werden ausschließlich folgende Felder geändert:

- `bge.method: autobge`,
- bekannte Werte unter `bge.autobge`,
- `bge.autobge.exclusion_polygons` nach Schemaerweiterung.

Andere `bge`-Felder, Classic-BGE-Werte, unbekannte YAML-Felder und alle anderen
Config-Abschnitte bleiben unverändert. Der YAML-Merge wird nicht durch lose
Regex-Ersetzung erweitert, sondern als gemeinsam getestete strukturierte
Config-Patch-Funktion umgesetzt. Damit funktionieren verschachtelte Listen der
Polygone zuverlässig.

## 14. Fehlerfälle

- Run nicht gefunden oder Run-Pfad nicht zulässig,
- `stacked_rgb_solve.fits` und `stacked_rgb.fits` fehlen,
- FITS beschädigt oder RGB-Geometrie inkonsistent,
- `canvas_mask.fits` fehlt oder passt geometrisch nicht,
- Mono-Eingabe bei `mono_mode: disabled`,
- zu wenige gültige Sample-Punkte,
- Polynomgrad benötigt mehr Punkte als verfügbar,
- RBF-System singulär oder numerisch instabil,
- Ausschlusspolygone entfernen zu viel gültige Samplingfläche,
- Guard lehnt Modell ab,
- ungültiger oder abgelaufener Preview-Token,
- veraltete Antwort trifft nach neuerem Request ein,
- YAML-Merge oder Resume-Start schlägt fehl,
- ein anderer Run/Resume-Job ist bereits aktiv.

Jeder Fehler erhält einen stabilen Fehlercode, eine lokalisierte Meldung und
strukturierte Details. Die letzte gültige Preview bleibt bei einem späteren
Fehler sichtbar.

## 15. Tests und Abnahmekriterien

### Core

- AutoBGE funktioniert ohne Classic-Tile-Metriken und Tile-Grid.
- `common_valid_mask` begrenzt Ausgabe und Sampling.
- `sampling_valid_mask` beeinflusst nur Sample-Auswahl, niemals Ausgabepixel.
- Polygonrasterisierung ist für Vollbild und Proxy geometrisch konsistent.
- deterministischer Seed erzeugt identische Punkte und Ergebnisse.
- Stufe-1- und Stufe-2-Punkte werden korrekt zurückgegeben.
- Mono-Modi, Guard-Annahme und Guard-Ablehnung sind abgedeckt.
- alle Parametergrenzen und ungerade Patch-Größe werden getestet.

### Backend

- sichere Run-Auflösung verhindert beliebige Dateizugriffe,
- korrekte Priorität `stacked_rgb_solve.fits` vor `stacked_rgb.fits`,
- BGE-korrigierte Dateien werden nie als Eingang verwendet,
- Proxy und Maske besitzen identische Geometrie,
- Cache-Hit lädt FITS nicht erneut,
- Artefaktänderung invalidiert den Cache,
- parallele identische Requests sind threadsicher,
- Token ist an Run und Ergebnis gebunden und läuft ab,
- Original/Korrektur/Hintergrund liefern gültige PNGs,
- Diagnostik und Sample-Punkte entsprechen dem direkten Core-Aufruf,
- ungültige Parameter nennen Feld, Wert und Bereich.

### Frontend

- Button erscheint nur bei ausgewählter Phase `BGE` und links neben dem Badge,
- Startwerte stammen aus der geladenen Resume-YAML,
- alle Controls sind vollständig DE/EN und besitzen Tooltips,
- abhängige Controls werden korrekt aktiviert/deaktiviert,
- ungültige Werte starten weder Preview noch Apply,
- Zoom, Pan, Ansichtsumschaltung und Histogramm funktionieren,
- Polygonerstellung, Abbruch, Auswahl und Löschen funktionieren,
- Polygone bleiben bei Zoom/Pan und Ansichtswechsel deckungsgleich,
- veraltete Responses überschreiben keine neuere Preview,
- Reset stellt Parameter und Polygone des Öffnungszustands wieder her,
- Apply verhindert Doppelstart.

### Integration

- Preview verändert keine FITS-Datei und keine `config.yaml`,
- Apply verändert nur die bekannten AutoBGE-Felder und Polygone,
- Resume startet exakt ab `BGE`,
- finaler Run verwendet die übernommenen Ausschlusspolygone,
- Preview und Vollbild-Run zeigen qualitativ dasselbe Gradientenmodell,
- `stacked_rgb_bge_linear.fits`, `stacked_rgb_bge.fits` und `bge.json` werden
  entsprechend dem bestehenden Runner-Vertrag erzeugt,
- der neue Job erscheint unmittelbar im Run Monitor.

## 16. Implementierungsreihenfolge

1. Schema-/Default-Vertrag für AutoBGE-Werte konsolidieren.
2. AutoBGE im Runner von Classic-Tile-Artefakten entkoppeln und testen.
3. Separate `sampling_valid_mask` im Core einführen.
4. Polygonschema, Rasterisierung und Runner-Übernahme implementieren.
5. `run_autobge_preview()` mit Background-, Corrected- und Sample-Ergebnis
   ergänzen.
6. Gemeinsame BGE-Eingangsartefaktauflösung extrahieren.
7. Proxy-Erzeugung und parameterabhängige Skalierung implementieren.
8. Thread-sicheren Input- und Ergebnis-Cache implementieren.
9. Preview-POST und tokenisierte PNG-Routen ergänzen.
10. Button links neben dem BGE-Badge und Modal-Grundgerüst ergänzen.
11. Parametercontrols, Validierung, Tooltips und i18n umsetzen.
12. Canvas-Ansichten, Histogramm und Sample-Punkt-Overlay umsetzen.
13. Polygoneditor und normalisierte Koordinaten umsetzen.
14. Strukturierten YAML-Merge und bestehenden Resume-Aufruf integrieren.
15. Core-, Backend-, Frontend- und Integrationstests ausführen.
16. Deutsche und englische Benutzeranleitung ergänzen.

## 17. Aufwandsschätzung

Für eine robuste AutoBGE-Preview einschließlich Core-Erweiterung,
Ausschlusspolygonen, Cache, Diagnostik, i18n und automatisierten Tests sind etwa
15–22 Entwicklungstage realistisch.

Aufteilung:

- Core-/Runner-Korrekturen und Maskenvertrag: 3–5 Tage,
- Preview-Service, Cache und API: 4–6 Tage,
- Modal, Bildansichten und Parameter: 3–4 Tage,
- Polygoneditor und Overlays: 2–3 Tage,
- YAML-Merge, Resume-Integration und i18n: 1–2 Tage,
- Tests, Dokumentation und Stabilisierung: 2–3 Tage.

Eine Version ohne Ausschlusspolygone wäre schneller, würde aber ein zentrales
Bedienelement der AutoBGE-Referenz auslassen und ist daher nicht Ziel dieses
Plans.
