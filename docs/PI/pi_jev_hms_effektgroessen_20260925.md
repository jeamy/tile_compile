# Stretch-Einstellungen: Effektgrößen an IC4605 und die Frage "warum ist der Hintergrund so gleichförmig?"

Stand: 2026-09-25. Messung an einer Kopie des Kontrolllaufs `jev_m5_ic4605_off_20260925` (nur der Schritt `HYPERMETRIC_STRETCH`
wurde wiederholt, je etwa 5 s; das Original im Archiv ist unberührt). Der Basislauf reproduziert `stacked_rgb_hms.fits` bit-identisch.

## Anlass

Das neue Endbild wirkt im Vergleich zu einem älteren Lauf (`ic4605_20260722_150501`, AQMH-Pipeline) im Hintergrund eintönig.

## Befunde

1. **Die Nebelstruktur ist im neuen linearen Stack vorhanden.** Beide linearen Stacks wurden über ihre WCS-Lösungen übereinander gelegt
   (Sterne entfernt, 48 px geglättet): Korrelation der großräumigen Hintergrundmuster 0,81 nach Abzug einer Ebene und 0,94 nach Abzug
   einer quadratischen Fläche. Halo um Antares, Leuchten am unteren Rand und die dunklen Zonen sind in beiden dieselben.
2. **Die Abflachung entsteht im Stretch.** Bei gleichem Median (Ziel 0,12) ist die großräumige Himmelsstruktur im alten Endbild etwa
   3,5-mal kontrastreicher (Spanne der 128-px-Blockmediane p5-p95: 0,0185 gegen 0,0052), im alten Bild aber auch das Pixelrauschen
   3-mal höher (0,019 gegen 0,0064; relativ zum Pegel 16 % gegen 5 %). Der alte Stretch hat Struktur und Rauschen stark angehoben, der
   neue tut das sehr zurückhaltend.
3. **Die Farbstichkorrektur** (`hypermetric_stretch.color_cast_correction`, im neuen Lauf an, im alten nicht vorhanden) wendet bei
   IC4605 mit Stärke 0,72 an (Verhältnis 1,88 auf 0,90) und halbiert die großräumige Farbstruktur.

## Effektgrößen (Stretch-Wiederholung, Median 0,12 Ziel, gleiche Daten)

| Variante | Pixelrauschen | Himmelsspanne | Farbspanne R-G / B-G | Anmerkung |
|---|---|---|---|---|
| Basis | 0,0064 | 0,0058 | 0,0037 / 0,0087 | |
| ohne Farbstichkorrektur | 0,0047 | 0,0072 | 0,0074 / 0,0141 | Farbstruktur x2, Rauschen -27 % |
| `neutralize_sky` aus | 0,0064 | 0,0058 | 0,0037 / 0,0087 | kein Effekt bei diesem Lauf |
| Sensorprofil `rec709` statt IMX415 | 0,0062 | 0,0057 | 0,0039 / 0,0087 | praktisch kein Effekt auf den Himmel |
| `adaptive_anchor` aus | 0,0088 | 0,0075 | 0,0041 / 0,0112 | Rauschen +37 % |
| `fixed_log_d` 2,5 | 0,0046 | 0,0043 | 0,0027 / 0,0063 | flacher |
| `fixed_log_d` 5,0 | 0,0087 | 0,0053 | 0,0059 / 0,0166 | mehr Farbe, mehr Rauschen |
| ohne Cast + `fixed_log_d` 4,5 | 0,0057 | 0,0068 | 0,0078 / 0,0155 | |
| `target_bg` 0,08 | 0,0044 | 0,0041 | 0,0026 / 0,0061 | verschiebt nur den Pegel |

## Schlüsse für die Kandidatenauswahl

- Das **Sensorprofil** hat auf den Himmel praktisch keinen Effekt (Rauschen und Spanne unter 4 % Unterschied): als Jev-Kandidat kaum relevant.
- **Farbstichkorrektur, `fixed_log_d` und `adaptive_anchor`** verändern Farbstruktur und Rauschen um 25 bis 100 %: das sind wirksame,
  datenabhängige Stretch-Parameter (kompakte Objekte gegen großflächige Nebel), also die aussichtsreichere Kandidatengruppe für die
  Objektklasse.
- Ein maßstabsselektiver Kontrast (nur die großräumige Komponente anheben) würde Struktur zeigen, ohne das Pixelrauschen zu verstärken;
  die bisherigen fehlgeschlagenen lokalen Kontrastversuche wirkten auf feinen Skalen.

## Grenzen

Ein Datensatz, ein Lauf; die Maße (Blockmedian-Spanne, Kanalunterschiede) sind einfache Kennzahlen, kein Bildqualitätsurteil.
Die Darstellung im Bericht (feste Fenster) hängt von der Bildschirmumrechnung ab.
