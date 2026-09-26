# Nachgelagerte Kandidaten: Vorschlag für die Bestätigung (Entwurf, 2026-09-26)

Status: **Vorschlag, nicht festgelegt.** Es wurde kein Lauf gestartet. Die Grenzen unten sind eine Wertentscheidung und werden erst mit deiner Zustimmung
in `release_policy_v3.json` festgelegt (der Entwurf `release_policy_v3_draft.json` wird von keinem Code geladen).

## Warum nachgelagerte Parameter anders sind

Bei `pixfrac` und Clipping ist ein niedrigeres Rauschen bei gleicher Sternschärfe ein Gewinn ohne Gegenleistung. Bei Denoise, BGE und Stretch kostet jede
Rauschminderung etwas anderes (Sternsignal, Himmelsstruktur, Farbe). Ein einzelner Endpunkt "weniger Rauschen" würde Änderungen freigeben, die genau
das entfernen, was du sehen willst. Deshalb hat der Entwurf einen Rauschendpunkt **und** Schutzgrenzen für Himmelsstruktur und Sterne.

## Vorgeschlagene Endpunkte (für alle Kandidaten gleich)

- Primär: Hintergrundrauschen Kandidat/Kontrolle, Obergrenze des 95-%-Intervalls <= 0,90.
- Schutz: Himmelsspanne (p5-p95 der 256-px-Blockmediane) im Bereich 0,75 bis 1,33; Sternbreite <= 1,05; Sternsignal >= 0,90; Elongation <= 1,05.
- Stichprobe wie bei Policy v2: 5 Sessions, 2 Objektgruppen mit je mindestens 2.
- Messung: Resume ab `HMS` auf den fertigen Läufen (`screen_downstream.py --from-phase HMS --variants confirmation_downstream_variants.json`), ein Kandidat je
  Kontrast, identischer Build.

## Ergebnis der Vorprüfung (Futilität) an IC4605

Verhältnisse für die Richtung "einschalten"; bei den Aus-Varianten wurde der Kehrwert genommen (Annahme: die Wirkung ist symmetrisch).

| Kandidat | Rauschen | Himmelsspanne | Sternbreite | Sternsignal | Ergebnis |
|---|---|---|---|---|---|
| `enable_adaptive_anchor` | 0,72 | 0,78 | 0,98 | 1,18 | besteht, geht in die Bestätigung |
| `enable_luma_denoise` | 0,28 | 0,81 | 0,96 | **0,71** | verworfen: Sternsignal-Verlust |
| `enable_chroma_denoise` | 0,30 | **1,93** | 1,04 | 1,16 | blockiert: verändert die Himmelsstruktur um das Doppelte |
| `set_bge_classic` | 0,86 | **0,33** | 1,03 | 1,05 | verworfen: entfernt zwei Drittel der Himmelsstruktur |
| `enable_color_cast_correction` | **1,34** | 0,82 | 0,96 | 0,89 | kein Rauschkandidat: Zweck ist Farbneutralität |

## Was das bedeutet

- Nur `enable_adaptive_anchor` bleibt im Verfahren. Seine Himmelsspanne (0,78) liegt nahe an der Untergrenze 0,75: bei der Bestätigung ist das der Punkt, an dem er
  am ehesten scheitert. Die Untergrenze ist eine Annahme, keine Messung.
- **Luma-Denoise** würde bei einer Sternsignal-Untergrenze unter 0,70 bestehen. Ob dieser Preis (Sternflügel werden entfernt) vertretbar ist, entscheidest du.
- **Chroma-Denoise** sollte zuerst verstanden werden: eine Chroma-Stufe darf die Luminanzstruktur nicht verdoppeln.
- **BGE classic** bleibt ausgeschlossen, solange Himmelsstruktur das Ziel ist. Die Einstellung `auto` greift in der Basis ohnehin nicht (Guard `background_chroma_worsened`).
- **Farbstichkorrektur** braucht einen eigenen Endpunkt (Farbspanne R-G/B-G), sonst ist sie mit dem Rauschendpunkt nicht bewertbar.
- Ein Datensatz, Kennzahlen statt Bildurteil. Die Vorprüfung ist ein Ausschlussfilter, keine Bestätigung.

## Was zur Ausführung fehlt

Die Bestätigung braucht fertige Läufe mit aktiver Downstream-Kette auf mindestens 5 Sessions. Die alten gepaarten Läufe (M31, M42, IC5070, M66, IC4605, M104) liegen
unter `/media/data/tile_compile_cache/jev-test/`; ob sich ein Resume ab `HMS` darauf ausführen lässt, ist nicht geprüft (Artefakte nach dem Aufräumen der Caches).
Jeder Resume ist ein Lauf und startet nur auf ausdrücklichen Auftrag.

## Entscheidungen, die ich von dir brauche

1. Sind die Schutzgrenzen (Himmelsspanne 0,75 bis 1,33, Sternsignal >= 0,90) so richtig?
2. Soll Luma-Denoise mit einer niedrigeren Sternsignal-Grenze als eigener Kandidat weiterverfolgt werden?
3. Soll für die Farbstichkorrektur ein Farb-Endpunkt definiert werden?
