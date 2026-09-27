# HMS-Schwarzpunkt: Variante 2 (Anchor aus Vor-Denoise-Referenz) — implementiert, getestet, ohne Nutzen

Stand: 2026-09-27. Nutzerauftrag: von den drei am 27.09. skizzierten Richtungen Variante 2 umsetzen
("Anchor aus der Verteilung vor der Rauschminderung berechnen") und mit M31- und M42-Kopien testen.
**Ergebnis: umgesetzt, auf echten Daten getestet, kein messbarer Nutzen.** Die eigentliche Ursache
der am 26.09. beobachteten Schwarzpunkt-Empfindlichkeit ist damit weiterhin ungeklärt.

## Was gebaut wurde

- `hypermetric_stretch.anchor_from_reference` (neuer Config-Schalter, Standard aus, wie beim
  Voll-Frame-Schätzer/`large_scale_contrast`).
- `image::run_hypermetric_stretch_rgb` nimmt jetzt optional eine Referenz-RGB entgegen; ist die
  Referenz gesetzt und der Schalter an, wird der Anchor (`calculate_anchor_adaptive`/
  `calculate_anchor_statistical`) aus der Referenz statt aus dem tatsächlich gestreckten Bild
  berechnet. Ohne Referenz oder bei falscher Dimension: unverändertes Verhalten, nie ein Fehler.
- Der Runner (`apps/runner_downstream.cpp`) sichert die lineare RGB direkt nach PCCs eigener
  Farbkorrektur — vor dem Speckle-Unterdrücker und vor dem `post_pcc`-Chroma-Denoise-Durchlauf — und
  reicht sie als Referenz an HMS weiter. Nur wenn `chroma_denoise.enabled` und
  `apply_stage: post_pcc` und der neue Schalter an sind, wird die Kopie überhaupt angefertigt.
- Für einen reinen HYPERMETRIC_STRETCH-Resume wird die Referenz zusätzlich als
  `outputs/pcc_predenoise_{R,G,B}.fit` abgelegt und beim Resume wieder eingelesen; fehlt die Datei
  (älterer Lauf, Funktion war aus), ist das ein normaler Rückfall auf das alte Verhalten, kein Fehler.
- Synthetischer Test (`hypermetric_anchor_from_reference_recovers_the_pre_denoise_anchor`,
  `tests/test_hypermetric_stretch.cpp`): ein Himmel mit demselben Pegel, aber 10x schmalerem Rauschen
  (simuliert eine Rauschminderung, die die Verteilung verengt) verschiebt den Anchor nachweisbar
  (> 0,002); mit der Vor-Rauschminderung-Referenz wird der ursprüngliche Anchor auf 0,0005 genau
  wiederhergestellt. Das beweist den MECHANISMUS im Konstrukt, nicht dass er der reale Fehler ist.

## Test auf echten Kopien

Methode wie beim `large_scale_contrast`-Test: Kopie ohne `cache/`, `chroma_denoise.apply_stage` auf
`post_pcc` gesetzt (Standard in diesem Repo), `resume-reconstruction --from-phase PCC`, drei Varianten
(`base`: Chroma-Denoise aus; `chroma_on`: an; `chroma_on_anchor_fixed`: an + `anchor_from_reference`
an), gemessen am `phase_end`-Ereignis von HYPERMETRIC_STRETCH.

**M31** (Kopie von `jev-test/20260923_212703_e8a50695`, 645 Lights):

| Variante | Anchor | Schwarzclip | Sternbreite/-signal/Elongation |
|---|---|---|---|
| `base` | 0,0001377 | 0,0 % | — |
| `chroma_on` | 0,0001374 | 0,0 % | ×1,011 / ×1,109 / ×0,998 |
| `chroma_on_anchor_fixed` | 0,0001371 | 0,0 % | ×1,011 / ×1,110 / ×0,998 |

**M42** (Kopie von `20260927_161504_810fb3ed`, 610 Lights):

| Variante | Anchor | Schwarzclip | Sternbreite/-signal/Elongation |
|---|---|---|---|
| `base` | 0,0002951 | 1,79e-5 % | — |
| `chroma_on` | 0,0002932 | 5,37e-5 % | ×1,018 / ×1,167 / ×0,999 |
| `chroma_on_anchor_fixed` | 0,0002914 | 5,37e-5 % | ×1,018 / ×1,164 / ×0,999 |

Auf beiden Datensätzen ist `chroma_on_anchor_fixed` gegenüber `chroma_on` **praktisch identisch**
(Schwarzclip exakt gleich, Anchor um denselben Bruchteil verschoben wie ohne Referenz, keine
Annäherung an `base`). Die Referenz-Anchor-Berechnung ändert auf echten Daten nichts.

## Warum: die Prämisse war falsch

Direkter Vergleich der Luminanz VOR (`outputs/pcc_predenoise_*.fit`) und NACH
(`outputs/pcc_*.fit`) dem `post_pcc`-Chroma-Denoise-Durchlauf, auf einem sauberen 400x400-Himmelsfeld
ohne Sterne (M42-Kopie):

| Größe | vor Chroma-Denoise | nach Chroma-Denoise |
|---|---|---|
| Luminanz-Std (Rauschen) | 50,684 | 50,681 (−0,007 %) |
| R−G-Std | 14,363 | 14,258 (−0,7 %) |
| B−G-Std | 20,561 | 20,411 (−0,7 %) |

Die Luminanzverteilung — genau das, was `calculate_anchor_adaptive_luminance` auswertet — ändert sich
durch `chroma_denoise` im `chroma_only`-Modus praktisch **gar nicht**. Das ist bei genauerem Hinsehen
konsequent: Der Denoiser arbeitet in einem Luma/Chroma-Farbraum und rührt laut Konfiguration
(`blend.mode: chroma_only`) die Luminanz per Konstruktion nicht an. Meine Annahme vom 27.09.
("Rauschminderung verengt die Himmelsverteilung, die der Anchor-Suche zugrunde liegt") trifft auf
diese Konfiguration nicht zu — der synthetische Test simuliert einen Fall (Rauschen wird tatsächlich
schmaler), der hier gar nicht eintritt.

## Was das für den ursprünglichen Befund (26.09.) bedeutet

Die frühere, deutlich größere Verschiebung (Schwarzclip 0 % → 1,8 %, Rauschen +24 %, Struktur +58 %,
an einem anderen M42-Lauf mit `target_bg: 0,12` statt `0,2`) hat demnach eine andere Ursache als
"Anchor aus verengter Verteilung". Mögliche Kandidaten, keiner geprüft:

- Eine Wechselwirkung mit `large_scale_contrast`, das in jenem Test zusammen mit `chroma_denoise`
  aktiviert war (hier separat, ohne `large_scale_contrast`, getestet).
- Eine Empfindlichkeit von `target_bg`/`adaptive_output_scaling` selbst, die bei einem tieferen
  Zielhintergrund (0,12) stärker auf kleine Eingabeänderungen reagiert als bei 0,2.
- Ein Effekt in `soft_floor`s Pro-Kanal-Logik (Kanäle rücken durch Chroma-Denoise näher zusammen,
  siehe R−G/B−G-Std oben, −0,7 %), der die Zahl der Pixel verändert, die in allen drei Kanälen
  knapp unter dem Anchor liegen — unabhängig vom Anchor-WERT selbst. Nicht geprüft.

## Entscheidung

`anchor_from_reference` bleibt Code, ist getestet und dokumentiert, aber **nicht empfohlen** und
nirgends aktiviert (weder in `tile_compile.yaml` noch in Beispiel-Configs) — es löst das gemeldete
Problem auf echten Daten nicht. Die eigentliche Ursache der M42-Schwarzpunkt-Empfindlichkeit vom
26.09. ist offen; eine der drei oben genannten Richtungen (oder Variante 1/3 aus der ursprünglichen
Liste) müsste als Nächstes geprüft werden.
