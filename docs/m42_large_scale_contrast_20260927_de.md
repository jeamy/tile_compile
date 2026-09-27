# Maßstabsselektiver Kontrast (`hypermetric_stretch.large_scale_contrast`) an M42: Test 2026-09-27

Stand: 2026-09-27. Anlass: Nutzer bemängelte, schwache Nebelstruktur (M42) hebe sich im gestreckten
Bild kaum vom Himmel ab. Getestet auf einer Kopie des Laufs `20260927_161504_810fb3ed`
(`input_dir /media/tc_ssd/M42_02.2026_lights_all`, 610 Frames, Backend `cuda_v2`, Voll-Frame-Schätzer
mit Clip 4/4 an, `chroma_denoise`/`luma_denoise` aus). Kopie unter `/media/tc_500/lab_m42_lsc_20260927`
(ohne `cache/`), Original unverändert. Methode: `web_backend_cpp/scripts/jev_screening/screen_downstream.py
--from-phase PCC`, vier Varianten, ein Resume je Variante, gemessen am Endbild `stacked_rgb_hms.fits`
gegen die Kontrolle (identischer Lauf ohne Änderung; reproduziert das Original bit-identisch, geprüft
per SHA-256).

## Konfiguration dieses Laufs vor dem Test

Der Lauf hatte **kein** `hypermetric_stretch.large_scale_contrast`-Feld in seiner Config (nicht einmal
`enabled: false`) — die Funktion existierte zum Zeitpunkt des Laufs im Schema, war hier aber nie gesetzt
und lief daher mit dem Schema-Default (aus).

## Varianten und Ergebnis

Getestete Werte: `amount: 2.0`, `chroma_amount: 1.0`, `sigma_px: 48.0` (Default), `remove_vignette: true`
(Default). `amount` und `chroma_amount` wurden **zusammen** in einer Variante getestet, nicht einzeln
isoliert.

| Variante | Pixelrauschen | Himmelsspanne | Farbspanne R-G/B-G | Sternbreite | Sternsignal | Elongation | Schwarzclip |
|---|---|---|---|---|---|---|---|
| Kontrolle (aus) | 0,0149 | 0,0232 | 0,0042 / 0,0091 | — | — | — | 1,79e-5 |
| `large_scale_contrast` an | 0,0148 (×0,99) | 0,0377 (**×1,63**) | 0,0068 / 0,0143 (×1,62/1,57) | ×1,000 | ×1,000 | ×1,000 | 1,79e-5 (unverändert) |
| `chroma_denoise` an (zum Vergleich) | 0,0161 (×1,08) | 0,0291 (×1,25) | 0,0057 / 0,0124 | ×1,018 | ×1,167 | ×0,999 | 5,37e-5 |
| beide an | 0,0160 (×1,07) | 0,0458 (×1,97) | 0,0092 / 0,0210 | ×1,018 | ×1,167 | ×0,999 | 5,37e-5 |

1458 gematchte Sterne (`matched_pair_metrics.py`). Himmelsspanne = p5-p95 der Luminanz über 256-px-Blockmediane.

**`large_scale_contrast` allein:** Himmelsstruktur um 63 % breiter, Sternmaße exakt (nicht nur ungefähr)
unverändert, Rauschen minimal niedriger, kein zusätzliches Schwarzclipping. Das ist ein sauberes Ergebnis
ohne messbaren Nachteil auf diesem Lauf.

**Zum Vergleich, `chroma_denoise`:** wirkt hier deutlich schwächer als am 26.09. an einem anderen
M42-Lauf gemessen (dort Rauschen +24 %, Struktur +58 % durch eine Schwarzpunktverschiebung im Stretch,
siehe `pi_jev_bestaetigung_pixfrac_clip_20260926_de.md`-Umfeld); hier nur +8 % Rauschen, +25 % Struktur,
mit einem kleinen Anstieg des Schwarzclippings. Kombiniert addieren sich beide Effekte; der
Stern-Einfluss stammt dabei ausschließlich von `chroma_denoise`.

## Sichtprobe

Visueller Vorher/Nachher-Crop (eigene Schnellansicht, nicht die Produktionsdarstellung) zeigte die
violette/rötliche Nebelstruktur um die hellen Trapez-Sterne mit `large_scale_contrast` klar sichtbar,
im Ausgangsbild kaum vom flachen Grau unterscheidbar. Eine Differenzkarte bestätigte, dass die Anhebung
räumlich auf die Nebelregion konzentriert ist und im leeren Sternfeld praktisch null bleibt.

## Grenzen

- Ein Datensatz (M42, diese eine Session), ein Parametersatz (`amount`/`chroma_amount` nicht einzeln
  variiert). Keine Bestätigung auf anderen Objekttypen (kompakte Objekte, Sternfelder, Galaxien).
- Keine formale Freigabe nach der Jev-Policy-Methodik (`release_policy_v2/v3`); dafür fehlen mehrere
  Sessions und Objektgruppen.
- „Sauber" heißt hier: kein messbarer Nachteil bei DIESEM Lauf und DIESER Kombination. Ob ein zentriertes,
  radialsymmetrisches Objekt (wo `remove_vignette` nicht mehr eindeutig zwischen Vignette und Objekt
  unterscheidet) betroffen wäre, ist nicht getestet.

## Entscheidung

Auf Nutzeranweisung 2026-09-27: `large_scale_contrast` mit den getesteten Werten
(`enabled: true`, `amount: 2.0`, `chroma_amount: 1.0`) eingeschaltet in
- der Config des Originallaufs `/media/tc_500/20260927_161504_810fb3ed` (Datei angepasst; die
  vorhandenen `outputs/` sind NICHT neu gerechnet — das bräuchte einen separaten Resume),
- `tile_compile_cpp/examples/m42_dwarf2_full_frame.example.yaml` und `.demo.yaml`,
- der ausgelieferten `tile_compile_cpp/tile_compile.yaml` (Standard für **alle** Objekte, nicht nur M42).

Code- und Schema-Default bleiben aus (wie beim Voll-Frame-Schätzer/Dynamic-Boost). Die Übertragung auf
den globalen Standard stützt sich auf einen einzigen Testlauf an einem Nebelobjekt; das ist schwächer
belegt als die 5-Sessions-Bestätigung, die für die Jev-Katalogkandidaten verlangt wird.
