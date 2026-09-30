# Effektgrößen der nachgelagerten Parameter (Denoise, BGE, PCC, Stretch) an IC4605

Stand: 2026-09-26. Rohdaten: `pi_jev_effektgroessen_nachgelagert_20260926.json`. Methode: Kopie des Kontrolllaufs
`jev_m5_ic4605_off_20260925`; je Variante ein Wiederaufsetzen ab `BGE` mit genau einer Änderung (`resume-reconstruction`, erlaubte
Abschnitte: `bge`, `pcc`, `chroma_denoise`, `luma_denoise`, `hypermetric_stretch`). Der Basislauf reproduziert das Original
bit-identisch (SHA-256). Kennzahlen am Endbild `stacked_rgb_hms.fits`: Pixelrauschen (Hochpass, robust), Himmelsspanne
(p5-p95 der 256-px-Blockmediane), Farbspanne R-G/B-G, Mittelskalen-Struktur (Bandpass 6-24 px relativ zum Rauschen), Sterne an
denselben Positionen (Breite, Signal in der Apertur, Elongation; Werkzeug `matched_pair_metrics.py`). Ein Datensatz, ein Lauf je Variante.

## Ergebnis (Änderung gegenüber der Basis)

| Variante | Rauschen | Himmelsspanne | Farbe R-G / B-G | Mittelskala | Sternbreite | Sternsignal |
|---|---|---|---|---|---|---|
| Luma-Denoise aus | +261 % | +24 % | +112 % / +71 % | -65 % | +4,5 % | +41 % |
| Luma Blend 0,5 (Basis 0,85) | +152 % | +23 % | +123 % / +72 % | -49 % | +4,4 % | +42 % |
| Luma Blend 1,0 | -1 % | +1 % | -2 % / -1 % | +1 % | 0 % | 0 % |
| Luma Wavelet-Schwelle 1,0 (Basis 1,5) | +122 % | +22 % | +107 % / +68 % | -43 % | +4,4 % | +42 % |
| Luma Wavelet-Schwelle 2,5 | +2 % | +9 % | +1 % / +3 % | +2 % | 0 % | +4 % |
| Luma Bilateral an | -4 % | 0 % | -3 % / -2 % | +5 % | 0 % | +1 % |
| Luma ohne Sternschutz | -2 % | +1 % | -1 % / -1 % | +3 % | 0 % | -3 % |
| Chroma-Denoise an | -70 % | +93 % | +10 % / -26 % | +292 % | +3,6 % | +16 % |
| Chroma an, Blend 1,0 | -71 % | +92 % | +8 % / -25 % | +294 % | +3,6 % | +16 % |
| Chroma an + `large_scale_bias` | -70 % | +93 % | +10 % / -26 % | +292 % | +3,6 % | +16 % |
| BGE `none` | 0 | 0 | 0 | 0 | 0 | 0 |
| BGE `classic` | -14 % | **-68 %** | -46 % / -49 % | +19 % | +2,5 % | +5 % |
| BGE `autobge` | 0 | 0 | 0 | 0 | 0 | 0 |
| Stretch: Farbstichkorrektur aus | -25 % | +23 % | +100 % / +63 % | +43 % | +4,2 % | +13 % |
| Stretch: Farbstich-Stärke max 0,3 | -15 % | +12 % | +47 % / +38 % | +21 % | +2,4 % | +7 % |
| Stretch `fixed_log_d` 2,5 | -28 % | -26 % | -28 % / -28 % | +31 % | -0,3 % | -20 % |
| Stretch `fixed_log_d` 5,0 | +37 % | -9 % | +57 % / +91 % | -48 % | +5,3 % | -25 % |
| Stretch `adaptive_anchor` aus | +38 % | +29 % | +10 % / +29 % | -35 % | +2,3 % | -15 % |
| Stretch `convergence_power` 2 | +1 % | +1 % | +1 % / +1 % | 0 % | 0 % | +2 % |
| Stretch `convergence_power` 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stretch `protect_b` 2 | +8 % | +4 % | +9 % / +9 % | -12 % | +1 % | +1 % |
| Stretch `color_grip` 0,5 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stretch `linear_expansion` 0,3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stretch `shadow_convergence` 0,5 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stretch Sensorprofil `rec709` | -3 % | -3 % | +6 % / 0 % | +5 % | 0 % | +4 % |
| PCC aus | -3 % | -3 % | +6 % / 0 % | +5 % | 0 % | +4 % |
| PCC `chroma_strength` 0,3 (Basis 0,7) | +39 % | +35 % | +14 % / +35 % | -27 % | +1 % | +10 % |

## Einteilung nach Wirkung (Kennzahl über 10 % Änderung = wirksam)

- **Stark wirksam:** `luma_denoise.enabled`, `blend_amount` unter 0,85 und `wavelet.threshold_scale` unter 1,5 (Rauschen um Faktor 2 bis 3,6);
  `chroma_denoise.enabled`; `bge.method = classic`; Stretch `fixed_log_d`, `adaptive_anchor`, Farbstichkorrektur (`enabled`, `max_amount`);
  `pcc.chroma_strength`.
- **Schwach (2 bis 10 %):** `luma_denoise.wavelet.threshold_scale` 2,5, `bilateral`, `star_protection`, `protect_b`, Sensorprofil (nur über den PCC-Pfad).
- **Ohne Wirkung in dieser Konfiguration (unter 0,5 %):** `convergence_power` 6, `color_grip`, `linear_expansion`, `shadow_convergence`,
  `chroma_denoise.blend.amount` und `large_scale_bias` (bei aktivem Chroma-Denoise), `luma_denoise.blend_amount` 1,0.

## Auffälligkeiten

1. **BGE `classic` entfernt zwei Drittel der großräumigen Himmelsstruktur** (Himmelsspanne -68 %). Wo BGE greift, ist es der stärkste
   Abflacher. In der Basis greift es nicht: `bge.method: auto` und `autobge` verwarfen das Ergebnis am Guard (`background_chroma_worsened`),
   deshalb sind beide identisch mit `none`.
2. **Luma-Denoise verändert die Sternphotometrie.** Ohne Luma-Denoise ist das Sternsignal in der 4-px-Apertur 41 % höher und die Sterne
   4,5 % breiter: die Stufe entfernt Flügel der Sterne, trotz Sternschutz. Rauschminderung um den Faktor 3,6 hat also diesen Preis.
3. **`pcc_off` und Sensorprofil `rec709` liefern identische Zahlen:** das Profil wirkt nur, wenn PCC angewendet wurde; ohne PCC verwendet der
   Stretch offenbar den `rec709`-Pfad.
4. **Mehrere Stretch-Parameter sind hier tot** (`color_grip`, `linear_expansion`, `shadow_convergence`, `convergence_power` ab 6): sie greifen
   in `mode: ready_to_use` mit festem `log_d` nicht oder nur in anderen Modi. Kandidaten dafür wären wirkungslos.
5. **Chroma-Denoise wirkt auch auf die Luminanz-Struktur** (Himmelsspanne +93 %, Mittelskala +292 %), nicht nur auf die Farbe. Das ist
   ungewöhnlich für eine reine Chroma-Stufe und sollte vor einer Freigabe als Kandidat geprüft werden.

## Grenzen

Ein Datensatz (IC4605, DWARF II), Kennzahlen statt Bildurteil; hohe Werte sind kein Qualitätsgewinn. Wirksamkeit ist eine Voraussetzung dafür,
dass ein Parameter Jev-Kandidat wird, kein Nachweis, dass ein bestimmter Wert besser ist.
