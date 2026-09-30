# Effektgrößen der Rekonstruktionsparameter an IC4605 (120 Frames)

Stand: 2026-09-26. Rohdaten: `pi_jev_effektgroessen_rekonstruktion_20260926.json`. Methode: volle `reconstruct`-Läufe auf 120 Frames von IC4605
(`--max-frames 120`), je Arm genau eine Änderung gegenüber der Kontrolle (adaptive Gewichtung aus, damit das Referenzbild gleich bleibt: in allen
Armen Referenzbild 74, gleiche Leinwand); PCC, Astrometrie, BGE und Stretch in allen Armen aus, weil PCC mit nur 120 Frames zu wenige Sterne findet
(7 von 24). Auswertung an gematchten Sternen (grüner Kanal, gewählter Ausgang `drizzle_raw`, Werkzeug `matched_pair_metrics.py`). Ein Lauf je Arm;
Läufe mit identischer Config sind bit-identisch (siehe M5-Bericht), Unterschiede sind also Effekte der Änderung.

## Ergebnis (Kandidat gegenüber Kontrolle)

| Variante (Basis) | Sternbreite | Elongation | Sternsignal | Rauschen | `n_eff` p10 | Wirkung |
|---|---|---|---|---|---|---|
| `drizzle.pixfrac` 1,0 (Basis 0,8) | 1,0003 | 0,9999 | 1,0005 | **0,908** | 37,8 (Basis 25,9) | Rauschen -9 %, Abdeckung +46 %, gleiche Schärfe |
| `drizzle.pixfrac` 0,6 | - | - | - | - | 16,8 | **Coverage-Gate nicht bestanden** (`n_eff` p10 unter 15 % der Frames) |
| Clipping 2,0 / 4,0 (Basis 4,0 / 4,0) | 1,0012 | 1,0017 | 1,067 | **1,485** | 25,9 | Rauschen +49 %, Sternsignal +7 % |
| Clipping 5,0 / 5,0 | 1,0009 | 0,9998 | 1,007 | 0,945 | 25,9 | Rauschen -5,5 % |
| `drizzle.robust_passes` 2 (Basis 4) | 1,0000 | 0,9999 | 1,001 | 0,993 | 25,9 | praktisch wirkungslos |
| `global_metrics.weight_exponent_scale` 1,8 (Basis 1,2) | 1,0002 | 0,9999 | 1,000 | 1,013 | 25,9 | schwach |
| `drizzle.internal_scale` 1 (Basis 2) | 1,0000 | 1,0000 | 1,000 | 1,000 | 50,6 | Ausgabe gleich (`n_eff`-Kennzahl ändert sich, das Bild nicht) |
| `registration.prewarp_interpolation` lanczos4 (Basis cubic) | 1,0000 | 1,0000 | 1,000 | 1,000 | 25,9 | Ausgabe gleich |
| `multiband.levels` 4 (Basis 3) | 1,0000 | 1,0000 | 1,000 | 1,000 | 25,9 | wirkungslos, auch am Multiband-Ausgang (max. 0,2 %) |
| Pyramide Schärfe 0,8 / SNR 0,2 | 1,0000 | 1,0000 | 1,001 | 1,002 | 25,9 | wirkungslos, auch am Multiband-Ausgang |
| Pyramide Schärfe 0,3 / SNR 0,7 | 1,0001 | 1,0001 | 1,000 | 0,998 | 25,9 | wirkungslos, auch am Multiband-Ausgang |
| `multiband.enabled` false (mit `full_frame_estimator` false) | - | - | - | - | - | **Lauf nicht möglich:** `FORWARD_DRIZZLE_V2_REQUIRES_MULTIBAND` |

Der Sternbreite-Wert liegt bei allen Varianten innerhalb 0,12 % der Kontrolle: **keine der Änderungen macht die Sterne schärfer oder breiter.**
Wirksam ist nur, was Rauschen und Abdeckung verändert (`pixfrac`, Clipping-Sigmas).

## Befunde

1. **`pixfrac` 1,0 statt 0,8 verbessert Rauschen (-9 %) und Abdeckung (+46 %) ohne Schärfeverlust** auf diesen Daten. Ein kleineres `pixfrac` ist durch das
   Coverage-Gate begrenzt: bei 0,6 liegt `n_eff` p10 bei 14,0 % der Frames, das Gate verlangt 15 %. Das Verhältnis hängt nicht von der Frame-Zahl ab
   (Kontrolle 21,6 %), die Untergrenze für `pixfrac` ist also eine Eigenschaft der Aufnahme (Dither, Abtastung), nicht des Umfangs. Ein Wertegitter
   für `pixfrac` bräuchte dieses `n_eff`-Verhältnis als Evidenz, das erst nach der Geometrie-Phase bekannt ist.
2. **Das aggressive Clipping aus dem Praxisprofil (2,0 / 4,0) macht das Rauschen um 49 % schlechter**; entspannteres Clipping (5,0 / 5,0) verbessert es
   um 5,5 %. Das Optimum hängt von Frame-Zahl und Artefaktlage ab: das ist eine datenabhängige Parametergruppe.
3. **Auf diesen Daten wirkungslos:** `internal_scale`, `prewarp_interpolation`, `robust_passes` (bis 4 auf 2), Multiband-Levels und Pyramidengewichte. Für Multiband
   gilt zusätzlich: es wird in allen Läufen nicht gewählt und unterscheidet sich von `drizzle_raw` um weniger als 0,1 % in der Sternbreite.
4. **`multiband.enabled` ist keine wählbare Option:** die Konfigurationsprüfung lässt `false` zu (nur zusammen mit `full_frame_estimator: false`), der Lauf verlangt
   Multiband aber zwingend. Ein Kandidat dafür wäre ungültig.
5. **Gate als Ergebnis:** Ein Lauf, der ein geschütztes Gate nicht besteht, ist ein gültiges Ergebnis und kein Betriebsfehler. Jeder Kandidat mit Wertegitter
   muss diese Gate-Grenzen als Bedingung kennen.

## Grenzen

Ein Datensatz (IC4605, DWARF II, 120 von 344 Frames), ein Lauf je Arm, grüner Kanal, ohne PCC/BGE/Stretch. Die Sternbreite wird mit einem Momentenschätzer im
Radius 6 px gemessen (Stern-FWHM etwa 5 px): das ist unempfindlich für kleine Schärfeunterschiede unter etwa 0,1 %. Ob `pixfrac` 1,0 auch bei anderer Frame-Zahl
oder anderem Seeing gilt, ist nicht gezeigt.
