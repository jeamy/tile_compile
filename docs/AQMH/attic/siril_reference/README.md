# Siril-Referenzstacks für den Schärfevergleich

Gehört zu Abschnitt 2.1 / 2.2 von
`../../aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md`.

Zweck: unabhängige lineare Vergleichsstacks aus denselben Dwarf-II-Rohframes,
die der jeweilige tile_compile-Lauf konsumiert hat, um den WCS-gematchten
Grünkanal-FWHM von tile_compile gegen Siril zu messen.

- Siril 1.4.4, `siril-cli`
- keine Dark/Flat/Bias-Master (wie die tile_compile-Läufe)
- kein Drizzle, lineare 32-bit-Ausgabe, native Sensorgeometrie 3840x2160
- Bayer GBRG aus dem Header
- alle Frames, Sirils eigene Winsorized-Sigma-Rejection (3.0 / 3.0)

## Aufruf

    siril-cli -d "<quellverzeichnis>" -s "<objekt>.ssf"

## Skriptvorlage (<objekt>.ssf)

    requires 1.2.0
    link light -out=<workdir>/process
    cd <workdir>/process
    calibrate light -debayer
    register pp_light
    stack r_pp_light rej 3 3 -norm=addscale -output_norm -out=<basis>/<objekt>_siril
    close

`run_all.sh` erzeugt diese Skripte pro Objekt, ruft Siril auf, verschiebt den
Stack nach `<basis>/<objekt>_siril.fit` und löscht das `process/`-Zwischen-
verzeichnis. `status.sh` zeigt den Fortschritt eines laufenden Batches.

## Quellzuordnung (Stand 2026-09-01, Basis /media/data/siril_compare)

| Objekt | Quellverzeichnis | Subs |
|---|---|---|
| m31    | /media/data/Astro/DwarfII/Astronomy/DWARF_RAW_M 31_EXP_10_GAIN_80_2024-10-07-20-51-46-987 | Referenz `result.fit` vorbestanden |
| m42    | /media/tc_ssd/M42_02.2026_lights_all   | 610 |
| ic434  | /media/tc_ssd/IC434_ligths_all         | 359 |
| m66    | /media/tc_ssd/M66_lights               | 975 |
| ic5070 | /media/tc_ssd/IC5070_2                 | 466 |

## Messung

Grünkanal, Sternerkennung, elliptischer 2D-Gauss-Fit, Cross-Match beider Stacks
über die WCS, Median über die gematchten Paare. tile_compile-Seite:
`outputs/stacked_rgb.fits` bzw. `outputs/aqmh_reconstructed_raw.fit`.
