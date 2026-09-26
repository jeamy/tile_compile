# Jev: Bestätigung von `set_pixfrac` und `set_clip_sigmas` (2026-09-26)

Bestätigung der in `release_policy_v2.json` vorab registrierten Kandidaten auf fünf unabhängigen Sessions
(`--max-frames 120`, Downstream-Stufen in allen Armen aus; innerhalb einer Session identischer Build, M104 lief mit einem älteren Runner-Build als die vier anderen Sessions). Kontrolle: pixfrac 0,8 und Clipping 4/4
(bei M42, IC5070, M66 auf diese Kontrollstufe normalisiert, `make_screening_configs.py --preset confirmation`).
IC4605 war die explorative Screening-Session und zählt nicht zur Bestätigung.

| Session (Gruppe) | pixfrac 1,0: Rauschen | n_eff | clip 5/5: Rauschen | Sterne | FWHM (px 1,0 / clip) |
|---|---|---|---|---|---|
| M31 (Galaxie) | 0,913 | 38,6 vs 26,5 | 0,955 | 2031 | 0,999 / 0,999 |
| M104 (Galaxie) | 0,907 | 37,4 vs 25,6 | 0,943 | 308 | 1,001 / 1,001 |
| M66 (Galaxie) | 0,906 | 37,4 vs 25,6 | 0,970 | 397 | 0,996 / 0,998 |
| M42 (Nebel) | 0,907 | 37,7 vs 25,8 | 0,968 | 1085 | 0,999 / 0,999 |
| IC5070 (Nebel) | 0,910 | 38,3 vs 26,3 | 0,967 | 2331 | 1,000 / 0,999 |

Ergebnis von `policy_v2.candidate_verdict` (Bootstrap über Sessions, Seed aus der Policy):
`set_pixfrac` hält (Rauschen-CI-Obergrenze <= 0,95, alle Sicherheitsgrenzen ok), `set_clip_sigmas` hält
(<= 0,97, alle Sicherheitsgrenzen ok). Strata erfüllt (3 Galaxien, 2 Nebel).

Grenzen: Rauschen wird auf 120-Frame-Teilmengen gemessen; das Urteil sagt nur, dass Endpunkt und Sicherheitsgrenzen der
Policy gelten, es gibt keine Freigabe. Pixfrac 0,6 scheitert am Coverage-Gate (M104 wie IC4605), clip 2/4 erhöht das Rauschen
um ca. 48 %. Rohdaten: `pi_jev_bestaetigung_20260926/`.

## Reproduzierbarkeit und Grenzen des Urteils

- Das Urteil ist mit `web_backend_cpp/scripts/jev_screening/confirm_candidates.py` erzeugt und mit den Bootstrap-Intervallen in
  `pi_jev_bestaetigung_20260926/verdikt.json` abgelegt.
- Obergrenze des 95-%-Intervalls des Rauschverhältnisses: `set_pixfrac` 0,913 (Grenze 0,95), `set_clip_sigmas` 0,970 (Grenze 0,97).
  Bei `set_clip_sigmas` ist der Abstand zur Grenze praktisch null; das Urteil hält, ist aber nicht robust gegen eine sechste, schwächere Session.
- Die Policy definiert "Objektgruppe" nicht. Verwendet wurde Galaxie (M31, M104, M66) gegen Nebel (M42, IC5070); das ist eine Auslegung.
- Kontrollstufen: pixfrac 0,8, Clipping 4/4 (die Configs von M42, IC5070 und M66 hatten selbst 2/4 und wurden im Bestätigungslauf normalisiert).
