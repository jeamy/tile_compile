# Jev-Screening: Werkzeuge zur Wirkungsmessung von Parametern

Diese Werkzeuge messen, **welche Config-Parameter das Ergebnis überhaupt merklich verändern** und ob eine Änderung besser ist. Sie
liefern die Evidenz für Jev-Kandidaten (siehe `docs/PI/pi_jev_kandidaten_inventar_de.md`, die Ergebnisse in
`docs/PI/pi_jev_effektgroessen_*.md` und die vorab festgelegten Endpunkte in
`web_backend_cpp/config/pi_decisions/release_policy_v2.json`). Sie liegen neben `evaluate_pi_decisions.py` und
`matched_pair_metrics.py` (`web_backend_cpp/scripts/`), auf denen sie aufbauen. Alle Werkzeuge arbeiten **lesend** an Run-Verzeichnissen,
außer wo unten steht, dass sie Läufe starten oder verschieben; neue Läufe starten nur auf ausdrücklichen Auftrag.

## Ablauf

1. **Konfigurationen erzeugen:** `make_screening_configs.py --base <run>/config.yaml --out-dir DIR --preset reconstruction|confirmation`
   (oder `--variants datei.json`). Jede Variante weicht in genau den angegebenen Schlüsseln von der Kontrolle ab; ein unbekannter
   Schlüssel ist ein Fehler. PCC, Astrometrie, BGE und Stretch sind standardmäßig in **allen** Armen aus, denn auf einer Frame-Stichprobe
   findet PCC zu wenige Sterne.
2. **Läufe fahren:** `run_reconstruction_screening.sh --config-dir DIR --input-dir INPUT --runs-dir /media/tc_500 --prefix scr_x_120 --max-frames 120`.
   Nacheinander, Kontrolle zuerst; ein fertiger Arm wird mit `tile_compile_cpp/scripts/prune_evaluation_run.py` bereinigt; ein Gate-Fehler
   (z. B. `FORWARD_STAGE_COVERAGE_GATE_FAILED`) ist ein **Ergebnis** und beendet die Kette nicht. Läufe gehören auf die NVMe: auf dem RAID
   `/media/data` dauert ein Lauf etwa fünfmal so lang.
3. **Auswerten:** `evaluate_reconstruction_screening.py --runs-dir RUNS --prefix scr_x_120 --out bericht.json` vergleicht jede Variante
   mit der Kontrolle an gematchten Sternen (Sternbreite, Elongation, Signal, Rauschen, `n_eff`) und prüft die Paarung (gleiches Referenzbild,
   gleiches Raster). Für die vorab festgelegten Endpunkte: `policy_v2.py` (`candidate_verdict`, `futility`, `session_valid`).
4. **Nachgelagerte Parameter (Denoise, BGE, PCC, Stretch)** ohne neue Rekonstruktion: Lauf kopieren (`cp -a`), dann
   `screen_downstream.py --lab-run KOPIE --out-dir DIR` (Wiederaufsetzen ab BGE, je Variante Sekunden bis Minuten; der Basislauf muss das
   Original bit-identisch reproduzieren). Nur Abschnitte, die die Phase erlaubt, dürfen sich ändern.
5. **Ist ein flacher Hintergrund echt?** `compare_background_structure.py --a A.fits --wcs-a A.wcs --b B.fits --wcs-b B.wcs`: legt zwei
   lineare Stacks über die Himmelskoordinaten und korreliert die großräumigen Muster (hohe Korrelation = Struktur steckt in beiden).
6. **Archivieren:** `archive_verified.sh --src /media/tc_500 --dst /media/data/tile_compile_cache/jev-test RUN_ID...` kopiert, prüft per Prüfsumme
   und löscht die Quelle erst bei Übereinstimmung; ein vorhandener archivierter Lauf wird nie überschrieben.

## Regeln

- **Eine Änderung je Arm**, gleiches Binary, gleiche Frames. Identisch konfigurierte Läufe sind bit-identisch (geprüft), ein Lauf je Arm genügt.
- **Adaptive Gewichtung aus** in allen Armen (sie verändert über `global_weights` das Referenzbild der Registrierung und damit die Geometrie).
- Kennzahlen sind **Wirkung, keine Qualität**: ein hoher Wert ist kein Gewinn. Wirkungsfreiheit heißt: kein Kandidat.
- Vor jeder Bestätigung die Endpunkte in `release_policy_v2.json` festschreiben und committen, bevor Ergebnisse angesehen werden.

## Tests

`python3 -m unittest discover -s web_backend_cpp/scripts/jev_screening -p 'test_*.py'` (20 Tests; die Shell-Skripte laufen gegen einen
Attrappen-Runner, `archive_verified.sh` benötigt `rsync`).
