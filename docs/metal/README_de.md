# Metal-GPU-Backend (Apple Silicon, M-Serie) - Implementierungsplan

Stand: 2026-09-26. Planungsdokumente, keine Implementierung. Keine Zeit- oder
Aufwandsschätzungen (Repo-Regel): nur Umfang, Risiko, Abhängigkeiten,
Verifikation.

## Dokumente

| Datei | Inhalt |
| --- | --- |
| `README_de.md` | Überblick, Entscheidungen, Nicht-Ziele, offene Fragen (diese Datei) |
| `01_architektur_und_quelltrennung_de.md` | Verzeichnislayout der getrennten Mac-Quellen, Schnittstellen zum gemeinsamen Code, Registrierung |
| `02_numerik_und_paritaetsvertrag_de.md` | Kein FP64 in Metal: Präzisionsstufen, Emulationsharness, Toleranzen, Kipprate |
| `03_kernel_portierungskarte_de.md` | CUDA-Kernel -> Metal-Kernel, Host-Klassen, Testzuordnung |
| `04_phasen_und_verifikation_de.md` | Abhängigkeitsgeordnete Phasen MP0-MP7 mit Eingangs-/Ausgangskriterien |
| `05_build_config_doku_de.md` | CMake, Config/Schema, GUI/i18n, Report, zu aktualisierende Dokumente |
| `06_github_ci_macos_de.md` | GitHub-Release-Build für Apple Silicon (`release-tile-compile-gui3.yml`, `build_local_macos.sh`): Shader-Einbettung, arm64-only, CI-Gates je Phase |

## Ausgangslage (verifiziert im Code am 2026-09-26)

- Produktionspfad der Rekonstruktion ist **Forward-Drizzle v2** (Gate 10,
  `cutover_complete`, Backend `cuda_v2`). Der v2-CUDA-Pfad ist auf der
  Referenzhardware deutlich schneller als der Legacy-/CPU-Pfad
  (Gate-10-Messung M42: 28.9 s vs 565 s Legacy). Frühere Aussagen "GPU ist
  langsamer" (M7-Slice-Stand) sind überholt.
- Der Treiber ist bereits backend-abstrahiert: `ForwardDrizzleV2Kernel`
  (`forward_drizzle_v2_cpu.hpp`) mit CPU- und CUDA-Implementierung;
  `run_forward_drizzle_v2` erledigt Band-Schleife, transaktionalen Store und
  den CPU-Neustart der gesamten Phase bei GPU-Fehler.
- Ein Metal-Backend ist damit im Kern **eine dritte Implementierung von
  `ForwardDrizzleV2Kernel`** plus zwei Bildoperationen (Prewarp,
  Source-Quality-Box-Filter), die heute über OpenCV-CUDA laufen.
- Der vorhandene CUDA-Code **nutzt** keinen Shared Memory (`__shared__`,
  in Metal "Threadgroup Memory"), keine Warp-Intrinsics (`__shfl`,
  `__ballot`) und kein thrust/cub (im `.cu` gezählt: je 0 Treffer). NVIDIA-
  GPUs haben diese Mittel; der Code verwendet sie nur nicht. Es sind
  Thread-pro-Element-Kernel mit `atomicAdd` (ca. 40 Stellen, überwiegend
  `double`) auf framelokale Ebenen. Für die Portierung entfällt dadurch
  jede Umschreibung von Shared-Memory-/Warp-Code; das Problem ist die
  Präzision, nicht die Kernelstruktur.

## Kernentscheidungen

1. **Quelltrennung:** Alle Apple-spezifischen Quellen liegen in einem eigenen
   Teilbaum `tile_compile_cpp/metal/` (Header, `.mm`, `.metal`, Tests,
   eigenes `CMakeLists.txt`). Er wird nur bei `APPLE` und
   `TILE_COMPILE_ENABLE_METAL` gebaut. Der gemeinsame Code kennt Metal nur
   über eine plattformneutrale Backend-Registry (siehe 01).
2. **Kein FP64:** Metal Shading Language hat keine Double-Precision. Der
   v2-Numerikvertrag (Gate 4: alle Akkumulatoren und Geometrie FP64, FP32
   und Neumaier-FP32 verworfen) ist deshalb nicht 1:1 übertragbar. Vorgabe:
   Double-Float-Arithmetik (2 x FP32) für Transformation und
   Langstrom-Akkumulatoren, FP32 nur für lokale Flächen, Toleranzen per
   Messung festlegen (siehe 02).
3. **Vertrag:** CPU <-> Metal wird als Toleranzparität mit dokumentierten
   Grenzen und gemessener Entscheidungs-Kipprate definiert, nicht als
   Bitgleichheit. Das ist konsistent mit dem v2-CUDA-Vertrag (1e-12,
   Atomics-Reihenfolge nicht deterministisch) und mit `AGENTS.md`
   (Toleranzen + getesteter CPU-Fallback).
4. **Fallback unverändert:** Jeder GPU-Fehler in irgendeinem Band verwirft die
   unveröffentlichte Generation und startet die gesamte Phase auf der CPU
   (Spec `wiring_contract/fallback`).
5. **Vollständigkeit vor Auswahl:** Die Config `full_frame_estimator` wird
   im Runner heute ohne CUDA mit Warnung (M42: 9632 s vs 383 s) behandelt.
   Ohne Metal-Implementierung der Estimator-Kernel (`k_full_seed`,
   `k_full_apply`) wäre diese Option auf dem Mac faktisch unbrauchbar; sie
   gehört daher in den Pflichtumfang (MP4), nicht in eine spätere Stufe.

6. **GitHub-Build bleibt grün:** Der Release-Job `package-macos` (Apple
   Silicon) baut Metal mit; Shader werden in die Binärdateien eingebettet
   (das Packaging-Skript kopiert nur Einzeldateien), Metal nur für arm64,
   Intel-/Linux-/Windows-Jobs bleiben unverändert. Ab MP1 ist jede Phase
   an ein CI-Gate gebunden (siehe 06).

## Nicht-Ziele

- Kein Umbau der CPU-Referenz und keine Änderung des CUDA-Pfads (nur
  mechanische Verschiebung backend-neutraler Typen, siehe 01).
- Keine Änderung der Methodik, der Config-Semantik oder der Phasenartefakte.
- Kein Start von Backend, Sidecar oder Läufen im Rahmen der Planung.
- `web_frontend/` (legacy) wird nicht angepasst.

## Rahmenbedingungen für die Umsetzung

- Entwicklungsrechner ist Linux/NVIDIA. Alles bis MP1 ist dort baubar und
  testbar; ab MP2 sind Apple-Hardware (M1 oder neuer) und macOS-Toolchain
  nötig. Umgebungsfehler (kein Device, fehlende SDK) und Codefehler werden
  getrennt berichtet.
- Reale Läufe (MP6) nur auf ausdrückliche Anforderung.
- Vor jeder Aussage über MSL-/Hardware-Fähigkeiten (Float-Atomics je
  GPU-Familie, `maxBufferLength`, Command-Buffer-Limits) gegen die aktuelle
  Apple-Dokumentation und das Zielgerät verifizieren; die Stellen sind in
  den Dokumenten als "zu verifizieren" markiert.

## Offene Fragen (vor MP4 zu entscheiden)

1. Resume über Backendgrenzen: Darf ein von CUDA/CPU committeter
   Band-Präfix von Metal adoptiert werden (und umgekehrt)? Prüfen, ob
   Plan-Hash/Commit-Hash das Backend oder die Toleranzklasse enthalten
   (`forward_drizzle_v2_store`, Gate 7). Wenn ja: eigene Toleranzklasse im
   Hash oder Adoption nur backendgleich.
2. Float-Atomics in MSL für den Framescatter (2.3 in 02) oder ein
   atomikfreier Ansatz (Ziel-Gather / Tile-Binning); Entscheidung nach
   Messung in MP0.
3. Mindest-Hardware: M1 (Apple GPU Family 7) als Untergrenze oder höher.
4. Gehostete CI-Runner mit Metal-Device ja/nein (05).
