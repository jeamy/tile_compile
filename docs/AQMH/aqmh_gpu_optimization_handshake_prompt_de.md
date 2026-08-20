# Handshake-Prompt: AQMH GPU-Optimierung lokal ausführen

Dieses Dokument enthält einen fertigen Prompt, um eine **lokale** Claude-Code-Session
(auf einer Maschine mit echtem `nvcc`/CUDA-GPU-Zugriff) mit dem vollen Kontext der
GPU-Optimierungsanalyse zu starten. Kopiere den Block zwischen den Markierungen
`=== PROMPT START ===` / `=== PROMPT END ===` als erste Nachricht in die neue Session.

Voraussetzung: Lies zuerst `docs/AQMH/aqmh_gpu_optimization_implementation_guide_de.md`
im Repo — der Prompt referenziert sie.

---

=== PROMPT START ===

Kontext: In einer vorherigen Analyse-Session (ohne lokalen GPU-/`nvcc`-Zugriff) wurden
die GPU-Implementierungen von AQMH_RECONSTRUCTION und AQMH_MAPS in diesem Repo
(`tile_compile`) auf Bottlenecks untersucht. Die Ergebnisse und ein priorisierter
Umsetzungsplan stehen in `docs/AQMH/aqmh_gpu_optimization_implementation_guide_de.md`.
Du hast jetzt echten GPU-Zugriff — nutze ihn, um die dort beschriebenen Work Packages
(WP-A bis WP-H) umzusetzen und **zu messen**, statt nur Code zu lesen.

**Pflichtlektüre vor dem ersten Edit:**
1. `docs/AQMH/aqmh_gpu_optimization_implementation_guide_de.md` (vollständig)
2. `AGENTS.md` (Repo-Root) — insbesondere die GPU/CPU-Parity-Vorgaben
3. `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_cuda.cu`
4. `tile_compile_cpp/src/reconstruction/aqmh_reconstruction_opencl.cpp`
5. `tile_compile_cpp/tests/test_aqmh_reconstruction.cpp` (Test
   `aqmh_cuda_reconstruction_matches_cpu_streaming_reference`, ist die Parity-Leine)

**Kondensierte Kernbefunde** (Details in der Anleitung, §0.2):
1. 4× redundante Rekonstruktionsdurchläufe (Luma/Core + R + G + B) mit
   redundantem Q-Map-/Masken-Reload und Buffer-Alloc pro Kanal.
2. Thread-private `values[MaxFrames]`/`weights[MaxFrames]`/`scores[...]`-Arrays
   spillen bei großen Tiers (512–1024 Frames) in Local Memory.
3. Pixel-major-Layout → unkoalesziertes Memory Access Pattern.
4. Serielle Host-Commit-Loops statt asynchronem Double-Buffering.
5. Buffer-Alloc/Free pro Aufruf statt Session-persistenter Puffer.
6. OpenCL-Pfad: Bitonic Sort immer über `MAX_FRAMES=1024` statt tatsächlichem `n`;
   synchrones `kernel.run(...,true)`; `auto_reject` fällt komplett auf CPU zurück.
7. AQMH_MAPS: nur lokale Varianz ist GPU-offloadet; Worker-Planung ist
   CPU-Bound-profiliert auch bei aktivem GPU-Backend.

**Verbindliche Guardrails** (siehe Anleitung §1 — hier nochmal explizit):
- CPU-Fallback bleibt immer vorhanden und funktionsfähig.
- Persistiertes Datenformat (Cache-Dateien, Q-Map-Layout auf Disk) ändert sich nicht
  ohne expliziten Auftrag.
- Jede WP ist flag-gated, bis Parität bewiesen ist.
- Ein Commit pro WP, klare Commit-Message mit WP-Kennung.
- Parity-Test `aqmh_cuda_reconstruction_matches_cpu_streaming_reference` muss nach
  jeder WP grün bleiben — Toleranzen dafür nicht aufweichen.
- **Nicht eigenmächtig entscheiden**, ob der separate Luma/Core-Rekonstruktionspass
  entfallen kann (Luma aus R/G/B ableiten). Das ist eine fachliche Frage
  (Validierungsmetriken hängen daran) — bei Bedarf zuerst fragen.
- fp16-Umstellung von Q-Maps (WP-E) und Bit-Packing von Masken ändern reale
  Zahlenwerte (Rundung) — nur nach expliziter Freigabe umsetzen.

**Build/Test:**
```bash
# CPU-only Baseline zuerst
cmake -S tile_compile_cpp -B build -DBUILD_TESTS=ON -DTILE_COMPILE_ENABLE_CUDA=OFF
cmake --build build --target tile_compile_runner tests -j$(nproc)
./build/tests "[aqmh]"

# CUDA-Build
cmake -S tile_compile_cpp -B build_cuda -DBUILD_TESTS=ON   # nvcc → auto-ON
cmake --build build_cuda --target tile_compile_runner tests -j$(nproc)
./build_cuda/tests "aqmh_cuda_reconstruction_matches_cpu_streaming_reference"
```

**Profiling (pro WP: vorher/nachher vergleichen, nicht nur behaupten):**
```bash
nsys profile -o /tmp/aqmh_wp_X ./build_cuda/tile_compile_runner <repräsentativer_lauf>
ncu --set full -o /tmp/aqmh_wp_X_ncu ./build_cuda/tile_compile_runner <repräsentativer_lauf>
```

**Arbeitsreihenfolge** (siehe Anleitung §11 für Details/Begründung):
1. WP-F — paralleler/asynchroner Host-Commit (risikoarm, kein Zahlenrisiko)
2. WP-A/R2 — Q-Maps/Maske/Gewichte nur einmal über alle 4 Kanäle hochladen
3. WP-G1 — OpenCL Bitonic Sort auf tatsächliches `n` begrenzen
4. WP-A/R1 — Fallback-Zwischenschritt, falls R2 zu riskant
5. WP-B — Two-Stage-Kernel (nur falls C/D den Registerdruck nicht lösen)
6. WP-C — Occupancy/Local-Memory-Tuning (mit `ncu` messen)
7. WP-E — fp16 Q-Maps/bit-gepackte Masken (nur nach expliziter Freigabe)
8. WP-D — Speicherlayout/Coalescing
9. WP-H — AQMH_MAPS GPU-Ausbau (nur nach Profiling-Beleg, nicht blind)
10. WP-C (Nachjustierung) — Occupancy nach allen Layoutänderungen erneut prüfen

**Definition of Done pro WP** (siehe Anleitung §13):
- Vorher-Messung vorhanden
- Flag-gated, CPU-Fallback unverändert
- Parity-Test grün, Toleranzen unverändert
- Nachher-Messung belegt tatsächlichen Gewinn
- Ein Commit mit WP-Kennung in der Message
- Domänenfragen vorher geklärt, nicht eigenmächtig entschieden

**Prozess/Kommunikation:**
- Erst Baseline messen, bevor die erste WP angefasst wird.
- Nach jeder WP kurz berichten: was gemessen, was verändert, was das Ergebnis war.
- Bei Unsicherheit über fachliche Auswirkungen (Genauigkeit, Validierungssemantik)
  nachfragen statt zu raten.
- Kein PR erstellen, es sei denn, explizit angefordert.

Beginne mit: Baseline-Build (CPU-only + CUDA), Baseline-Messung eines repräsentativen
Laufs, dann WP-F.

=== PROMPT END ===

---

## Referenzen

- `docs/AQMH/aqmh_gpu_optimization_implementation_guide_de.md` — die vollständige
  Implementationsanleitung, auf die dieser Prompt verweist.
- `docs/AQMH/aqmh_reconstruction_optimierung_de.md` — bereits umgesetzte
  CPU-seitige Optimierungen der Rekonstruktionsphase (Kontext/Historie).
- `AGENTS.md` (Repo-Root) — Build-/Test-Kommandos, GPU/CPU-Parity-Vorgaben.
