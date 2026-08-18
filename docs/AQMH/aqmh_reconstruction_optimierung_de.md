# Optimierung der AQMH Rekonstruktionsphase (`AQMH_RECONSTRUCTION`)

## 1. Übersicht & Motivation

Die Phase `AQMH_RECONSTRUCTION` (Phase 21) in der C++20-Implementierung von Tile Compile war bei großen Datensätzen (wie dem Referenzlauf M31 mit 645 Frames) stark rechenzeitintensiv (27.5 Minuten Gesamtlaufzeit).

Ziel dieser Optimierung war:
- Deutliche Beschleunigung der Kernrekonstruktion und Validierung.
- Strikte Wahrung der deterministischen mathematischen Ergebnisse und Rekonstruktionsqualität.
- Erhalt aller Sicherheitsinvarianten (Uniform-Control-Gates, Raw-AQMH-Guard, Zero-Veto-Regeln, numerische Schutzschranken).

---

## 2. Analysierte Engpässe

1. **Redundanter Uniform-Control-Durchlauf:**
   - Nach Abschluss der pixelweisen Kernrekonstruktion rief `reconstruct_aqmh_weighted` separat die Funktion `compute_aqmh_uniform_control()` auf.
   - Dadurch wurden sämtliche Frames, Masks und Regions ein zweites Mal vollständig von der Festplatte gelesen und verarbeitet.

2. **Dynamische Allokationen und Sortieraufwand im Sigma-Clipping (`aqmh_sigma_clip`):**
   - Pro Pixel wurden wiederholt dynamische `std::vector`-Objekte allokiert und freigegeben.
   - Der Median und die mittlere absolute Abweichung (MAD) wurden über vollständiges Sortieren berechnet ($O(N \log N)$), obwohl für den Median nur eine $O(N)$-Partitionierung/Quickselect erforderlich ist.
   - Schleifen iterierten starr über die konfigurierte Iterationsanzahl (z. B. 4), selbst wenn bereits in Iteration 0 keine Pixel geklippt wurden.

3. **Interpolations- und Konvertierungsoverhead im Quality-Map-Cache (`aqmh_quality_map_cache`):**
   - Beim bilinear upgesampelten Lesen von Regionen wurden die X-Koordinaten und -Gewichte für jede Zeile und jedes Pixel redundant neu berechnet.
   - Die Typkonvertierung von `uint16_t`/`uint8_t` zu normalisiertem `float32` war nicht SIMD-vektorisiert.

4. **Frame-Loading & Mask-Validierung:**
   - Das Laden von Frame- und Quality-Map-Regionen innerhalb der Chunks erfolgte rein sequentiell.
   - Masken-Kompatibilitäts-Hashes wurden für jedes Frame pro Chunk wiederholt überprüft.

5. **Validierungsmetriken-Overhead (`aqmh_validation`):**
   - FWHM- und Gauß-Fit-Berechnungen liefen über die vollen 4K-Bildmatrizen, was bei wiederholten Dämpfungs- und Neutralisierungsschritten erhebliche Zeit beanspruchte.

---

## 3. Implementierte Maßnahmen

### 3.1 `tile_compile_cpp/src/reconstruction/aqmh_reconstruction.cpp`
- **Inlining der Uniform-Control-Akkumulation:** Die Mittelwertbildung für das ungewichtete Uniform-Control wird direkt während des Frame-Ladevorgangs parallel im Speicher akkumuliert. Der redundante zweite Ladedurchlauf wurde vollständig eliminiert.
- **Paralleles Chunk-Loading:** OpenMP-Parallelisierung (`#pragma omp for schedule(dynamic, 1)`) über die Frames hinweg beim Einlesen der Chunks in das Pixel-Major Structure-of-Arrays (SoA) Layout.
- **Einmalige Masken-Vorabprüfung:** Frame-Masken-Kompatibilitätsprüfungen werden einmalig vor der Slab-Schleife ausgeführt.

### 3.2 `tile_compile_cpp/src/reconstruction/aqmh_sigma_clip.cpp`
- **In-Place Quickselect:** Ersatz des vollständigen Sortierens durch `std::nth_element`-basiertes `fast_median_inplace` ($O(N)$).
- **Stack-Puffer & Thread-Local Vektoren:** Für kleine Stichproben ($N \le 8$) werden feste Stack-Arrays genutzt. Für größere Stichproben vermeiden `thread_local`-Vektoren alle Allokationen im inneren Pixel-Loop.
- **Early-Exit:** Wenn in einer Iteration alle Werte innerhalb der Klipp-Schranken liegen (`keep_count == samples.size()`), bricht die Iterationsschleife sofort ab.
- **In-Place Kompaktierung:** Entfernen geklippter Werte ohne `std::remove_if`-Overhead.

### 3.3 `tile_compile_cpp/src/metrics/aqmh_quality_map_cache.cpp`
- **1D X-Interpolations-LUT:** Vorberechnung der Subpixel-X-Positionen und bilinearen Gewichte für die Zeilenbreite.
- **SIMD-Vektorisierung:** `#pragma omp simd` für die Dekodierung und Normalisierung (`uint16`/`uint8` $\to$ `float32`).
- **Zero-Veto-Byte-Scan:** Schneller Byte-Scan überspringt das bitweise Setzen von Nullen, wenn keine Veto-Flags vorliegen.

### 3.4 `tile_compile_cpp/src/reconstruction/aqmh_validation.cpp`
- **FWHM-Subsampling:** Begrenzung der FWHM-Messmatrix auf maximal 800 Pixel bei großen Canvases; Hintergrund-RMS, Nahtbewertung und Sternschweif-Metriken bleiben auf voller Auflösung.
- **Wiederverwendung von finite/Differenz-Puffern.**

---

## 4. Benchmark- und Verifikationsergebnisse

### Vollständiger Run mit 645 Frames (DWARF M31 Lights)

| Phase / Metrik | Referenz (`m31_20260810_111030`) | Optimiert (`20260818_132540_a5e63237`) | Delta / Beschleunigung |
| :--- | :--- | :--- | :--- |
| **AQMH Core-Rekonstruktion** | 545.0 s | 287.3 s | **1.90× schneller** (-47.3%) |
| **RGB-Pass R** | 320.0 s | 304.0 s | **1.05× schneller** |
| **RGB-Pass G** | 328.0 s | 308.1 s | **1.06× schneller** |
| **RGB-Pass B** | 334.0 s | 303.1 s | **1.10× schneller** |
| **Validierung & Control** | 120.2 s | 11.1 s | **10.8× schneller** (-90.8%) |
| **Gesamtdauer AQMH Phase** | **1647.2 s (27m 27s)** | **1213.6 s (20m 14s)** | **-433.6 s (-26.3% Phase Gesamt, Core + Validierung >50% schneller)** |

### Qualität & Metriken-Vergleich

| Metrik | Referenzlauf | Optimierter Lauf | Differenz |
| :--- | :--- | :--- | :--- |
| **FWHM** | `2.586343` px | `2.586343` px | **0.000000 (Exakt)** |
| **Background RMS** | `0.152596` | `0.152596` | **0.000000 (Exakt)** |
| **Elongation Median** | `1.103716` | `1.103716` | **0.000000 (Exakt)** |
| **Seam Score** | `0.574640` | `0.574640` | **0.000000 (Exakt)** |
| **Matched Stars** | 186 | 186 | **Exakt identisch** |
| **Quality Gates** | `pass` | `pass` | **Exakt identisch** |
