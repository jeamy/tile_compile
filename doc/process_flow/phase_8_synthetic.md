# Phase 8: Synthetic Frames (C++)

## Ziel

Pro Cluster wird ein **synthetisches Frame** erzeugt – **nicht** durch simples Mittel, sondern durch **TLR‑Rekonstruktion** aus den Cluster‑Frames.

## Verhältnis zu Phase 6 (wichtig)

Phase 8 **ersetzt Phase 6 nicht**. Es ist eine **zusätzliche** Rekonstruktion auf **Cluster‑Subsets**:

- **Phase 6** erzeugt ein **Gesamtbild** aus **allen** Frames (TLR über alle Frames).
- **Phase 8** erzeugt **mehrere** synthetische Frames, jeweils aus **homogenen Clustern**.

Das ist **bewusste Mehrarbeit**: Die Cluster trennen Seeing/Drift/Qualität, wodurch spätere Stacks weniger Misch‑Artefakte enthalten.

## C++‑Implementierung

**Referenz:** `tile_compile_cpp/apps/runner_main.cpp` (Phase 8)

### Ablauf

1. **Cluster‑Auswahl**
   - Für jedes Cluster die zugehörigen Frames sammeln.
   - Wenn `cluster_size < synthetic.frames_min` → Cluster wird übersprungen.

2. **TLR‑Rekonstruktion je Cluster**
   - `reconstruct_tlr_subset(use_frame)`
   - identischer Ablauf wie Phase 6 (per‑Tile ECC, Gewichtung, Hanning‑Overlap)

3. **Abbruchlogik**
   - Wenn keine synthetischen Frames und `frames < frames_min` → Phase 8 **skip**.
   - Sonst **error** (Spezifikation verlangt synthetische Frames, wenn möglich).

4. **Artefakte**
   - `outputs/synthetic_*.fit`
   - `artifacts/synthetic_frames.json`

## Progress‑Darstellung

Phase 8 meldet pro Cluster Fortschritt:
- „Cluster X von N“
- „synthetic S/T“ (S erzeugt, T Ziel nach frames_min/frames_max)

## Wann bringt Phase 8 zusätzlichen Gewinn?

Phase 8 ist dann ein **klarer Zusatzgewinn**, wenn sich die Frames **messbar in mehrere Zustände aufteilen**. Praktische Indikatoren:

- **Mehrere Cluster mit ausreichender Größe** (`cluster_size >= synthetic.frames_min`)
- **Hohe Varianz** in `mean_local_quality` / `var_local_quality` und/oder `mean_warp_var_tiles`
- **Deutlich unterschiedliche `G_f`‑Gewichte** zwischen Clustern

Typische Szenarien mit Gewinn:
- **Alt/Az‑Feldrotation** und lokale Drift
- **Stark schwankendes Seeing**
- **Smart‑Teleskope** mit kurzen Belichtungen und wechselndem SNR

Wenig zusätzlicher Gewinn:
- Sehr **konstante** Frames (kaum Cluster‑Trennung)
- **Nur 1 Cluster** mit ausreichend Frames
- `frames < synthetic.frames_min`

## Programmgesteuerte Entscheidung (optional)

Die Pipeline **kann** Phase 8 automatisch auslassen, wenn Clustering keinen Mehrwert zeigt. Sinnvolle Heuristiken:

1. **Cluster‑Vielfalt**: Wenn nur 1 Cluster `>= frames_min` existiert → Phase 8 auslassen.\n2. **Qualitäts‑Spread**: Wenn die Spannweite von `G_f` oder `mean_local_quality` zwischen Clustern unter einem Schwellwert liegt → Phase 8 auslassen.\n3. **Warp‑Varianz**: Wenn `mean_warp_var_tiles` über alle Cluster ähnlich ist → Phase 8 auslassen.\n
Diese Entscheidung kann in Phase 7 getroffen werden (nach `state_clustering.json`), bevor Phase 8 startet. Aktuell ist Phase 8 **immer aktiv**, außer im Reduced‑Mode oder bei `frames < frames_min`.

## C++‑Skizze

```cpp
for (cluster c) {
  mark_frames(use_frame);
  if (count < frames_min) continue;
  syn = reconstruct_tlr_subset(use_frame);
  synthetic_frames.push_back(syn);
}
```

## Parameter (Auszug)

- `synthetic.frames_min`, `synthetic.frames_max`
- `synthetic.clustering.cluster_count_range`
