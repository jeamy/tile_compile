# SlopCodeBench-Analyse: Code-Sloppiness-Metriken

**Datum**: 2026-09-22
**Quelle**: [Earendil Blog: Measuring Code Sloppiness](https://earendil.com/posts/measuring-code-sloppiness/) und [SlopCodeBench (arXiv:2603.24755)](https://arxiv.org/html/2603.24755v1)
**Scope**: `tile_compile_cpp/` (src + apps), `web_backend_cpp/`

## Die drei Metriken

### 1. LOC-Änderung (einfachste Metrik)
Die Anzahl der hinzugefügten Zeilen Code. In den letzten 5 Tagen: **~49.675 Zeilen eingefügt, ~1.858 gelöscht** über 179 Dateien. Das ist eine Netto-Produktion von ~47.817 Zeilen in 5 Tagen.

### 2. Verbosity
Misst den Anteil redundanter/duplizierter Zeilen:

```
Verbosity = |AST-Grep flagged lines ∪ Clone lines| / LOC
```

- **AST-Grep flagged lines**: Zeilen, die von 137 handgefertigten ast-grep-Regeln als "verbose Patterns" markiert wurden (z.B. unnötige Zwischenvariablen, Identity-List-Comprehensions, unnötige Wrapper).
- **Clone lines**: Duplizierte Zeilen (Token-basierte Clone-Detection).
- Union beider, normalisiert durch Gesamt-LOC.

### 3. Erosion (Strukturelle Erosion)
Misst, wie viel der Komplexitätsmasse in hochkomplexen Funktionen konzentriert ist:

```
mass(f) = CC(f) × √SLOC(f)

Erosion = Σ mass(f) für CC(f) > 10  /  Σ mass(f) für alle f
```

- `CC(f)` = Cyclomatic Complexity der Funktion f
- `SLOC(f)` = Source Lines of Code der Funktion f
- Die Quadratwurzel komprimiert den Größenfaktor, sodass Komplexität dominiert.
- Schwellwert: CC > 10 (folgt Radon-Tool-Konvention).

## Benchmark-Referenzwerte (SlopCodeBench Tabelle 2)

| Gruppe | n | Violation % | Clone Ratio | Verbosity | Erosion |
|--------|---|-------------|-------------|-----------|---------|
| Human (Niche <1k★) | 8 | 0.17 ± 0.13 | 0.05 ± 0.02 | 0.18 ± 0.11 | 0.39 ± 0.30 |
| Human (Established 1k-10k★) | 12 | 0.10 ± 0.07 | 0.06 ± 0.03 | 0.13 ± 0.06 | 0.26 ± 0.13 |
| Human (Major >10k★) | 28 | 0.10 ± 0.04 | 0.07 ± 0.04 | 0.14 ± 0.03 | 0.31 ± 0.12 |
| **Human (alle)** | **48** | **0.11 ± 0.07** | **0.06 ± 0.03** | **0.15 ± 0.06** | **0.31 ± 0.17** |
| **Agent** | **990** | **0.32 ± 0.11** | **0.08 ± 0.07** | **0.33 ± 0.10** | **0.68 ± 0.20** |

Referenzwerte einzelner Repos:
- scikit-learn: Erosion 0.411
- scipy: Erosion 0.457
- Beide liegen am oberen Ende der humanen Repos, aber noch deutlich unter Agent-Level.

---

## Messergebnisse: tile_compile_cpp

### Tools
- **lizard 1.24.0** — Cyclomatic Complexity + SLOC pro Funktion
- **jscpd 5.2.0** — Token-basierte Clone-Detection (min 5 lines, min 50 tokens)
- **grep** — Doxygen-Boilerplate-Zählung (Proxy für AST-Grep flagged lines)

### Gesamt-Codebase (src/ + apps/, nur .cpp)

| Metrik | Wert | Benchmark Human | Benchmark Agent | Bewertung |
|--------|------|-----------------|-----------------|-----------|
| **LOC** | 74.413 | — | — | — |
| **Funktionen** | 1.275 | — | — | — |
| **Clone lines** | 3.147 (4,23%) | 0.06 ± 0.03 | 0.08 ± 0.07 | Im humanen Bereich |
| **Doxygen-Boilerplate** | 2.080 (520 Blöcke × 4 Zeilen) | — | — | — |
| **Dead Code** | ~450 | — | — | — |
| **AST-Grep flagged (est.)** | ~2.590 | — | — | — |
| **Verbosity** | **0.069 - 0.077** | 0.15 ± 0.06 | 0.33 ± 0.10 | ✅ Unter humanem Niveau |
| **Erosion** | **0.9433** | 0.31 ± 0.17 | 0.68 ± 0.20 | ❌ Weit über Agent-Level |

### Aufschlüsselung nach Komponente

| Komponente | Funktionen | High-CC (>10) | Erosion | Bewertung |
|------------|-----------|---------------|---------|-----------|
| **src/reconstruction/** | 276 | 56 (20,3%) | 0.8370 | ❌ Über Agent-Level |
| **apps/** | 191 | 45 (23,6%) | 0.9815 | ❌ Extrem |
| **web_backend_cpp/** | 822 | 167 (20,3%) | 0.8603 | ❌ Über Agent-Level |
| **Alle geänderten Dateien** | 795 | 181 (22,8%) | 0.9484 | ❌ Extrem |

### Cyclomatic Complexity Stats

| Metrik | Wert |
|--------|------|
| Avg CC | 11.39 |
| Median CC | 4 |
| Max CC | **968** |

---

## Interpretation

### Verbosity: Überraschend niedrig (0.069 - 0.077)

Die Verbosity liegt **unter** dem humanen Durchschnitt (0.15 ± 0.06). Das bedeutet:

- **Clone Ratio (4,23%)**: Im humanen Bereich (0.06 ± 0.03). Die Codebase hat relativ wenig exakte Code-Duplikate.
- **AST-Grep flagged lines**: Die geschätzten ~2.590 verbose Zeilen (Doxygen-Boilerplate + Dead Code + Wrapper) sind nur ~3,5% der 74K LOC.

**Aber**: Diese Metrik unterschätzt das tatsächliche Problem, weil:
1. Die 137 AST-Grep-Regeln aus SlopCodeBench auf Python zugeschnitten sind, nicht C++. C++-spezifische verbose Patterns (manuelle JSON-Serialisierung, repetitive Doxygen-Blöcke) werden nicht erfasst.
2. Die Doxygen-Boilerplate (520 Blöcke × 4 Zeilen = 2.080 Zeilen) ist ein C++-spezifisches Muster, das in Python-Repositories nicht existiert. Würde man diese als "verbose" zählen, läge die Verbosity bei ~(2.080 + 3.147) / 74.413 = 0.070, immer noch niedrig.
3. Die Codebase ist groß (74K LOC), was den Nenner erhöht und die Metrik künstlich drückt.

**Fazit Verbosity**: Die Codebase hat wenig exakte Duplikate, aber signifikante "semantische" Duplikation (gleiche Logik, unterschiedliche Typen/Namespaces), die von jscpd nicht erfasst wird.

### Erosion: Kritisch hoch (0.9433)

Die Erosion liegt **weit über** dem Agent-Level (0.68 ± 0.20) und ist **3× höher** als der humane Durchschnitt (0.31 ± 0.17). Das bedeutet: **94,3% der Komplexitätsmasse** konzentriert sich in Funktionen mit CC > 10.

**Hauptverursacher (Top 10 nach Mass):**

| Funktion | CC | SLOC | Mass | Anteil |
|----------|-----|------|------|--------|
| `run_pipeline_command` | **968** | 5.250 | 70.138 | 25,3% |
| `run_phase_registration_prewarp` | **780** | 3.992 | 49.282 | 17,8% |
| `Config::from_yaml` | **428** | 1.044 | 13.829 | 5,0% |
| `Config::validate` | **353** | 744 | 9.629 | 3,5% |
| `run_phase_aqmh_reconstruction` | **221** | 1.397 | 8.260 | 3,0% |
| `resume_command` | **211** | 1.189 | 7.276 | 2,6% |
| `run_phase_local_metrics` | **182** | 1.094 | 6.020 | 2,2% |
| `run_rgb_downstream` | **185** | 1.025 | 5.923 | 2,1% |
| `apply_background_extraction` | **165** | 884 | 4.906 | 1,8% |
| `reconstruct_aqmh_weighted` | **144** | 355 | 2.713 | 1,0% |

Die Top 3 Funktionen (`run_pipeline_command`, `run_phase_registration_prewarp`, `Config::from_yaml`) allein accounten für **48,1% der gesamten Komplexitätsmasse**.

**Ursachen:**
1. **Monolithische Runner-Funktionen**: `run_pipeline_command` (CC=968, 5.250 SLOC) ist eine einzelne Funktion, die die gesamte Pipeline orchestriert. Jede Phase, jeder Config-Pfad, jeder Resume-Zweig ist in dieser einen Funktion.
2. **Config-Parsing in einer Funktion**: `Config::from_yaml` (CC=428, 1.044 SLOC) parst das gesamte YAML-Config in einer einzigen Funktion mit hunderten von if/else-Zweigen.
3. **Phase-Orchestrierung ohne Polymorphismus**: Statt einer Phasen-Pipeline mit Strategy-Pattern sind alle Phasen inline in einer Funktion.

**Vergleich mit bekannten Repos:**
- scikit-learn (Erosion 0.411) und scipy (Erosion 0.457) sind mathematische Bibliotheken mit komplexen Algorithmen, aber sie haben ihre Komplexität über viele mittelgroße Funktionen verteilt.
- Diese Codebase konzentriert Komplexität in wenigen gigantischen Funktionen.

---

## Vergleich mit dem Blog-Post

Der Blog-Post von Earendil beschreibt:
> "On average the verbosity in the repos is 0.15 ± 0.06 and in the agents code is 0.33 ± 0.10. For erosion the repos achieve 0.31 ± 0.17 and the agents 0.68 ± 0.20."

Unsere Messung:
- **Verbosity 0.069-0.077**: Unter dem humanen Niveau. Die Codebase ist "effizient" im Sinne von wenig exakter Duplikation, was auf die manuelle Duplikatsbereinigung zurückzuführen ist, die in dieser Session durchgeführt wurde.
- **Erosion 0.9433**: 3× über dem humanen Niveau, 1,4× über dem Agent-Niveau. Die Codebase hat ein massives Problem mit Komplexitätskonzentration.

Der Blog-Post sagt auch:
> "bad coding decisions accumulate over time and for the strict solve rate, where all tests have to be passed at all checkpoints, even state of the art models achieve 0% pass rate"

Dies ist bei dieser Codebase relevant: Die Tests bestehen (2.756.148 Assertionen, 549 Testfälle), aber die Code-Struktur verschlechtert sich mit jeder hinzugefügten Phase, weil neue Logik in bestehende monolithische Funktionen eingefügt wird statt neue Funktionen zu extrahieren.

---

## Empfehlungen

### Erosion reduzieren (höchste Priorität)

Die Erosion von 0.94 auf ~0.50 (unter Agent-Level) zu senken, erfordert:

1. **`run_pipeline_command` (CC=968, SLOC=5250) zerlegen**:
   - Extrahiere jede Phase in eine separate Funktion/Methode.
   - Verwende ein Phase-Registry-Pattern: `std::vector<PhaseStep>` mit `run_step(config, context)`.
   - Geschätzte CC-Reduktion: 968 → ~20 pro extrahierter Phase.
   - Geschätzte Erosions-Reduktion: -25% (Mass von 70K → ~5K).

2. **`run_phase_registration_prewarp` (CC=780, SLOC=3992) zerlegen**:
   - Extrahiere Unterphasen (Star-Detection, Matching, Warp-Estimation, Validation).
   - Geschätzte Erosions-Reduktion: -18%.

3. **`Config::from_yaml` (CC=428, SLOC=1044) zerlegen**:
   - Verwende Section-Parser: `parse_drizzle_config(node)`, `parse_aqmh_config(node)`, etc.
   - Geschätzte Erosions-Reduktion: -5%.

4. **`Config::validate` (CC=353, SLOC=744) zerlegen**:
   - Validierung pro Sektion in separate Funktionen.
   - Geschätzte Erosions-Reduktion: -3,5%.

**Potenzielle Erosion nach Refactoring**: ~0.50-0.60 (unter Agent-Level, im oberen humanen Bereich).

### Verbosity im Auge behalten

Die aktuelle Verbosity ist niedrig, aber:
- Die Doxygen-Boilerplate (2.080 Zeilen) sollte entfernt werden — sie ist C++-spezifischer "Slop", der von jscpd nicht als Clone erfasst wird.
- Semantische Duplikate (gleiche Logik, unterschiedliche Typen) sollten durch Templates/Overloads reduziert werden.

---

## Methodische Einschränkungen

1. **AST-Grep-Regeln**: SlopCodeBench verwendet 137 Python-spezifische ast-grep-Regeln. Wir haben als Proxy Doxygen-Boilerplate + Dead Code verwendet, was C++-spezifische Patterns erfasst, aber nicht direkt vergleichbar ist.

2. **Clone-Detection**: jscpd erfasst nur exakte Token-Clones. Semantische Duplikate (gleiche Logik, unterschiedliche Variablennamen) werden nicht erfasst. Die tatsächliche Verbosity könnte höher sein.

3. **Erosion-Schwellwert**: CC > 10 ist für Python optimiert. C++ hat naturgemäß höhere CC durch manuelle Speicherverwaltung, Templates und fehlende Pattern-Matching-Syntax. Ein Schwellwert von CC > 15 oder CC > 20 könnte fairer sein.

4. **Codebase-Größe**: Die SlopCodeBench-Referenzrepos sind Python-Repositories mit typischerweise 1K-50K LOC. Diese Codebase hat 74K LOC in C++ allein, was direkte Vergleiche erschwert.

5. **Domain**: Bildverarbeitungs-Pipelines haben naturgemäß viele Verzweigungen (Config-Pfade, Format-Checks, Fehlerbehandlung). Ein direkter Vergleich mit allgemeinen Python-Repos ist nicht ganz fair.

---

## Rohdaten

### Clone-Detection (jscpd)

```
Total clones:      305
Duplicated lines:   3.147 (4,23%)
Total lines:        74.413
Sources:            92 Dateien
Min lines:          5
Min tokens:         50
```

### Erosion-Top-25 (lizard)

| # | CC | SLOC | Mass | Funktion |
|---|-----|------|------|---------|
| 1 | 968 | 5.250 | 70.138 | run_pipeline_command |
| 2 | 780 | 3.992 | 49.282 | run_phase_registration_prewarp |
| 3 | 428 | 1.044 | 13.829 | Config::from_yaml |
| 4 | 353 | 744 | 9.629 | Config::validate |
| 5 | 221 | 1.397 | 8.260 | run_phase_aqmh_reconstruction |
| 6 | 211 | 1.189 | 7.276 | resume_command |
| 7 | 185 | 1.025 | 5.923 | run_rgb_downstream |
| 8 | 182 | 1.094 | 6.020 | run_phase_local_metrics |
| 9 | 165 | 884 | 4.906 | apply_background_extraction |
| 10 | 144 | 355 | 2.713 | reconstruct_aqmh_weighted |
| 11 | 143 | 759 | 3.940 | run_forward_drizzle_stages |
| 12 | 134 | 630 | 3.363 | run_phase_channel_split_normalization_global_metrics |
| 13 | 131 | 424 | 2.697 | stream_forward_drizzle_uniform_and_raw |
| 14 | 97 | 386 | 1.906 | run_pcc |
| 15 | 88 | 314 | 1.559 | accumulate_pair_impl |
| 16 | 83 | 210 | 1.203 | generate_autobge_sample_points |
| 17 | 77 | 227 | 1.160 | main (cli_main) |
| 18 | 76 | 449 | 1.610 | run_preprocess_pipeline |
| 19 | 75 | 227 | 1.130 | overlap_add |
| 20 | 75 | 318 | 1.337 | reconstruct_aqmh_weighted_opencl |
| 21 | 72 | 270 | 1.183 | extract_autotune_prepared_tile_samples |
| 22 | 68 | 171 | 889 | preprocessing::validate |
| 23 | 63 | 334 | 1.151 | persist_forward_drizzle_multiband |
| 24 | 61 | 191 | 843 | fit_color_matrix |
| 25 | 61 | 308 | 1.071 | compute_geometric_coverage |

### Web-Backend Top-10

| # | CC | SLOC | Mass | Funktion |
|---|-----|------|------|---------|
| 1 | 376 | 1.368 | 13.907 | register_pi_routes |
| 2 | 202 | 919 | 6.124 | register_runs_routes |
| 3 | 201 | 807 | 5.710 | register_tools_routes |
| 4 | 180 | 877 | 5.331 | register_ai_routes |
| 5 | 109 | 359 | 2.065 | register_scan_routes |
| 6 | 107 | 216 | 1.573 | read_run_status |
| 7 | 101 | 111 | 1.064 | pi::validate_op |
| 8 | 93 | 246 | 1.459 | register_config_routes |
| 9 | 90 | 306 | 1.574 | register_preprocessing_routes |
| 10 | 85 | 209 | 1.229 | fallback_parse_message |
