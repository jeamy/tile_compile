# Implementierungsplan: Registration Pipeline Optimierungen

> **Basierend auf:** `doc/registration_pipeline.md` (kombinierte Fassung)
> **Stand:** April 2026
> **Ziel:** Priorisierte Umsetzung aller identifizierten Optimierungen

---

## Übersicht der Änderungen

| # | Feature | Abschnitt | Priorität |
|---|---------|-----------|-----------|
| 1 | Blind-Chain-Parameter konfigurierbar | §4.1, §8.B | Hoch |
| 2 | Chain-Tiefen-Warnung | §7.3, §8.I | Hoch |
| 3 | use_astrometry Parameter | §4.13 | Hoch |
| 4 | Lokale Hintergrundsubtraktion | §4.4, §8.D | Mittel |
| 5 | Adaptiver Polynomial-Grad | §4.7, §8.C | Mittel |
| 6 | Differenzierte Modell-CC | §4.10, §8.G | Mittel |
| 7 | Beidseitige Abdeckung Local Reference | §4.11 | Mittel |
| 8 | Gate-Policy-Funktion | §4.0, §8.A | Niedrig (Architektur) |
| 9 | Szenenklassifikation | §4.12, §8.E | Niedrig (Architektur) |
| 10 | Astrometrische Rescue | §4.13 | Niedrig (Architektur) |
| 11 | Erweiterte JSON-Telemetrie | §7.1, §8.H | Mittel |

---

## Code-Implementierung

### 1. Blind-Chain-Parameter konfigurierbar (§4.1, §8.B)

**Dateien:**
- `tile_compile_cpp/include/tile_compile/config/configuration.hpp`
- `tile_compile_cpp/src/io/config.cpp`
- `tile_compile_cpp/apps/runner_phase_registration.cpp`

```cpp
// configuration.hpp:
struct RegistrationConfig {
    int max_blind_chain_depth = 0;               // 0 = auto (N/10), >0 = manuell
    float blind_chain_strong_anchor_cc = 0.08f;
    float blind_chain_drift_threshold_px = 2.0f;
};

// Berechnung effektiver Tiefe:
int get_effective_chain_depth(int num_frames, const RegistrationConfig& cfg) {
    if (cfg.max_blind_chain_depth > 0) {
        return cfg.max_blind_chain_depth;  // Manuelle Überschreibung
    }
    // Auto: N/10, mindestens 12, maximal 50
    return std::clamp(num_frames / 10, 12, 50);
}
```

**Logging der effektiven Tiefe:**
```cpp
int effective_depth = get_effective_chain_depth(num_frames, registration_cfg);

emitter.info(run_id, "[REG-CHAIN] Using max_blind_chain_depth=" + 
    std::to_string(effective_depth) + 
    " (config=" + (registration_cfg.max_blind_chain_depth == 0 ? "auto" : 
                  std::to_string(registration_cfg.max_blind_chain_depth)) + 
    ", N=" + std::to_string(num_frames) + ")");
```

### 2. use_astrometry Parameter (§4.13)

**Dateien:**
- `tile_compile_cpp/include/tile_compile/config/configuration.hpp`
- `tile_compile_cpp/src/io/config.cpp`

```cpp
// configuration.hpp:
bool use_astrometry = true;  // default: true
```

### 3. Weitere Implementierungen

Siehe vollständige Code-Beispiele in `doc/registration_pipeline.md` §8.

---

## Dokumentations-Checkliste

### A. Schema-Dateien

| Datei | Pfad | Änderungen |
|-------|------|------------|
| `tile_compile.schema.yaml` | `tile_compile_cpp/` | Neue Parameter hinzufügen |
| `tile_compile.schema.json` | `tile_compile_cpp/` | JSON-Schema aktualisieren |

**Neue Parameter im Schema:**
```yaml
registration:
  type: object
  properties:
    # Bestehende Parameter...
    
    # Neu:
    max_blind_chain_depth:
      type: integer
      default: 0
      minimum: 0
      maximum: 100
      description: "0 = auto (N/10), >0 = manual override"
    
    blind_chain_strong_anchor_cc:
      type: number
      default: 0.08
      minimum: 0.01
      maximum: 0.5
    
    blind_chain_drift_threshold_px:
      type: number
      default: 2.0
      minimum: 0.5
      maximum: 10.0
    
    use_astrometry:
      type: boolean
      default: true
    
    enable_local_background_subtraction:
      type: boolean
      default: false
```

---

### B. Konfigurationsreferenz

| Datei | Sprache | Änderungen |
|-------|---------|------------|
| `configuration_reference.md` | DE | §8 Registration erweitern |
| `configuration_reference_en.md` | EN | §8 Registration erweitern |

**Inhalt für beide Dateien:**

```markdown
### `registration.max_blind_chain_depth`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | integer |
| **Default** | `0` (auto = N/10, clamped 12-50) |
| **Minimum** | `0` |
| **Maximum** | `100` |

**Zweck:** Maximale Tiefe für Blind-Chain-Rettung. 
- `0` = automatische Berechnung: `min(max(N/10, 12), 50)`
- `>0` = manuelle Überschreibung

**Wann anpassen:**
- `20` bei sehr langen Wolkenblöcken (>auto-Tiefe)
- `10` bei konservativerem Verhalten (weniger aggressive Chains)
- `0` (default) für die meisten Sequenzen

---

### `registration.blind_chain_strong_anchor_cc`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | number |
| **Default** | `0.08` |
| **Minimum** | `0.01` |
| **Maximum** | `0.5` |

**Zweck:** CC-Schwelle für "starke Anker" in Blind-Chains.
Frames mit höherem CC können tiefere Ketten starten.

---

### `registration.use_astrometry`

| Eigenschaft | Wert |
|-------------|------|
| **Typ** | boolean |
| **Default** | `true` |

**Zweck:** Astrometrische Rescue für unresolved Frames aktivieren.
Erfordert ASTAP-Binary und Katalog in `astrometry.*`.

**Wann deaktivieren:**
- Bekannte Probleme mit ASTAP
- Schnellere Verarbeitung ohne Plate-Solving
```

---

### C. Praktische Beispiele

| Datei | Sprache | Änderungen |
|-------|---------|------------|
| `configuration_examples_practical_de.md` | DE | Neue Szenarien |
| `configuration_examples_practical_en.md` | EN | Neue Szenarien |

**Neue Szenarien für beide Dateien:**

```yaml
### Szenario: Langer Wolkenblock (maximale Chain-Tiefe)

```yaml
registration:
  engine: triangle_star_matching
  use_astrometry: true
  max_blind_chain_depth: 20        # Erhöht für lange Wolkenblöcke
  blind_chain_strong_anchor_cc: 0.05  # Niedriger für mehr Anker
  star_topk: 200
  star_inlier_tol_px: 6.0
```

### Szenario: Astrometrie deaktiviert (schnellere Verarbeitung)

```yaml
registration:
  engine: triangle_star_matching
  use_astrometry: false            # Keine astrometrische Rescue
  # Fallback auf Polynom-Modell bei unresolved Frames
```

### Szenario: Mondlicht mit Hintergrundsubtraktion

```yaml
registration:
  engine: triangle_star_matching
  enable_local_background_subtraction: true
  star_topk: 200
  star_inlier_tol_px: 5.0
```
```

---

### D. YAML-Beispiele (examples/)

| Datei | Änderungen |
|-------|------------|
| `M42.global_medium.yaml` | `use_astrometry: true` hinzufügen |
| `M45_high_altitude_strong_rotation.example.yaml` | `max_blind_chain_depth` dokumentieren |
| `bright_star.example.yaml` | `use_astrometry` (bei sehr hellen Sternen evtl. false) |
| `canon_equatorial_balanced.example.yaml` | Standard-Config aktualisieren |
| `canon_low_n_high_quality.example.yaml` | `max_blind_chain_depth: 0` (auto-default) |
| `emergency_mode.example.yaml` | `use_astrometry: false` (Schnelligkeit) |
| `full_mode.example.yaml` | Alle neuen Parameter mit Defaults |
| `ic434.example.yaml` | `use_astrometry: true` |
| `ic434_background_gradient.example.yaml` | `enable_local_background_subtraction: true` |
| `m31_background_gradient_balanced.example.yaml` | `enable_local_background_subtraction: true` |
| `m66_galaxy_background_balanced.example.yaml` | `enable_local_background_subtraction: true` |
| `mono_full_mode.example.yaml` | `use_astrometry: true` |
| `mono_small_n_anti_grid.example.yaml` | `max_blind_chain_depth: 15` (override für kleine N) |
| `mono_small_n_ultra_conservative.example.yaml` | `max_blind_chain_depth: 10` (konservativer override) |
| `reduced_mode.example.yaml` | `use_astrometry: false` (Schnelligkeit) |
| `smart_telescope_dwarf_seestar.example.yaml` | `max_blind_chain_depth: 0` (auto) |
| `smart_telescope_very_bright_star.example.yaml` | `use_astrometry: false` (ASTAP Probleme) |
| `very_bright_star_anti_seam.example.yaml` | `use_astrometry: false` |

**Beispiel-Update für `smart_telescope_very_bright_star.example.yaml`:**

```yaml
registration:
  engine: triangle_star_matching
  enable_star_pair_fallback: false
  use_astrometry: false           # Sehr helle Sterne: ASTAP oft problematisch
  max_blind_chain_depth: 15
```

---

### E. Pipeline-Dokumentation

| Datei | Änderungen |
|-------|------------|
| `registration_pipeline.md` | Bereits aktualisiert (Quell-Referenz) |
| `registration_optimierung_tile_compile_cpp.md` | Querdokument — Abgleich |

---

## Vollständige Datei-Liste

### Code (C++)
```
tile_compile_cpp/
├── include/tile_compile/config/configuration.hpp
├── include/tile_compile/registration/
│   ├── gate_policy.hpp              (NEU)
│   ├── scene_profile.hpp            (NEU)
│   └── astrometric_rescue.hpp       (NEU)
├── src/
│   ├── io/config.cpp
│   ├── io/global_registration_json.cpp
│   ├── registration/
│   │   ├── star_detection.cpp
│   │   ├── polynomial_model.cpp
│   │   ├── local_reference.cpp
│   │   ├── gate_policy.cpp          (NEU)
│   │   ├── scene_profile.cpp        (NEU)
│   │   └── astrometric_rescue.cpp   (NEU)
│   └── tests/
│       ├── test_gate_policy.cpp     (NEU)
│       ├── test_polynomial_model.cpp
│       ├── test_astrometric_rescue.cpp (NEU)
│       └── test_scene_profile.cpp   (NEU)
└── apps/runner_phase_registration.cpp
```

### Dokumentation
```
doc/
├── registration_pipeline.md          (Referenz — bereits aktualisiert)
├── registration_implementation_plan.md   (Diese Datei)
├── configuration_reference.md        (§8 Registration erweitern)
├── configuration_reference_en.md     (§8 Registration erweitern)
├── configuration_examples_practical_de.md  (Neue Szenarien)
├── configuration_examples_practical_en.md  (Neue Szenarien)
└── tile_compile.schema.yaml          (Neue Parameter)
```

### YAML-Beispiele (18 Dateien)
```
tile_compile_cpp/examples/
├── M42.global_medium.yaml
├── M45_high_altitude_strong_rotation.example.yaml
├── bright_star.example.yaml
├── canon_equatorial_balanced.example.yaml
├── canon_low_n_high_quality.example.yaml
├── emergency_mode.example.yaml
├── full_mode.example.yaml
├── ic434.example.yaml
├── ic434_background_gradient.example.yaml
├── m31_background_gradient_balanced.example.yaml
├── m66_galaxy_background_balanced.example.yaml
├── mono_full_mode.example.yaml
├── mono_small_n_anti_grid.example.yaml
├── mono_small_n_ultra_conservative.example.yaml
├── reduced_mode.example.yaml
├── smart_telescope_dwarf_seestar.example.yaml
├── smart_telescope_very_bright_star.example.yaml
└── very_bright_star_anti_seam.example.yaml
```

---

## Logging & JSON-Ausgaben

### Logfile-Ausgaben (runner_phase_registration.cpp)

| Feature | Log-Pattern | Level |
|---------|-------------|-------|
| **Chain-Tiefe adaptiv** | `[REG-CHAIN] Using max_blind_chain_depth=25 (auto=1, N=250)` | INFO |
| **Chain-Warnung** | `[REG-WARN] High blind-chain depth 18/20` | WARNING |
| **Astrometrie-Rescue** | `[REG-ASTRO] Frame 42: astrometric_rescue successful (5 matches, CC=0.35)` | INFO |
| **Astrometrie fehlgeschlagen** | `[REG-ASTRO] Frame 42: astrometric_rescue failed (ASTAP unavailable)` | DEBUG |
| **Szenenprofil** | `[REG-SCENE] star_density=2.5 fwhm=3.2 (dense field detected)` | INFO |
| **Gate-Policy** | `[REG-GATE] Frame 42: accepted (margin=0.008, shift=45px)` | DEBUG |
| **Polynom-Grad** | `[REG-POLY] Using degree=3 for 30 frames (adaptive)` | INFO |
| **Modell-CC-Klasse** | `[REG-MODEL] Frame 42: model_cc=5e-4 (high_confidence)` | DEBUG |
| **Local-Reference Abgelehnt** | `[REG-LOCAL] Frame 42: rejected (unilateral support: 6/0)` | DEBUG |
| **Hintergrundsubtraktion** | `[REG-BG] Local background subtraction enabled` | INFO |

### JSON-Ausgaben (global_registration.json)

| Feld | Beschreibung | Wann gesetzt |
|------|--------------|--------------|
| `max_blind_chain_depth_used` | Effektiv verwendete Tiefe | Immer |
| `max_blind_chain_depth_config` | Konfigurierter Wert (0 = auto) | Immer |
| `use_astrometry` | Ob astrometrische Rescue versucht wurde | Immer |
| `astrometric_matches` | Anzahl ASTAP-Matches | Bei `astrometric_rescue` |
| `astrometric_solver` | Welcher Solver verwendet (`astap`, `internal`) | Bei `astrometric_rescue` |
| `scene_star_density` | Sterne/100x100px | Wenn Szenenklassifikation aktiv |
| `scene_fwhm_estimate` | Median-FWHM | Wenn Szenenklassifikation aktiv |
| `gate_margin_used` | Tatsächlich verwendeter Margin | Bei allen Rescues |
| `polynomial_degree` | Verwendeter Polynom-Grad | Bei `model_*` Provenances |
| `model_confidence` | `high`/`medium`/`low` | Bei `model_*` Provenances |
| `local_ref_bilateral` | Ob beidseitige Abdeckung vorlag | Bei `local_reference_rescue` |
| `background_subtraction` | Ob Hintergrundsubtraktion aktiv war | Immer |

**Erweiterte JSON-Struktur:**

```json
{
  "registration_config_applied": {
    "max_blind_chain_depth_config": 0,
    "max_blind_chain_depth_used": 25,
    "use_astrometry": true,
    "scene_adaptations": {
      "star_density": 2.5,
      "fwhm_estimate": 3.2,
      "adaptations_applied": ["star_topk", "inlier_tol"]
    }
  },
  "warps": [
    {
      "frame_idx": 42,
      "cc": 0.35,
      "source": "astrometric_rescue",
      "astrometric_matches": 5,
      "astrometric_solver": "astap",
      "chain_depth": 0
    },
    {
      "frame_idx": 43,
      "cc": 0.0005,
      "source": "model_blended",
      "polynomial_degree": 3,
      "model_confidence": "high",
      "span": 8
    }
  ],
  "statistics": {
    "chain_max_depth_observed": 18,
    "astrometric_rescue_count": 12,
    "scene_classification": "dense",
    "gate_policy_version": "v2"
  }
}
```

---

## Rollout-Checkliste

### Schritt 1: Code-Implementierung — ERLEDIGT
- [x] `configuration.hpp` — Neue Parameter + `get_effective_chain_depth()` Helper
- [x] `config.cpp` — YAML Parsing, Serialisierung, Validierung, JSON-Schema
- [x] `runner_phase_registration.cpp` — Chain-Tiefen-Berechnung, Config-basierte Konstanten, Logging, **Astrometrische Rescue-Integration**
- [x] `global_registration.cpp` — `enable_local_background_subtraction` in `detect_stars_simple`, Funktions-Signaturen aktualisiert
- [x] `global_registration.hpp` — Neue Parameter in Deklarationen
- [x] `astrometric_rescue.hpp/.cpp` — **Neue ASTAP-Integration für Plate-Solving Rescue**
- [x] `test_registration_new_features.cpp` — **Unit-Tests für neue Funktionen**
- [x] `CMakeLists.txt` — Neue Dateien zu LIB_SOURCES und tests hinzugefügt
- [ ] Build erfolgreich

### Schritt 2: Schema & Dokumentation
- [x] `tile_compile.schema.yaml` aktualisieren — Spezifikation im Plan erstellt
- [x] `configuration_reference.md` erweitern — §8 Registration mit 5 neuen Parametern ergänzt
- [x] `configuration_reference_en.md` erweitern — §8 Registration mit 5 neuen Parametern ergänzt
- [x] `web_frontend/i18n/de.json` erweitern — 5 neue Parameter eingetragen
- [x] `web_frontend/i18n/en.json` erweitern — 5 neue Parameter eingetragen
- [x] Beispiel-Szenarien dokumentieren — Practical Examples DE/EN mit neuen Parametern ergänzt

### Schritt 2a: Parameter Studio Frontend — NICHT BENÖTIGT
- [x] ~~Statische Frontend-Änderungen~~ — Nicht erforderlich: Frontend generiert UI dynamisch aus JSON-Schema
- [x] i18n-Einträge sind bereits vorhanden für Labels und Hilfetexte

### Schritt 3: YAML-Beispiele aktualisieren (18 Dateien) — ERLEDIGT
- [x] `M42.global_medium.yaml` — use_astrometry: true
- [x] `M45_high_altitude_strong_rotation.example.yaml` — max_blind_chain_depth: 0 (auto)
- [x] `bright_star.example.yaml` — use_astrometry: false
- [x] `canon_equatorial_balanced.example.yaml` — Standard-Config
- [x] `canon_low_n_high_quality.example.yaml` — max_blind_chain_depth: 0
- [x] `emergency_mode.example.yaml` — use_astrometry: false
- [x] `full_mode.example.yaml` — Alle neuen Parameter mit Defaults
- [x] `ic434.example.yaml` — use_astrometry: true
- [x] `ic434_background_gradient.example.yaml` — enable_local_background_subtraction: true
- [x] `m31_background_gradient_balanced.example.yaml` — enable_local_background_subtraction: true
- [x] `m66_galaxy_background_balanced.example.yaml` — enable_local_background_subtraction: true
- [x] `mono_full_mode.example.yaml` — use_astrometry: true
- [x] `mono_small_n_anti_grid.example.yaml` — max_blind_chain_depth: 15
- [x] `mono_small_n_ultra_conservative.example.yaml` — max_blind_chain_depth: 10
- [x] `reduced_mode.example.yaml` — use_astrometry: false
- [x] `smart_telescope_dwarf_seestar.example.yaml` — max_blind_chain_depth: 0
- [x] `smart_telescope_very_bright_star.example.yaml` — use_astrometry: false
- [x] `very_bright_star_anti_seam.example.yaml` — use_astrometry: false

### Schritt 4: Review
- [ ] Technische Review (Code)
- [ ] Dokumentations-Review
- [ ] YAML-Validierung

---

## Erfolgskriterien

### Qualitätsmetriken

| Metrik | Vorher | Ziel |
|--------|--------|------|
| `unresolved` Frames | 5-10% | < 3% (mit Astrometrie) |
| `model_*` Anteil | 10-20% | < 10% (bessere Rescue) |
| Chain-Drift | unkontrolliert | < 2px pro Kette |

### Dokumentations-Metriken

| Metrik | Vorher | Ziel |
|--------|--------|------|
| Dokumentations-Abdeckung | — | 100% aller Parameter |
| YAML-Beispiele aktualisiert | 0/18 | 18/18 |
| Log-Patterns implementiert | 0/10 | 10/10 |
| JSON-Felder erweitert | 4/14 | 14/14 |

### Monitoring-Metriken (via Logs/JSON)

| Metrik | Quelle | Nutzen |
|--------|--------|--------|
| `max_blind_chain_depth_used` | JSON | Überprüfung automatischer Adaption |
| `astrometric_rescue_count` | JSON | Anteil geretteter Frames |
| `scene_classification` | JSON | Automatische Parameterwahl nachvollziehen |
| `model_confidence` | JSON | Qualität der Modell-Fallbacks |
| `chain_max_depth_observed` | JSON | Frühwarnung für Grenzbereiche |
| `gate_policy_version` | JSON | Feature-Tracking |

### Debuggability-Metriken

| Kriterium | Ziel |
|-----------|------|
| Alle Rescue-Entscheidungen | Via `[REG-*]` Log-Patterns nachvollziehbar |
| Automatische Adaptionen | Via `[REG-CHAIN]`, `[REG-SCENE]` transparent |
| Fehlerursachen | Via spezifischen DEBUG-Logs identifizierbar |
| Performance-Impact | Via JSON `statistics` Block messbar |

---

**Dokument-Version:** 2.0
**Letzte Aktualisierung:** April 2026

