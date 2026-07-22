# PI – KI-gestützte Konfigurationsempfehlungen

> 🇬🇧 [English version](pi_ai_recommendations_en.md)

**Status:** Produktiv (v0.3.3, 2026-06-16)  
**Modul:** `agent_service` (TypeScript-Sidecar) + `web_backend_cpp` AI-Routen  

---

## Übersicht

Das **PI** (Parameter Intelligence) Modul analysiert das Ergebnis eines Scan-Jobs und erzeugt
KI-gestützte Konfigurationsempfehlungen für `tile_compile`. Die Empfehlungen werden gegen das
JSON-Schema validiert, bei Bedarf einzeln geprüft (per-Update-Validierung) und können direkt im
Parameter Studio übernommen werden.

---

## Architektur

```
Frontend (app.js)
  ├── POST /api/scan/metrics        → scan-metrics CLI → Bildqualitäts-Aggregate
  ├── POST /api/scan/analysis       → web_backend_cpp AI-Route
  │       ├── scan_result           (Frame-Metadaten: Anzahl, Gain, Exposure …)
  │       ├── scan_metrics          (FWHM, SNR, Noise, Roundness, Star-Count – Aggregate)
  │       ├── base_config           (aktuelle tile_compile.yaml, flach)
  │       ├── config_schema         (Pfad → Typ, Enum, Min, Max, Beschreibung)
  │       └── allowed_config_paths  (freigegebene Pfade für Empfehlungen)
  │
  └── agent_service (Node/TypeScript)
        ├── buildPrompt()           → strukturierter Prompt mit allen Datensektionen
        ├── Claude / Anthropic API  → JSON-Antwort (schema_version: pi.scan-analysis.v1)
        └── validate_updates_against_schema()
              ├── Gesamtvalidierung (fast path)
              └── Per-Update-Validierung (isoliert ungültige Empfehlungen)
```

---

## Prompt-Struktur

Der Prompt enthält sechs Sektionen:

| Sektion | Inhalt |
|---------|--------|
| **System-Anweisung** | Rolle, Ausgabeformat (JSON), strikte Regeln |
| **CONFIG SCHEMA** | Alle erlaubten Pfade mit `type`, `enum`, `min`, `max`, Beschreibung |
| **CURRENT CONFIG** | Aktuelle Werte als `pfad = wert` |
| **SCAN RESULT** | Frame-Metadaten (Anzahl, Kamera, Gain, Belichtung, Fehler …) |
| **IMAGE QUALITY METRICS** | Aggregate aus `scan-metrics` (FWHM, Noise, Roundness, SNR, Star-Count) |
| **FRAME STATISTICS** | Verteilung pro Frame (sofern verfügbar) |

### Strikte KI-Regeln im Prompt

- Empfohlener Wert muss sich vom aktuellen Wert unterscheiden.
- Typ muss mit Schema übereinstimmen (`integer`, `number`, `boolean`, `string`).
- `enum`-Werte: nur erlaubte Werte.
- **`min`/`max`-Grenzen: Wert MUSS im erlaubten Bereich liegen.**
- Keine `object`- oder `array`-Pfade.
- Keine Datei-/Verzeichnispfade.
- Keine CRITICAL-Warnungen für Annahmen oder allgemeine Astrofotografie-Hinweise.

---

## Validierung

### Gesamtvalidierung (fast path)

Alle empfohlenen Updates werden als Patch auf die aktuelle Config angewendet und via
`tile_compile_cli validate-config --stdin` geprüft. Bei Erfolg werden alle übernommen.

### Per-Update-Validierung (Fallback)

Schlägt der Gesamtpatch fehl, wird jedes Update **einzeln und kumulativ** geprüft:

```
für jedes Update u:
  trial = current_base + u
  validate(trial)
  → OK:   current_base = trial, u gilt als "applicable: true"
  → FAIL: u wird abgelehnt (reject_reason: "config_validation_failed")
```

Dies verhindert, dass ein einziges ungültiges Update alle anderen Empfehlungen blockiert.

### Häufige Ablehnungsgründe

| Grund | Ursache |
|-------|---------|
| `config_validation_failed` | Empfohlener Wert verletzt Schema-Constraint (z. B. `max: 16`) |
| `unknown_path` | Pfad existiert nicht im Schema |
| `type_mismatch` | Falscher Datentyp |
| `same_value` | Neuer Wert identisch mit aktuellem |

---

## Bildqualitäts-Metriken (`scan_metrics`)

Die `scan-metrics`-Phase berechnet folgende Aggregate pro Datensatz:

| Metrik | Bedeutung |
|--------|-----------|
| `fwhm` | Sternschärfe (Full Width at Half Maximum) in Pixel |
| `noise` | Hintergrundrauschen (σ) |
| `background` | Hintergrundhelligkeit (Median) |
| `roundness` | Sterndichte (1.0 = perfekt rund) |
| `star_count` | Erkannte Sterne pro Frame |

Aggregate enthalten `median`, `mean`, `std`, `min`, `max`, `p10`, `p90`.

Die KI nutzt diese Werte als **gemessene Fakten** – nicht als Annahmen – für Empfehlungen
zu `aqmh.cherry_pick`, `local_metrics`, `global_metrics.weights` und Sigma-Clip-Parametern.

### Metrik-Cache

`POST /api/scan/metrics` verwendet bereits berechnete Bildstatistiken erneut, wenn der
Cache-Key identisch ist:

```
input_path | normalisierter object_name | frame_count
```

Erfolgreiche Ergebnisse werden sowohl im Backend-Jobstore als auch auf Disk gespeichert:

```
runs/.pi_memory/scan_metrics_cache/
```

Bei einem Cache-Hit antwortet der Endpunkt sofort:

```json
{
  "cached": true,
  "state": "ok",
  "result": {
    "cache_hit": true,
    "cache_source": "job_store | disk",
    "cache_source_job_id": "...",
    "cache_key": "..."
  }
}
```

Das Frontend protokolliert dies als `Image statistics cache hit` und startet keinen neuen
`scan-metrics`-Job. Neu berechnet wird nur, wenn sich Input-Pfad, Objektname oder Frame-Anzahl
ändern.

### Sidecar-Timeouts

Backend-Aufrufe an den PI-Sidecar verwenden einen kurzen Verbindungs-Timeout und einen langen
Analyse-Timeout:

| Env-Variable | Default | Zweck |
|--------------|---------|-------|
| `AI_AGENT_CONNECT_TIMEOUT_MS` | `10000` | Schnelles Scheitern nur, wenn der Sidecar nicht erreichbar ist. |
| `AI_AGENT_ANALYSIS_TIMEOUT_MS` | `1200000` | Mindest-Timeout für `/analyze`; verhindert, dass lange Provider-Aufrufe als Sidecar-Ausfall gemeldet werden. |

`AI_SCAN_TIMEOUT_MS` bleibt der allgemeine AI-Timeout. Für `/analyze` wird aber der größere Wert
aus `AI_SCAN_TIMEOUT_MS` und `AI_AGENT_ANALYSIS_TIMEOUT_MS` verwendet.

---

## Ausgabeformat

```json
{
  "schema_version": "pi.scan-analysis.v1",
  "summary": "…",
  "confidence": 0.83,
  "detected_scenarios": ["osc_short_exposure", "large_frame_count", …],
  "recommendations": [
    {
      "id": "rec_sigma_high",
      "path": "stacking.sigma_clip.sigma_high",
      "value": 2.5,
      "current_value": 3.0,
      "confidence": 0.91,
      "review_required": false,
      "rationale": "…"
    }
  ],
  "warnings": ["…"]
}
```

Gespeichert unter `.ai_analyses/<Target>_<Datum>.json`.

---

## Konfiguration

| Env-Variable | Bedeutung | Default |
|---|---|---|
| `TILE_COMPILE_AI_AGENT_AUTOSTART` | Sidecar automatisch starten | `1` |
| `TILE_COMPILE_AI_MODEL` | KI-Modell | `claude-sonnet-4-6` |
| `TILE_COMPILE_BACKEND_SUBPROCESS_CAPTURE_BYTES` | Max. stdout für CLI-Subprozesse | `1048576` (1 MB) |
| `ANTHROPIC_API_KEY` | Anthropic API-Schlüssel (in `.env`) | – |

---

## Implementierungsdateien

| Datei | Rolle |
|---|---|
| `agent_service/src/services/frameAnalysisService.ts` | Prompt-Bau, KI-Aufruf, Traffic-Log |
| `web_backend_cpp/src/routes/ai_routes.cpp` | Schema-Export, Payload-Bau, Per-Update-Validierung |
| `web_backend_cpp/src/services/ai_service.cpp` | Sidecar-HTTP-Client |
| `web_frontend/src/app.js` | scan-metrics Abruf, Analyse-Trigger, UI-Integration |
| `tile_compile_cpp/tile_compile.schema.json` | Autoritäre Schema-Quelle (min/max/enum/desc) |

---

## Verwandte Dokumente

- [PI Parameter Studio](scan_ai_parameterstudio.md)
- [Konfigurationsreferenz](../configuration_reference.md)
- [AQMH Methodik](../AQMH/aqmh_methodik_en_v0.2.1.md)
- 🇬🇧 [English version](pi_ai_recommendations_en.md)
