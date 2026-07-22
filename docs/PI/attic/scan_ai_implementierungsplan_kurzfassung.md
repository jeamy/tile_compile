# PI Scan-AI - Kurzfassung des Implementierungsplans

**Status:** Kurzfassung des Implementierungsplans  
**Ziel:** PI Scan-AI optional in Scan und Parameter Studio integrieren.  
**Wichtigster Punkt:** AI ist default aus und benoetigt explizite Aktivierung plus Credentials.

---

## Kernarchitektur

Die PI-Integration wird als lokaler Node/TypeScript-Sidecar umgesetzt, weil `@earendil-works/pi-coding-agent` dort direkt genutzt werden kann:

```text
Frontend -> C++ Backend -> Node PI Sidecar -> @earendil-works/pi-coding-agent -> externe AI
```

Das C++ Backend bleibt fuer Scan, Jobs, Config-Patches und Validierung verantwortlich. Der Sidecar uebernimmt Provider-/Modellverwaltung, AuthStorage, `.env` und die eigentliche PI-Agent-Session.

---

## Was bereits vorhanden ist

| Bereich | Vorhanden |
|---|---|
| Scan | `/api/scan`, `/api/scan/latest`, JobStore |
| Config | `/api/config/schema`, `/api/config/defaults`, `/api/config/patch`, `/api/config/validate` |
| Revisions | `ConfigRevisionStore` |
| Frontend | `web_frontend/src/api.js`, `src/app.js`, `input-scan.html`, `parameter-studio.html` |
| Validierung | CLI/Backend Config-Validate |

---

## Feste Vorgaben

- AI ist default aus.
- Externe Requests passieren nur nach expliziter Aktivierung oder bestaetigtem One-shot.
- API-Keys kommen aus PI `AuthStorage`, Prozess-Environment, `agent_service/.env` oder Projektwurzel `.env`.
- Secrets werden nie in Config, Revisions, JobStore, Logs oder UI-State geschrieben.
- Alle PI-Provider/Modelle kommen aus `ModelRegistry`.
- Intern wird `updates: [{path, value}]` als kanonisches Patch-Format verwendet.
- Jede AI-Empfehlung wird gegen Schema und Config-Validierung geprueft.

---

## Ziel-Dateien

### Neu: Sidecar

```text
agent_service/
  package.json
  tsconfig.json
  .env.example
  src/
    server.ts
    types.ts
    config.ts
    services/
      authService.ts
      modelService.ts
      frameAnalysisService.ts
```

### Neu: C++ Backend

```text
web_backend_cpp/include/routes/ai_routes.hpp
web_backend_cpp/src/routes/ai_routes.cpp
web_backend_cpp/include/services/ai_service.hpp
web_backend_cpp/src/services/ai_service.cpp
web_backend_cpp/tests/test_ai_routes.cpp
web_backend_cpp/tests/test_ai_service.cpp
```

### Zu aendern

```text
web_backend_cpp/src/main.cpp
web_backend_cpp/CMakeLists.txt
web_backend_cpp/include/backend_runtime.hpp
web_backend_cpp/src/backend_runtime.cpp
web_frontend/src/constants.js
web_frontend/src/api.js
web_frontend/src/app.js
web_frontend/input-scan.html
web_frontend/parameter-studio.html
web_frontend/i18n/de.json
web_frontend/i18n/en.json
web_frontend/style.css
web_frontend/layout-panels.css
.gitignore
```

---

## Neue APIs

### C++ Backend

| Route | Zweck |
|---|---|
| `GET /api/ai/config` | AI Settings ohne Secrets lesen |
| `PATCH /api/ai/config` | AI Settings ohne Secrets speichern |
| `GET /api/ai/models` | PI Provider/Modelle/Auth-Status lesen |
| `POST /api/ai/auth` | API-Key fuer Provider speichern |
| `POST /api/ai/test` | Verbindung/Modell testen |
| `DELETE /api/ai/auth/<provider>` | gespeicherten Key entfernen |
| `POST /api/scan/analysis` | Scan analysieren |
| `GET /api/scan/analysis/latest` | letzte Analyse abrufen |
| `POST /api/scan/analysis/apply` | validierte Empfehlungen anwenden |

### Node Sidecar

| Route | Zweck |
|---|---|
| `GET /health` | Sidecar Health |
| `GET /models` | PI ModelRegistry |
| `POST /auth` | Key in PI AuthStorage speichern |
| `DELETE /auth/:provider` | gespeicherten Key entfernen |
| `POST /test` | Testprompt |
| `POST /analyze` | Scan-AI Analyse |

---

## AI-Konfiguration

Default:

```yaml
ai:
  scan_analysis:
    enabled: false
    mode: manual
    provider: ""
    model: ""
    temperature: 0
    max_tokens: 8000
    timeout_ms: 120000
    send_paths: false
    persist_recommendations: false
```

Credential-Prioritaet:

1. PI `AuthStorage`
2. Prozess-Environment
3. `agent_service/.env`
4. Projektwurzel `.env`

Typische `.env`-Keys:

```dotenv
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
```

Secrets werden nie in Config, Revisions, JobStore, Logs oder UI-State geschrieben.

---

## Implementierungsroadmap

### Phase 0: Setup

- Node-Version festlegen.
- `agent_service` anlegen.
- `@earendil-works/pi-coding-agent` einbinden.
- Sidecar-Port `127.0.0.1:3001` als Default.

### Phase 1: Sidecar

- `/health`
- `.env` laden
- PI `AuthStorage`
- PI `ModelRegistry`
- `/models`, `/auth`, `/test`

### Phase 2: Analyse

- `FrameAnalysisService`
- PI Session mit `createAgentSession`
- Prompt + JSON-only Contract
- `/analyze`

### Phase 3: C++ Backend

- `ai_routes`
- `ai_service` als Sidecar-Client
- `/api/ai/*`
- `/api/scan/analysis*`
- Fehler bei fehlendem Sidecar strukturiert melden

### Phase 4: Validierung

- AI Response ist untrusted.
- Nur bekannte Schema-Pfade erlauben.
- Kanonisch `updates: [{path, value}]`.
- Patch gegen Config-Schema und `validate-config` pruefen.
- Nur validierte Updates anwendbar machen.

### Phase 5: Frontend

- AI Settings Panel.
- Provider/Modell-Auswahl.
- API-Key-Eingabe/Test.
- Scan-Analysepanel.
- Empfehlungen mit Confidence/Risiko/Warnungen.
- Apply ausgewählter Empfehlungen.

### Phase 6: Tests

- Default-off.
- `.env` erkannt.
- Keine Secret-Leaks.
- Sidecar unavailable.
- AI disabled.
- Unknown path rejected.
- Wrong type rejected.
- Apply erzeugt Revision `pi_scan_ai`.

---

## Must-have Akzeptanzkriterien

- [ ] Scan funktioniert unveraendert ohne AI.
- [ ] AI ist default aus.
- [ ] Keine externen Requests ohne Aktivierung/Credentials.
- [ ] Alle PI-Modelle aus `ModelRegistry` koennen angezeigt werden.
- [ ] API-Keys aus AuthStorage, Environment oder `.env` werden erkannt.
- [ ] Secrets werden nirgends ausgegeben oder persistiert.
- [ ] `/api/scan/analysis` liefert validierte Empfehlungen.
- [ ] Ungueltige Empfehlungen werden verworfen.
- [ ] Apply erzeugt validierte Config und optional Revision.
- [ ] Parameter Studio zeigt Confidence, Risiko, Begruendung und Warnungen.

---

## Wichtigste Risiken

| Risiko | Schutz |
|---|---|
| Unerwartete externe AI-Kosten | `enabled=false`, `mode=manual`, sichtbare UI-Aktion |
| Secret-Leak | Redaction, keine Secret-Responses, Tests |
| Ungueltige AI-Config | Schema-Filter + Config-Validate |
| Sidecar down | Backend bleibt nutzbar |
| Falsches Patch-Format | nur `updates[]` intern verwenden |
| Absolute Pfade an AI | `send_paths=false` default |

---

## Rollback

1. AI deaktivieren.
2. Sidecar nicht starten.
3. Optional `register_ai_routes(...)` entfernen.
4. Scan, Config und Parameter Studio laufen weiter wie bisher.

---

**Vollstaendiger Plan:** [`scan_ai_implementierungsplan.md`](./scan_ai_implementierungsplan.md)  
**Grundlagen-Spezifikation:** [`scan_ai_parameterstudio.md`](../scan_ai_parameterstudio.md)
