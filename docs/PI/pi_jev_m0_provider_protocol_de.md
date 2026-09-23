# PI Jev M0 — Verifizierter Provider-Vertrag (TypeSafe / Jev via OpenRouter)

> **Stand:** 2026-09-22, zweite Fassung — korrigiert nach echtem API-Test.
> **Status:** M0-Ergebnis, empirisch verifiziert (nicht nur Doku-Abgleich).
> Deckt den Checklistenpunkt "Provider-Protokoll aus offiziellen Quellen
> verifizieren" aus
> [Implementierungsplan Abschnitt 4](pi_jev_implementierungsplan_de.md#4-m0--vertr%C3%A4ge-quellen-und-testgrundlage-einfrieren)
> ab.

## 0. Korrektur der ersten Fassung dieses Dokuments (wichtig)

Die erste Fassung (2026-09-22, früher am selben Tag) kam über reines
Doku-Fetching von `docs.typesafe.ai` zu dem Schluss, Jev laufe **nicht** über
OpenRouter, weil `openrouter.ai/typesafe/jev-1.13/` als Modellseite 404
liefert und Jev im öffentlichen `/api/v1/models`-Katalog fehlt. Das war
**falsch** — der Nutzer hat auf einen vorhandenen, funktionierenden
`.env`-Key (`JEV_OPENROUTER_API_KEY`) hingewiesen. Ein echter Testaufruf
damit belegt das Gegenteil:

- `typesafe/jev-1.13` **ist** eine gültige OpenRouter-Modell-ID, nur nicht
  für `chat/completions` nutzbar — die Fehlermeldung dazu lautet wörtlich:
  *"typesafe/jev-1.13 is a decisions model and cannot be used with the
  chat/completions endpoint. Use the /api/alpha/decisions endpoint
  instead."*
- Der Grund, warum das Modell im normalen `/models`-Katalog und auf einer
  gewöhnlichen Modell-Detailseite fehlt: es gehört zu einer separaten
  Alpha-Kategorie ("decisions model"), die außerhalb der normalen
  Chat-Modell-Auflistung geführt wird — kein Beleg dafür, dass es nicht
  existiert.
- Ein vollständiger End-to-End-Aufruf gegen `POST
  https://openrouter.ai/api/alpha/decisions` mit dem vorhandenen Key
  liefert eine reguläre, korrekt geformte Antwort (§2).

**Lektion für die eigene Vorgehensweise:** Ein 404 auf einer Marketing-/
Modellseite und ein leerer Treffer im öffentlichen Modellkatalog sind kein
Beweis für "existiert nicht" — bei einem Alpha-/Decisions-Endpunkt zählt nur
ein echter API-Aufruf. Die docs.typesafe.ai-Inhalte aus der ersten Fassung
bleiben als Hintergrund zum zugrundeliegenden Primitiv-Vertrag relevant
(§4), sind aber **nicht** der tatsächliche Integrationsweg für dieses
Projekt — der ist OpenRouter.

Alle Korrekturen aus der ersten Fassung, die "OpenRouter -> TypeSafe direkt"
umgeschrieben hatten (`pi_jev_decisions_plan_de.md` §3,
`pi_jev_implementierungsplan_de.md` M3 + Eingriffspunkte-Tabelle + §2.1),
sind in dieser Fassung **zurückgenommen**. Der ursprüngliche Stand vor der
ersten (falschen) Korrektur war in der Sache richtig: OpenRouter, mit einem
vom übrigen `OPENROUTER_API_KEY` getrennten, dediziert für Jev benannten
Key. Das ist bereits exakt so vorbereitet:
`.env` enthält `JEV_OPENROUTER_API_KEY` (Format `sk-or-v1-...`, gültiger,
aktiver OpenRouter-Key, Stand 2026-09-22 unbenutzt, Freikontingent
50/Monat).

## 1. HTTP-Vertrag (empirisch verifiziert, 2026-09-22)

```
POST https://openrouter.ai/api/alpha/decisions
Authorization: Bearer <JEV_OPENROUTER_API_KEY>
Content-Type: application/json
```

Nicht der normale `POST /api/v1/chat/completions`-Pfad — ein separater
Alpha-Endpunkt mit eigenem Vertrag.

### Request (per Zod-Validierungsfehler bei leerem Body ermittelt + per Erfolgsaufruf bestätigt)

| Feld | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `model` | string | ja | OpenRouter-Modell-ID im `vendor/model`-Format, z. B. `typesafe/jev-1.13` |
| `state` | string \| object \| array | ja | Zu bewertender Inhalt (Union-Typ, per Validierungsfehler bestätigt) |
| `questions` | object (record) | ja | Map `question_id -> Question` |

Getestete Question-Form (`type: "choice"`):

```json
{
  "type": "choice",
  "instructions": "<Freitext>",
  "criteria": {"<candidate_id>": "<Beschreibung oder null>", "...": null}
}
```

`noul`/`score` sind laut der zugrundeliegenden TypeSafe-API-Dokumentation
(`docs.typesafe.ai/api.md`, §4) ebenfalls Question-Typen; über den
OpenRouter-Alpha-Endpunkt selbst nur `choice` empirisch getestet. Vor
produktivem `noul`-Einsatz (Zielbild §3: "optionale unabhängige
`noul`-Fragen") einen eigenen Testaufruf nachholen — **nicht** ungeprüft aus
der TypeSafe-Direktdoku übernehmen, siehe §0-Lektion.

### Response (realer Aufruf, Payload redigiert — Inhalte synthetisch)

```json
{
  "model": "typesafe/jev-1.13-20260917",
  "answers": {
    "category": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 1, "other": 0, "technical": 0},
      "confidence": 1
    }
  },
  "usage": {"input_tokens": 350, "output_tokens": 38, "cost": 0.0000147},
  "id": "gen-dec-0000000000-SyntheticFixtureId00",
  "provider": "TypeSafe"
}
```

Abweichungen gegenüber der (weiterhin gültigen) TypeSafe-Direkt-API-Form aus
§4 — **wichtig für die Schema-Definition in §3**:

- `model` im Response ist **nicht** die reine TypeSafe-Versions-ID
  (`jev-1.13.0`), sondern OpenRouter-eigen datumsgestempelt
  (`typesafe/jev-1.13-<YYYYMMDD>`, hier `-20260917`). Das ist die tatsächlich
  verarbeitende Build-Kennung und erfüllt trotzdem den Zweck von
  `model_reported` in `pi.decisions.response.v1` — nur das Format
  unterscheidet sich vom direkten TypeSafe-Vertrag.
- Zusätzliche, bei OpenRouter übliche Felder: `usage.cost` (USD,
  Kostentransparenz pro Aufruf — relevant für M3s Kostenkontrolle-Punkt),
  `id` (OpenRouter-Generation-ID, `gen-dec-...`-Präfix für Decisions-Calls),
  `provider` (expliziter Upstream-Provider-Name, hier `"TypeSafe"`).
- Kein `usage.input_tokens`/`output_tokens`-Abweichung zur Direkt-API —
  gleiche Feldnamen.

### Fehlerverhalten (teilweise empirisch, teilweise aus Zod-Validierung abgeleitet)

- Fehlerform durchgängig `{"error": {"message": "<string>", "code": <int>}}`
  — `error.message` ist **immer ein String**, auch im Validierungsfehlerfall;
  dort ist der Stringinhalt selbst JSON-formatierter Text (ein
  Zod-Fehlerarray als eingebettetes, noch einmal zu parsendes JSON, nicht ein
  natives verschachteltes Array-Feld). Eigenes Detail des OpenRouter-Proxys,
  im Adapter zu berücksichtigen (M3: "Fehler in stabile Anwendungscodes
  übersetzen" — ein einfacher `error.message`-Read reicht als Rohtext, ein
  zweiter `JSON.parse`-Versuch ist optional für strukturierte
  Feldreferenzen, niemals Voraussetzung für die Fehlerbehandlung selbst).
- Falsches `model`-Chat-Pfad-Modell (`chat/completions` statt
  `alpha/decisions`) -> eigener, klarer `400`-Fehlertext (§0-Zitat).
- 401/429/5xx: **nicht separat gegen den Alpha-Endpunkt getestet** (kein
  ungültiger Key verfügbar, Rate Limit nicht ausgelöst). Angenommen: Standard-
  OpenRouter-HTTP-Semantik (401 ungültiger Key, 429 Rate Limit, 5xx
  Providerausfall) — vor M3-Freigabe mit einem absichtlich ungültigen Key
  nachzuholen, nicht ungeprüft übernehmen.

## 2. Modellkennung und Pinning

`typesafe/jev-1.13` ist die vom Alpha-Endpunkt akzeptierte ID. Ob OpenRouter
zusätzlich Alias-Varianten (`jev-latest` o. ä.) für denselben Endpunkt
anbietet, wurde **nicht getestet** — nur `typesafe/jev-1.13`,
`typesafe/jev-1.13.0`, `typesafe/jev-latest`, `typesafe/jev` wurden probiert
(nur die erste war gültig; die anderen drei ergaben `"is not a valid model
ID"`). Für M3s "gepinnte Modellkennung" ist `typesafe/jev-1.13` damit die
korrekte, tatsächlich funktionierende Referenz. Das `model`-Feld der
Response (`typesafe/jev-1.13-<Datum>`) zusätzlich protokollieren, um
Build-Drift innerhalb dieser Versionslinie sichtbar zu machen — auch ohne
separate Alias-Ebene.

## 3. Wire-Format-Schemas dieses M0-Schritts (korrigiert)

- `web_backend_cpp/config/pi_decisions/schemas/openrouter.decisions.request.v1.schema.json`
- `web_backend_cpp/config/pi_decisions/schemas/openrouter.decisions.response.v1.schema.json`
- `web_backend_cpp/config/pi_decisions/schemas/openrouter.decisions.error.v1.schema.json`

Ersetzen die in der ersten Fassung angelegten
`typesafe.systemone.request/response.v1.schema.json` (TypeSafe-Direktvertrag)
als **primäre**, tatsächlich zu verwendende Wire-Schemas. Die
TypeSafe-Direktschemas bleiben als Dokumentation des zugrundeliegenden
Primitiv-Vertrags im Repo (§4), sind aber nicht der Integrationsweg.

## 4. Zugrundeliegender TypeSafe-Primitiv-Vertrag (Hintergrund, nicht der Integrationsweg)

`docs.typesafe.ai` beschreibt denselben `choice`/`score`/`noul`-Vertrag,
`confidence`/`probabilities`-Semantik (siehe Zitate unten) und die
Versionierungsphilosophie (Alias vs. gepinnte Version) — inhaltlich
weiterhin nützlich, weil OpenRouter erkennbar ein dünner Proxy auf genau
diese Primitive ist (identische `answers.<id>.{type,choice,probabilities,
confidence}`-Form). Nur der tatsächliche Netzwerkpfad in diesem Projekt ist
OpenRouter, nicht `api.typesafe.ai` direkt.

Herstellerzitat zu Confidence (deckt sich mit Zielbild §3): *"The correct
threshold values depend on your domain and the performance of the model for
your use case. Start with conservative thresholds, test with your own data,
and adjust as you observe results."* M5 muss eigene Schwellen anhand realer
Paarvergleiche festlegen; `confidence` allein ist kein Freigabekriterium.

## 5. SDK-Frage, korrigiert: `@earendil-works/pi-ai` geprüft und bewusst nicht wiederverwendet

Für den OpenRouter-Alpha-Endpunkt existiert kein bekanntes offizielles SDK
(anders als `@typesafe-ai/sdk` für die TypeSafe-Direkt-API, die hier nicht
der Integrationsweg ist). Zusätzlich zu prüfen (Nutzerhinweis, 2026-09-23):
"PI" in diesem Projekt ist über `agent_service`s Abhängigkeit
`@earendil-works/pi-coding-agent` transitiv auf `@earendil-works/pi-ai`
aufgebaut (bestätigt: `agent_service/node_modules/@earendil-works/pi-ai`
liegt real vor; dessen `providers/`-Verzeichnis listet exakt dieselben
~30 Provider-IDs wie `agent_service/src/config.ts`s
`redactedEnvSources()`-Allowlist — kein Zufall, sondern derselbe
Provider-Katalog).

Geprüft, ob dessen bestehender `openrouterProvider()` für den
Decisions-Aufruf wiederverwendbar ist
(`dist/providers/openrouter.js`, Paketversion 0.80.10):

```js
export function openrouterProvider() {
  return createProvider({
    id: "openrouter", name: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    auth: { apiKey: envApiKeyAuth("OpenRouter API key", ["OPENROUTER_API_KEY"]) },
    models: Object.values(OPENROUTER_MODELS),
    api: openAICompletionsApi(),
  });
}
```

**Ergebnis: nicht direkt wiederverwendbar, bewusst nicht versucht.**
`api: openAICompletionsApi()` ist fest auf Chat-Completions verdrahtet
(Streaming, `messages`-Array als `Context`, inkrementelle Tokens). Zwar ist
`createProvider`s `api`-Feld strukturell pluggable (eine
`byApi`-Dispatch-Map wäre möglich), aber der Decisions-Vertrag aus §1
(`state`/`questions` rein, ein atomares JSON-Objekt zurück, kein Streaming,
kein Message-Array) passt konzeptionell nicht auf das
Context/Stream-Modell, das die gesamte Bibliothek voraussetzt (Zweck laut
eigener Beschreibung: *"context persistence and hand-off to other models
mid-session"* — für einen einzelnen atomaren Entscheidungsaufruf ohne
Sitzungskontext irrelevant). Eine Anpassung würde mehr Komplexität durch
Fehlanpassung erzeugen, als sie an Wiederverwendung spart.

**Was doch übernommen wird:** das Auth-Resolution-Muster
(`dist/auth/helpers.js::envApiKeyAuth`, gestützt Credential zuerst, dann
Env-Var-Fallback) ist simpel (~10 Zeilen) und deckt sich mit der Konvention,
die `pi_storage_paths.hpp` an anderer Stelle im Projekt bereits etabliert.
Nicht importierbar (`auth/helpers.js` ist intern, kein Re-Export über
`dist/index.js`/das Package-`exports`-Feld) — `decisionsService.ts`
repliziert dasselbe kleine Muster lokal (`JEV_OPENROUTER_API_KEY` zuerst aus
gespeichertem Credential, dann Env-Var), statt einen internen, nicht
unterstützten Tiefpfad zu importieren.

Damit bleibt die SDK-vs-HTTP-Frage bei demselben Ergebnis wie in der ersten
Fassung, jetzt aber gegen die tatsächlich vorhandene Bibliothek geprüft
statt nur gegen ein fehlendes offizielles TypeSafe-SDK für diesen Endpunkt:
**eigener, injizierbarer HTTP-Client in `decisionsService.ts`**, mit dem
Auth-Muster von `@earendil-works/pi-ai` als Vorbild, nicht als Abhängigkeit.

## 6. Offene Punkte vor M3-Start

- 401/429/5xx-Verhalten des Alpha-Endpunkts real testen (nur mit bewusst
  ungültigem Key/absichtlicher Überlast, nicht im Rahmen von M0 gemacht).
- `noul`-Question-Typ real gegen den Alpha-Endpunkt testen, nicht nur aus
  der TypeSafe-Direktdoku übernehmen.
- Klären, ob `JEV_OPENROUTER_API_KEY` das für den Produktivbetrieb
  vorgesehene Secret ist oder ein Test-/Entwicklungs-Key (Freikontingent,
  50/Monat) — Implementierungsplan M3 sieht "bestehenden Secret-Mechanismus"
  vor, ohne bereits zwischen Dev- und Prod-Key zu unterscheiden.
