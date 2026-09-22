# PI Jev M0 — Verifizierter Provider-Vertrag (TypeSafe / Jev)

> **Stand:** 2026-09-22, verifiziert per Live-Abruf von `docs.typesafe.ai`.
> **Status:** M0-Teilergebnis; deckt den Checklistenpunkt "Provider-Protokoll aus
> offiziellen Quellen verifizieren" aus
> [Implementierungsplan Abschnitt 4](pi_jev_implementierungsplan_de.md#4-m0--vertr%C3%A4ge-quellen-und-testgrundlage-einfrieren)
> ab. Live-Verifikation, ausdrücklich getrennt von den Mock-Fixtures unter
> `web_backend_cpp/config/pi_decisions/fixtures/`.

## 1. Korrektur gegenüber den ursprünglichen Referenzen

[Zielbild §3](pi_jev_decisions_plan_de.md#3-jev-vertrag-und-aussagegrenzen) zitierte
drei Links als Integrationsreferenz. Beim Live-Abruf (2026-09-22):

| Referenz | Ergebnis |
|---|---|
| `https://openrouter.ai/labs/jev/compile` | Erreichbar, aber reines Demo-/Rezept-Beispiel ("Prompt to questions"), keine API-Dokumentation. |
| `https://openrouter.ai/typesafe/jev-1.13/` | **HTTP 404.** Jev ist auf OpenRouter nicht als eigenständige, buchbare Modellseite gelistet. |
| `https://typesafe.ai/blog/introducing-system-one-models-and-jev` | Erreichbar, reiner Marketing-Announcement-Post ohne technischen Vertrag; verweist selbst auf `docs.typesafe.ai`. |

**Korrigierter Befund:** Jev ist **kein über OpenRouter geroutetes Modell**,
sondern ein eigenständiges Produkt von TypeSafe mit eigener, direkter HTTP-API
unter `api.typesafe.ai`. Die OpenRouter-Seiten sind Marketing-/Cross-Promotion,
keine Integrationsquelle. Jede Erwähnung von "unabhängiger OpenRouter-Key" in
[Implementierungsplan M3](pi_jev_implementierungsplan_de.md#7-m3--jev-adapter-fehlerbehandlung-und-betriebsmodus)
und [Zielbild §3](pi_jev_decisions_plan_de.md#3-jev-vertrag-und-aussagegrenzen)
ist entsprechend zu lesen als: eigener, unabhängiger **TypeSafe**-Key
(`TYPESAFE_API_KEY`), nicht OpenRouter. Beide Dokumente sind an den
betroffenen Stellen aktualisiert (2026-09-22).

Maßgebliche, tatsächlich technische Quelle: `https://docs.typesafe.ai/api.md`
(HTTP-Referenz), `https://docs.typesafe.ai/models.md` (Versionierung),
`https://docs.typesafe.ai/confidence.md` (Confidence-Semantik),
`https://docs.typesafe.ai/sdk/javascript.md` (TS/JS-SDK).

## 2. HTTP-Vertrag

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>
Content-Type: application/json
```

### Request

| Feld | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `state` | string \| object \| array | ja | Zu bewertender Inhalt — bei uns die Allowlist-Projektion aus `pi.decision-state.v1` |
| `model` | string | ja | Gepinnte Version, siehe §3 — **nicht** `jev-latest` in freigegebener Policy |
| `questions` | map<string, Question> | ja | Eine Frage pro Kandidatenentscheidung/Noul-Frage |

Question-Typen (jede mit `type` + `instructions`):

- **`choice`**: `criteria` ist eine map<string, string\|object\|array\|null> mit
  max. 255 Optionen — das ist unser Kandidaten-Auswahlfeld
  (`allowed_candidates` aus `pi.decisions.request.v1`).
- **`noul`** (ja/nein): optionale `criteria.true`/`criteria.false`-Beschreibung
  — Kandidat für die in den Plänen erwähnten "unabhängigen `noul`-Fragen".
- **`score`**: `criteria` ein Array mit 2-10 Stufen — im aktuellen Scope nicht
  genutzt (Jev liefert bei uns nur eine Kandidaten-`choice`, keine Ratingskala).

### Response

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "<question_id>": {
      "type": "choice",
      "choice": "<candidate_id>",
      "probabilities": {"<candidate_id>": 0.0},
      "confidence": 0.0
    }
  },
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

`answers.<id>.model`-Feld existiert **nicht** pro Antwort; die tatsächlich
verarbeitende Version steht ausschließlich im Top-Level-`model`-Feld. Das
deckt sich mit dem in `pi.decisions.response.v1.schema.json` vorgesehenen
`model_reported`.

### Fehlercodes

| HTTP-Status | Bedeutung | Mapping auf `pi.decisions.response.v1.status` |
|---|---|---|
| 401 | Fehlender/ungültiger Key | `unavailable` |
| 422 | Request-Body-Validierung fehlgeschlagen | `invalid_response` (unser Request war falsch geformt — Programmfehler, kein Providerausfall) |
| 429 | Rate Limit überschritten | `unavailable`; laut Doku Backoff, Limits "ändern sich ohne Ankündigung" — kein fester Wert im Adapter hartcodierbar |
| 529 | TypeSafe temporär überlastet | `unavailable` |

## 3. Modellversionierung

| Kennung | Bedeutung |
|---|---|
| `jev-1.13.0` | Gepinnte Version — **für freigegebene Policy zu verwenden** (Implementierungsplan M3: "Aliaswechsel invalidiert Kalibrierungsfreigabe") |
| `jev-latest` | Alias, zeigt aktuell auf `jev-1.13.0`; wandert bei neuem Release automatisch |
| `jev-preview` | Alias, aktuell identisch mit `jev-latest`; kein separater Preview-Build vorhanden |

Herstellerzitat (Calibration-Hinweis, deckt sich mit unserer bestehenden
Vorgabe "wenn Confidence-Schwellen kalibriert wurden, die Versions-ID pinnen,
nicht den Alias"): *"If you have tuned confidence thresholds against a
specific version, pin that version's ID instead of the alias and move to the
new one on your own schedule."*

## 4. Confidence/Probabilities — Herstellerposition

Verifiziert deckungsgleich mit der bestehenden Vorgabe in
[Zielbild §3](pi_jev_decisions_plan_de.md#3-jev-vertrag-und-aussagegrenzen)
("Anbieter-Confidence gilt ohne lokale Evaluation nicht als astrophotografisch
kalibriert"):

- `confidence` ist eine vom Anbieter aus `probabilities` abgeleitete
  Konzentrationsmetrik, keine domänenspezifisch kalibrierte Erfolgsschätzung.
- Herstellerzitat: *"The correct threshold values depend on your domain and
  the performance of the model for your use case. Start with conservative
  thresholds, test with your own data, and adjust as you observe results."*
- Konsequenz unverändert: M5 muss eigene Schwellen anhand realer,
  astrophotografischer Paarvergleiche festlegen; `confidence` allein ist kein
  Freigabekriterium.

## 5. SDK-Frage (neu, für M3 zu entscheiden)

Es existiert ein offizielles `@typesafe-ai/sdk`-NPM-Paket (`TypeSafeClient`,
`systemOne()`-Methode, `choice()`-Helper). Der Implementierungsplan (M3)
verlangt einen "injizierbaren HTTP-Transport" für Tests — mit dem offiziellen
SDK ist unklar, ob dessen Transport-Layer testfreundlich injizierbar ist oder
ob `decisionsService.ts` stattdessen direkt gegen die rohe HTTP-Schnittstelle
aus §2 implementiert werden sollte (mehr Kontrolle über Timeouts/Retries/
Fake-Transport, aber Pflege eines eigenen Wire-Client). **Offene
Entscheidung, vor M3-Start zu treffen; hier nicht vorweggenommen.**

## 6. Artefakte dieses M0-Schritts

- `web_backend_cpp/config/pi_decisions/schemas/typesafe.systemone.request.v1.schema.json`
- `web_backend_cpp/config/pi_decisions/schemas/typesafe.systemone.response.v1.schema.json`
- `web_backend_cpp/config/pi_decisions/fixtures/` — redigierte Beispiel-Request/
  -Response/-Fehlerpayloads für die Fake-Transport-Tests aus M3.

Diese Wire-Format-Schemas sind bewusst getrennt von
`pi.decisions.request.v1`/`pi.decisions.response.v1`: Letztere sind unser
Anwendungsvertrag, den `decisionsService.ts` explizit in den hier
dokumentierten Provider-Vertrag übersetzt (Implementierungsplan §3.3) — keine
Weiterleitung des Providerformats ungefiltert an den State-Builder.
