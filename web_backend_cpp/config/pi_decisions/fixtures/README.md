# Jev Provider Fixtures (M0/M3)

Fixtures for the OpenRouter Alpha Decisions endpoint
(`POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`),
the verified integration path documented in
[`docs/PI/pi_jev_m0_provider_protocol_de.md`](../../../../docs/PI/pi_jev_m0_provider_protocol_de.md).
No real dataset/scan data, no real API key in any file. For
`agent_service/tests/decisionsService.test.ts` (M3, not yet implemented).

| File | Case | Verified? |
|---|---|---|
| `request_ok.json` | Valid `openrouter.decisions.request.v1` for the `enable_adaptive_weights` candidate decision | shape matches the real endpoint's accepted request |
| `response_ok.json` | 200 response selecting `keep_current` | shape from a real captured response, IDs/content synthesized |
| `response_invalid_unknown_candidate.json` | 200 body naming a `choice` value absent from the request's `criteria` -- must be rejected, never auto-mapped | synthetic, schema-derived |
| `response_invalid_nan_probability.json` | 200 body with a non-finite probability -- must reject the whole response, not clamp | synthetic, schema-derived |
| `error_400_validation.json` | Empty request body | **empirically captured**, real error text |
| `error_400_invalid_model.json` | Well-formed request, wrong (TypeSafe-direct-style) model id | **empirically captured**, real error text |
| `error_400_wrong_endpoint.json` | Correct model id posted to `chat/completions` instead of `alpha/decisions` | **empirically captured**, real error text |
| `error_401_assumed.json`, `error_429_assumed.json`, `error_5xx_assumed.json` | Auth/rate-limit/upstream failure | **NOT verified** against this endpoint -- standard OpenRouter conventions assumed; confirm before relying on them (provider-protocol doc section 6) |
| `scan_metrics_synthetic.json` | Synthetic `scan-metrics` output, shape verified against `cli_main.cpp` | for `pi_decision_state` (M1) builder tests, not the provider contract |

Every `error_*` fixture's `_fixture_note` states plainly whether its content
was actually observed from the live endpoint or is an unverified assumption
-- keep that distinction when adding more cases; do not silently upgrade an
assumed fixture to look verified without a real call to back it.
