# Jev Provider Fixtures (M0/M3)

Redacted request/response fixtures for `agent_service/tests/decisionsService.test.ts`
(M3, not yet implemented). Wire format verified against
[`docs/PI/pi_jev_m0_provider_protocol_de.md`](../../../../docs/PI/pi_jev_m0_provider_protocol_de.md)
on 2026-09-22 — synthetic content, no real dataset/scan data, no real API key.

| File | Case |
|---|---|
| `request_ok.json` | Valid `typesafe.systemone.request.v1` for the `enable_adaptive_weights` candidate decision |
| `response_ok.json` | Valid 200 response selecting `keep_current` |
| `response_invalid_unknown_candidate.json` | 200 body naming a `choice` value absent from the request's `criteria` -- must be rejected, never auto-mapped |
| `response_invalid_nan_probability.json` | 200 body with a non-finite probability -- must reject the whole response, not clamp |
| `error_401.json`, `error_422.json`, `error_429.json`, `error_529.json` | Provider error bodies for each documented status code |
| `scan_metrics_synthetic.json` | Synthetic `scan-metrics` output (M0 checklist "lokale Fixtures"), shape verified against `cli_main.cpp`; for `pi_decision_state` (M1) builder tests. Deliberately includes an `ok=false` frame and a frame with `fwhm<=0`/`roundness<=0` (dropped from the aggregate, not zeroed) to exercise the missing-value paths in [pi_jev_m0_field_inventory_de.md](../../../../docs/PI/pi_jev_m0_field_inventory_de.md). |

Each `error_*.json` pairs an HTTP status (in the filename) with whatever body
shape the fixture assumes for that status; M3's actual adapter test decides
how much of the body it trusts (docs/PI/pi_jev_m0_provider_protocol_de.md
section 2: provider JSON is untrusted input even on an error path).
