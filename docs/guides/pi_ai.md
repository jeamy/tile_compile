# PI – AI-Assisted Configuration Recommendations

The PI (Parameter Intelligence) module uses an AI sidecar to analyse scan results and generate validated parameter recommendations directly in Parameter Studio.

## How it works

1. **Scan metrics** — Frame quality metrics (FWHM, noise, background, roundness, star count) from `scan-metrics` are passed to the AI as measured facts.
2. **Schema constraints** — The AI receives all relevant configuration parameters with descriptions and the complete schema constraints (`min`, `max`, `enum`).
3. **Session context** — Session geometry (mount type, field rotation estimate, session duration) is forwarded alongside scan metrics.
4. **Validated output** — The AI produces data-driven configuration recommendations. Per-update validation ensures only valid recommendations are applied.

## Documentation

- Full documentation: [PI AI Recommendations](../PI/pi_ai_recommendations_en.md)
- German version: [PI KI-Empfehlungen](../PI/pi_ki_empfehlungen_de.md)
