# PI – KI-gestützte Konfigurationsempfehlungen

Das PI-Modul (Parameter Intelligence) verwendet einen AI-Sidecar, um Scan-Ergebnisse zu analysieren und validierte Parameterempfehlungen direkt im Parameter Studio zu erzeugen.

## Funktionsweise

1. **Scan-Metriken** — Frame-Qualitätsmetriken (FWHM, Noise, Hintergrund, Rundheit, Sternanzahl) aus `scan-metrics` werden der KI als gemessene Fakten übergeben.
2. **Schema-Constraints** — Die KI erhält alle relevanten Konfigurationsparameter mit Beschreibungen und den vollständigen Schema-Constraints (`min`, `max`, `enum`).
3. **Session-Kontext** — Sitzungsgeometrie (Montierungstyp, Feldrotations-Schätzung, Session-Dauer) wird neben den Scan-Metriken weitergegeben.
4. **Validierte Ausgabe** — Die KI erzeugt datengetriebene Konfigurationsempfehlungen. Per-Update-Validierung stellt sicher, dass nur gültige Empfehlungen angewendet werden.

## Dokumentation

- Vollständige Dokumentation: [PI KI-Empfehlungen](../PI/pi_ki_empfehlungen_de.md)
- Englische Version: [PI AI Recommendations](../PI/pi_ai_recommendations_en.md)
