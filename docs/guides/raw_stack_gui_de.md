# Raw Stack GUI

> **Hinweis:** Diese Funktion ist nicht optimiert und wird nur aus historischen Gründen vorgehalten.

Die Raw-Stack-Seite bietet eigenständige Vorverarbeitung von FITS-Light-Frames bis zum finalen gestackten und nachbearbeiteten Bild, vollständig getrennt vom normalen Tile-Compile-Run-Studio.

## Pipeline-Phasen

Kalibrierung → CFA/Mono-Vorbereitung → Registrierung → Qualitätsanalyse → Frame-Filterung → Stacking (Sigma/Median/Winsor) → Astrometrie (ASTAP, danach lokaler Gaia-DR3-Fallback) → BGE → PCC → HyperMetric Stretch

Alle konfigurierbaren Parameter (Sigma-Clip, Rejection-Methode, Stacking-Gewichtung, BGE, PCC, Astrometrie und HyperMetric Stretch) werden direkt aus der Parameter-Studio-Konfiguration übernommen — keine fest codierten Werte.

## Dokumentation

- Englisch: [Raw Stack GUI (EN)](../raw_stack_gui_en.md)
- Deutsch: [Raw Stack GUI (DE)](../raw_stack_gui_de.md)
