
## Abgrenzung zu etablierten Software-Pipelines

### PixInsight

PixInsight bietet eine modulare Pipeline mit leistungsfähigen Werkzeugen wie *SubframeSelector*, *ImageIntegration* und *LocalNormalization*. Die Qualitätsbewertung erfolgt primär framebasiert, optional ergänzt durch lokale Normalisierung, die jedoch als photometrische Korrektur und nicht als physikalische Qualitätsmodellierung konzipiert ist.

Im Gegensatz dazu:
- keine explizite Trennung globaler und lokaler Qualitätsachsen,
- keine seeing-adaptive Tile-Geometrie,
- lokale Normalisierung wirkt auf Intensitäten, nicht auf Gewichtung.

Die hier vorgestellte Methodik kann als physikalisch motivierte Verallgemeinerung des gewichteten ImageIntegration-Prozesses verstanden werden.

---

### Astro Pixel Processor (APP)

Astro Pixel Processor verfolgt einen stark automatisierten Ansatz mit integrierter Normalisierung, Registrierung und Stacking. Qualitätsbewertungen sind überwiegend global (SNR, Hintergrund, FWHM) und dienen primär der Frame-Gewichtung oder -Ausschlussentscheidung.

Abgrenzung:
- keine lokale Qualitätsrekonstruktion,
- keine Tile-basierte Gewichtung,
- implizite Frame-Selektion durch Gewichtsschwellen.

Die tile-basierte Methode ersetzt diese globale Heuristik durch kontinuierliche, lokale Gewichtsfelder.

---

### Siril

Siril ist eine lineare, reproduzierbare Pipeline mit Fokus auf klassische astronomische Verarbeitungsschritte. Frame-Gewichtungen sind möglich, jedoch ausschließlich global. Lokale Analysefunktionen dienen der Diagnose, nicht der Rekonstruktion.

Abgrenzung:
- streng framebasierte Gewichtung,
- keine lokale Rekonstruktion,
- keine seeing-abhängige Adaptivität.

Die Referenzmethodik erweitert den linearen Siril-Ansatz um eine zusätzliche räumliche Dimension der Qualitätsmodellierung.
