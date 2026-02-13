# Abweichungsanalyse: C++ Implementierung vs. Spezifikation (Methodik v3)

**Datum:** 2026-02-12
**Erstellt durch:** Automatische Codeanalyse
**Referenzversion:** Methodik v3.1 (2026-01-09)

Diese Diagnose beschreibt Abweichungen zwischen der C++ Implementierung in `tile_compile_cpp` und der Spezifikation in `tile_basierte_qualitatsrekonstruktion_methodik_v_3.md`, mit besonderem Fokus auf technische Verbesserungen und fortschrittliche Methoden in der Implementierung.

## 1. Zusammenfassung

Die Implementierung folgt in weiten Teilen der Spezifikation, geht jedoch in mehreren Bereichen über diese hinaus und verwendet teilweise fortschrittlichere Methoden. Gleichzeitig gibt es einige unvollständige Aspekte, die noch implementiert werden müssen, um vollständig spezifikationskonform zu sein.

## 2. Überlegene Aspekte der Implementierung

### 2.1 Fortschrittliche Registrierung

**Status:** Erheblich fortschrittlicher als spezifiziert

**Technische Überlegenheit:**
- Implementierung eines **mehrstufigen Registrierungssystems** mit automatischen Fallbacks
- Kombination **mehrerer Registrierungsalgorithmen** anstelle einer einzelnen Methode:
  - `hybrid_phase_ecc`: Kombiniert Phasenkorrelation mit ECC (Enhanced Correlation Coefficient)
  - `robust_phase_ecc`: Robuste Version mit Ausreißererkennung
  - `triangle_star_matching`: Geometrische Übereinstimmung von Sternmustern
  - `star_registration_similarity`: RANSAC-basierte robuste Registrierung
  - `feature_registration_similarity`: Feature-basierte Registrierung (z.B. AKAZE)

**Vorteile:**
- Deutlich höhere Erfolgsrate bei schwierigen Bildern
- Bessere Robustheit gegen Bildartefakte
- Automatische Auswahl der besten Methode für jedes Bildpaar

### 2.2 CFA-aware Transformation

**Status:** Methodisch sauberer als einfaches Debayer+Warp

**Technische Überlegenheit:**
- Implementierung der `warp_cfa_mosaic_via_subplanes`-Methode, die das Bayer-Muster während der Transformation erhält
- Separate Behandlung der R/G/G/B-Subplanes, wodurch Farbinterpolationsartefakte vermieden werden
- Präzise Phasenkohärenz durch Berücksichtigung der Bayer-Offsets

**Vorteile:**
- Verhindert Farbfehler und Moiré-Artefakte an Sternrändern
- Erhält die ursprüngliche Sensorauflösung ohne Interpolationsartefakte
- Methodisch korrekter als die konventionelle Debayer-dann-Warp Strategie

### 2.3 Qualitätsbasierte Referenzauswahl

**Status:** Erweitert und optimiert

**Technische Überlegenheit:**
- Anstelle einer einfachen zeitlichen Mittenwahl verwendet die Implementierung qualitätsbasierte Referenzauswahl
- Mehrere Auswahlkriterien implementiert:
  - Globales Qualitätsgewicht (wie in Spezifikation)
  - Gradientenenergie (für strukturreiche Referenzen)
  - Rauscharmut
  - NCC-Stabilität (Normalized Cross Correlation)

**Vorteile:**
- Deutlich bessere Registrierungsergebnisse durch optimale Referenz
- Höhere Wahrscheinlichkeit erfolgreicher Registrierung aller Frames

### 2.4 Wiener-Filter für Tiles

**Status:** Zusätzliche Funktionalität, nicht in Spezifikation

**Technische Überlegenheit:**
- Implementierung eines adaptiven Wiener-Filters (`wiener_tile_filter`) für Rauschunterdrückung auf Tile-Ebene
- Frequenzraum-basierte Rauschunterdrückung mit SNR-Adaption
- Selektive Anwendung nur auf relevante Tiles (Struktur-Tiles, nicht Stern-Tiles)

**Vorteile:**
- Verbesserte Detailerhaltung bei gleichzeitiger Rauschunterdrückung
- Adaptive Anpassung an lokale Signal- und Rauschcharakteristik
- Keine Überglättung von Sternbereichen

### 2.5 Robuste Statistik

**Status:** Durchgängig implementiert, robuster als Spezifikation

**Technische Überlegenheit:**
- Konsequente Verwendung von robusten statistischen Methoden:
  - MAD-basierte Rauschschätzung (Median Absolute Deviation)
  - Biweight-Location für robuste zentrale Tendenz
  - RANSAC für Ausreißererkennung in der Registrierung
  - Sigma-Clipping mit adaptiven Schwellwerten

**Vorteile:**
- Höhere Robustheit gegenüber Ausreißern und Artefakten
- Bessere Leistung bei verrauschten oder problematischen Daten

## 3. Abweichungen und unvollständige Implementierungen

### 3.1 Unvollständige Tile-Geometrie

**Status:** Abweichung

Die Spezifikation definiert eine seeing-adaptive Tile-Größe, während die Implementierung einen festen `tile_size` verwendet. Dies ist ein klarer Bereich für Verbesserungen.

### 3.2 Unvollständige CFA-Pfad-Integration

**Status:** Teilweise implementiert

Die grundlegenden CFA-Funktionen sind implementiert, aber die vollständige Pipeline-Integration für Pfad B fehlt noch.

### 3.3 Zustandsbasierte Clusterung

**Status:** Abweichend

Die dynamische Cluster-Anzahl und der spezifizierte Zustandsvektor sind nicht vollständig gemäß Spezifikation implementiert.

### 3.4 Validierung

**Status:** Unvollständig

Die in der Spezifikation geforderten Validierungsplots und -kriterien sind nicht vollständig implementiert.

## 4. Verbesserungspotenzial

### 4.1 Integration der fortschrittlichen Methoden in die Spezifikation

Die überlegenen Aspekte der Implementierung sollten in die Spezifikation aufgenommen werden:

1. **Mehrstufige Registrierung**: Die fortschrittlichere Registrierungskaskade sollte als normativer Teil in die Spezifikation einfließen
2. **CFA-aware Transformation**: Die präziseren Methoden für CFA-Daten sollten detaillierter spezifiziert werden
3. **Qualitätsbasierte Referenzauswahl**: Die verbesserte Auswahllogik sollte formalisiert werden
4. **Wiener-Filter Option**: Die zusätzliche Rauschunterdrückungsoption sollte als optionale Erweiterung aufgenommen werden

### 4.2 Vervollständigung der fehlenden Elemente

1. **Tile-Geometrie**: Implementierung der seeing-adaptiven Tile-Größe nach §3.3
2. **CFA-Pfad (B)**: Vollständige Pipeline-Integration
3. **Clusterung**: Implementierung der dynamischen Cluster-Anzahl nach §3.7
4. **Validierung**: Implementierung der Validierungsplots aus Anhang B

## 5. Technische Detailanalyse

### 5.1 Registrierung und Vorverarbeitungspfade

**Status:** Eigenständige, fortschrittlichere Implementierung

Die Implementierung verwendet einen kaskadierten Ansatz für die Registrierung:

1. **Vorverarbeitung**:
   - OSC-Daten: CFA-aware Luminanz-Extraktion via `cfa_green_proxy_downsample2x2`
   - Mono-Daten: Optimiertes Downsampling für Geschwindigkeitsgewinn

2. **Referenzauswahl**:
   - Qualitätsbasierte Selektion statt einfacher zeitlicher Mittenwahl
   - Berücksichtigung multipler Qualitätsfaktoren

3. **Registrierungskaskade**:
   ```
   Für jedes Frame:
     1. Versuche hybrid_phase_ecc
     2. Falls fehlgeschlagen: robust_phase_ecc
     3. Falls fehlgeschlagen: star_registration_similarity
     4. Falls fehlgeschlagen: feature_registration_similarity
     5. Falls alle fehlschlagen: Fallback auf Identitätstransformation mit Warnung
   ```

4. **Validierung**:
   - NCC-basierte Bewertung jeder Registrierung
   - Vergleich mit unregistriertem Baseline-NCC

5. **Transformation**:
   - CFA-Daten: Phasenerhaltende Transformation
   - Mono/RGB: Optimierte Affintransformation

### 5.2 Normalisierung und Metriken

**Status:** Teilweise verbessert, teilweise konform

Die Implementierung bietet verbesserte robuste statistische Methoden:

1. **Hintergrundschätzung**:
   - Verbesserte Masken für Objektausschluss
   - Robuste Sigma-Clipping-Verfahren

2. **Rauschschätzung**:
   - MAD-basierte Schätzung für höhere Robustheit
   - Verbesserte Outlier-Rejection

3. **Qualitätsmetriken**:
   - Verbesserte Gradientenmessung via Sobel/Scharr-Operatoren
   - Optimierte Rundheitsberechnung

### 5.3 Tile-Rekonstruktion und -Verarbeitung

**Status:** Erweitert mit zusätzlichen Optionen

1. **Wiener-Filtering**:
   - Frequenzraumbasierte adaptive Rauschunterdrückung
   - Selektive Anwendung basierend auf Tile-Typ und SNR

2. **Gewichtete Sigma-Clipping**:
   - Erweitert die einfache Pixel-Rejection um Gewichte
   - Optimiert für lokale Strukturen

## 6. Empfehlungen

1. **Bidirektionale Aktualisierung**:
   - Aktualisierung der Spezifikation, um die fortschrittlichen Methoden der Implementierung zu inkludieren
   - Vervollständigung der fehlenden Elemente in der Implementierung

2. **Priorisierung**:
   - Hohe Priorität: Seeing-adaptive Tile-Geometrie, CFA-Pfad-Integration
   - Mittlere Priorität: Dynamische Cluster-Anzahl, Validierung
   - Niedrige Priorität: Dokumentation der Erweiterungen

3. **Validierung und Tests**:
   - Entwicklung von Benchmark-Datensätzen zum Vergleich beider Pfade
   - Automatisierte Validierung gegen Spezifikation

## 7. Fazit

Die C++ Implementierung stellt in mehreren Bereichen eine technisch überlegene Umsetzung der Methodik v3 dar. Insbesondere die Registrierungsphase, die CFA-aware Transformation und die robusten statistischen Methoden bieten signifikante Vorteile gegenüber der Spezifikation.

Gleichzeitig gibt es Bereiche, in denen die Implementierung noch vervollständigt werden muss, um vollständig spezifikationskonform zu sein.

Die optimale Strategie wäre eine bidirektionale Aktualisierung: Die Spezifikation sollte um die fortschrittlichen Methoden der Implementierung erweitert werden, während die Implementierung in den Bereichen vervollständigt wird, in denen sie von der Spezifikation abweicht.