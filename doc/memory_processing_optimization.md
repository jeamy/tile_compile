# Memory Processing Optimization in Tile-Compile

## Überblick

In Version 3.1 wurden umfassende Optimierungen zur Speicher- und Ressourcenverwaltung eingeführt. Ziel war es, die Verarbeitung großer astronomischer Datensätze zu verbessern und Speicher-Overflow-Probleme zu minimieren.

**Implementierungsansatz:** Disk-basierte Verarbeitung statt RAM-intensiver Streaming-Ansätze.

## Implementierte Komponenten

### 1. Disk-basierte Bildverarbeitung
- Datei: `runner/image_processing.py`
- Funktionen: `normalize_frame()`, `compute_frame_medians()`, `split_cfa_channels()`
- Strategie:
  - Frames werden auf Disk geschrieben (`work/channels/`)
  - Normalisierung in 2 Passes: Mediane berechnen, dann Frame-für-Frame normalisieren
  - Maximaler RAM-Verbrauch: 1 Frame statt alle Frames
- Vorteile:
  - Skaliert auf beliebig große Datasets (40GB+ getestet)
  - Kein OOM bei 422 Frames à 95MB
  - Einfache, wartbare Implementierung

### 2. Erweiterte Fehlerbehandlung
- Datei: `runner/error_handling.py`
- Klassen: `ProcessingError`, `MemoryManagementError`
- Decorators: `robust_processing()`, `log_exception()`
- Vorteile:
  - Zentralisierte Fehlerprotokollierung
  - Robuste Fehlerbehandlung
  - Detaillierte Fehleranalyse

### 3. Logging-System
- Datei: `runner/logging_config.py`
- Funktionen: `setup_logging()`, `get_logger()`
- Vorteile:
  - Zentralisierte Logging-Konfiguration
  - Rotative Logdateien
  - Konfigurierbare Log-Level

### 4. Memory Mapping
- Datei: `runner/memory_mapping.py`
- Klasse: `MemoryMappedArray`
- Funktionen: `memory_mapped_processing()`
- Vorteile:
  - Speichereffiziente Array-Verarbeitung
  - Disk-basierte temporäre Speicherung
  - Dynamische Ressourcenverwaltung

### 5. OOM-Prävention
- Datei: `runner/oom_prevention.py`
- Klassen: `ResourceManager`, `ChunkedProcessor`
- Funktionen: `prevent_oom()` Decorator
- Vorteile:
  - Echtzeitüberwachung von CPU, Speicher, Festplatte
  - Chunk-basierte Verarbeitung mit adaptiver Größe
  - Automatische Ressourcenprüfung

### 6. Fallback-Mechanismen
- Datei: `runner/fallback_mechanism.py`
- Klasse: `FallbackMechanism`
- Funktionen: 
  - Retry-Strategien
  - Fehler-Handling
  - Generatoren-Schutz
- Vorteile:
  - Robuste Fehlerbehandlung
  - Automatische Wiederholungsstrategien
  - Graceful Degradation

## Hauptänderungen in der Verarbeitungsstrategie

### Vorher (v3.0)
- Vollständiges Laden aller Frames in den Speicher (RAM-Listen)
- Channels als `Dict[str, List[np.ndarray]]` → 120GB+ RAM bei großen Datasets
- OOM-Crashes bei >400 Frames
- Minimales Logging

### Nachher (v3.1 - Disk-basiert)
- Frames werden auf Disk geschrieben (`work/channels/`)
- Channels als `Dict[str, List[Path]]` → nur 1 Frame im RAM
- 2-Pass Normalisierung: Mediane berechnen, dann einzeln normalisieren
- Skaliert auf beliebig große Datasets
- Umfassende Fehlerbehandlung und Logging

## Konfigurationsoptionen

Die neuen Komponenten sind vollständig konfigurierbar:

```python
# Beispielkonfiguration
resource_config = {
    'memory_limit_mb': 8192,  # 8 GB
    'chunk_size': 50,
    'logging_level': 'INFO',
    'fallback_attempts': 3
}
```

## Performance-Empfehlungen

- Setzen Sie realistische Ressourcenlimits
- Überwachen Sie Logdateien
- Passen Sie Chunk-Größen an Ihre Hardware an
- Nutzen Sie Fallback-Mechanismen

## Bekannte Einschränkungen

- Erhöhter Verarbeitungsoverhead
- Leicht reduzierte Verarbeitungsgeschwindigkeit
- Abhängigkeit von Systemressourcen

## Zukünftige Entwicklungen

- Verteilte Verarbeitung
- GPU-Beschleunigung
- Noch dynamischere Ressourcenanpassung

## Migration

1. Updaten Sie alle Abhängigkeiten
2. Migrieren Sie Verarbeitungsfunktionen zu Streaming-Varianten
3. Konfigurieren Sie Ressourcenlimits
4. Überprüfen Sie Logging und Fehlerbehandlung

---

*Tile-Compile Methodik v3.1 - Memory Processing Optimization*