# Zeitplan und Meilensteine

## Übersicht

**Geschätzte Gesamtdauer**: 16-22 Wochen (4-5.5 Monate)

---

## Phasen-Übersicht

| Phase | Beschreibung | Dauer | Abhängigkeiten |
|-------|--------------|-------|----------------|
| 1 | Core-Infrastruktur | 2-3 Wochen | - |
| 2 | I/O und Konfiguration | 1-2 Wochen | Phase 1 |
| 3 | Bildverarbeitung | 2-3 Wochen | Phase 2 |
| 4 | Registrierung | 2-3 Wochen | Phase 3 |
| 5 | Metriken und Clustering | 2-3 Wochen | Phase 3 |
| 6 | Rekonstruktion und Synthetic | 2-3 Wochen | Phase 5 |
| 7 | Pipeline-Integration | 2-3 Wochen | Phase 4, 6 |
| 8 | Testing und Validierung | 2-3 Wochen | Phase 7 |

---

## Detaillierter Zeitplan

### Woche 1-3: Phase 1 - Core-Infrastruktur

**Ziele**:
- [ ] Projektstruktur erstellen
- [ ] CMake Build-System aufsetzen
- [ ] vcpkg/Conan Dependency-Management
- [ ] Core-Typen definieren
- [ ] Fehlerbehandlung implementieren
- [ ] Utilities implementieren
- [ ] Event-System implementieren

**Meilenstein**: Build-System funktioniert, erste Unit-Tests laufen

---

### Woche 4-5: Phase 2 - I/O und Konfiguration

**Ziele**:
- [ ] CFITSIO-Integration
- [ ] FITS lesen/schreiben
- [ ] Bayer-Pattern-Erkennung
- [ ] YAML-Config-Parsing
- [ ] Config-Validierung
- [ ] Schema-Validierung (optional)

**Meilenstein**: FITS-Dateien können gelesen und geschrieben werden

---

### Woche 6-8: Phase 3 - Bildverarbeitung

**Ziele**:
- [ ] CFA-Channel-Splitting
- [ ] CFA-Reassembly
- [ ] Demosaicing via OpenCV
- [ ] Normalisierung
- [ ] Hotpixel-Korrektur
- [ ] CFA-Warping

**Meilenstein**: Alle Bildverarbeitungsfunktionen portiert und getestet

---

### Woche 9-11: Phase 4 - Registrierung

**Ziele**:
- [ ] ECC-Bildvorbereitung
- [ ] Phase-Korrelation
- [ ] ECC-Registrierung
- [ ] Rotations-Sweep
- [ ] Warp-Anwendung

**Meilenstein**: Registrierung funktioniert identisch zur Python-Version

---

### Woche 9-11: Phase 5 - Metriken und Clustering (parallel zu Phase 4)

**Ziele**:
- [ ] Globale Metriken
- [ ] Tile-Metriken
- [ ] Wiener-Filter
- [ ] K-Means Clustering
- [ ] Standard-Scaler
- [ ] Silhouette-Score

**Meilenstein**: Metriken und Clustering produzieren identische Ergebnisse

---

### Woche 12-14: Phase 6 - Rekonstruktion und Synthetic

**Ziele**:
- [ ] Tile-Rekonstruktion
- [ ] Hann-Fenster-Blending
- [ ] Fallback-Mechanismus
- [ ] Synthetic-Frame-Generierung
- [ ] Cluster-basierte Generierung
- [ ] Quantil-Fallback
- [ ] Sigma-Clipping

**Meilenstein**: Rekonstruktion produziert identische Ergebnisse

---

### Woche 15-17: Phase 7 - Pipeline-Integration

**Ziele**:
- [ ] Pipeline-Kontext
- [ ] Phase-Handler
- [ ] Event-Emission
- [ ] Progress-Reporting
- [ ] Stop-Mechanismus
- [ ] Resume-Funktionalität
- [ ] CLI-Hauptprogramm

**Meilenstein**: Vollständige Pipeline läuft durch

---

### Woche 18-20: Phase 8 - Testing und Validierung

**Ziele**:
- [ ] Unit-Tests vervollständigen
- [ ] Regressionstests gegen Python
- [ ] Performance-Benchmarks
- [ ] End-to-End-Tests
- [ ] CI/CD-Pipeline
- [ ] Dokumentation

**Meilenstein**: Alle Tests bestehen, Performance-Ziele erreicht

---

### Woche 21-22: Puffer und Feinschliff

**Ziele**:
- [ ] Bug-Fixes
- [ ] Performance-Optimierung
- [ ] GUI-Integration finalisieren
- [ ] Dokumentation vervollständigen
- [ ] Release vorbereiten

**Meilenstein**: Production-ready Release

---

## Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|--------|-------------------|------------|------------|
| OpenCV-API-Unterschiede | Mittel | Mittel | Früh testen, Wrapper-Funktionen |
| CFITSIO-Komplexität | Niedrig | Mittel | Bestehende Wrapper nutzen |
| Numerische Abweichungen | Hoch | Niedrig | Toleranzen in Tests, Dokumentation |
| Performance-Probleme | Mittel | Mittel | Profiling, Parallelisierung |
| Dependency-Konflikte | Mittel | Hoch | vcpkg/Conan, Docker |

---

## Ressourcen

### Entwickler-Zeit
- 1 Entwickler Vollzeit: 16-22 Wochen
- 2 Entwickler parallel: 10-14 Wochen (Phase 4+5 parallel)

### Hardware
- Build-Server mit ≥16 GB RAM
- Test-Daten (FITS-Dateien)

### Software
- C++17 Compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.16+
- vcpkg oder Conan
- Git

---

## Erfolgs-Kriterien

1. **Funktional**: Identische Ergebnisse wie Python-Implementierung
2. **Performance**: ≥2x Speedup gegenüber Python
3. **Stabilität**: Keine Crashes, saubere Fehlerbehandlung
4. **Wartbarkeit**: Sauberer Code, gute Dokumentation
5. **Integration**: Nahtlose GUI-Integration
