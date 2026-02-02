# PATCH 05 – Entfernen globaler Pfade (v4-only)

## Dateien vollständig entfernen

```
runner/opencv_registration.py
runner/phases.py
runner/phases_impl.py
ref_siril_registration.py
ref_siril_registration_call.py
```

## Tests entfernen oder deaktivieren

```
tests/test_registration.py
validation/methodik_v3_compliance.py
```

Empfehlung:
- verschieben nach `legacy/`
- oder löschen

---

## Imports bereinigen

In folgenden Dateien sicherstellen, dass **keine Imports** der obigen Module mehr existieren:

- `tile_compile_runner.py`
- `tile_compile_backend/*.py`
- `runner/*.py`

---

## Konfigurationsbereinigung

In `tile_compile.yaml`:

- alle Optionen zu globaler Registrierung ignorieren oder löschen
- einzig zulässiger Modus:

```yaml
registration:
  mode: local_tiles
```

---

## Test-Suite aktualisieren

- neue v4-Tests aktiv
- alte Registrierungstests dürfen **nicht mehr im CI laufen**

