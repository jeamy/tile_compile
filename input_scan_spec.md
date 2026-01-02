# Tile Compile – Input Scan Specification

**Zweck:** deterministische Ableitung von Input-Metadaten aus dem Input-Verzeichnis

---

## 1. Ziel

Dieses Dokument definiert, wie die GUI / das Backend ein Input-Verzeichnis analysiert, um automatisch alle **ableitbaren Metadaten** zu bestimmen.

---

## 2. Dateierkennung

* Erlaubte Endungen: `.fit`, `.fits`
* Sortierung: lexikographisch, stabil
* Ergebnis: `frames_manifest`

---

## 3. FITS-Header-Felder

### Pflichtfelder (Hard Error falls inkonsistent)

| Feld   | Bedeutung  |
| ------ | ---------- |
| NAXIS1 | Bildbreite |
| NAXIS2 | Bildhöhe   |

Alle Frames müssen identische Werte besitzen.

---

### Optionale Felder (Soft Detection)

| Feld                | Zweck         |
| ------------------- | ------------- |
| BAYERPAT            | Bayer-Pattern |
| XBAYROFF / YBAYROFF | Bayer-Offset  |
| INSTRUME            | Kamera        |

---

## 4. Abgeleitete Werte

| Wert            | Quelle         |
| --------------- | -------------- |
| image_width     | NAXIS1         |
| image_height    | NAXIS2         |
| frames_detected | Anzahl Dateien |
| color_mode      | BAYERPAT → OSC |

---

## 5. Validierung

* Falls `frames_detected < frames_min` → Hard Error
* Falls inkonsistente Bildgrößen → Hard Error
* Falls `color_mode` nicht eindeutig → Warnung + Bestätigung

---

## 6. Frames-Manifest

Inhalt:

* sortierte Dateiliste
* optionale Checksums (SHA256)

Hash:

```
frames_manifest_id = sha256(manifest)
```

---

## 7. GUI-Anzeige

GUI zeigt vor Run-Start:

* Anzahl Frames
* Bildgröße
* detektierten Farbmodus

Alle Werte sind read-only.
