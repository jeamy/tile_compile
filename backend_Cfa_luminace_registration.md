# Backend – CFA‑Luminanz‑Registrierung

**Status:** Entwurfs‑ und Implementierungsspezifikation
**Rolle:** Default‑Registrierungsbackend
**Fallback:** Siril (optional, austauschbar)

---

## 1. Zielsetzung

Diese Spezifikation beschreibt eine **vollständige Backend‑Implementierung der geometrischen Registrierung** auf Basis einer **CFA‑Luminanz**, geeignet für OSC‑Daten.

Ziele:

* Trennung von **Geometrie** und **Farbe**
* Vermeidung chromatischer Registrierungsartefakte
* Reproduzierbarkeit ohne externe Tool‑Abhängigkeit
* Modulare Erweiterbarkeit (Siril, OpenCV, GPU, …)

---

## 2. Grundprinzip

> **Registrierung ist eine rein geometrische Operation.**

* Sie darf keine Farb‑ oder PSF‑Information einbringen.
* Sie muss für alle Farbkanäle identisch sein.
* Sie wird auf einer **skalaren CFA‑Luminanz** durchgeführt.

Debayering ist **kein Bestandteil** der Registrierung.

---

## 3. Architekturübersicht

```text
backend/
├── registration/
│   ├── base.py              # Abstraktes Interface
│   ├── cfa_luminance.py     # Default‑Backend
│   ├── siril.py             # Optionaler Adapter
│   └── models.py            # Transformationsmodelle
│
├── io/
│   ├── fits_reader.py
│   └── cfa.py
│
├── pipeline/
│   └── registration_stage.py
│
└── run_context.py
```

---

## 4. Gemeinsames Interface

```python
class RegistrationResult:
    def __init__(self, transforms, reference_index):
        self.transforms = transforms
        self.reference_index = reference_index

class RegistrationBackend:
    def register(self, frames, workdir):
        raise NotImplementedError
```

Die Pipeline kennt **ausschließlich dieses Interface**.

---

## 5. CFA‑Luminanz‑Definition

Für ein CFA‑Frame `I(x,y)` wird die Luminanz definiert als:

```
L(x,y) = R + G1 + G2 + B
```

* nur reale CFA‑Samples
* keine Interpolation
* grüne Samples doppelt gewichtet

---

## 6. CFA‑Extraktion

```python
def cfa_luminance(frame, bayer):
    L = zeros_like(frame)
    if bayer == "RGGB":
        L[0::2,0::2] = frame[0::2,0::2]
        L[0::2,1::2] = frame[0::2,1::2]
        L[1::2,0::2] = frame[1::2,0::2]
        L[1::2,1::2] = frame[1::2,1::2]
    return L
```

Andere Bayer‑Pattern analog.

---

## 7. Sternfindung (minimal)

```python
def detect_stars(img):
    blurred = gaussian_filter(img, sigma=1.2)
    return peak_local_max(
        blurred,
        min_distance=6,
        threshold_abs=percentile(blurred, 99.5)
    )
```

---

## 8. Transformationsschätzung

```python
def estimate_affine(src, dst):
    return estimate_transform("affine", src, dst)
```

* Translation + Rotation
* Affin optional

---

## 9. CFA‑Luminanz‑Registrierungsbackend

```python
class CFALuminanceRegistration(RegistrationBackend):
    def register(self, frames, workdir):
        lum_frames = []
        star_lists = []

        for path in frames:
            data, header = read_fits(path)
            L = cfa_luminance(data, header['BAYERPAT'])
            lum_frames.append(L)
            star_lists.append(detect_stars(L))

        ref = argmax(len(s) for s in star_lists)
        transforms = []

        for i, stars in enumerate(star_lists):
            if i == ref:
                transforms.append(identity())
            else:
                transforms.append(estimate_affine(stars, star_lists[ref]))

        return RegistrationResult(transforms, ref)
```

---

## 10. Anwendung der Transformationen

Transformationen werden **auf rohe CFA‑Frames** angewandt:

```python
def apply_transform(frame, T):
    return warp(frame, inverse_map=T.inverse, preserve_range=True)
```

* identische Geometrie für alle Kanäle
* keine chromatischen Residuen

---

## 11. Siril‑Backend (Fallback)

```python
class SirilRegistration(RegistrationBackend):
    def register(self, frames, workdir):
        prepare_links(frames, workdir)
        run_siril(workdir)
        transforms = read_siril_transforms(workdir)
        return RegistrationResult(transforms, 0)
```

Siril ist:

* optional
* austauschbar
* kein methodischer Kernbestandteil

---

## 12. Pipeline‑Integration

```python
def run_registration(ctx):
    if ctx.config.registration.backend == "cfa_luminance":
        backend = CFALuminanceRegistration()
    elif ctx.config.registration.backend == "siril":
        backend = SirilRegistration()

    result = backend.register(ctx.frames, ctx.run_dir)
    ctx.transforms = result.transforms
```

---

## 13. Konfiguration (YAML)

```yaml
registration:
  backend: cfa_luminance
  fallback: siril
```

---

## 14. Methodische Begründung

* Registrierung ist geometrisch → CFA‑Luminanz
* Farbe bleibt unangetastet
* Keine Debayer‑Resampling‑Artefakte
* Vollständig reproduzierbar

---

## 15. Zusammenfassung

* CFA‑Luminanz‑Registrierung ist **Default**
* Siril ist **Adapter**, kein Kernmodul
* Architektur bleibt offen für Erweiterungen

---

**Ende der Spezifikation**
