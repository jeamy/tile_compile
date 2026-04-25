# Wiener-Tile Noise Reduction – Normative Formulierung & Referenzimplementierung

## 1. Zweck und Geltungsbereich

Dieses Dokument spezifiziert eine **konforme Noise-Reduction-Methode auf Tile-Ebene** für die `tile_compile`-Pipeline.

**Ziel:** Reduktion rein stochastischer Varianz in *rekonstruierten Tiles* ohne Beeinflussung von

* Qualitätsmetriken
* Gewichten
* Zustandsclusterung

Die Methode ist **linear**, **deterministisch** und **metrikenblind**.

---

## 2. Zwingende Randbedingungen (normativ)

Noise Reduction (NR) darf ausschließlich:

* **nach** Tile-Rekonstruktion
* **vor** Overlap-Add
* **nie** auf Einzel-Frames
* **nie** vor Metrikberechnung

angewendet werden.

Nicht zulässig sind:

* Thresholding (Wavelet, Soft/Hard)
* nichtlineare Filter (NLMeans, BM3D, KI)
* Multi-Scale-Operatoren

---

## 3. Mathematisches Modell

### 3.1 Signalannahme

Für ein rekonstruiertes Tile $t$:

$$
I_t(p) = S_t(p) + N_t(p)
$$

mit

* $S_t(p)$: unbekanntes kohärentes Signal
* $N_t(p)$: additives, weißes Rauschen

$$
\mathbb{E}[N_t] = 0, \quad \mathrm{Var}(N_t) = \sigma_t^2
$$

---

### 3.2 Wiener-Filter (Frequenzraum)

Die Wiener-Übertragungsfunktion ist definiert als:

$$
H_t(k) = \frac{P_{S,t}(k)}{P_{S,t}(k) + P_{N,t}(k)}
$$

mit

$$
P_{N,t}(k) = \sigma_t^2
$$

Da $P_{S,t}$ unbekannt ist, verwenden wir die klassische Approximation:

$$
P_{S,t}(k) \approx \max(|I_t(k)|^2 - \sigma_t^2, 0)
$$

Daraus folgt:

$$
H_t(k) = \frac{\max(|I_t(k)|^2 - \sigma_t^2, 0)}{|I_t(k)|^2}
$$

und das gefilterte Signal:

$$
\hat S_t(k) = H_t(k) \cdot I_t(k)
$$

Inverse FFT liefert:

$$
\hat I_t(p) = \mathcal{F}^{-1}[\hat S_t(k)]
$$

---

## 4. Numerische Stabilität

* $|I_t(k)|^2 < \varepsilon \Rightarrow H_t(k)=0$
* $H_t(k) \in [0,1]$
* empfohlene Parameter:

```
ε = 1e-12
precision = float64
```

---

## 5. Tile-selektive Aktivierung (verbindlich)

NR wird **nur** aktiviert, wenn:

$$
\mathrm{SNR}*t < \tau*{\mathrm{snr}} \land Q_{struct,t} > Q_{min}
$$

Standardwerte:

```
τ_snr = 5.0
Q_min = -0.5
```

**Stern-dominierte Tiles:**

```
NR ≡ deaktiviert
```

---

## 6. Exakte Position im Ablauf

```
rekonstruiere Tile I_t
→ (optional) Wiener-NR
→ Hintergrundsubtraktion
→ Tile-Normalisierung
→ Fensterfunktion
→ Overlap-Add
```

---

## 7. Python-Referenzimplementierung (Minimal)

```python
import numpy as np

def wiener_tile_filter(
    tile: np.ndarray,
    sigma: float,
    *,
    snr_tile: float,
    q_struct_tile: float,
    is_star_tile: bool,
    snr_threshold: float = 5.0,
    q_min: float = -0.5,
    eps: float = 1e-12
) -> np.ndarray:
    if is_star_tile:
        return tile
    if snr_tile >= snr_threshold:
        return tile
    if q_struct_tile <= q_min:
        return tile

    padded = symmetric_pad(tile)
    F = np.fft.fft2(padded).astype(np.complex128)

    power = np.abs(F) ** 2
    H = np.maximum(power - sigma * sigma, 0.0) / np.maximum(power, eps)
    H = np.clip(H, 0.0, 1.0)

    F_filtered = H * F
    filtered = np.real(np.fft.ifft2(F_filtered))
    return crop_to_original(filtered, tile.shape)
```

---

## 8. C++-Pseudo-Referenz

```cpp
Tile apply_wiener_filter(
    const Tile& tile,
    double sigma,
    const TileMeta& meta,
    double snr_threshold = 5.0,
    double q_min = -0.5,
    double eps = 1e-12
) {
    if (meta.is_star_tile) return tile;
    if (meta.snr_tile >= snr_threshold) return tile;
    if (meta.q_struct_tile <= q_min) return tile;

    Tile padded = symmetric_pad(tile);
    FFTComplex F = fft2(padded);

    for (auto& c : F) {
        double power = norm(c);
        double H = (power < eps) ? 0.0 : (power - sigma * sigma) / power;
        H = std::clamp(H, 0.0, 1.0);
        c *= H;
    }

    Tile filtered = ifft2_real(F);
    return crop_to_original(filtered, tile.width(), tile.height());
}
```

---

## 9. Nicht-Ziele (explizit ausgeschlossen)

Diese Methode:

* verstärkt keine Details
* verändert keine PSF
* greift nicht in Gewichtung oder Clustering ein
* ersetzt kein Post-Processing

---

## 10. Kernaussage

> Der Wiener-Tile-Filter ist ein **lokal optimaler, linearer MSE-Minimierer**.
>
> Er ist **nur dann konform**, wenn er ausschließlich auf **rekonstruierten Tiles** angewendet wird und keinerlei Rückwirkung auf Metriken oder Zustandsbildung besitzt.

Dieses Dokument ist **normativ** für `tile_compile`-konforme Implementierungen.
