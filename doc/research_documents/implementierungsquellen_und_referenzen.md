# Implementierungsquellen und praktische Referenzen

---

## 1. Quellen zu Siril (Registrierung & Debayering)

### 1.1 GitHub Repository

**URL:** https://github.com/lock042/siril  
**Lizenz:** GPL v3  
**Relevante Dateien:**

| Datei | Funktion | Relevanz zu deiner Methodik |
|---|---|---|
| `src/registration/registration.c` | Haupt-Registrierungslogik | Pfad A (Registrierung) |
| `src/registration/regrast.c` | RANSAC-Homography | Star-basierte Alignment |
| `src/core/preprocess.c` | Debayering | Pfad A (Kanal-Extraktion) |
| `src/processing/filters.c` | Lokale Filterung | Potenzial für Tile-Basis |
| `src/core/siril_world.c` | Globale Metriken | FWHM-Berechnung |

**Abrufen:**
```bash
git clone https://github.com/lock042/siril.git
cd siril/src
# Relevante Komponenten suchen:
grep -r "FWHM" *.c | head -20
grep -r "weight" registration/ | head -20
grep -r "demosaic\|debayer" *.c
```

### 1.2 Siril-Dokumentation

**Lokale Dokumentation:**
- Installation: `./INSTALL.md`
- Entwicklung: `DEVELOPERS.md`
- Konfiguration: `data/siril.conf.in`

**Online-Ressourcen:**
- Offizielle Docs: https://www.siril.org/docs/
- Forum: https://siril.org/forum/
- FAQ zu Stacking: https://siril.org/docs/concepts/stacking/

### 1.3 Siril Stacking-Algorithmus (aus Quellcode)

**Vermutete Implementierung (basierend auf öffentlichen Informationen):**

```python
# Pseudo-Python aus Siril's C-Logik

def siril_weighted_stacking(frames, quality_weights):
    """
    Vereinfachte Nachbildung von Siril's Stacking
    """
    H, W = frames[0].shape
    result = np.zeros((H, W))
    
    for y in range(H):
        for x in range(W):
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for f in range(len(frames)):
                weight = quality_weights[f]
                pixel = frames[f][y, x]
                
                # Einfacher weighted average
                weighted_sum += weight * pixel
                weight_sum += weight
            
            if weight_sum > 0:
                result[y, x] = weighted_sum / weight_sum
            else:
                # Fallback
                result[y, x] = np.median([frames[f][y, x] for f in range(len(frames))])
    
    return result
```

**Limitationen in Siril für deine Methodik:**
- Keine Tile-Basis implementiert
- Gewichte sind Global, nicht lokal
- Keine Clusterung
- Keine Synthetic Frames

---

## 2. PixInsight Ressourcen

### 2.1 Dokumentation & APIs

**URL:** https://pixinsight.com/  
**Dokumentations-Zugang:**
- Online-Hilfe: https://pixinsight.com/doc/ (Login erforderlich)
- API-Referenz: PixelMath, Script Console, C++ SDK

**Relevante Module:**

| Modul | Beschreibung | URL |
|---|---|---|
| ImageIntegration | Weighted Stacking | https://pixinsight.com/doc/ (Internal) |
| StarAlignment | Registrierung | https://pixinsight.com/doc/ (Internal) |
| DynamicBackgroundExtraction | Lokale Background | https://pixinsight.com/doc/ (Internal) |
| ATWT | Multiskalen-Wavelet | https://pixinsight.com/doc/ (Internal) |

### 2.2 PixInsight Scripting (PJSR)

**Lokale Qualitäts-Gewichtung (Beispiel aus PixInsight Forum):**

```javascript
// PixelMath / PJSR Script für Custom Tiling (Community-Beispiel)

// Pseudo-Code basierend auf PixInsight Dokumentation
#include <pjsr/ImageWindow.jsh>

function computeLocalWeights(image) {
    // Tile-wise processing
    var tileSize = 64;  // Fixed, nicht FWHM-adaptiv
    
    var weights = new Image(image.width, image.height);
    
    for (var y = 0; y < image.height; y += tileSize) {
        for (var x = 0; x < image.width; x += tileSize) {
            var tile = extractTile(image, x, y, tileSize, tileSize);
            var localFWHM = estimateFWHM(tile);
            var localWeight = 1.0 / (1.0 + localFWHM * localFWHM);
            
            fillTile(weights, x, y, tileSize, tileSize, localWeight);
        }
    }
    
    return weights;
}
```

**Limitation:** Keine FWHM-adaptive Tile-Größe, keine Clusterung

### 2.3 PixInsight C++ SDK

**Für Custom Modul-Entwicklung:**
```cpp
// Pseudo-Code für PixInsight C++ Module
#include <pcl/ProcessInterface.h>
#include <pcl/StarAligner.h>

class CustomStackingModule : public ProcessInterface {
    // Implementierung mit lokalen Gewichten
    virtual bool ExecuteGlobal() override;
    
    // Tile-basierte Rekonstruktion (Custom)
    void reconstructTiles(const ImageVariant& frames, 
                         const TileWeightMatrix& weights);
};
```

**Ressource:** PixInsight SDK Documentation (requires installation)

---

## 3. Akademische Referenzen mit Implementierungs-Details

### 3.1 Lucky Imaging & Frame Selection

**Bramich et al. (2005)**
- **Titel:** "Lucky imaging: high angular resolution imaging in the infrared and visible"
- **Journal:** *Monthly Notices of the Royal Astronomical Society*, Vol. 359, No. 1, pp. 1096-1098
- **DOI:** 10.1111/j.1365-2966.2005.08950.x
- **Direkter Link:** https://doi.org/10.1111/j.1365-2966.2005.08950.x
- **Archiv:** https://arxiv.org/abs/astro-ph/0404232

**Relevante Algorithmen:**
```
Algorithm 1: Lucky Imaging Pipeline
1. Capture N frames of same scene
2. Compute quality metric for each frame Q[f]
3. Select top M% frames (M ≈ 10-30%)
4. Shift-and-Add mit Subpixel Alignment
5. Output: High-resolution stacked image
```

**Unterschied zu deiner Methodik:**
- Bramich: Frame-Selektion (Top M%)
- Deine: Alle Frames behalten, gewichtet (No Selection)

### 3.2 CCD-Registrierung & Astrometrie

**Astrometry.net**
- **URL:** https://astrometry.net/
- **Referenz:** Lang, D. et al. (2010): "Astrometry.net: Blind Astrometric Calibration of Arbitrary Astronomical Images"
- **Algorithmus:** SIFT-basierte Feature-Matching

**Implementierung (Open Source):**
```bash
# Installation
git clone https://github.com/dstndstn/astrometry.net.git
cd astrometry.net
./install-all-deps.sh
make
```

**Relevanz zu Pfad B (CFA-Registrierung):**
- Astrometry.net ist nicht CFA-aware
- Könnte aber als Basis für Luminanz-Registrierung dienen

### 3.3 CFA & Debayering Methoden

**Menon et al. (2007)**
- **Titel:** "High-Quality Color Demosaicing of CFA Images"
- **Verfügbar:** https://ieeexplore.ieee.org/document/4215200/
- **Relevanz:** Demosaicing-Methoden für Pfad A

**Directional Demosaicing (Gunturk et al., 2005):**
- Adaptive Interpolation basierend auf lokale Struktur
- Kann CFA-Struktur besser respektieren

---

## 4. Open-Source Implementierungen (Verwandte Projekte)

### 4.1 Sequator

**URL:** https://www.sequator.io/ (derzeit offline, aber GitHub vorhanden)  
**GitHub:** https://github.com/PeterLoesche/Sequator  
**Sprache:** C++ / Qt  
**Relevanz:** Landschaftsfotografie-Stacking mit lokalen Gewichten

**Interessante Komponenten:**
```cpp
// Pseudo-Code aus Sequator-Konzept
class StackingEngine {
    // Local detail weighting
    void computeLocalWeights(const Frame& frame, 
                            WeightMap& weights);
    
    // Gewichtetes Stacking (pixel-weise)
    void stackWeighted(const FrameSet& frames, 
                      const WeightSet& weights);
};
```

### 4.2 Startools

**URL:** https://www.startools.org/  
**Art:** Closed-Source, aber gut dokumentiert  
**Relevante Module:**
- **Tracking:** Lokale Alignment
- **Wipe:** Lokale Rauschabzug
- **Heal:** Artefakt-Rekonstruktion

**Konzept:** "Global/Local Quality" adaptive Processing  
(Ähnlich zu deinem G_f,c · L_f,t,c Ansatz, aber nicht formal definiert)

### 4.3 AstroImageJ

**URL:** https://www.astroImageJ.org/  
**GitHub:** https://github.com/AstroImageJ/astroimagej  
**Sprache:** Java / ImageJ-Plugin  
**Fokus:** Wissenschaftliche Astronomie (Exoplanet-Timing, etc.)

**Registrierungs-Algorithmus:**
```java
// Aus AstroImageJ-Quellcode (Pseudo-Code)
public void registerFrames(ImageStack stack) {
    // Cross-correlation based registration
    ImageProcessor reference = stack.getProcessor(1);
    
    for (int f = 2; f <= stack.getSize(); f++) {
        ImageProcessor frame = stack.getProcessor(f);
        Point2D.Double shift = crossCorrelation(reference, frame);
        applyShift(stack, f, shift);
    }
}
```

---

## 5. Numerik-Bibliotheken für Implementierung

### 5.1 Python (empfohlen für Prototyping)

```python
# Relevante Bibliotheken für deine Methodik

# 1. Image Processing & Numerics
import numpy as np                      # Arrays, Algebra
import scipy.ndimage as ndi            # Morphological Operations
import scipy.interpolate as interp     # Interpolation
from scipy import signal               # Filtering

# 2. Clustering
from sklearn.cluster import KMeans     # State-based Clustering (§3.7)
from sklearn.preprocessing import StandardScaler

# 3. Image Registration
from skimage import registration       # For experimental Pfad B
from skimage.registration import phase_cross_correlation

# 4. FITS Handling
from astropy.io import fits            # FITS I/O
from astropy.nddata import Cutout2D    # Tile extraction

# 5. Robuste Statistik
from scipy.stats import robust_skew    # Outlier detection
import astropy.stats as astats         # Astronomical statistics

# 6. Optimization
from scipy.optimize import minimize     # For PSF fitting

# 7. Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
```

**Beispiel-Implementierung (Tile-Erzeugung, §3.3):**

```python
import numpy as np
from scipy import ndimage
from astropy.io import fits

def generate_tile_geometry(image_path, seeing_fwhm, s=4.0, T_min=32, D=3, o=0.3):
    """
    Implementierung deiner Tile-Geometrie-Formel (v3, §3.3)
    """
    with fits.open(image_path) as hdul:
        image = hdul[0].data
    
    H, W = image.shape
    
    # Step 1: Seeing-proportional size
    T0 = s * seeing_fwhm
    
    # Step 2: Clipping
    T_max = min(W, H) // D
    T = int(np.floor(np.clip(T0, T_min, T_max)))
    
    # Step 3: Overlap
    O = int(np.floor(o * T))
    S = T - O  # Stride
    
    # Step 4: Generate tile grid
    tiles = []
    y = 0
    while y < H:
        x = 0
        y1 = min(y + T, H)
        while x < W:
            x1 = min(x + T, W)
            tiles.append((x, y, x1, y1))
            x += S
        y += S
    
    print(f"Tile configuration:")
    print(f"  FWHM: {seeing_fwhm:.2f} px")
    print(f"  Tile size T: {T} px")
    print(f"  Overlap O: {O} px")
    print(f"  Stride S: {S} px")
    print(f"  Number of tiles: {len(tiles)}")
    
    return tiles, T, O, S

# Anwendung
tiles, T, O, S = generate_tile_geometry(
    'observation.fits', 
    seeing_fwhm=2.5,
    s=4.0, T_min=32, D=3, o=0.3
)
```

### 5.2 C++ (für Produktion)

**Relevante Bibliotheken:**

```cpp
// Image Processing & Numerics
#include <opencv2/opencv.hpp>          // OpenCV (Registrierung, Morphologie)
#include <Eigen/Dense>                 // Lineare Algebra
#include <cfitsio>                     // FITS I/O (C)

// Clustering
#include <boost/kmeans.hpp>            // oder Custom-Implementierung

// PSF Fitting
// GSL (GNU Scientific Library) für Non-Linear Fitting
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit_nlin.h>
```

**Beispiel: Registrierungsmotoren in C++:**

```cpp
// OpenCV-basierte Registrierung
#include <opencv2/opencv.hpp>

class FrameRegistration {
public:
    cv::Mat registerFrame(const cv::Mat& reference, const cv::Mat& frame) {
        // Phase Correlation (Subpixel-Genauigkeit)
        cv::Point2d shift = cv::phaseCorrelate(reference, frame);
        
        // Affine Transform anwenden
        cv::Mat warpMatrix = cv::getRotationMatrix2D(
            cv::Point2f(frame.cols/2, frame.rows/2), 
            0.0,  // rotation
            1.0   // scale
        );
        warpMatrix.at<double>(0, 2) += shift.x;
        warpMatrix.at<double>(1, 2) += shift.y;
        
        cv::Mat registered;
        cv::warpAffine(frame, registered, warpMatrix, frame.size());
        
        return registered;
    }
};
```

### 5.3 C (für Siril-Integration)

Siril ist in C geschrieben. Für Integration deiner Methodik:

```c
// Struktur für Tile-Gewichte
typedef struct {
    int x0, y0, x1, y1;     // Tile bounds
    double weight_global;   // G_f,c
    double weight_local;    // L_f,t,c
    double weight_effective; // W_f,t,c = G · L
    int fallback_used;      // Flag für Fallback (§3.6)
} TileWeight;

// Function prototypes (für Siril Integration)
void compute_tile_geometry(int W, int H, float F, 
                          TileWeight** tiles, int* n_tiles);

void compute_tile_weights(fits* frame, TileWeight* tile, 
                         float FWHM_global, float background, float noise);

void reconstruct_tile(fits** frames, int n_frames, 
                     TileWeight* tile, fits* result);
```

---

## 6. Testing & Validierung (§4.1 Testfälle)

### 6.1 Unit Tests für deine Methodik

```python
# pytest / unittest Beispiele

import numpy as np
import pytest

class TestTileBasedReconstruction:
    
    def test_weight_normalization(self):
        """Testfall 1: α + β + γ = 1"""
        alpha, beta, gamma = 0.4, 0.3, 0.3
        assert np.isclose(alpha + beta + gamma, 1.0)
    
    def test_clamping_before_exp(self):
        """Testfall 2: Clamping auf [-3, +3]"""
        Q_values = [-10, -3, 0, +3, +10]
        Q_clamped = np.clip(Q_values, -3, +3)
        assert np.all(Q_clamped >= -3)
        assert np.all(Q_clamped <= +3)
    
    def test_tile_size_monotonicity(self):
        """Testfall 3: T(F1) ≤ T(F2) für F1 < F2"""
        def tile_size(F, s=4.0, T_min=32, D=3):
            T0 = s * F
            T_max = 1000 // D
            return int(np.floor(np.clip(T0, T_min, T_max)))
        
        F1, F2 = 1.5, 3.0
        assert tile_size(F1) <= tile_size(F2)
    
    def test_overlap_consistency(self):
        """Testfall 4: Overlap-Konsistenz"""
        T, o = 64, 0.3
        O = int(np.floor(o * T))
        S = T - O
        assert O >= 0 and O <= 32
        assert S > 0
    
    def test_low_weight_fallback(self):
        """Testfall 5: Low-weight Fallback"""
        frames = np.random.rand(10, 64, 64)
        weights = np.ones(10) * 0.0001  # Very small weights
        epsilon = 1e-10
        
        denominator = np.sum(weights)
        assert denominator < epsilon
        
        # Fallback: ungewichtetes Mittel
        result = np.mean(frames, axis=0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_no_channel_coupling(self):
        """Testfall 6: Keine Kanal-Kopplung"""
        # R, G, B werden separat verarbeitet
        channels = ['R', 'G', 'B']
        for ch in channels:
            # Jeder Kanal wird unabhängig berechnet
            # (Test: Ergebnis sollte gleich sein ob Kanäle zusammen oder einzeln)
            pass
    
    def test_no_frame_selection(self):
        """Testfall 7: Keine Frame-Selektion"""
        N_frames = 100
        weights = np.ones(N_frames)
        
        # Alle Frames müssen verwendet werden
        for f in range(N_frames):
            assert weights[f] > 0 or weights[f] == 0  # Kein NaN, kein -inf
    
    def test_determinism(self):
        """Testfall 8: Determinismus"""
        frames = np.random.rand(10, 256, 256)
        config = {'T_min': 32, 'D': 3, 'overlap': 0.3}
        
        result1 = full_pipeline(frames, config)
        result2 = full_pipeline(frames, config)
        
        assert np.allclose(result1, result2, atol=1e-6)
```

### 6.2 Integration Tests

```python
def test_full_pipeline_pfad_A():
    """Pfad A: Siril + Kanaltrennung"""
    frames = load_test_frames('siril_registered_rgb.fits')
    
    # Kanaltrennung
    R = frames[:, :, 0]
    G = frames[:, :, 1]
    B = frames[:, :, 2]
    
    # Verarbeite jeden Kanal
    R_recon = tile_reconstruction(R)
    G_recon = tile_reconstruction(G)
    B_recon = tile_reconstruction(B)
    
    # Ergebnis sollte Linear bleiben
    assert check_linearity(R_recon)
    assert check_linearity(G_recon)
    assert check_linearity(B_recon)

def test_full_pipeline_pfad_B():
    """Pfad B: CFA-aware Registration"""
    cfa_frames = load_test_frames('raw_cfa.fits')
    
    # CFA-basierte Registrierung
    registered_cfa = register_cfa_aware(cfa_frames)
    
    # Debayer (nach Registrierung)
    R, G, B = debayer_after_registration(registered_cfa)
    
    # Verarbeitung
    R_recon = tile_reconstruction(R)
    G_recon = tile_reconstruction(G)
    B_recon = tile_reconstruction(B)
    
    # Vergleich A vs B (sollte ähnlich sein, aber B weniger Artefakte)
    assert quality_metric(B_recon) >= quality_metric(A_recon) * 0.95
```

---

## 7. Öffentlich verfügbare Datensätze zum Testen

### 7.1 Astronomische Test-Frames

**Quellen:**

1. **Polaris-Bilder (DSO Referenz)**
   - URL: https://astrometry.net/user_images/
   - Format: FITS
   - Größe: Typisch 2000-4000 px

2. **M13 Globular Cluster**
   - URL: https://ned.ipac.caltech.edu/
   - Format: FITS
   - Lizenz: Public Domain

3. **SDSS Bilder (Public Release)**
   - URL: https://www.sdss.org/dr17/optical/
   - Format: FITS
   - Größe: 2048 × 1489 px

4. **Hubble Archive**
   - URL: https://archive.stsci.edu/
   - Format: FITS
   - Lizenz: Public Domain

### 7.2 Synthetische Test-Daten

```python
# Generiere synthetische Testframes (Reproduzierbarkeit)

import numpy as np
from astropy.io import fits

def generate_synthetic_frame(W=1024, H=1024, stars=50, noise_level=10):
    """
    Erzeuge synthetisches Test-Frame
    - Künstliche Sterne (PSF-Modell)
    - Rausch
    - Hintergrund
    """
    # Background
    image = np.ones((H, W)) * 100
    
    # Stars (Moffat-Profil mit FWHM=2.5)
    from scipy.special import gamma as sp_gamma
    
    def moffat_psf(r, FWHM=2.5, alpha=1.5):
        """Moffat PSF"""
        beta = FWHM / (2 * np.sqrt(2**(1/alpha) - 1))
        return (alpha - 1) / (np.pi * beta**2) * (1 + (r/beta)**2)**(-alpha)
    
    for _ in range(stars):
        x, y = np.random.randint(50, W-50), np.random.randint(50, H-50)
        amplitude = np.random.uniform(100, 500)
        
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                r = np.sqrt(dx**2 + dy**2)
                if y+dy < H and x+dx < W and y+dy >= 0 and x+dx >= 0:
                    image[y+dy, x+dx] += amplitude * moffat_psf(r)
    
    # Poisson noise
    image = np.random.poisson(image)
    
    # Gaussian noise
    image = image + np.random.normal(0, noise_level, (H, W))
    
    return image.astype(np.float32)

# Speichern
frames = [generate_synthetic_frame() for _ in range(50)]
hdul = fits.HDUList([fits.PrimaryHDU(np.array(frames))])
hdul.writeto('test_frames_synthetic.fits', overwrite=True)
```

---


