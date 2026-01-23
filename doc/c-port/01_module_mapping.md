# Modul-Mapping: Python → C++

## Übersicht

Dieses Dokument beschreibt die detaillierte Zuordnung aller Python-Module zu ihren C++ Äquivalenten.

---

## runner/ → tile_compile_cpp/src/runner/

### phases_impl.py → pipeline/phases_impl.cpp

**Größe**: ~4400 Zeilen (größte Datei)

**Funktionen zu portieren**:
```cpp
namespace tile_compile::pipeline {

class PhasesImpl {
public:
    bool run_phases_impl(
        const std::string& run_id,
        std::ostream& log_fp,
        bool dry_run,
        const fs::path& run_dir,
        const fs::path& project_root,
        const Config& cfg,
        const std::vector<fs::path>& frames,
        bool stop_flag = false,
        std::optional<int> resume_from_phase = std::nullopt
    );

private:
    // Phase-Implementierungen
    bool phase_0_calibration(...);
    bool phase_1_registration(...);
    bool phase_2_channel_split(...);
    bool phase_3_normalization(...);
    bool phase_4_global_metrics(...);
    bool phase_5_tile_grid(...);
    bool phase_6_local_metrics(...);
    bool phase_7_quality_indices(...);
    bool phase_8_clustering(...);
    bool phase_9_synthetic_frames(...);
    bool phase_10_reconstruction(...);
    bool phase_11_output(...);
};

} // namespace tile_compile::pipeline
```

**Abhängigkeiten**:
- OpenCV (cv2 → cv::)
- Eigen3 (numpy → Eigen::)
- CFITSIO (astropy.io.fits)

---

### image_processing.py → image/processing.cpp

**Funktionen**:
```cpp
namespace tile_compile::image {

// CFA/Bayer-Verarbeitung
std::map<std::string, Eigen::MatrixXf> split_cfa_channels(
    const Eigen::MatrixXf& mosaic,
    const std::string& bayer_pattern
);

Eigen::MatrixXf reassemble_cfa_mosaic(
    const Eigen::MatrixXf& r_plane,
    const Eigen::MatrixXf& g_plane,
    const Eigen::MatrixXf& b_plane,
    const std::string& bayer_pattern
);

// Demosaicing via OpenCV
Eigen::Tensor<float, 3> demosaic_cfa(
    const Eigen::MatrixXf& mosaic,
    const std::string& bayer_pattern
);

// Normalisierung
Eigen::MatrixXf normalize_frame(
    const Eigen::MatrixXf& frame,
    float frame_median,
    float target_median,
    const std::string& mode
);

// Hotpixel-Korrektur
Eigen::MatrixXf cosmetic_correction(
    const Eigen::MatrixXf& data,
    float sigma_threshold = 8.0f,
    bool hot_only = true
);

// CFA Warping
Eigen::MatrixXf warp_cfa_mosaic_via_subplanes(
    const Eigen::MatrixXf& mosaic,
    const Eigen::Matrix<float, 2, 3>& warp,
    std::optional<std::pair<int, int>> out_shape = std::nullopt,
    const std::string& border_mode = "replicate",
    float border_value = 0.0f,
    const std::string& interpolation = "linear"
);

// Hilfsfunktionen
Eigen::MatrixXf cfa_downsample_sum2x2(const Eigen::MatrixXf& mosaic);
std::map<std::string, Eigen::MatrixXf> split_rgb_frame(const Eigen::Tensor<float, 3>& data);
std::pair<std::vector<float>, float> compute_frame_medians(const std::vector<Eigen::MatrixXf>& frames);

} // namespace tile_compile::image
```

---

### opencv_registration.py → registration/opencv_registration.cpp

**Funktionen**:
```cpp
namespace tile_compile::registration {

// ECC-Bildvorbereitung
cv::Mat opencv_prepare_ecc_image(const cv::Mat& img);

// Stern-Zählung
int opencv_count_stars(const cv::Mat& img01);

// ECC-Warp-Berechnung
std::pair<cv::Mat, float> opencv_ecc_warp(
    const cv::Mat& moving01,
    const cv::Mat& ref01,
    bool allow_rotation,
    const cv::Mat& init_warp
);

// Phase-Korrelation
std::pair<float, float> opencv_phasecorr_translation(
    const cv::Mat& moving01,
    const cv::Mat& ref01
);

// Alignment-Score
float opencv_alignment_score(
    const cv::Mat& moving01,
    const cv::Mat& ref01
);

// Beste initiale Translation finden
cv::Mat opencv_best_translation_init(
    const cv::Mat& moving01,
    const cv::Mat& ref01,
    bool rotation_sweep = true,
    float rotation_range_deg = 5.0f,
    int rotation_steps = 11
);

} // namespace tile_compile::registration
```

---

### calibration.py → calibration/calibration.cpp

**Funktionen**:
```cpp
namespace tile_compile::calibration {

struct MasterFrame {
    Eigen::MatrixXf data;
    FitsHeader header;
};

std::optional<MasterFrame> build_master_mean(const std::vector<fs::path>& paths);

std::optional<MasterFrame> bias_correct_dark(
    const std::optional<MasterFrame>& dark_master,
    const std::optional<MasterFrame>& bias_master
);

std::optional<MasterFrame> prepare_flat(
    const std::optional<MasterFrame>& flat_master,
    const std::optional<MasterFrame>& bias_master,
    const std::optional<MasterFrame>& dark_master
);

Eigen::MatrixXf apply_calibration(
    const Eigen::MatrixXf& img,
    const Eigen::MatrixXf* bias_arr,
    const Eigen::MatrixXf* dark_arr,
    const Eigen::MatrixXf* flat_arr,
    float denom_eps = 1e-6f
);

} // namespace tile_compile::calibration
```

---

### siril_utils.py → ENTFÄLLT

**Hinweis**: Siril wird in der C++ Portierung **nicht mehr verwendet**. Die gesamte Registrierung erfolgt nativ mit OpenCV (ECC, Phase Correlation). Diese Datei muss nicht portiert werden.

---

### fits_utils.py → io/fits_utils.cpp

```cpp
namespace tile_compile::io {

bool is_fits_image_path(const fs::path& path);

std::pair<Eigen::MatrixXf, FitsHeader> read_fits_float(const fs::path& path);

void write_fits_float(
    const fs::path& path,
    const Eigen::MatrixXf& data,
    const FitsHeader& header
);

bool fits_is_cfa(const FitsHeader& header);
std::string fits_get_bayerpat(const FitsHeader& header);

} // namespace tile_compile::io
```

---

### events.py → core/events.cpp

```cpp
namespace tile_compile::core {

class EventEmitter {
public:
    void emit(const nlohmann::json& event, std::ostream& log_fp);
    
    void phase_start(const std::string& run_id, int phase, 
                     const std::string& name, std::ostream& log_fp);
    void phase_end(const std::string& run_id, int phase,
                   const std::string& status, std::ostream& log_fp);
    void phase_progress(const std::string& run_id, int phase,
                        float progress, const std::string& message,
                        std::ostream& log_fp);
    
    bool stop_requested() const;
    void request_stop();

private:
    std::atomic<bool> stop_flag_{false};
};

} // namespace tile_compile::core
```

---

### utils.py → core/utils.cpp

```cpp
namespace tile_compile::core {

void safe_symlink_or_copy(const fs::path& src, const fs::path& dst);
void safe_hardlink_or_copy(const fs::path& src, const fs::path& dst);
fs::path pick_output_file(const fs::path& dir, const std::string& prefix, 
                          const std::string& ext);

std::vector<fs::path> discover_frames(const fs::path& input_dir, 
                                       const std::string& pattern);
std::vector<uint8_t> read_bytes(const fs::path& path);
std::string sha256_bytes(const std::vector<uint8_t>& data);
void copy_config(const fs::path& src, const fs::path& dst);
fs::path resolve_project_root(const fs::path& config_path);
} // namespace tile_compile::core
```

---

## tile_compile_backend/ → tile_compile_cpp/src/backend/

### metrics.py → metrics/metrics.cpp

```cpp
namespace tile_compile::metrics {

// Wiener-Filter
Eigen::MatrixXf wiener_tile_filter(
    const Eigen::MatrixXf& tile,
    float sigma,
    float snr_tile,
    float q_struct_tile,
    bool is_star_tile,
    float snr_threshold = 5.0f,
    float q_min = -0.5f,
    float eps = 1e-12f
);

struct GlobalMetrics {
    std::vector<float> background_level;
    std::vector<float> noise_level;
    std::vector<float> gradient_energy;
    std::vector<float> Q_f;
    std::vector<float> Q_f_clamped;
    std::vector<float> G_f_c;
    std::map<std::string, float> weights;
    int n_frames;
};

class MetricsCalculator {
public:
    static GlobalMetrics calculate_global_metrics(
        const std::vector<Eigen::MatrixXf>& frames,
        const std::map<std::string, float>& weights = {},
        std::pair<float, float> clamp_range = {-3.0f, 3.0f}
    );
    
private:
    static float calculate_gradient_energy(const Eigen::MatrixXf& frame);
};

struct TileMetrics {
    std::vector<float> fwhm;
    std::vector<float> roundness;
    std::vector<float> contrast;
    std::vector<float> background_level;
    std::vector<float> noise_level;
    std::vector<float> gradient_energy;
};

class TileMetricsCalculator {
public:
    TileMetricsCalculator(int tile_size = 64, float overlap = 0.25f);
    
    TileMetrics calculate_tile_metrics(
        const Eigen::MatrixXf& frame,
        const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>* valid_mask = nullptr
    );

private:
    int tile_size_;
    float overlap_;
    
    float calculate_fwhm(const Eigen::MatrixXf& tile);
    float calculate_roundness(const Eigen::MatrixXf& tile);
    float calculate_contrast(const Eigen::MatrixXf& tile);
    float calculate_gradient_energy(const Eigen::MatrixXf& tile);
};

} // namespace tile_compile::metrics
```

---

### clustering.py → clustering/clustering.cpp

```cpp
namespace tile_compile::clustering {

struct ClusteringResult {
    std::vector<int> cluster_labels;
    std::vector<std::vector<float>> cluster_centers;
    std::map<int, std::map<std::string, std::variant<int, std::vector<float>>>> cluster_stats;
    int n_clusters;
    float silhouette_score;
    std::string method;  // "kmeans" oder "quantile_fallback"
};

class StateClustering {
public:
    static ClusteringResult cluster_frames(
        const std::vector<Eigen::MatrixXf>& frames,
        const std::map<std::string, std::any>& metrics,
        const std::map<std::string, std::any>& config = {}
    );
    
    static ClusteringResult cluster_frames_quantile_fallback(
        const std::vector<Eigen::MatrixXf>& frames,
        const std::map<std::string, std::any>& metrics,
        const std::map<std::string, std::any>& config = {}
    );

private:
    static Eigen::MatrixXf compute_state_vectors(
        const std::vector<Eigen::MatrixXf>& frames,
        const std::map<std::string, std::any>& metrics
    );
    
    static ClusteringResult find_optimal_clustering(
        const Eigen::MatrixXf& data,
        int min_clusters,
        int max_clusters
    );
};

std::map<std::string, ClusteringResult> cluster_channels(
    const std::map<std::string, std::vector<Eigen::MatrixXf>>& channels,
    const std::map<std::string, std::map<std::string, std::any>>& metrics,
    const std::map<std::string, std::any>& config = {}
);

} // namespace tile_compile::clustering
```

---

### reconstruction.py → reconstruction/reconstruction.cpp

```cpp
namespace tile_compile::reconstruction {

constexpr float DEFAULT_EPSILON = 1e-10f;

struct ReconstructionResult {
    Eigen::MatrixXf reconstructed;
    std::vector<std::pair<int, int>> fallback_tiles;
    int n_fallback;
    bool fallback_used;
};

class TileReconstructor {
public:
    TileReconstructor(int tile_size = 64, float overlap = 0.25f, 
                      float epsilon = DEFAULT_EPSILON);
    
    ReconstructionResult reconstruct_channel(
        const std::vector<Eigen::MatrixXf>& frames,
        const std::map<std::string, std::any>& metrics
    );

private:
    int tile_size_;
    float overlap_;
    float epsilon_;
    std::vector<std::pair<int, int>> fallback_tiles_;
    
    Eigen::VectorXf get_local_quality(
        const std::map<std::string, std::any>& metrics,
        int tile_idx,
        int n_frames
    );
    
    Eigen::MatrixXf create_blending_window(int h, int w);
};

struct ChannelReconstructionResult {
    std::map<std::string, ReconstructionResult> channels;
    int total_fallback_tiles;
    bool any_fallback;
};

ChannelReconstructionResult reconstruct_channels(
    const std::map<std::string, std::vector<Eigen::MatrixXf>>& channels,
    const std::map<std::string, std::map<std::string, std::any>>& metrics,
    const std::map<std::string, std::any>& config = {}
);

} // namespace tile_compile::reconstruction
```

---

### synthetic.py → synthetic/synthetic.cpp

```cpp
namespace tile_compile::synthetic {

class SyntheticFrameGenerator {
public:
    static std::vector<Eigen::MatrixXf> generate_synthetic_frames(
        const std::vector<Eigen::MatrixXf>& input_frames,
        const std::map<std::string, std::any>& metrics,
        const std::map<std::string, std::any>& config = {},
        const std::map<std::string, std::any>& clustering_results = {}
    );

private:
    static std::vector<Eigen::MatrixXf> generate_from_clusters(
        const std::vector<Eigen::MatrixXf>& frames,
        const std::map<std::string, std::any>& metrics,
        const std::map<std::string, std::any>& clustering_results
    );
    
    static std::vector<Eigen::MatrixXf> generate_quantile_based(
        const std::vector<Eigen::MatrixXf>& frames,
        const std::map<std::string, std::any>& metrics,
        int n_synthetic
    );
    
    static Eigen::MatrixXf weighted_average(
        const std::vector<Eigen::MatrixXf>& frames,
        const Eigen::VectorXf& weights
    );
};

std::map<std::string, std::vector<Eigen::MatrixXf>> generate_channel_synthetic_frames(
    const std::map<std::string, std::vector<Eigen::MatrixXf>>& channels,
    const std::map<std::string, std::map<std::string, std::any>>& metrics,
    const std::map<std::string, std::any>& config = {},
    const std::map<std::string, std::any>& clustering_results = {}
);

} // namespace tile_compile::synthetic
```

---

### sigma_clipping.py → stacking/sigma_clipping.cpp

```cpp
namespace tile_compile::stacking {

struct SigmaClipConfig {
    float sigma_low = 3.0f;
    float sigma_high = 3.0f;
    int max_iters = 3;
    float min_fraction = 0.5f;
    
    SigmaClipConfig clamp() const;
};

struct SigmaClipResult {
    Eigen::MatrixXf clipped_mean;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask;
    std::map<std::string, std::variant<int, float, std::string>> stats;
};

SigmaClipResult sigma_clip_stack_nd(
    const std::vector<Eigen::MatrixXf>& frames,
    const SigmaClipConfig& cfg = {},
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>* valid_mask = nullptr
);

Eigen::MatrixXf simple_mean_stack_nd(
    const std::vector<Eigen::MatrixXf>& frames,
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>* valid_mask = nullptr
);

} // namespace tile_compile::stacking
```

---

### tile_grid.py → grid/tile_grid.cpp

```cpp
namespace tile_compile::grid {

int compute_tile_size_v3(
    int image_width,
    int image_height,
    float fwhm,
    float size_factor = 8.0f,
    int min_size = 32,
    int max_divisor = 4
);

std::pair<int, int> compute_overlap_v3(int tile_size, float overlap_fraction = 0.25f);

struct FrameAnalysis {
    float mean_intensity;
    float std_intensity;
    float gradient_complexity;
    int star_density;
    std::pair<int, int> shape;
};

struct GridMetadata {
    int total_tiles;
    std::vector<std::tuple<int, int, int, int>> tile_coordinates;
    float coverage_percentage;
};

struct TileGridResult {
    std::vector<Eigen::MatrixXf> tiles;
    int tile_size;
    float overlap;
    int overlap_px;
    FrameAnalysis frame_metadata;
    GridMetadata grid_metadata;
};

class TileGridGenerator {
public:
    static TileGridResult generate_adaptive_grid(
        const Eigen::MatrixXf& frame,
        const std::map<std::string, std::any>& config = {}
    );

private:
    static FrameAnalysis analyze_frame_characteristics(const Eigen::MatrixXf& frame);
    static int compute_adaptive_tile_size(
        std::pair<int, int> frame_shape,
        const FrameAnalysis& analysis,
        int min_tile_size,
        int max_tile_size
    );
};

std::map<std::string, TileGridResult> generate_multi_channel_grid(
    const std::map<std::string, Eigen::MatrixXf>& channels,
    const std::map<std::string, std::any>& config = {}
);

} // namespace tile_compile::grid
```

---

### linearity.py → validation/linearity.cpp

```cpp
namespace tile_compile::validation {

struct MomentTest {
    float skewness;
    float kurtosis;
    float variance_coefficient;
};

struct SpectralTest {
    float wavelet_coherence;
    float energy_ratio;
};

struct SpatialTest {
    float gradient_consistency;
    float edge_uniformity;
};

struct LinearityResult {
    bool is_linear;
    MomentTest moment_test;
    SpectralTest spectral_test;
    SpatialTest spatial_test;
    float linearity_score;
    std::vector<std::string> diagnostics;
};

class LinearityValidator {
public:
    static LinearityResult validate_frame_linearity(
        const Eigen::MatrixXf& frame,
        const std::map<std::string, std::any>& config = {}
    );

private:
    static MomentTest moment_linearity_test(const Eigen::MatrixXf& frame);
    static SpectralTest spectral_linearity_test(const Eigen::MatrixXf& frame);
    static SpatialTest spatial_linearity_test(const Eigen::MatrixXf& frame);
};

struct FramesLinearityResult {
    std::vector<LinearityResult> results;
    std::vector<Eigen::MatrixXf> valid_frames;
    std::vector<Eigen::MatrixXf> rejected_frames;
    float overall_linearity;
};

FramesLinearityResult validate_frames_linearity(
    const std::vector<Eigen::MatrixXf>& frames,
    const std::map<std::string, std::any>& config = {}
);

} // namespace tile_compile::validation
```

---

### validate.py → config/validate.cpp

```cpp
namespace tile_compile::config {

struct ValidationIssue {
    std::string severity;  // "error" | "warning"
    std::string code;
    std::string path;
    std::string message;
};

struct ValidationResult {
    bool valid;
    std::vector<ValidationIssue> errors;
    std::vector<ValidationIssue> warnings;
};

ValidationResult validate_config_yaml_text(
    const std::string& yaml_text,
    const std::optional<fs::path>& schema_path = std::nullopt
);

} // namespace tile_compile::config
```

---

## Zusammenfassung

| Python-Modul | C++ Namespace | Priorität |
|--------------|---------------|-----------|
| phases_impl.py | tile_compile::pipeline | Hoch |
| image_processing.py | tile_compile::image | Hoch |
| opencv_registration.py | tile_compile::registration | Hoch |
| metrics.py | tile_compile::metrics | Hoch |
| reconstruction.py | tile_compile::reconstruction | Hoch |
| clustering.py | tile_compile::clustering | Mittel |
| synthetic.py | tile_compile::synthetic | Mittel |
| sigma_clipping.py | tile_compile::stacking | Mittel |
| tile_grid.py | tile_compile::grid | Mittel |
| calibration.py | tile_compile::calibration | Mittel |
| linearity.py | tile_compile::validation | Niedrig |
| validate.py | tile_compile::config | Niedrig |
| fits_utils.py | tile_compile::io | Hoch |
| events.py | tile_compile::core | Mittel |
| utils.py | tile_compile::core | Mittel |
| siril_utils.py | ENTFÄLLT | - (nicht portiert, OpenCV-native Registrierung) |
