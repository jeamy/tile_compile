# Astronomical Image Reconstruction Methods Comparison

## 1. Traditional Stacking Methods

### Conventional Approach
- Simple mean or median stacking
- Uniform frame contribution
- No quality-based weighting

### Tile-Based Method Advantages
- Adaptive quality weighting
- Orthogonal metric assessment
- Preserves local image details
- Dynamically handles varying observing conditions

## 2. Drizzle Technique (Hubble Space Telescope)

### Original Purpose
- Increase spatial resolution
- Subpixel image combination
- Primarily used with space-based observations

### Tile-Based Method Differences
- Explicitly rejects drizzle technique
- Focuses on quality reconstruction
- Optimized for ground-based observations
- Preserves original spatial sampling

## 3. LRGB Combination Methods

### Traditional Approach
- Separate processing of luminance and color channels
- Manual weighting
- Subjective quality assessment

### Tile-Based Method Innovations
- Objective, metrics-driven channel processing
- Automatic quality assessment
- Channel-independent reconstruction
- Systematic state-based clustering

## 4. Machine Learning Reconstruction

### Deep Learning Approaches
- Neural network-based image enhancement
- Black-box quality improvement
- Requires extensive training data

### Tile-Based Method Advantages
- Physical model-based
- Interpretable quality metrics
- No training data required
- Preserves astronomical signal characteristics

## 5. Frame Selection Methods

### Traditional Approach
- Manual frame rejection
- Subjective quality criteria
- Potential significant information loss

### Tile-Based Method Innovations
- No frame selection
- Every frame contributes
- Quality-weighted contribution
- Preserves all available information

## 6. Registration Techniques

### Conventional Methods
- Global transformation estimation
- Limited local detail preservation
- Uniform registration approach

### Tile-Based Method Advancements
- Two parallel registration paths
  1. Siril-based (proven)
  2. CFA-aware (experimental)
- Seeing-adaptive tile geometry
- Local transformation quality assessment

## Unique Methodological Characteristics

### 1. Orthogonal Quality Metrics
- Global atmospheric quality assessment
- Local seeing condition evaluation
- Structural information preservation

### 2. State-Based Clustering
- Comprehensive frame state vector representation
- 15-30 cluster approach
- Synthetic frame generation

### 3. Computational Approach
- Strictly linear pipeline
- No feedback loops
- Deterministic processing
- Channel-independent reconstruction

## Comparative Performance Expectations

### Anticipated Improvements
- Enhanced signal-to-noise ratio
- Improved local detail preservation
- Reduced atmospheric artifacts
- More consistent image reconstruction

### Validation Metrics
- FWHM (Full Width at Half Maximum) improvement
- Signal-to-noise ratio enhancement
- Background noise reduction
- Detail preservation assessment

## Limitations and Considerations

### Computational Aspects
- Higher computational complexity
- Requires significant computational resources
- Potential for parallel processing optimization

### Parameter Sensitivity
- Requires careful parameter tuning
- Needs robust validation of quality thresholds
- Adaptable to various observing conditions

## 7. Lucky Imaging Techniques

### Traditional Approach
- Select only best frames (typically <10%)
- Very high quality per selected frame
- Massive information loss

### Tile-Based Method Comparison
- Uses **all** frames weighted by quality
- No arbitrary selection threshold
- Tile-local "lucky" weighting preserves best local regions
- Information-theoretically optimal for large datasets

## 8. Sigma Clipping / Rejection Methods

### Traditional Approach
- Statistical outlier rejection per pixel
- Removes cosmic rays, satellites, hot pixels
- Global threshold (e.g., 3σ)

### Tile-Based Method Integration
- Compatible as pre-processing step
- Weight-based approach naturally down-weights outliers
- No explicit rejection needed for atmospheric variations
- Recommendation: Use sigma-clipping only for transient artifacts (cosmic rays)

## 9. HDR / Multi-Exposure Fusion

### Traditional Approach
- Combine exposures of different lengths
- Tone mapping for display
- Often non-linear

### Tile-Based Method Position
- **Strictly linear** – no HDR tone mapping
- Designed for uniform short exposures
- HDR combination is external post-processing (outside methodology)

## 10. Wavelet-Based Enhancement

### Traditional Approach
- Multi-scale decomposition
- Selective sharpening per scale
- Risk of over-processing

### Tile-Based Method Position
- No wavelet enhancement in core pipeline
- Quality metrics could use wavelet features (extension)
- Sharpening is post-processing concern

## Quantitative Comparison Matrix

| Method | Uses All Frames | Spatially Adaptive | Linear | No Training Data | Interpretable |
|--------|-----------------|-------------------|--------|------------------|---------------|
| Mean Stack | ✓ | ✗ | ✓ | ✓ | ✓ |
| Median Stack | ✓ | ✗ | ✗ | ✓ | ✓ |
| Lucky Imaging | ✗ | ✗ | ✓ | ✓ | ✓ |
| Drizzle | ✓ | ✗ | ✓ | ✓ | ✓ |
| ML Denoising | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Tile-Based** | **✓** | **✓** | **✓** | **✓** | **✓** |

## Conclusion

The tile-based quality reconstruction method represents a sophisticated, physically-grounded approach to astronomical image processing, offering a systematic, objective alternative to traditional stacking techniques.

### Key Differentiators
1. **Physical model** rather than heuristics
2. **Spatio-temporal quality map** instead of frame selection
3. **Orthogonal metric decomposition** (global atmospheric + local seeing)
4. **Deterministic and reproducible** processing
5. **Scales with dataset size** – benefits from more frames