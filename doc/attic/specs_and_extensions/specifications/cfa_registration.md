# CFA-aware Registration Specification (Methodik v3)

## 1. Objective
Develop a robust, physics-aware registration method for Color Filter Array (CFA) astronomical images that:
- Preserves color channel information
- Minimizes interpolation artifacts
- Provides precise geometric alignment

## 2. Key Requirements

### 2.1 Input Constraints
- Raw OSC/CFA frames
- Minimal prior processing
- Linear, unscaled data

### 2.2 Registration Principles
- Subplane-aware transformation
- No color channel interpolation
- Minimal information loss
- Robust to varying seeing conditions

## 3. Transformation Estimation

### 3.1 Reference Frame Selection
- Criteria:
  - Maximum star count
  - Highest signal-to-noise ratio
  - Minimal optical aberrations

### 3.2 Star Detection
- CFA-aware star detection algorithm
- Multi-scale detection
- Bayer pattern consideration
- Minimum star match threshold (10-15 stars)

### 3.3 Transformation Types
1. Translation
2. Rigid rotation
3. Affine transformation
4. Homography (optional)

### 3.4 Estimation Methods
- RANSAC for outlier rejection
- Enhanced correlation techniques
- Sub-pixel registration

## 4. Transformation Validation

### 4.1 Geometric Consistency
- Maximum allowed translation
- Rotation angle limits
- Scale change constraints

### 4.2 Quality Metrics
- Correlation coefficient
- Geometric error
- Star matching score
- Residual transformation parameters

## 5. Subplane Processing

### 5.1 Bayer Pattern Handling
Support patterns:
- RGGB
- BGGR
- GBRG
- GRBG

### 5.2 Independent Subplane Warping
- R, G1, G2, B planes processed separately
- Consistent transformation across subplanes
- No interpolation between planes

## 6. Performance Constraints
- Computational complexity: O(n log n)
- Memory usage: Linear with frame count
- Deterministic behavior

## 7. Error Handling
- Detailed error classification
- Graceful failure modes
- Comprehensive logging

## 8. Output Specifications
- Registered CFA frames
- Transformation matrices
- Registration quality metrics
- Diagnostic information

## 9. Validation Criteria
1. Geometric accuracy
2. Minimal information loss
3. Consistent across different datasets
4. Reproducible results

## 10. Rejection Criteria
- Insufficient star matches
- Excessive geometric distortion
- Non-linear transformations
- Outlier frames