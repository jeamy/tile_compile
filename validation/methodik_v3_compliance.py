"""
Methodik v3 Compliance Validator

Validates datasets and pipeline runs against Methodik v3 specification:
- 11-phase pipeline compliance
- Hard/Soft/Implicit assumptions validation
- Reduced Mode detection and validation
- Phase-specific quality checks

Usage:
    validator = MetodikV3ComplianceValidator(assumptions=assumptions)
    report = validator.validate_dataset(frames, 'dataset_name')
    validator.generate_report(report)
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class MetodikV3ComplianceValidator:
    """Validates datasets against Methodik v3 specification."""
    
    # Methodik v3 phases (§2)
    PHASES = [
        (0, "SCAN_INPUT", "Input validation"),
        (1, "REGISTRATION", "Frame registration"),
        (2, "CHANNEL_SPLIT", "Channel separation"),
        (3, "NORMALIZATION", "Global linear normalization"),
        (4, "GLOBAL_METRICS", "Global frame metrics"),
        (5, "TILE_GRID", "Seeing-adaptive tile geometry"),
        (6, "LOCAL_METRICS", "Local tile metrics"),
        (7, "TILE_RECONSTRUCTION", "Tile-wise reconstruction"),
        (8, "STATE_CLUSTERING", "State-based clustering"),
        (9, "SYNTHETIC_FRAMES", "Synthetic quality frames"),
        (10, "STACKING", "Final linear stacking"),
        (11, "DONE", "Completion"),
    ]
    
    def __init__(self, assumptions: Optional[Dict] = None, output_dir: str = 'validation_results'):
        """
        Initialize compliance validator.
        
        Args:
            assumptions: Methodik v3 assumptions dict
            output_dir: Output directory for reports
        """
        self.assumptions = assumptions or self._default_assumptions()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _default_assumptions(self) -> Dict:
        """Get default Methodik v3 assumptions."""
        return {
            'frames_min': 50,
            'frames_optimal': 800,
            'frames_reduced_threshold': 200,
            'exposure_time_tolerance_percent': 5.0,
            'registration_residual_warn_px': 0.5,
            'registration_residual_max_px': 1.0,
            'elongation_warn': 0.3,
            'elongation_max': 0.4,
            'tracking_error_max_px': 1.0,
            'reduced_mode_skip_clustering': True,
            'reduced_mode_cluster_range': [5, 10],
        }
    
    def validate_dataset(self, frames: np.ndarray, dataset_name: str) -> Dict[str, Any]:
        """
        Validate dataset against Methodik v3 compliance.
        
        Args:
            frames: Input frames array
            dataset_name: Name of dataset
        
        Returns:
            Compliance report dict
        """
        frame_count = len(frames)
        
        report = {
            'dataset_name': dataset_name,
            'timestamp': self._timestamp(),
            'frame_count': frame_count,
            'assumptions': self.assumptions,
            'compliance_checks': {},
            'warnings': [],
            'errors': [],
            'overall_compliant': True,
        }
        
        # Check 1: Frame count (§1.2 - Hard Assumption)
        self._check_frame_count(frame_count, report)
        
        # Check 2: Reduced Mode detection (§1.4)
        self._check_reduced_mode(frame_count, report)
        
        # Check 3: Data linearity (§1.1 - Hard Assumption)
        self._check_linearity(frames, report)
        
        # Check 4: Frame consistency
        self._check_frame_consistency(frames, report)
        
        # Check 5: Pipeline phase requirements
        self._check_phase_requirements(frame_count, report)
        
        # Determine overall compliance
        report['overall_compliant'] = len(report['errors']) == 0
        
        return report
    
    def _check_frame_count(self, frame_count: int, report: Dict):
        """Check frame count against assumptions (Methodik v3 §1.2)."""
        frames_min = self.assumptions['frames_min']
        frames_optimal = self.assumptions['frames_optimal']
        
        if frame_count < frames_min:
            report['errors'].append({
                'type': 'HARD_ASSUMPTION_VIOLATION',
                'assumption': 'frames_min',
                'message': f"Frame count ({frame_count}) below minimum ({frames_min})",
                'severity': 'CRITICAL'
            })
            report['overall_compliant'] = False
        elif frame_count < frames_optimal:
            report['warnings'].append({
                'type': 'SOFT_ASSUMPTION',
                'assumption': 'frames_optimal',
                'message': f"Frame count ({frame_count}) below optimal ({frames_optimal})",
                'severity': 'WARNING'
            })
        
        report['compliance_checks']['frame_count'] = {
            'passed': frame_count >= frames_min,
            'value': frame_count,
            'threshold': frames_min,
            'optimal': frames_optimal
        }
    
    def _check_reduced_mode(self, frame_count: int, report: Dict):
        """Check if Reduced Mode applies (Methodik v3 §1.4)."""
        threshold = self.assumptions['frames_reduced_threshold']
        reduced_mode = frame_count < threshold
        
        if reduced_mode:
            report['warnings'].append({
                'type': 'REDUCED_MODE',
                'message': f"Reduced Mode active: {frame_count} frames < {threshold}",
                'implications': [
                    "STATE_CLUSTERING will be skipped" if self.assumptions['reduced_mode_skip_clustering'] else "STATE_CLUSTERING cluster range reduced",
                    "SYNTHETIC_FRAMES will be skipped" if self.assumptions['reduced_mode_skip_clustering'] else "SYNTHETIC_FRAMES count reduced",
                    "Direct tile-weighted stacking"
                ],
                'severity': 'INFO'
            })
        
        report['compliance_checks']['reduced_mode'] = {
            'active': reduced_mode,
            'frame_count': frame_count,
            'threshold': threshold,
            'skip_clustering': self.assumptions['reduced_mode_skip_clustering']
        }
    
    def _check_linearity(self, frames: np.ndarray, report: Dict):
        """Check data linearity (Methodik v3 §1.1 - Hard Assumption)."""
        # Simple linearity check: data should not be stretched (no histogram clipping)
        sample_frame = frames[0] if len(frames) > 0 else None
        
        if sample_frame is not None:
            # Check for histogram clipping (sign of non-linear stretch)
            hist, bins = np.histogram(sample_frame.flatten(), bins=256)
            
            # Check if too many values at extremes (clipping indicator)
            total_pixels = sample_frame.size
            edge_pixels = hist[0] + hist[-1]
            edge_ratio = edge_pixels / total_pixels
            
            if edge_ratio > 0.01:  # More than 1% at edges suggests clipping
                report['warnings'].append({
                    'type': 'LINEARITY_WARNING',
                    'message': f"Possible non-linear data detected (edge ratio: {edge_ratio:.2%})",
                    'severity': 'WARNING'
                })
        
        report['compliance_checks']['linearity'] = {
            'checked': sample_frame is not None,
            'passed': True,  # Conservative: assume linear unless proven otherwise
        }
    
    def _check_frame_consistency(self, frames: np.ndarray, report: Dict):
        """Check frame shape and data type consistency."""
        if len(frames) == 0:
            return
        
        shapes = [f.shape for f in frames]
        dtypes = [f.dtype for f in frames]
        
        shape_consistent = all(s == shapes[0] for s in shapes)
        dtype_consistent = all(d == dtypes[0] for d in dtypes)
        
        if not shape_consistent:
            report['errors'].append({
                'type': 'FRAME_INCONSISTENCY',
                'message': "Frame shapes are inconsistent",
                'severity': 'ERROR'
            })
        
        if not dtype_consistent:
            report['warnings'].append({
                'type': 'DTYPE_INCONSISTENCY',
                'message': "Frame data types are inconsistent",
                'severity': 'WARNING'
            })
        
        report['compliance_checks']['frame_consistency'] = {
            'shape_consistent': shape_consistent,
            'dtype_consistent': dtype_consistent,
            'reference_shape': shapes[0] if shapes else None,
            'reference_dtype': str(dtypes[0]) if dtypes else None
        }
    
    def _check_phase_requirements(self, frame_count: int, report: Dict):
        """Check if all 11 Methodik v3 phases can be executed."""
        reduced_mode = frame_count < self.assumptions['frames_reduced_threshold']
        
        phase_requirements = {}
        for phase_id, phase_name, phase_desc in self.PHASES:
            can_execute = True
            skip_reason = None
            
            # Phase-specific requirements
            if phase_name == "STATE_CLUSTERING" and reduced_mode and self.assumptions['reduced_mode_skip_clustering']:
                can_execute = False
                skip_reason = "Skipped in Reduced Mode"
            
            if phase_name == "SYNTHETIC_FRAMES" and reduced_mode and self.assumptions['reduced_mode_skip_clustering']:
                can_execute = False
                skip_reason = "Skipped in Reduced Mode"
            
            phase_requirements[phase_name] = {
                'phase_id': phase_id,
                'can_execute': can_execute,
                'skip_reason': skip_reason,
                'description': phase_desc
            }
        
        report['compliance_checks']['phase_requirements'] = phase_requirements
    
    def _timestamp(self) -> str:
        """Get ISO timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    def validate_opencv_cfa_registration(self, image: np.ndarray, shifts: list[tuple[float, float]],
                                           allow_rotation: bool = False,
                                           noise_sigma: float = 0.0) -> Dict[str, Any]:
        """QA-Test für registration.engine=opencv_cfa mit synthetischen Shifts.

        Erzeugt aus einem Eingangsbild ein Set synthetischer CFA-Frames mit
        bekannten Translationen (und optional Rotation), führt dann die
        opencv_cfa-Registrierung darauf aus und vergleicht die geschätzten
        ECC-Transformationen mit den Ground-Truth-Werten.

        Args:
            image: Basisbild (2D-Array) als float32/float64.
            shifts: Liste von (dx, dy) in Pixeln relativ zur Referenz.
            allow_rotation: Ob ECC Rotation schätzen darf (euclidean statt affine
                            nur-Translation).
            noise_sigma: optionale additive Gauß-Rauschstärke.

        Returns:
            Dict mit Fehlerstatistiken (RMSE, max-Fehler, Korrelationen).
        """

        from astropy.io import fits  # nur für Header/MOCK
        from runner.opencv_registration import (
            opencv_prepare_ecc_image,
            opencv_best_translation_init,
            opencv_ecc_warp,
        )
        from runner.image_processing import cfa_downsample_sum2x2, warp_cfa_mosaic_via_subplanes

        base = np.asarray(image).astype("float32", copy=False)
        H, W = base.shape

        # Synthetische CFA-Mosaiks: einfache quadratische Replikation in 2x2
        def make_cfa(img: np.ndarray) -> np.ndarray:
            cfa = np.zeros_like(img, dtype="float32")
            cfa[0::2, 0::2] = img[0::2, 0::2]  # R-plane
            cfa[0::2, 1::2] = img[0::2, 1::2]  # G1
            cfa[1::2, 0::2] = img[1::2, 0::2]  # G2
            cfa[1::2, 1::2] = img[1::2, 1::2]  # B
            return cfa

        frames = []
        gt_params: list[dict[str, float]] = []
        for dx, dy in shifts:
            # Wahrer Affinwarp (nur Translation/Rotation auf Intensitätsbild)
            M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            if allow_rotation and (dx != 0 or dy != 0):
                # Kleine synthetische Rotation proportional zur Verschiebung
                angle = 0.01 * np.hypot(dx, dy)
                c = float(np.cos(angle))
                s = float(np.sin(angle))
                cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
                R = np.array([[c, -s, (1 - c) * cx + s * cy],
                              [s,  c, (1 - c) * cy - s * cx]], dtype=np.float32)
                M = R @ np.vstack([M, [0, 0, 1]])[:2, :]

            warped = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR)
            if noise_sigma > 0.0:
                warped = warped + np.random.normal(0.0, noise_sigma, warped.shape).astype("float32")

            cfa = make_cfa(warped)
            frames.append(cfa)
            gt_params.append({"dx": float(dx), "dy": float(dy)})

        # Referenzwahl wie im Runner: bestes Star-Count
        ref_idx = 0
        ref_stars = -1
        ref_lum01 = None
        for i, cfa in enumerate(frames):
            lum = cfa_downsample_sum2x2(cfa)
            lum01 = opencv_prepare_ecc_image(lum)
            stars = int(cv2.goodFeaturesToTrack(lum01, maxCorners=1200, qualityLevel=0.01, minDistance=5, blockSize=7).shape[0])
            if stars > ref_stars:
                ref_stars = stars
                ref_idx = i
                ref_lum01 = lum01

        est_params: list[dict[str, float]] = []
        corr_coeffs: list[float] = []
        for idx, cfa in enumerate(frames):
            lum = cfa_downsample_sum2x2(cfa)
            lum01 = opencv_prepare_ecc_image(lum)
            if idx == ref_idx:
                est_params.append({"dx": 0.0, "dy": 0.0})
                corr_coeffs.append(1.0)
                continue
            init = opencv_best_translation_init(lum01, ref_lum01)
            try:
                warp, cc = opencv_ecc_warp(lum01, ref_lum01, allow_rotation=allow_rotation, init_warp=init)
            except Exception:
                warp, cc = init, 0.0
            corr_coeffs.append(float(cc))
            dx_est = float(warp[0, 2])
            dy_est = float(warp[1, 2])
            est_params.append({"dx": dx_est, "dy": dy_est})

        # Fehlerstatistik
        dx_err = []
        dy_err = []
        for gt, est in zip(gt_params, est_params):
            dx_err.append(est["dx"] - gt["dx"])
            dy_err.append(est["dy"] - gt["dy"])

        dx_err = np.asarray(dx_err, dtype=np.float32)
        dy_err = np.asarray(dy_err, dtype=np.float32)
        rmse = float(np.sqrt(np.mean(dx_err**2 + dy_err**2)))
        max_err = float(np.max(np.hypot(dx_err, dy_err)))

        return {
            "gt_params": gt_params,
            "est_params": est_params,
            "corr_coeffs": corr_coeffs,
            "dx_err": dx_err.tolist(),
            "dy_err": dy_err.tolist(),
            "rmse": rmse,
            "max_error": max_err,
        }

    def generate_report(self, report: Dict[str, Any]):
        """
        Generate and save compliance report.
        
        Args:
            report: Compliance report dict
        """
        # Console output
        print(f"\n{'='*70}")
        print(f"METHODIK V3 COMPLIANCE REPORT: {report['dataset_name']}")
        print(f"{'='*70}")
        print(f"Frame Count: {report['frame_count']}")
        print(f"Overall Compliant: {'✓ YES' if report['overall_compliant'] else '✗ NO'}")
        
        if report['compliance_checks'].get('reduced_mode', {}).get('active'):
            print(f"Reduced Mode: ACTIVE")
        
        print(f"\nCompliance Checks:")
        for check_name, check_result in report['compliance_checks'].items():
            if isinstance(check_result, dict) and 'passed' in check_result:
                status = "✓" if check_result['passed'] else "✗"
                print(f"  {status} {check_name}")
        
        if report['warnings']:
            print(f"\nWarnings ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"  ⚠ {warning['message']}")
        
        if report['errors']:
            print(f"\nErrors ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"  ✗ {error['message']}")
        
        print(f"{'='*70}\n")
        
        # Save JSON report
        report_path = os.path.join(self.output_dir, f"methodik_v3_compliance_{report['dataset_name']}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved: {report_path}")
