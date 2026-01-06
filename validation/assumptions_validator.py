"""
Methodik v3 Assumptions Validator

Validates datasets against all Methodik v3 assumptions:
- Hard assumptions (violations → abort)
- Soft assumptions (warnings)
- Implicit assumptions (tracking, stability)

Usage:
    validator = AssumptionsValidator(assumptions=assumptions)
    report = validator.validate(frames, 'dataset_name')
    validator.generate_report(report)
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class AssumptionsValidator:
    """Validates datasets against Methodik v3 assumptions."""
    
    def __init__(self, assumptions: Optional[Dict] = None, output_dir: str = 'validation_results'):
        """
        Initialize assumptions validator.
        
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
    
    def validate(self, frames: np.ndarray, dataset_name: str) -> Dict[str, Any]:
        """
        Validate dataset against all assumptions.
        
        Args:
            frames: Input frames array
            dataset_name: Name of dataset
        
        Returns:
            Validation report dict
        """
        frame_count = len(frames)
        
        report = {
            'dataset_name': dataset_name,
            'timestamp': self._timestamp(),
            'frame_count': frame_count,
            'assumptions': self.assumptions,
            'hard_assumptions': {},
            'soft_assumptions': {},
            'implicit_assumptions': {},
            'violations': [],
            'warnings': [],
            'valid': True,
        }
        
        # Hard Assumptions (Methodik v3 §1.1)
        self._validate_hard_assumptions(frames, report)
        
        # Soft Assumptions (Methodik v3 §1.2)
        self._validate_soft_assumptions(frames, report)
        
        # Implicit Assumptions (Methodik v3 §1.3)
        self._validate_implicit_assumptions(frames, report)
        
        # Overall validity
        report['valid'] = len(report['violations']) == 0
        
        return report
    
    def _validate_hard_assumptions(self, frames: np.ndarray, report: Dict):
        """
        Validate hard assumptions (violations → abort).
        
        Hard assumptions (Methodik v3 §1.1):
        - Data is linear (no stretch, no non-linear operators)
        - No frame selection (pixel-level artifact rejection allowed)
        - Channel-separated processing
        - Strictly linear pipeline (no feedback loops)
        - Uniform exposure time (tolerance: ±5%)
        """
        frame_count = len(frames)
        
        # 1. Frame count minimum
        frames_min = self.assumptions['frames_min']
        if frame_count < frames_min:
            report['violations'].append({
                'assumption': 'frames_min',
                'type': 'HARD',
                'message': f"Frame count ({frame_count}) below minimum ({frames_min})",
                'severity': 'CRITICAL',
                'action': 'ABORT'
            })
        
        report['hard_assumptions']['frames_min'] = {
            'passed': frame_count >= frames_min,
            'value': frame_count,
            'threshold': frames_min
        }
        
        # 2. Data linearity (basic check)
        linearity_ok = self._check_linearity_basic(frames)
        report['hard_assumptions']['data_linearity'] = {
            'passed': linearity_ok,
            'checked': True
        }
        
        if not linearity_ok:
            report['violations'].append({
                'assumption': 'data_linearity',
                'type': 'HARD',
                'message': "Data appears non-linear (histogram clipping detected)",
                'severity': 'CRITICAL',
                'action': 'ABORT'
            })
        
        # 3. Exposure time uniformity (if metadata available)
        # Note: This would require FITS headers, skipped for now
        report['hard_assumptions']['exposure_uniformity'] = {
            'passed': True,
            'checked': False,
            'note': 'Requires FITS header metadata'
        }
    
    def _validate_soft_assumptions(self, frames: np.ndarray, report: Dict):
        """
        Validate soft assumptions (warnings with tolerances).
        
        Soft assumptions (Methodik v3 §1.2):
        - Frame count: optimal ≥ 800, minimum ≥ 50, reduced mode 50-199
        - Registration residual: < 0.3 px optimal, < 1.0 px maximum
        - Star elongation: < 0.2 optimal, < 0.4 maximum
        """
        frame_count = len(frames)
        
        # 1. Frame count thresholds
        frames_optimal = self.assumptions['frames_optimal']
        frames_reduced = self.assumptions['frames_reduced_threshold']
        
        if frame_count < frames_reduced:
            report['warnings'].append({
                'assumption': 'frames_reduced_threshold',
                'type': 'SOFT',
                'message': f"Reduced Mode: {frame_count} frames < {frames_reduced}",
                'severity': 'WARNING'
            })
        
        if frame_count < frames_optimal:
            report['warnings'].append({
                'assumption': 'frames_optimal',
                'type': 'SOFT',
                'message': f"Frame count ({frame_count}) below optimal ({frames_optimal})",
                'severity': 'INFO'
            })
        
        report['soft_assumptions']['frame_count'] = {
            'optimal': frame_count >= frames_optimal,
            'reduced_mode': frame_count < frames_reduced,
            'value': frame_count,
            'thresholds': {
                'optimal': frames_optimal,
                'reduced': frames_reduced
            }
        }
        
        # 2. Registration residual (would require registration data)
        report['soft_assumptions']['registration_residual'] = {
            'checked': False,
            'note': 'Requires registration results'
        }
        
        # 3. Star elongation (would require star detection)
        report['soft_assumptions']['star_elongation'] = {
            'checked': False,
            'note': 'Requires star detection results'
        }
    
    def _validate_implicit_assumptions(self, frames: np.ndarray, report: Dict):
        """
        Validate implicit assumptions (Methodik v3 §1.3).
        
        Implicit assumptions:
        - Stable optical configuration (focus, field curvature)
        - Tracking error < 1 pixel per exposure
        - No systematic drift during session
        """
        # 1. Frame stability (basic variance check)
        if len(frames) >= 2:
            frame_medians = [np.median(f) for f in frames]
            median_variance = np.var(frame_medians) / np.mean(frame_medians) if np.mean(frame_medians) > 0 else 0
            
            # High variance suggests instability
            if median_variance > 0.1:
                report['warnings'].append({
                    'assumption': 'optical_stability',
                    'type': 'IMPLICIT',
                    'message': f"High frame variance detected (CV: {median_variance:.3f})",
                    'severity': 'WARNING'
                })
            
            report['implicit_assumptions']['optical_stability'] = {
                'checked': True,
                'median_variance': float(median_variance),
                'stable': median_variance <= 0.1
            }
        else:
            report['implicit_assumptions']['optical_stability'] = {
                'checked': False,
                'note': 'Requires multiple frames'
            }
        
        # 2. Tracking error (would require registration data)
        report['implicit_assumptions']['tracking_error'] = {
            'checked': False,
            'threshold_px': self.assumptions['tracking_error_max_px'],
            'note': 'Requires registration results'
        }
        
        # 3. Systematic drift (would require time-series analysis)
        report['implicit_assumptions']['systematic_drift'] = {
            'checked': False,
            'note': 'Requires temporal metadata'
        }
    
    def _check_linearity_basic(self, frames: np.ndarray) -> bool:
        """Basic linearity check via histogram analysis."""
        if len(frames) == 0:
            return True
        
        sample_frame = frames[0]
        hist, bins = np.histogram(sample_frame.flatten(), bins=256)
        
        # Check for histogram clipping (sign of non-linear stretch)
        total_pixels = sample_frame.size
        edge_pixels = hist[0] + hist[-1]
        edge_ratio = edge_pixels / total_pixels
        
        # More than 1% at edges suggests clipping
        return edge_ratio <= 0.01
    
    def _timestamp(self) -> str:
        """Get ISO timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    def generate_report(self, report: Dict[str, Any]):
        """
        Generate and save assumptions validation report.
        
        Args:
            report: Validation report dict
        """
        # Console output
        print(f"\n{'='*70}")
        print(f"ASSUMPTIONS VALIDATION REPORT: {report['dataset_name']}")
        print(f"{'='*70}")
        print(f"Frame Count: {report['frame_count']}")
        print(f"Valid: {'✓ YES' if report['valid'] else '✗ NO'}")
        
        # Hard Assumptions
        print(f"\nHard Assumptions (violations → abort):")
        for name, result in report['hard_assumptions'].items():
            if result.get('checked', True):
                status = "✓" if result.get('passed', True) else "✗"
                print(f"  {status} {name}")
        
        # Soft Assumptions
        print(f"\nSoft Assumptions (with tolerances):")
        for name, result in report['soft_assumptions'].items():
            if result.get('checked', True):
                if 'optimal' in result:
                    status = "✓" if result['optimal'] else "⚠"
                    print(f"  {status} {name}")
                else:
                    print(f"  - {name} (not checked)")
        
        # Implicit Assumptions
        print(f"\nImplicit Assumptions:")
        for name, result in report['implicit_assumptions'].items():
            if result.get('checked', False):
                status = "✓" if result.get('stable', True) else "⚠"
                print(f"  {status} {name}")
            else:
                print(f"  - {name} (not checked)")
        
        # Violations
        if report['violations']:
            print(f"\nViolations ({len(report['violations'])}):")
            for violation in report['violations']:
                print(f"  ✗ [{violation['type']}] {violation['message']}")
                print(f"    Action: {violation['action']}")
        
        # Warnings
        if report['warnings']:
            print(f"\nWarnings ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"  ⚠ [{warning['type']}] {warning['message']}")
        
        print(f"{'='*70}\n")
        
        # Save JSON report
        report_path = os.path.join(self.output_dir, f"assumptions_validation_{report['dataset_name']}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved: {report_path}")
