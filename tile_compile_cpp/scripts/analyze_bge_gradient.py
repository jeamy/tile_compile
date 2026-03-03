#!/usr/bin/env python3
"""
Analyze FITS file for BGE gradient smoothing evaluation.
"""

import numpy as np
from astropy.io import fits
import sys

def load_fits(filepath):
    """Load FITS file and return data."""
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
    return data, header

def analyze_gradient(data, region="left"):
    """Analyze gradient characteristics in specified region."""
    if len(data.shape) == 3:
        # RGB image
        h, w = data.shape[1], data.shape[2]
        brightness = np.mean(data, axis=0)
    else:
        h, w = data.shape
        brightness = data
    
    # Define regions
    if region == "left":
        # Left 20% of image
        region_data = brightness[:, :int(w * 0.2)]
        region_name = "Left 20%"
    elif region == "right":
        region_data = brightness[:, int(w * 0.8):]
        region_name = "Right 20%"
    elif region == "top":
        region_data = brightness[:int(h * 0.2), :]
        region_name = "Top 20%"
    elif region == "bottom":
        region_data = brightness[int(h * 0.8):, :]
        region_name = "Bottom 20%"
    else:
        region_data = brightness
        region_name = "Full Frame"
    
    # Calculate statistics
    mean_val = np.mean(region_data)
    median_val = np.median(region_data)
    std_val = np.std(region_data)
    min_val = np.min(region_data)
    max_val = np.max(region_data)
    
    # Calculate gradient strength (variation)
    gradient_x = np.abs(np.diff(region_data, axis=1))
    gradient_y = np.abs(np.diff(region_data, axis=0))
    
    mean_grad_x = np.mean(gradient_x)
    mean_grad_y = np.mean(gradient_y)
    max_grad_x = np.max(gradient_x)
    max_grad_y = np.max(gradient_y)
    
    # Calculate smoothness (inverse of gradient variance)
    smoothness_x = 1.0 / (np.std(gradient_x) + 1e-10)
    smoothness_y = 1.0 / (np.std(gradient_y) + 1e-10)
    
    return {
        'region': region_name,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'range': max_val - min_val,
        'mean_grad_x': mean_grad_x,
        'mean_grad_y': mean_grad_y,
        'max_grad_x': max_grad_x,
        'max_grad_y': max_grad_y,
        'smoothness_x': smoothness_x,
        'smoothness_y': smoothness_y
    }

def analyze_vertical_profile(data, x_percent=10):
    """Analyze vertical profile at specified x position."""
    if len(data.shape) == 3:
        brightness = np.mean(data, axis=0)
    else:
        brightness = data
    
    h, w = brightness.shape
    x_pos = int(w * x_percent / 100.0)
    
    profile = brightness[:, x_pos]
    
    # Calculate smoothness of profile
    diff = np.abs(np.diff(profile))
    
    return {
        'x_percent': x_percent,
        'x_pixel': x_pos,
        'mean': np.mean(profile),
        'std': np.std(profile),
        'min': np.min(profile),
        'max': np.max(profile),
        'range': np.max(profile) - np.min(profile),
        'mean_gradient': np.mean(diff),
        'max_gradient': np.max(diff),
        'gradient_std': np.std(diff)
    }

def compare_files(solve_file, pcc_file):
    """Compare solve and PCC files for BGE evaluation."""
    print("="*60)
    print("BGE GRADIENT ANALYSIS")
    print("="*60)
    
    print(f"\nLoading INPUT (solve): {solve_file}")
    data_solve, header_solve = load_fits(solve_file)
    
    print(f"Loading OUTPUT (PCC): {pcc_file}")
    data_pcc, header_pcc = load_fits(pcc_file)
    
    print(f"\nSolve shape: {data_solve.shape}")
    print(f"PCC shape: {data_pcc.shape}")
    
    # Analyze different regions
    regions = ['left', 'right', 'top', 'bottom', 'full']
    
    print("\n" + "="*60)
    print("SOLVE FILE (INPUT) - REGIONAL ANALYSIS")
    print("="*60)
    
    solve_stats = {}
    for region in regions:
        stats = analyze_gradient(data_solve, region)
        solve_stats[region] = stats
        
        print(f"\n{stats['region']}:")
        print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        print(f"  Range: {stats['range']:.2f} (min={stats['min']:.2f}, max={stats['max']:.2f})")
        print(f"  Mean Gradient X: {stats['mean_grad_x']:.4f}, Y: {stats['mean_grad_y']:.4f}")
        print(f"  Max Gradient X: {stats['max_grad_x']:.2f}, Y: {stats['max_grad_y']:.2f}")
        print(f"  Smoothness X: {stats['smoothness_x']:.4f}, Y: {stats['smoothness_y']:.4f}")
    
    print("\n" + "="*60)
    print("PCC FILE (OUTPUT) - REGIONAL ANALYSIS")
    print("="*60)
    
    pcc_stats = {}
    for region in regions:
        stats = analyze_gradient(data_pcc, region)
        pcc_stats[region] = stats
        
        print(f"\n{stats['region']}:")
        print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print(f"  Range: {stats['range']:.6f} (min={stats['min']:.6f}, max={stats['max']:.6f})")
        print(f"  Mean Gradient X: {stats['mean_grad_x']:.8f}, Y: {stats['mean_grad_y']:.8f}")
        print(f"  Max Gradient X: {stats['max_grad_x']:.6f}, Y: {stats['max_grad_y']:.6f}")
        print(f"  Smoothness X: {stats['smoothness_x']:.4f}, Y: {stats['smoothness_y']:.4f}")
    
    # Vertical profiles at different x positions (left side focus)
    print("\n" + "="*60)
    print("VERTICAL PROFILES - LEFT SIDE ANALYSIS")
    print("="*60)
    
    x_positions = [5, 10, 15, 20]  # Percentages from left
    
    print("\nSOLVE FILE:")
    for x_pct in x_positions:
        profile = analyze_vertical_profile(data_solve, x_pct)
        print(f"\n  At {x_pct}% from left (x={profile['x_pixel']}):")
        print(f"    Mean: {profile['mean']:.2f}, Std: {profile['std']:.2f}")
        print(f"    Range: {profile['range']:.2f}")
        print(f"    Mean Gradient: {profile['mean_gradient']:.4f}, Max: {profile['max_gradient']:.2f}")
        print(f"    Gradient Std: {profile['gradient_std']:.4f} (lower = smoother)")
    
    print("\nPCC FILE:")
    for x_pct in x_positions:
        profile = analyze_vertical_profile(data_pcc, x_pct)
        print(f"\n  At {x_pct}% from left (x={profile['x_pixel']}):")
        print(f"    Mean: {profile['mean']:.6f}, Std: {profile['std']:.6f}")
        print(f"    Range: {profile['range']:.6f}")
        print(f"    Mean Gradient: {profile['mean_gradient']:.8f}, Max: {profile['max_gradient']:.6f}")
        print(f"    Gradient Std: {profile['gradient_std']:.8f} (lower = smoother)")
    
    # BGE effectiveness analysis
    print("\n" + "="*60)
    print("BGE EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    # Compare left region gradient reduction
    left_grad_before = solve_stats['left']['mean_grad_x']
    left_grad_after = pcc_stats['left']['mean_grad_x']
    
    # Note: Need to account for scaling difference
    if data_solve.max() > 100:  # Likely 16-bit
        scale_factor = data_solve.max() / (data_pcc.max() + 1e-10)
        left_grad_after_scaled = left_grad_after * scale_factor
    else:
        left_grad_after_scaled = left_grad_after
    
    reduction = (left_grad_before - left_grad_after_scaled) / left_grad_before * 100
    
    print(f"\nLeft Region Gradient Reduction:")
    print(f"  Before BGE: {left_grad_before:.4f}")
    print(f"  After BGE (scaled): {left_grad_after_scaled:.4f}")
    print(f"  Reduction: {reduction:.2f}%")
    
    # Smoothness comparison
    smooth_before = solve_stats['left']['smoothness_x']
    smooth_after = pcc_stats['left']['smoothness_x']
    
    print(f"\nLeft Region Smoothness:")
    print(f"  Before BGE: {smooth_before:.4f}")
    print(f"  After BGE: {smooth_after:.4f}")
    print(f"  Improvement: {(smooth_after/smooth_before - 1)*100:.2f}%")
    
    # Recommendations
    print("\n" + "="*60)
    print("BGE IMPROVEMENT RECOMMENDATIONS")
    print("="*60)
    
    print("\nBased on the analysis:")
    
    if left_grad_after_scaled > left_grad_before * 0.5:
        print("\n⚠️  LEFT SIDE GRADIENT STILL SIGNIFICANT")
        print("   BGE has reduced gradients but residual variation remains.")
        print("\n   Recommendations to improve smoothing:")
        print("   1. Increase BGE interpolation degree (e.g., from 3 to 4 or 5)")
        print("   2. Increase smoothing_sigma parameter")
        print("   3. Reduce sample_grid_spacing for finer background model")
        print("   4. Check if gradient is from actual nebulosity (should preserve)")
    else:
        print("\n✓  LEFT SIDE GRADIENT WELL CONTROLLED")
        print("   BGE has effectively smoothed the background.")
    
    # Check for over-smoothing
    if pcc_stats['left']['range'] < solve_stats['left']['range'] * 0.1:
        print("\n⚠️  POSSIBLE OVER-SMOOTHING")
        print("   Background may be too flat, potentially removing real structure.")
    
    # Write detailed report
    output_file = "/tmp/bge_analysis_report.txt"
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BGE GRADIENT ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("FILES ANALYZED:\n")
        f.write(f"  Input (solve): {solve_file}\n")
        f.write(f"  Output (PCC): {pcc_file}\n\n")
        
        f.write("LEFT REGION STATISTICS:\n")
        f.write(f"  Before BGE:\n")
        f.write(f"    Mean Gradient: {left_grad_before:.4f}\n")
        f.write(f"    Smoothness: {smooth_before:.4f}\n")
        f.write(f"  After BGE:\n")
        f.write(f"    Mean Gradient: {left_grad_after_scaled:.4f}\n")
        f.write(f"    Smoothness: {smooth_after:.4f}\n")
        f.write(f"  Improvement: {reduction:.2f}% gradient reduction\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        if left_grad_after_scaled > left_grad_before * 0.5:
            f.write("  - Increase interpolation_degree\n")
            f.write("  - Increase smoothing_sigma\n")
            f.write("  - Reduce sample_grid_spacing\n")
        else:
            f.write("  - Current BGE settings appear effective\n")
    
    print(f"\nDetailed report written to: {output_file}")

if __name__ == "__main__":
    solve_file = "/home/mux/programme/tile_compile/tile_compile_cpp/build/runs/20260303_092027_250beffd/IC434_ligths_all/outputs/stacked_rgb_solve.fits"
    pcc_file = "/home/mux/programme/tile_compile/tile_compile_cpp/build/runs/20260303_092027_250beffd/IC434_ligths_all/outputs/stacked_rgb_pcc.fits"
    
    compare_files(solve_file, pcc_file)
