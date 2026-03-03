#!/usr/bin/env python3
"""
Compare two FITS files from different runs.
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

def analyze_color_neutrality(data, region_name="full frame"):
    """Analyze color neutrality of RGB data."""
    if len(data.shape) == 3:
        r, g, b = data[0], data[1], data[2]
    elif len(data.shape) == 2 and data.shape[0] == 3:
        r, g, b = data[0], data[1], data[2]
    else:
        print(f"Warning: Unexpected data shape {data.shape}")
        return None
    
    r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
    r_median, g_median, b_median = np.median(r), np.median(g), np.median(b)
    r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
    
    rg_ratio = r_mean / g_mean if g_mean > 0 else 0
    rb_ratio = r_mean / b_mean if b_mean > 0 else 0
    gb_ratio = g_mean / b_mean if b_mean > 0 else 0
    
    print(f"\n=== {region_name} ===")
    print(f"R: mean={r_mean:.6f}, median={r_median:.6f}, std={r_std:.6f}")
    print(f"G: mean={g_mean:.6f}, median={g_median:.6f}, std={g_std:.6f}")
    print(f"B: mean={b_mean:.6f}, median={b_median:.6f}, std={b_std:.6f}")
    print(f"\nColor Ratios:")
    print(f"R/G = {rg_ratio:.6f}")
    print(f"R/B = {rb_ratio:.6f}")
    print(f"G/B = {gb_ratio:.6f}")
    
    rg_dev = abs(rg_ratio - 1.0) * 100
    rb_dev = abs(rb_ratio - 1.0) * 100
    gb_dev = abs(gb_ratio - 1.0) * 100
    print(f"\nDeviation from neutral (%):")
    print(f"R/G: {rg_dev:.2f}%")
    print(f"R/B: {rb_dev:.2f}%")
    print(f"G/B: {gb_dev:.2f}%")
    
    return {
        'means': (r_mean, g_mean, b_mean),
        'medians': (r_median, g_median, b_median),
        'stds': (r_std, g_std, b_std),
        'ratios': (rg_ratio, rb_ratio, gb_ratio),
        'deviations': (rg_dev, rb_dev, gb_dev)
    }

def extract_bright_core(data, percentile=99):
    """Extract bright core region."""
    if len(data.shape) == 3:
        brightness = np.mean(data, axis=0)
    else:
        brightness = data
    
    threshold = np.percentile(brightness, percentile)
    mask = brightness >= threshold
    
    masked_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        masked_data[i] = np.where(mask, data[i], np.nan)
    
    return masked_data, mask

def compare_files(file1, file2, output_dir):
    """Compare two FITS files."""
    print(f"Loading Run 1: {file1}...")
    data1, header1 = load_fits(file1)
    
    print(f"Loading Run 2: {file2}...")
    data2, header2 = load_fits(file2)
    
    print(f"\nRun 1 shape: {data1.shape}")
    print(f"Run 2 shape: {data2.shape}")
    
    print("\n" + "="*60)
    print("RUN 1 (20260303_074247_b4_dedup) - FULL FRAME")
    print("="*60)
    stats1_full = analyze_color_neutrality(data1, "Full Frame")
    
    print("\n" + "="*60)
    print("RUN 2 (20260303_080623_10a3183e) - FULL FRAME")
    print("="*60)
    stats2_full = analyze_color_neutrality(data2, "Full Frame")
    
    print("\n" + "="*60)
    print("BRIGHT CORE ANALYSIS (top 1% brightest pixels)")
    print("="*60)
    
    core1, mask1 = extract_bright_core(data1, percentile=99)
    core2, mask2 = extract_bright_core(data2, percentile=99)
    
    print("\n" + "="*60)
    print("RUN 1 - BRIGHT CORE")
    print("="*60)
    mask1_flat = ~np.isnan(core1[0].flatten())
    core1_clean = np.array([
        core1[0].flatten()[mask1_flat],
        core1[1].flatten()[mask1_flat],
        core1[2].flatten()[mask1_flat]
    ])
    stats1_core = analyze_color_neutrality(core1_clean, "Bright Core")
    
    print("\n" + "="*60)
    print("RUN 2 - BRIGHT CORE")
    print("="*60)
    mask2_flat = ~np.isnan(core2[0].flatten())
    core2_clean = np.array([
        core2[0].flatten()[mask2_flat],
        core2[1].flatten()[mask2_flat],
        core2[2].flatten()[mask2_flat]
    ])
    stats2_core = analyze_color_neutrality(core2_clean, "Bright Core")
    
    print("\n" + "="*60)
    print("DIFFERENCE ANALYSIS")
    print("="*60)
    
    diff = data1 - data2
    abs_diff = np.abs(diff)
    
    print(f"\nAbsolute difference statistics:")
    print(f"R channel: mean={np.mean(abs_diff[0]):.6f}, max={np.max(abs_diff[0]):.6f}")
    print(f"G channel: mean={np.mean(abs_diff[1]):.6f}, max={np.max(abs_diff[1]):.6f}")
    print(f"B channel: mean={np.mean(abs_diff[2]):.6f}, max={np.max(abs_diff[2]):.6f}")
    
    rel_diff = np.abs(diff / (data2 + 1e-10)) * 100
    print(f"\nRelative difference (%):")
    print(f"R channel: mean={np.mean(rel_diff[0]):.2f}%, max={np.max(rel_diff[0]):.2f}%")
    print(f"G channel: mean={np.mean(rel_diff[1]):.2f}%, max={np.max(rel_diff[1]):.2f}%")
    print(f"B channel: mean={np.mean(rel_diff[2]):.2f}%, max={np.max(rel_diff[2]):.2f}%")
    
    print(f"\n" + "="*60)
    print("CREATING COMPARISON FITS FILES")
    print("="*60)
    
    diff_file = f"{output_dir}/difference_run1_minus_run2.fits"
    fits.writeto(diff_file, diff, header1, overwrite=True)
    print(f"Created: {diff_file}")
    
    abs_diff_file = f"{output_dir}/abs_difference_runs.fits"
    fits.writeto(abs_diff_file, abs_diff, header1, overwrite=True)
    print(f"Created: {abs_diff_file}")
    
    ratio = data1 / (data2 + 1e-10)
    ratio_file = f"{output_dir}/ratio_run1_div_run2.fits"
    fits.writeto(ratio_file, ratio, header1, overwrite=True)
    print(f"Created: {ratio_file}")
    
    average = (data1 + data2) / 2.0
    avg_file = f"{output_dir}/average_both_runs.fits"
    fits.writeto(avg_file, average, header1, overwrite=True)
    print(f"Created: {avg_file}")
    
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT")
    print("="*60)
    
    print("\n### Color Neutrality Comparison:")
    print(f"\nFull Frame:")
    print(f"  Run 1 (b4_dedup): avg deviation = {np.mean(stats1_full['deviations']):.2f}%")
    print(f"  Run 2 (10a3183e): avg deviation = {np.mean(stats2_full['deviations']):.2f}%")
    
    print(f"\nBright Core:")
    print(f"  Run 1 (b4_dedup): avg deviation = {np.mean(stats1_core['deviations']):.2f}%")
    print(f"  Run 2 (10a3183e): avg deviation = {np.mean(stats2_core['deviations']):.2f}%")
    
    full_better = "Run 1" if np.mean(stats1_full['deviations']) < np.mean(stats2_full['deviations']) else "Run 2"
    core_better = "Run 1" if np.mean(stats1_core['deviations']) < np.mean(stats2_core['deviations']) else "Run 2"
    
    print(f"\n### Better Color Neutrality:")
    print(f"  Full Frame: {full_better}")
    print(f"  Bright Core: {core_better}")
    
    avg_diff_magnitude = np.mean(abs_diff)
    print(f"\n### Overall Quality Assessment:")
    print(f"  Average absolute difference: {avg_diff_magnitude:.6f}")
    
    if avg_diff_magnitude < 0.001:
        print(f"  => Runs are very similar (difference < 0.1%)")
    elif avg_diff_magnitude < 0.01:
        print(f"  => Runs have minor differences")
    else:
        print(f"  => Runs have significant differences")
    
    # Write summary to text file
    summary_file = f"{output_dir}/comparison_summary_runs.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPARISON: Run 1 (b4_dedup) vs Run 2 (10a3183e)\n")
        f.write("="*60 + "\n\n")
        
        f.write("FULL FRAME COLOR NEUTRALITY:\n")
        f.write(f"  Run 1: {np.mean(stats1_full['deviations']):.2f}% avg deviation\n")
        f.write(f"  Run 2: {np.mean(stats2_full['deviations']):.2f}% avg deviation\n")
        f.write(f"  Better: {full_better}\n\n")
        
        f.write("BRIGHT CORE COLOR NEUTRALITY:\n")
        f.write(f"  Run 1: {np.mean(stats1_core['deviations']):.2f}% avg deviation\n")
        f.write(f"  Run 2: {np.mean(stats2_core['deviations']):.2f}% avg deviation\n")
        f.write(f"  Better: {core_better}\n\n")
        
        f.write("DIFFERENCE STATISTICS:\n")
        f.write(f"  Average absolute difference: {avg_diff_magnitude:.6f}\n")
        f.write(f"  R channel: mean={np.mean(abs_diff[0]):.6f}, max={np.max(abs_diff[0]):.6f}\n")
        f.write(f"  G channel: mean={np.mean(abs_diff[1]):.6f}, max={np.max(abs_diff[1]):.6f}\n")
        f.write(f"  B channel: mean={np.mean(abs_diff[2]):.6f}, max={np.max(abs_diff[2]):.6f}\n\n")
        
        f.write("FILES CREATED:\n")
        f.write(f"  - difference_run1_minus_run2.fits\n")
        f.write(f"  - abs_difference_runs.fits\n")
        f.write(f"  - ratio_run1_div_run2.fits\n")
        f.write(f"  - average_both_runs.fits\n")
    
    print(f"\nSummary written to: {summary_file}")

if __name__ == "__main__":
    file1 = "/home/mux/programme/tile_compile/tile_compile_cpp/build/runs/20260303_074247_b4_dedup/lights/outputs/stacked_rgb_pcc.fits"
    file2 = "/home/mux/programme/tile_compile/tile_compile_cpp/build/runs/20260303_080623_10a3183e/lights/outputs/stacked_rgb_pcc.fits"
    output_dir = "/home/mux/programme/tile_compile/tile_compile_cpp/build/runs/20260303_074247_b4_dedup/lights/outputs"
    
    compare_files(file1, file2, output_dir)
