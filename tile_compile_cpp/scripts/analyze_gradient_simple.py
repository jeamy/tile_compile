#!/usr/bin/env python3
"""Simple gradient analyzer for optimization loop."""

import numpy as np
from astropy.io import fits
import sys

def analyze_left_gradient(filepath):
    """Analyze left region gradient."""
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(np.float64)
    
    if len(data.shape) == 3:
        brightness = np.mean(data, axis=0)
    else:
        brightness = data
    
    h, w = brightness.shape
    left_region = brightness[:, :int(w * 0.2)]
    
    # Calculate mean gradient
    gradient_x = np.abs(np.diff(left_region, axis=1))
    mean_grad = np.mean(gradient_x)
    
    print(f"Left gradient: {mean_grad:.8f}")
    print(f"Left std: {np.std(left_region):.6f}")
    print(f"Left range: {np.max(left_region) - np.min(left_region):.6f}")
    
    return mean_grad

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze_gradient_simple.py <fits_file>")
        sys.exit(1)
    
    analyze_left_gradient(sys.argv[1])
