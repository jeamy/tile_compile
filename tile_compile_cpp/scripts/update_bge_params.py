#!/usr/bin/env python3
"""Update BGE parameters in YAML config file."""

import sys
import re

def update_bge_params(config_file, mu, lambda_val, epsilon, n_g, g_min):
    """Update BGE parameters in config file."""
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update parameters using regex
    content = re.sub(
        r'(\s+rbf_mu_factor:\s+)[0-9.]+',
        r'\g<1>' + str(mu),
        content
    )
    
    content = re.sub(
        r'(\s+rbf_lambda:\s+)[0-9.e-]+',
        r'\g<1>' + str(lambda_val),
        content
    )
    
    content = re.sub(
        r'(\s+rbf_epsilon:\s+)[0-9.]+',
        r'\g<1>' + str(epsilon),
        content
    )
    
    content = re.sub(
        r'(\s+N_g:\s+)[0-9]+',
        r'\g<1>' + str(n_g),
        content
    )
    
    content = re.sub(
        r'(\s+G_min_px:\s+)[0-9]+',
        r'\g<1>' + str(g_min),
        content
    )
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {config_file}:")
    print(f"  rbf_mu_factor: {mu}")
    print(f"  rbf_lambda: {lambda_val}")
    print(f"  rbf_epsilon: {epsilon}")
    print(f"  N_g: {n_g}")
    print(f"  G_min_px: {g_min}")

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: update_bge_params.py <config_file> <mu> <lambda> <epsilon> <n_g> <g_min>")
        sys.exit(1)
    
    update_bge_params(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6]
    )
