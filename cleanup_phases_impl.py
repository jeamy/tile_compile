#!/usr/bin/env python3
"""
Script to clean up phases_impl.py by removing REGISTRATION phase.

This script:
1. Removes REGISTRATION phase (lines ~1669-2217)
2. Updates all phase_id numbers
3. Removes registration-related config parsing
"""

import re
from pathlib import Path

def cleanup_phases_impl():
    """Clean up phases_impl.py by removing REGISTRATION phase."""
    
    file_path = Path("tile_compile_python/runner/phases_impl.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file: {len(lines)} lines")
    
    # Find REGISTRATION phase start and end
    reg_start = None
    reg_end = None
    
    for i, line in enumerate(lines):
        # Find start: phase_id = 1, phase_name = "REGISTRATION"
        if 'phase_id = 1' in line and i + 1 < len(lines) and 'REGISTRATION' in lines[i + 1]:
            reg_start = i
            print(f"Found REGISTRATION phase start at line {i + 1}")
        
        # Find end: next phase_id = 2
        if reg_start is not None and reg_end is None:
            if 'phase_id = 2' in line and 'CHANNEL_SPLIT' in lines[i + 1] if i + 1 < len(lines) else False:
                reg_end = i
                print(f"Found REGISTRATION phase end at line {i + 1}")
                break
    
    if reg_start is None or reg_end is None:
        print("Error: Could not find REGISTRATION phase boundaries")
        return False
    
    print(f"Removing lines {reg_start + 1} to {reg_end} ({reg_end - reg_start} lines)")
    
    # Remove REGISTRATION phase
    new_lines = lines[:reg_start] + [
        "\n",
        "    # Phase 1 (REGISTRATION) removed in Methodik v4\n",
        "    # Registration is now performed tile-wise locally during TILE_RECONSTRUCTION_TLR\n",
        "    \n",
        "    # For backward compatibility: use raw frames directly\n",
        "    registered_files = frames\n",
        "\n"
    ] + lines[reg_end:]
    
    print(f"New file: {len(new_lines)} lines (removed {len(lines) - len(new_lines)} lines)")
    
    # Update phase_id numbers (2→1, 3→2, etc.)
    print("Updating phase_id numbers...")
    phase_id_map = {
        'phase_id = 2': 'phase_id = 1',   # CHANNEL_SPLIT
        'phase_id = 3': 'phase_id = 2',   # NORMALIZATION
        'phase_id = 4': 'phase_id = 3',   # GLOBAL_METRICS
        'phase_id = 5': 'phase_id = 4',   # TILE_GRID
        'phase_id = 6': 'phase_id = 5',   # LOCAL_METRICS
        'phase_id = 7': 'phase_id = 6',   # TILE_RECONSTRUCTION
        'phase_id = 8': 'phase_id = 7',   # STATE_CLUSTERING
        'phase_id = 9': 'phase_id = 8',   # SYNTHETIC_FRAMES
        'phase_id = 10': 'phase_id = 9',  # STACKING
        'phase_id = 11': 'phase_id = 10', # DEBAYER
        'phase_id = 12': 'phase_id = 11', # DONE
    }
    
    for i, line in enumerate(new_lines):
        for old, new in phase_id_map.items():
            if old in line:
                new_lines[i] = line.replace(old, new)
                print(f"  Line {i + 1}: {old} → {new}")
    
    # Write backup
    backup_path = file_path.with_suffix('.py.backup')
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    # Write cleaned file
    print(f"Writing cleaned file: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✓ Cleanup complete")
    return True

if __name__ == "__main__":
    import sys
    success = cleanup_phases_impl()
    sys.exit(0 if success else 1)
