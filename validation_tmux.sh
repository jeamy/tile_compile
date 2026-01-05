#!/bin/bash

# Ensure we're in the tile_compile directory
cd "$(dirname "$0")"

# Create a new tmux session
tmux new-session -d -s tile_compile_validation

# First pane: Activate venv and generate datasets
tmux send-keys -t tile_compile_validation 'source .venv/bin/activate' C-m
tmux send-keys -t tile_compile_validation 'python generate_datasets.py' C-m

# Split window vertically
tmux split-window -v -t tile_compile_validation

# Second pane: Run validation
tmux send-keys -t tile_compile_validation.1 'source .venv/bin/activate' C-m
tmux send-keys -t tile_compile_validation.1 'python run_validation.py' C-m

# Attach to the session to monitor
tmux attach-session -t tile_compile_validation
```