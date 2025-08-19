#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# Ensure venv is active if available
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Set a couple of example paths
MC_DIR="$ROOT/../visualization_methods/outputs/Active Inference Mountain car__mountain_car"
MEDIA_DIR="$ROOT/output/manim_renders"
mkdir -p "$MEDIA_DIR"

# Save last frames to quickly smoke-test
DOT_PATH="$MC_DIR/mountain_car.dot" TITLE="Mountain Car" \
python3 -m manim -qk --save_last_frame manim_project/scenes/generic_dot_graph.py DotGraphScene --media_dir "$MEDIA_DIR" --output_file "smoke_graph.png"

SRC_PATH="$MC_DIR/mountain_car_source.jl" \
python3 -m manim -qk --save_last_frame manim_project/scenes/source_code_panel.py SourceCodePanel --media_dir "$MEDIA_DIR" --output_file "smoke_source.png"

echo "Smoke tests completed. Outputs in $MEDIA_DIR"
