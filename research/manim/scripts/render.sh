#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SCENE=${1:-MountainCarGraph}

python3 -m manim -qh manim_project/scenes/mountain_car_graph.py "$SCENE" --media_dir ../output/manim_renders
