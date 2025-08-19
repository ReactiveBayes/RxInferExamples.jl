from __future__ import annotations
import pathlib

# Go up to repository root: configs -> manim_project -> manim -> research -> repo
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
OUTPUTS_ROOT = REPO_ROOT / "research/visualization_methods/outputs/Active Inference Mountain car__mountain_car"

MOUNTAIN_CAR_SOURCE = OUTPUTS_ROOT / "mountain_car_source.jl"
MOUNTAIN_CAR_PNG = OUTPUTS_ROOT / "mountain_car.png"
MOUNTAIN_CAR_DOT = OUTPUTS_ROOT / "mountain_car.dot"
MOUNTAIN_CAR_MMD = OUTPUTS_ROOT / "mountain_car.mmd"

RENDERS_DIR = REPO_ROOT / "output/manim_renders"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)
