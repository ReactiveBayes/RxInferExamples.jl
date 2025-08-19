#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install manim
python -m pip install pytest
echo "Virtual env ready. Activate with: source research/manim/.venv/bin/activate"
