#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_BASE="$ROOT/../visualization_methods/outputs"
MEDIA_DIR="$ROOT/output/manim_renders"
mkdir -p "$MEDIA_DIR"
mkdir -p "$MEDIA_DIR/videos/generic_dot_graph/1080p60/partial_movie_files/DotGraphScene" \
         "$MEDIA_DIR/videos/source_code_panel/1080p60/partial_movie_files/SourceCodePanel"

# simple retry helper: retry CMD up to N times
retry() {
  local n=0; local max=${2:-3}
  until [ $n -ge $max ]; do
    { eval "$1" && return 0; } || true
    n=$((n+1))
    echo "Retry $n/$max for: $1"
    sleep 1
  done
  echo "Command failed after $max attempts: $1" >&2
  return 1
}

# Render helpers
render_dot() {
  local dot="$1"; local png="$2"; local title="$3"
  local safe_title=$(echo "$title" | tr ' /' '__')
  mkdir -p "$MEDIA_DIR/videos/generic_dot_graph/1080p60/partial_movie_files/DotGraphScene"
  DOT_PATH="$dot" PNG_PATH="$png" TITLE="$title" \
  python3 -m manim -qh "$ROOT/manim_project/scenes/generic_dot_graph.py" DotGraphScene --media_dir "$MEDIA_DIR" --output_file "${safe_title}.mp4"
}

render_code() {
  local src="$1"
  local title=$(basename "$(dirname "$src")")
  local safe_title=$(echo "$title" | tr ' /' '__')
  mkdir -p "$MEDIA_DIR/videos/source_code_panel/1080p60/partial_movie_files/SourceCodePanel"
  SRC_PATH="$src" \
  python3 -m manim -qh "$ROOT/manim_project/scenes/source_code_panel.py" SourceCodePanel --media_dir "$MEDIA_DIR" --output_file "${safe_title}__source.mp4"
}

# Mountain Car example-specific
MC_DIR="$OUT_BASE/Active Inference Mountain car__mountain_car"
if [ -d "$MC_DIR" ]; then
  echo "Rendering Mountain Car visuals..."
  retry "render_dot '$MC_DIR/mountain_car.dot' '$MC_DIR/mountain_car.png' 'Mountain Car'" 2 || echo "Failed MC dot"
  retry "render_code '$MC_DIR/mountain_car_source.jl'" 2 || echo "Failed MC source"
  sleep 1
fi

# Iterate over all *.dot in outputs and render thumbnails
while IFS= read -r dir; do
  dot="$(ls "$dir"/*.dot 2>/dev/null | head -n1 || true)"
  src="$(ls "$dir"/*_source.jl 2>/dev/null | head -n1 || true)"
  png="$(ls "$dir"/*.png 2>/dev/null | head -n1 || true)"
  title=$(basename "$dir")
  if [ -n "$dot" ]; then
    echo "Rendering graph: $title"
    retry "render_dot '$dot' '${png:-}' '$title'" 2 || echo "Failed DOT: $dot"
    sleep 1
  fi
  if [ -n "$src" ]; then
    echo "Rendering source panel: $title"
    retry "render_code '$src'" 2 || echo "Failed SRC: $src"
    sleep 1
  fi
done < <(find "$OUT_BASE" -type d | sort)

echo "All renders written to: $MEDIA_DIR"
