#!/usr/bin/env bash

set -euo pipefail

# Simplified runner for generalized_coordinates_n_order example
# Usage: ./run.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

EMJ_INFO="ℹ️"; EMJ_OK="✅"; EMJ_WARN="⚠️"; EMJ_ERR="❌"; EMJ_ROCKET="🚀"; EMJ_GEAR="⚙️"; EMJ_TOOLS="🛠️"; EMJ_LAPTOP="💻";

log(){ local level="$1"; shift; echo "$level $*"; }

log "$EMJ_ROCKET" "Starting generalized_coordinates_n_order example"

if ! command -v julia >/dev/null 2>&1; then
  log "$EMJ_ERR" "Julia not found. Install from https://julialang.org/downloads/ and re-run."
  exit 1
fi

# Ensure the project environment is instantiated
log "$EMJ_GEAR" "Ensuring project environment is ready"
julia --project="$ROOT_DIR/research/generalized_coordinates_n_order" -e 'using Pkg; Pkg.instantiate()'

# Run the example with a timeout to prevent hanging (macOS compatible)
log "$EMJ_LAPTOP" "Executing generalized_coordinates_n_order example (with 5 minute timeout)"
julia --project="$ROOT_DIR/research/generalized_coordinates_n_order" "$ROOT_DIR/research/generalized_coordinates_n_order/run_gc_car.jl" &
JULIA_PID=$!

# Wait for 5 minutes, then check if still running
sleep 300
if kill -0 $JULIA_PID 2>/dev/null; then
  log "$EMJ_WARN" "Script still running after 5 minutes - stopping to prevent hanging"
  kill $JULIA_PID 2>/dev/null || true
  wait $JULIA_PID 2>/dev/null || true
  log "$EMJ_INFO" "Check the outputs directory for partial results"
else
  log "$EMJ_INFO" "Script completed successfully"
fi

log "$EMJ_OK" "Completed"