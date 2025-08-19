#!/usr/bin/env bash

set -euo pipefail

echo "Starting test"
echo "Julia check:"
command -v julia
echo "After julia check"
echo "Config path: research/run_research/run_config.yaml"
echo "File exists: $([[ -f research/run_research/run_config.yaml ]] && echo "yes" || echo "no")"
echo "End of test"
