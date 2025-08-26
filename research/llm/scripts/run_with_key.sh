#!/usr/bin/env bash
# Example: run the example with the key set in the environment for one-off runs
if [ -z "$1" ]; then
  echo "Usage: $0 <OPENAI_KEY>"
  exit 1
fi

OPENAI_KEY="$1" julia --project=.. bin/run_rxinfer_llm.jl


