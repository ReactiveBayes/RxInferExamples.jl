# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Multi-agent Trajectory Planning/Multi-agent Trajectory Planning.ipynb
# by notebooks_to_scripts.jl at 2025-06-03T10:14:28.748
#
# Source notebook: Multi-agent Trajectory Planning.ipynb

# Activate the local project environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Set GR backend for headless operation
ENV["GKSwstype"] = "100"

# Load the module and run the experiments
println("Loading TrajectoryPlanning module...")
include("TrajectoryPlanning.jl")
using .TrajectoryPlanning
using Plots
using Dates

# Set plot backend
gr()

# Run all experiments and get the output directory
println("Starting Multi-agent Trajectory Planning experiments...")
output_dir = run_all_experiments()
println("All experiments completed. Results saved to: $output_dir") 