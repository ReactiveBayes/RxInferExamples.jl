# Multi-agent Trajectory Planning
# This file is the main entry point for the Multi-agent Trajectory Planning example

# Activate the local project environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Set GR backend for headless operation
ENV["GKSwstype"] = "100"

# Load modules with safeguards against redefinition
println("Loading modules...")

# Define modules only if they don't exist
if !@isdefined(Environment)
    include("Environment.jl")
end

if !@isdefined(Visualizations)
    include("Visualizations.jl")
end

if !@isdefined(Models)
    include("Models.jl")
end

if !@isdefined(Experiments)
    include("Experiments.jl")
end

# Import modules
using .Environment
using .Visualizations
using .Models
using .Experiments
using Plots

# Set plot backend
gr()

# Run all experiments
println("Starting Multi-agent Trajectory Planning experiments...")
run_all_experiments() 