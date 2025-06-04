"""
    Models

Multi-agent Trajectory Planning Models

This module integrates all model components:
- HalfspaceNode: Custom node definition for constraints
- DistanceFunctions: Functions for distance calculations
- InferenceModel: Core probabilistic model and inference

All components are re-exported for convenience.
"""
module Models

# Include the modular components
include("HalfspaceNode.jl")
include("DistanceFunctions.jl")
include("InferenceModel.jl")

# Re-export all the components
using .HalfspaceNode
using .DistanceFunctions
using .InferenceModel

# Import and re-export visualization functions needed for ELBO plots
using ..Visualizations: plot_elbo_convergence

# Export all necessary functions and types
export Halfspace
export distance, g, h, softmin, configure_softmin
export path_planning, path_planning_model, path_planning_constraints, compute_diagnostics
export plot_elbo_convergence

end # module 