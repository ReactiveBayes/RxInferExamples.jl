module Models

# Include the modular components
include("HalfspaceNode.jl")
include("DistanceFunctions.jl")
include("InferenceModel.jl")

# Re-export all the components
using .HalfspaceNode
using .DistanceFunctions
using .InferenceModel

# Export all necessary functions and types
export Halfspace
export distance, g, h, softmin, configure_softmin
export path_planning, path_planning_model, path_planning_constraints

end # module 