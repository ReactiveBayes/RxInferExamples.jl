module HierarchicalGaussianFilterScripts

# Re-expose the HGF module used by scripts and tests
include(joinpath(@__DIR__, "..", "HGF.jl"))
using .HGF

export HGF

end # module


