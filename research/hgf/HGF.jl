module HGF

using RxInfer
using Distributions
using Random
using StableRNGs
using Statistics
using Logging
using Plots

# Submodules (logical separation via include)
include("Utils.jl")
include("Model.jl")
include("Viz.jl")
include("Run.jl")

using .Utils: HGFParams, default_hgf_params, generate_data
using .Model: hgf, hgfconstraints, hgfmeta, hgf_smoothing, hgfconstraints_smoothing, hgfmeta_smoothing
using .Viz: plot_hidden_states, plot_free_energy, plot_param_posteriors
using .Run: run_filter, run_smoothing, run_hgf

export HGFParams,
       default_hgf_params,
       generate_data,
       run_filter,
       run_smoothing,
       plot_hidden_states,
       plot_free_energy,
       plot_param_posteriors,
       run_hgf

end # module


