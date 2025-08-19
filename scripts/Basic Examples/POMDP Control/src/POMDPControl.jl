module POMDPControl

using RxInfer
using Distributions
using RxEnvironments
using Plots
using Statistics

include("env.jl")
include("utils.jl")
include("model.jl")
include("runner.jl")

export WindyGridWorld, WindyGridWorldAgent,
       reset_env!, plot_environment, create_environment,
       grid_location_to_index, index_to_grid_location, index_to_one_hot,
       build_pomdp, run_pomdp_experiments

end


