include(joinpath(@__DIR__, "src", "POMDPControl.jl"))
using .POMDPControl
using Statistics

# Allow overrides via ENV
const N_EXPERIMENTS = parse(Int, get(ENV, "POMDP_N_EXP", "100"))
const PLANNING_HORIZON = parse(Int, get(ENV, "POMDP_HOR", "4"))

result = POMDPControl.run_pomdp_experiments(n_experiments=N_EXPERIMENTS, T=PLANNING_HORIZON)

env = result.env
agent = result.agent
successes = result.successes
global p_A = result.p_A
global p_B = result.p_B

# Export canonical variables for downstream scripts
global n_experiments = length(successes)
global T = PLANNING_HORIZON

mean(successes)

plot_environment(env)