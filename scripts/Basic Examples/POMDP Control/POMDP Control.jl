include(joinpath(@__DIR__, "src", "POMDPControl.jl"))
using .POMDPControl

result = POMDPControl.run_pomdp_experiments(n_experiments=100, T=4)

env = result.env
agent = result.agent
successes = result.successes
global p_A = result.p_A
global p_B = result.p_B

mean(successes)

plot_environment(env)