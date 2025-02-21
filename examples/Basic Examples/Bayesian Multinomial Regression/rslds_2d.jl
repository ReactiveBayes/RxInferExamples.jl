using RxInfer, Plots, StableRNGs, Distributions, ExponentialFamily, LinearAlgebra, Optim, Random
## LOAD DATA
include("data_generators.jl")
## LOAD MODEL
include("rslds_model.jl")

hyperparameters = RSLDSHyperparameters(
    a_w = 1.0,
    b_w = 1.0,
    Ψ_w = diageye(2),
    Ψ_R = 0.1diageye(2),
    ν_R = 1.0,
    α = [100.0 1.0; 1.0 100.0],
    C = 1.0*[1 0 0 0 0 0; 0 1 0 0 0 0]
)   
rslds_result = fit_rslds([y_drone[:, t] for t in 1:100], 2, 6; iterations = 40, hyperparameters = hyperparameters, progress = true)
plot(rslds_result.free_energy[3:end])

plot(y_drone[1,:], y_drone[2,:], label="Observations", color="black", linewidth=2)
scatter(states_to_categorical(rslds_result.posteriors[:s][end]), label="Estimated States", color="blue", linewidth=2)

println("estimated H: ", reshape.(mean.(rslds_result.posteriors[:H][end]), 6,6))
println("true H: ", [A1, A2])
println("estimated transition matrix: ", mean(rslds_result.posteriors[:P][end]))



m_states = mean.(rslds_result.posteriors[:x][end])

var_states = var.(rslds_result.posteriors[:x][end])

plot(getindex.(m_states, 5), getindex.(m_states, 6), label="Estimated States", color="blue", linewidth=2)
plot!(x_drone[5,:], x_drone[6,:], label="True States", color="red", linewidth=2)

plot(getindex.(m_states, 4), label="Estimated States", color="blue", linewidth=2)
plot!(x_drone[1,:][1:100], label="True States", color="red", linewidth=2)