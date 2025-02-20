using RxInfer, Plots, StableRNGs, Distributions, ExponentialFamily, LinearAlgebra, Optim, Random
## LOAD DATA
include("data_generators.jl")
## LOAD MODEL
include("rslds_model.jl")

hyperparameters = RSLDSHyperparameters(
    a_w = 1.0,
    b_w = 1.0,
    Ψ_w = 1.0diageye(2),
    Ψ_R = 1*diageye(2),
    ν_R = 1.0,
    α = [1.0 1.0; 1.0 1.0],
    C = c
)   
rslds_result = fit_rslds(y, 2, 2; iterations = 50, hyperparameters = hyperparameters, progress = true)

plot(rslds_result.free_energy)

scatter(states_to_categorical(rslds_result.posteriors[:s][end]), label="Estimated States", color="blue", linewidth=2)

println("estimated H: ", reshape.(mean.(rslds_result.posteriors[:H][end]), 2,2))
println("true H: ", [A1, A2])
println("estimated transition matrix: ", mean(rslds_result.posteriors[:P][end]))


m_states = mean.(rslds_result.posteriors[:x][end])

var_states = var.(rslds_result.posteriors[:x][end])
index = 1
from = 500
to = 700

# Create main figure with layout
p = plot(layout=grid(2,1, heights=[0.8,0.2]), 
    size=(1200,800), 
    background_color=:white, 
    foreground_color=:black,
    margin=10Plots.mm
)

# Main plot with states and observations
plot!(p[1], 
    getindex.(m_states, 1)[from+1:to],
    ribbon = getindex.(var_states, 1)[from+1:to],
    label = "Estimated States (Dim 1)",
    legend=:topleft, 
    color=:royalblue, 
    linewidth=1.5, 
    fillalpha = 0.2,
    title="State Estimation and Regime Switching",
    titlefont=font(12, "Computer Modern"),
    xlabel="Time",
    ylabel="State Value"
)



# Add true states
plot!(p[1], 
    getindex.(x, 1)[from:to], 
    label = "True States (Dim 1)",
    color=:maroon, 
    linewidth=1.5
)



# Add observations
scatter!(p[1], 
    getindex.(y, 1)[from:to], 
    label = "Observations (Dim 1)", 
    ms = 2.0, 
    color=:black, 
    alpha = 1.6,
    markerstrokewidth=0
)



# Get the inferred regimes
regimes = states_to_categorical(rslds_result.posteriors[:s][end])[from:to-1]

# Create colors for each regime
colors = [:indianred, :steelblue]

# Plot regimes in the second subplot
plot!(p[2], regimes, 
    label="Regimes",
    color=colors[regimes], 
    linewidth=3,
    legend=false,
    ylims=(0.5,2.5),
    yticks=(1:2, ["Regime 1", "Regime 2"]),
    xlabel="Time",
    ylabel="Regime"
)