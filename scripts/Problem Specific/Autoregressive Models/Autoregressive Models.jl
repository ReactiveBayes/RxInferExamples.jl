# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Autoregressive Models/Autoregressive Models.ipynb
# by notebooks_to_scripts.jl at 2025-03-27T06:11:20.351
#
# Source notebook: Autoregressive Models.ipynb

using RxInfer, Distributions, LinearAlgebra, Plots, StableRNGs, DataFrames, CSV, Dates

# The following coefficients correspond to stable poles
coefs_ar_1 = [-0.27002517200218096]
coefs_ar_2 = [0.4511170798064709, -0.05740081602446657]
coefs_ar_5 = [0.10699399235785655, -0.5237303489793305, 0.3068897071844715, -0.17232255282458891, 0.13323964347539288];

function generate_synthetic_dataset(; n, θ, γ = 1.0, τ = 1.0, rng = StableRNG(42), states1 = randn(rng, length(θ)))
    order = length(θ)

    # Convert precision parameters to standard deviation
    τ_std = sqrt(inv(τ))

    # Initialize states and observations
    states       = Vector{Vector{Float64}}(undef, n + 3order)
    observations = Vector{Float64}(undef, n + 3order)

    # `NormalMeanPrecision` is exported by `RxInfer.jl`
    # and is a part of `ExponentialFamily.jl`
    states[1]       = states1
    observations[1] = rand(rng, NormalMeanPrecision(states[1][1], γ))
    
    for i in 2:(n + 3order)
        previous_state  = states[i - 1]
        transition      = dot(θ, previous_state)
        next_x          = rand(rng, NormalMeanPrecision(transition, τ))
        states[i]       = vcat(next_x, previous_state[1:end-1])
        observations[i] = rand(rng, NormalMeanPrecision(next_x, γ))
    end
    
    return states[1+3order:end], observations[1+3order:end]
end

function plot_synthetic_dataset(; dataset, title)
    states, observations = dataset
    p = plot(first.(states), label = "Hidden states", title = title)
    p = scatter!(p, observations, label = "Observations")
    return p
end

dataset_1 = generate_synthetic_dataset(n = 100, θ = coefs_ar_1)
dataset_2 = generate_synthetic_dataset(n = 100, θ = coefs_ar_2)
dataset_5 = generate_synthetic_dataset(n = 100, θ = coefs_ar_5)

p1 = plot_synthetic_dataset(dataset = dataset_1, title = "AR(1)")
p2 = plot_synthetic_dataset(dataset = dataset_2, title = "AR(2)")
p3 = plot_synthetic_dataset(dataset = dataset_5, title = "AR(5)")

plot(p1, p2, p3, layout = @layout([ a b ; c ]))

@model function lar_multivariate(y, order, γ)
    # `c` is a unit vector of size `order` with first element equal to 1
    c = ReactiveMP.ar_unit(Multivariate, order)
    
    τ  ~ Gamma(α = 1.0, β = 1.0)
    θ  ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    
    x_prev = x0
    
    for i in eachindex(y)
 
        x[i] ~ AR(x_prev, θ, τ) 
        y[i] ~ Normal(mean = dot(c, x[i]), precision = γ)
        
        x_prev = x[i]
    end
end

@constraints function ar_constraints() 
    q(x0, x, θ, τ) = q(x0, x)q(θ)q(τ)
end

@meta function ar_meta(order)
    AR() -> ARMeta(Multivariate, order, ARsafe())
end

@initialization function ar_init(order)
    q(τ) = GammaShapeRate(1.0, 1.0)
    q(θ) = MvNormalMeanPrecision(zeros(order), diageye(order))
end

real_θ = coefs_ar_5
real_τ = 0.5
real_γ = 2.0

order = length(real_θ)
n     = 500 

states, observations = generate_synthetic_dataset(n = n, θ = real_θ, τ = real_τ, γ = real_γ)

result = infer(
    model          = lar_multivariate(order = order, γ = real_γ), 
    data           = (y = observations, ),
    constraints    = ar_constraints(),
    meta           = ar_meta(order),
    initialization = ar_init(order),
    options        = (limit_stack_depth = 500, ),
    returnvars     = (x = KeepLast(), τ = KeepEach(), θ = KeepEach()),
    free_energy    = true,
    iterations     = 20
)

mean(result.posteriors[:θ][end])

cov(result.posteriors[:θ][end])

real_θ

posterior_states       = result.posteriors[:x]
posterior_τ            = result.posteriors[:τ]

p1 = plot(first.(states), label="Hidden state")
p1 = scatter!(p1, observations, label="Observations")
p1 = plot!(p1, first.(mean.(posterior_states)), ribbon = 3first.(std.(posterior_states)), label="Inferred states (+-3σ)", legend = :bottomright)
p1 = lens!(p1, [20, 40], [-2, 2], inset = (1, bbox(0.2, 0.0, 0.4, 0.4)))

p2 = plot(mean.(posterior_τ), ribbon = 3std.(posterior_τ), label = "Inferred τ (+-3σ)", legend = :topright)
p2 = plot!([ real_τ ], seriestype = :hline, label = "Real τ")


plot(p1, p2, layout = @layout([ a; b ]))

plot(result.free_energy, label = "Bethe Free Energy")

posterior_coefficients = result.posteriors[:θ]

pθ = []
cθ = Plots.palette(:tab10)

θms = mean.(posterior_coefficients)
θvs = 3std.(posterior_coefficients)

for i in 1:length(first(θms))
    push!(pθ, plot(getindex.(θms, i), ribbon = getindex.(θvs, i), label = "Estimated θ[$i]", color = cθ[i]))
end

for i in 1:length(real_θ)
    plot!(pθ[i], [ real_θ[i] ], seriestype = :hline, label = "Real θ[$i]", color = cθ[i], linewidth = 2)
end

plot(pθ..., size = (800, 300), legend = :bottomright)

function generate_sinusoidal_coefficients(; f) 
    a1 = 2cos(2pi*f)
    a2 = -1
    return [a1, a2]
end

# Generate coefficients
predictions_coefficients = generate_sinusoidal_coefficients(f = 0.03)

# Generate dataset
predictions_dataset = generate_synthetic_dataset(n = 350, θ = predictions_coefficients, τ = 1.0, γ = 0.01)

# Plot dataset
plot_synthetic_dataset(dataset = predictions_dataset, title = "Sinusoidal AR(2)")

number_of_predictions = 100

predictions_states, predictions_observations = predictions_dataset

# We inject `missing` values to the observations to simulate 
# the future values that we want to predict
predictions_observations_with_predictions = vcat(
    predictions_observations,
    [ missing for _ in 1:number_of_predictions ]
)

# It is better to use `UnfactorizedData` for prediction
predictions_result = infer(
    model          = lar_multivariate(order = 2, γ = 0.01), 
    data           = (y = UnfactorizedData(predictions_observations_with_predictions), ),
    constraints    = ar_constraints(),
    meta           = ar_meta(2),
    initialization = ar_init(2),
    options        = (limit_stack_depth = 500, ),
    returnvars     = (x = KeepLast(), τ = KeepEach(), θ = KeepEach()),
    free_energy    = false,
    iterations     = 20
)

# Extract the inferred coefficients (mean of posterior)
inferred_coefficients = predictions_result.posteriors[:θ][end]

println("True coefficients: ", predictions_coefficients)
println("Inferred coefficients (mean value): ", mean(inferred_coefficients))

μ_true = predictions_coefficients
μ_inferred = mean(inferred_coefficients)

# Create grid of points
x = range(μ_true[1]-0.025, μ_true[1]+0.025, length=100)
y = range(μ_true[2]-0.025, μ_true[2]+0.025, length=100)

# Create contour plot
coefficients_plot = contour(x, y, (x, y) -> pdf(inferred_coefficients, [x, y]), 
    fill=true, 
    title="True vs Inferred AR Coefficients",
    xlabel="θ₁",
    ylabel="θ₂",
    levels = 14, 
    color=:turbo,
    colorbar = false
)

# Add point for true coefficients
scatter!(coefficients_plot, [μ_true[1]], [μ_true[2]], 
    label="True coefficients",
    marker=:star,
    markersize=20,
    color=:red
)

# Add point for inferred mean
scatter!(coefficients_plot, [μ_inferred[1]], [μ_inferred[2]], 
    label="Inferred mean",
    markersize=8,
    color=:white
)

predictions_posterior_states = predictions_result.predictions[:y][end]

predictions_posterior_states_mean = mean.(predictions_posterior_states)
predictions_posterior_states_std = std.(predictions_posterior_states)

pred_p = scatter(predictions_observations, label="Observations", ms=2)
pred_p = plot!(pred_p, predictions_posterior_states_mean, ribbon=3predictions_posterior_states_std, label="Predictions")


function predict_manual(; number_of_predictions, coefficients, precision, first_state, rng = StableRNG(42))
    states = [ first_state ]
    for i in 1:(number_of_predictions + 1)
        next_x     = rand(rng, NormalMeanPrecision(dot(coefficients, states[end]), precision))
        next_state = vcat(next_x, states[end][1:end-1])
        push!(states, next_state)
    end
    return states[2:end]
end

predicted_manually = predict_manual(; 
    number_of_predictions = number_of_predictions, 
    coefficients = predictions_coefficients, 
    precision = 0.1, 
    first_state = predictions_states[end]
)

plot(1:length(predictions_states), first.(predictions_states), label = "Real state")
scatter!(1:length(predictions_observations), first.(predictions_observations), label = "Observations", ms = 2)
plot!((length(predictions_observations)+1):length(predictions_observations) + number_of_predictions + 1, first.(predicted_manually), label = "Predictions manually")

pred_p_manual = scatter(predictions_observations, label="Observations", ms=2)
pred_p_manual = plot!(pred_p_manual, predictions_posterior_states_mean, ribbon=3predictions_posterior_states_std, label="Predictions")
pred_p_manual = plot!(pred_p_manual, (length(predictions_observations)+1):length(predictions_observations) + number_of_predictions + 1, first.(predicted_manually), label = "Predictions manual")

x_df = CSV.read("aal_stock.csv", DataFrame)

# We will use "close" column
x_data = filter(!ismissing, x_df[:, 5])

# Plot data
plot(x_data, xlabel="Day", ylabel="Price", label="Close")

observed_size = length(x_data) - 50

# Observed part
x_observed    = Float64.(x_data[1:observed_size])

# We need to predict this part
x_to_predict   = Float64.(x_data[observed_size+1:end])

x_observed_length   = length(x_observed)
x_to_predict_length = length(x_to_predict)

plot(1:x_observed_length, x_observed, label = "Observed signal")
plot!((x_observed_length + 1):(x_observed_length + x_to_predict_length), x_to_predict, label = "To predict")

stock_observations_with_predictions = vcat(
    x_observed,
    [ missing for _ in 1:length(x_to_predict) ]
)

stock_predictions_result = infer(
    model          = lar_multivariate(order = 50, γ = 1.0), 
    data           = (y = UnfactorizedData(stock_observations_with_predictions), ),
    constraints    = ar_constraints(),
    meta           = ar_meta(50),
    initialization = ar_init(50),
    options        = (limit_stack_depth = 500, ),
    returnvars     = (x = KeepLast(), τ = KeepEach(), θ = KeepEach()),
    free_energy    = false,
    iterations     = 20
)

plot(1:x_observed_length, x_observed, label = "Observed signal")
plot!((x_observed_length + 1):(x_observed_length + x_to_predict_length), x_to_predict, label = "To predict")

stock_predictions = stock_predictions_result.predictions[:y][end]

plot!(mean.(stock_predictions), ribbon = std.(stock_predictions), label = "Predictions")

plot(1:x_observed_length, x_observed, label = "Observed signal")
plot!((x_observed_length + 1):(x_observed_length + x_to_predict_length), x_to_predict, label = "To predict")

stock_hidden_states = stock_predictions_result.posteriors[:x]

plot!(first.(mean.(stock_hidden_states)), ribbon = first.(std.(stock_hidden_states)), label = "x[1]")

function shift(dim)
    S = Matrix{Float64}(I, dim, dim)
    for i in dim:-1:2
        S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    return S
end

shift(2)

@model function ARMA(x, x_prev, priors, p_order, q_order)
    
    # arguments
    c = zeros(q_order); c[1] = 1.0;
    S = shift(q_order); # MA

    # set priors
    γ    ~ priors[:γ]
    η    ~ priors[:η]
    θ    ~ priors[:θ]
    τ    ~ priors[:τ]
    
    h[1] ~ priors[:h]
    z[1] ~ AR(h[1], η, τ)
    e[1] ~ Normal(mean = 0.0, precision = γ)
    x[1] ~ dot(c, z[1]) + dot(θ, x_prev[1]) + e[1]

    for t in 1:length(x)-1
        h[t+1] ~ S * h[t] + c * e[t]
        z[t+1] ~ AR(h[t+1], η, τ)
        e[t+1] ~ Normal(mean = 0.0, precision = γ)
        x[t+1] ~ dot(c, z[t+1]) + dot(θ, x_prev[t+1]) + e[t+1]
    end
end

@constraints function arma_constraints()
    q(z, h, η, τ, γ,e) = q(z, h)q(η)q(τ)q(γ)q(e)
end

@initialization function arma_initialization(priors) 
    q(h)   = priors[:h]
    μ(h)   = priors[:h]
    q(γ)   = priors[:γ]
    q(τ)   = priors[:τ]
    q(η)   = priors[:η]
    q(θ)   = priors[:θ]
end

p_order = 10 # AR
q_order = 4  # MA

priors  = (
    h = MvNormalMeanPrecision(zeros(q_order), diageye(q_order)),
    γ = GammaShapeRate(1e4, 1.0),
    τ = GammaShapeRate(1e2, 1.0),
    η = MvNormalMeanPrecision(ones(q_order), diageye(q_order)),
    θ = MvNormalMeanPrecision(zeros(p_order), diageye(p_order))
)

arma_x_data = Float64.(x_data[p_order+1:end])[1:observed_size]
arma_x_prev_data = [Float64.(x_data[i+p_order-1:-1:i]) for i in 1:length(x_data)-p_order][1:observed_size]

result = infer(
    model = ARMA(priors=priors, p_order = p_order, q_order = q_order), 
    data  = (x = arma_x_data, x_prev = arma_x_prev_data),
    initialization = arma_initialization(priors),
    constraints    = arma_constraints(),
    meta           = ar_meta(q_order),
    returnvars     = KeepLast(),
    iterations     = 20,
    options        = (limit_stack_depth = 400, ),
)

plot(mean.(result.posteriors[:e]), ribbon = var.(result.posteriors[:e][end]), label = "eₜ")