# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Predicting Bike Rental Demand/Predicting Bike Rental Demand.ipynb
# by notebooks_to_scripts.jl at 2025-04-21T06:26:05.290
#
# Source notebook: Predicting Bike Rental Demand.ipynb

using RxInfer

@model function example_model(y)

    h ~ NormalMeanPrecision(0, 1.0)
    x ~ NormalMeanPrecision(h, 1.0)
    y ~ NormalMeanPrecision(x, 10.0)
end

# Implicit Prediction
result = infer(model = example_model(), data = (y = missing,))

# Explicit Prediction
result = infer(model = example_model(), predictvars = (y = KeepLast(),))

using CSV, DataFrames, Plots

# Load the data
df = CSV.read("modified_bicycle.csv", DataFrame)
df[1:10, :]

# we reserve few samples for prediction
n_future = 24

# `x` is a sequence of observed features
X = Union{Vector{Float64}, Missing}[[row[i] for i in 2:(ncol(df))-1] for row in eachrow(df)][1:end-n_future]

# `y` is a sequence of "count" bicycles
y = Union{Float64, Missing}[df[:, "count"]...][1:end-n_future]

state_dim = length(X[1]); # dimensionality of feature space

# # We augument the dataset with missing entries for 24 hours ahead
append!(X, repeat([missing], n_future))
append!(y, repeat([missing], n_future));

# Function to perform the state transition in the model.
# It reshapes vector `a` into a matrix and multiplies it with vector `x` to simulate the transition.
function transition(a, x)
    nm, n = length(a), length(x)
    m = nm ÷ n  # Calculate the number of rows for reshaping 'a' into a matrix
    A = reshape(a, (m, n))  
    return A * x
end


# The dotsoftplus function combines a dot product and softplus transformation.
# While useful for ensuring positivity, it may not be the optimal choice for all scenarios,
# especially if the data suggests other forms of relationships or distributions.
import StatsFuns: softplus
dotsoftplus(a, x) = softplus(dot(a, x))

# model definction
@model function bicycle_ssm(x, y, h0, θ0, a0, Q, s)

    a ~ a0
    θ ~ θ0
    h_prior ~ h0

    h_prev = h_prior
    for i in eachindex(y)
        
        h[i] ~ MvNormal(μ=transition(a, h_prev), Σ=Q)
        x[i] ~ MvNormal(μ=h[i], Σ=diageye(state_dim))
        y[i] ~ Normal(μ=dotsoftplus(θ, h[i]), σ²=s)
        h_prev = h[i]
    end

end

# In this example, we opt for a basic Linearization approach for the transition and dotsoftplus functions.
# However, alternative methods like Unscented or CVI approximations can also be considered.
bicycle_ssm_meta = @meta begin 
    transition() -> Linearization()
    dotsoftplus() -> Linearization()
end

# prior_h: Based on first observation, assuming initial state is similar with equal variance.
prior_h = MvNormalMeanCovariance(X[1], diageye(state_dim))
# prior_θ, prior_a: No initial bias, parameters independent with equal uncertainty.
prior_θ = MvNormalMeanCovariance(zeros(state_dim), diageye(state_dim))
prior_a = MvNormalMeanCovariance(zeros(state_dim^2), diageye(state_dim^2));

# the deterministic relationsships (transition) and (dotsoftplus) will induce loops in the graph representation of our model, this necessiates the initialization of the messages
imessages = @initialization begin
    μ(h) = prior_h
    μ(a) = prior_a
    μ(θ) = prior_θ
end
# Assumptions about the model parameters:
# Q: Process noise based on observed features' variance, assuming process variability reflects observed features variability.
# s: Observation noise based on observed data variance, directly estimating variance in the data, important for predictions
bicycle_model = bicycle_ssm(h0=prior_h, θ0=prior_θ, a0=prior_a, Q=var(filter(!ismissing, X)).*diageye(state_dim), s=var(filter(!ismissing, y)))

result = infer(
    model = bicycle_model,
    data  = (x=X, y=UnfactorizedData(y)), 
    options = (limit_stack_depth = 500, ), 
    returnvars = KeepLast(),
    predictvars = KeepLast(),
    initialization = imessages,
    meta = bicycle_ssm_meta,
    iterations = 20,
    showprogress=true,
)

# For a sake of this example, we extract only predictions
mean_y, cov_y = mean.(result.predictions[:y]), cov.(result.predictions[:y])
mean_x, cov_x = mean.(result.predictions[:x]), var.(result.predictions[:x])

mean_x1, cov_x1 = getindex.(mean_x, 1), getindex.(cov_x, 1)
mean_x2, cov_x2 = getindex.(mean_x, 2), getindex.(cov_x, 2)
mean_x3, cov_x3 = getindex.(mean_x, 3), getindex.(cov_x, 3)
mean_x4, cov_x4 = getindex.(mean_x, 4), getindex.(cov_x, 4);


slice = (300, length(y))
data = df[:, "count"][length(y)-n_future:length(y)]

p = scatter(y, 
            color=:darkblue, 
            markerstrokewidth=0,
            label="Observed Count", 
            alpha=0.6)

# Plotting the mean prediction with variance ribbon
plot!(mean_y, ribbon=sqrt.(cov_y), 
      color=:orange, 
      fillalpha=0.3,
      label="Predicted Mean ± Std Dev")

# Adding a vertical line to indicate the start of the future prediction
vline!([length(y)-n_future], 
       label="Prediction Start", 
       linestyle=:dash, 
       linecolor=:green)

# Future (unobserved) data
plot!(length(y)-n_future:length(y), data, label="Future Count")

# Adjusting the limits
xlims!(slice)

# Enhancing the plot with titles and labels
title!("Bike Rental Demand Prediction")
xlabel!("Time")
ylabel!("Bike Count")

# Adjust the legend
legend=:topright

# Show the final plot
display(p)

using Plots

# Define a color palette
palette = cgrad(:viridis)

# Plot the hidden states with observations
p1 = plot(mean_x1, ribbon=sqrt.(cov_x1), color=palette[1], label="Hidden State 1", legend=:topleft)
plot!(df[!, :temp], color=:grey, label="Temperature")
vline!([length(y)-n_future], linestyle=:dash, color=:red, label="Prediction Start")
xlabel!("Time")
ylabel!("Value")
title!("Temperature vs Hidden State 1")

p2 = plot(mean_x2, ribbon=sqrt.(cov_x2), color=palette[2], label="Hidden State 2", legend=:topleft)
plot!(df[!, :atemp], color=:grey, label="Feels-Like Temp")
vline!([length(y)-n_future], linestyle=:dash, color=:red, label="")
xlabel!("Time")
ylabel!("Value")
title!("Feels-Like Temp vs Hidden State 2")

p3 = plot(mean_x3, ribbon=sqrt.(cov_x3), color=palette[3], label="Hidden State 3", legend=:topleft)
plot!(df[!, :humidity], color=:grey, label="Humidity")
vline!([length(y)-n_future], linestyle=:dash, color=:red, label="Prediction Start")
xlabel!("Time")
ylabel!("Value")
title!("Humidity vs Hidden State 3")

p4 = plot(mean_x4, ribbon=sqrt.(cov_x4), color=palette[4], label="Hidden State 4", legend=:topleft)
plot!(df[!, :windspeed], color=:grey, label="Windspeed")
vline!([length(y)-n_future], linestyle=:dash, color=:red, label="Prediction Start")
xlabel!("Time")
ylabel!("Value")
title!("Windspeed vs Hidden State 4")

for p in [p1, p2, p3, p4]
    xlims!(p, first(slice), last(slice))
end

plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 400))

transformation = a -> reshape(a, state_dim, state_dim)

# model definction
@model function bicycle_ssm_advanced(x, y, h0, θ0, a0, P0, γ0)

    a ~ a0
    θ ~ θ0
    h_prior ~ h0
    P ~ P0
    γ ~ γ0

    h_prev = h_prior
    for i in eachindex(y)
        
        h[i] ~ CTransition(h_prev, a, P)
        x[i]  ~ MvNormal(μ=h[i], Λ=diageye(state_dim))
        _y[i] ~ softdot(θ, h[i], γ)
        y[i] ~ Normal(μ=softplus(_y[i]), γ=1e4)
        h_prev = h[i]
    end

end

bicycle_ssm_advanced_meta = @meta begin 
    softplus() -> Linearization()
    CTransition() -> CTMeta(transformation)
end

bicycle_ssm_advanced_constraints = @constraints begin
    q(h_prior, h, a, P, γ, _y, y, θ) = q(h_prior, h)q(a)q(P)q(γ)q(_y, y)q(θ)
end

prior_P = ExponentialFamily.WishartFast(state_dim+2, inv.(var(filter(!ismissing, X))) .* diageye(state_dim))
prior_a = MvNormalMeanPrecision(ones(state_dim^2), diageye(state_dim^2));

prior_γ = GammaShapeRate(1.0, var(filter(!ismissing, y)))
prior_h = MvNormalMeanPrecision(X[1], diageye(state_dim))
prior_θ = MvNormalMeanPrecision(ones(state_dim), diageye(state_dim))

imarginals = @initialization begin 
    q(h) = prior_h
    q(a) = prior_a
    q(P) = prior_P
    q(γ) = prior_γ
    q(θ) = prior_θ
end

bicycle_model_advanced = bicycle_ssm_advanced(h0=prior_h, θ0=prior_θ, a0=prior_a, P0=prior_P, γ0=prior_γ)

result_advanced = infer(
    model = bicycle_model_advanced,
    data  = (x=X, y=y), 
    options = (limit_stack_depth = 500, ), 
    returnvars = KeepLast(),
    predictvars = KeepLast(),
    initialization = imarginals,
    constraints = bicycle_ssm_advanced_constraints,
    meta = bicycle_ssm_advanced_meta,
    iterations = 10,
    showprogress=true,
)

# For a sake of this example, we extract only predictions
mean_y, cov_y = mean.(result_advanced.predictions[:y]), cov.(result_advanced.predictions[:y])

slice = (300, length(y))
data = df[:, "count"][length(y)-n_future:length(y)]

pa = scatter(y, 
            color=:darkblue, 
            markerstrokewidth=0,
            label="Observed Count", 
            alpha=0.6)

# Plotting the mean prediction with variance ribbon
plot!(mean_y, ribbon=sqrt.(cov_y), 
      color=:orange, 
      fillalpha=0.3,
      label="Predicted Mean ± Std Dev")

# Adding a vertical line to indicate the start of the future prediction
vline!([length(y)-n_future], 
       label="Prediction Start", 
       linestyle=:dash, 
       linecolor=:green)

# Future (unobserved) data
plot!(length(y)-n_future:length(y), data, label="Future Count")

# Adjusting the limits
xlims!(slice)

# Enhancing the plot with titles and labels
title!("Advanced model")
xlabel!("Time")
ylabel!("Bike Count")

# Adjust the legend
legend=:topright

# Show the final plot
plot(pa, p, size=(800, 400))