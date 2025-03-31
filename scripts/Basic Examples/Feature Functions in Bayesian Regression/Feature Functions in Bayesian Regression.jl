# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Feature Functions in Bayesian Regression/Feature Functions in Bayesian Regression.ipynb
# by notebooks_to_scripts.jl at 2025-03-31T09:50:41.133
#
# Source notebook: Feature Functions in Bayesian Regression.ipynb

using RxInfer, StableRNGs, LinearAlgebra, Plots, DataFrames

N = 40
Λ = I
X = range(-8, 8, length=N)

rng = StableRNG(42)

# Arbitrary non-linear function, which is hidden
f(x) = -((-x / 3)^3 - (-x / 2)^2 + x + 10) 

Y = rand(rng, MvNormalMeanCovariance(f.(X), Λ))

# Can be loaded from a file or a database
df = DataFrame(X = X, Y = Y)

# Split data into train/test sets
# Forward split - first half train, second half test
dataset_1 = let mid = N ÷ 2
    (
        y_train = Y[1:mid], x_train = X[1:mid],
        y_test = Y[mid+1:end], x_test = X[mid+1:end]
    )
end

# Reverse split - first half test, second half train  
dataset_2 = let mid = N ÷ 2
    (
        y_test = Y[1:mid], x_test = X[1:mid],
        y_train = Y[mid+1:end], x_train = X[mid+1:end]
    )
end

# Interleaved split - first/last quarters train, middle half test
dataset_3 = let q1 = N ÷ 4, q3 = 3N ÷ 4
    (
        y_train = [Y[1:q1]..., Y[q3+1:end]...],
        x_train = [X[1:q1]..., X[q3+1:end]...],
        y_test = Y[q1+1:q3],
        x_test = X[q1+1:q3]
    )
end

datasets = [dataset_1, dataset_2, dataset_3]

# Create visualization for each dataset split
ps = map(enumerate(datasets)) do (i, dataset)
    p = plot(
        xlim = (-10, 10), 
        ylim = (-30, 30),
        title = "Dataset $i",
        xlabel = "x",
        ylabel = "y"
    )
    scatter!(p, 
        dataset[:x_train], dataset[:y_train],
        yerror = Λ,
        label = "Train dataset",
        color = :blue,
        markersize = 4
    )
    scatter!(p,
        dataset[:x_test], dataset[:y_test], 
        yerror = Λ,
        label = "Test dataset",
        color = :red,
        markersize = 4
    )
    return p
end

plot(ps..., size = (1200, 400), layout = @layout([a b c]))

@model function parametric_regression(ϕs, x, y, μ, Σ, Λ)
    # Prior distribution over parameters ω
    ω ~ MvNormal(mean = μ, covariance = Σ)
    
    # Design matrix Φₓ where each element is ϕᵢ(xⱼ)
    Φₓ = [ϕ(xᵢ) for xᵢ in x, ϕ in ϕs]
    
    # Likelihood of observations y given parameters ω
    y ~ MvNormal(mean = Φₓ * ω, covariance = Λ)
end

function infer_ω(; ϕs, x, y)
    # Create probabilistic model, 
    # RxInfer will construct the graph of this model auutomatically
    model = parametric_regression(
        ϕs = ϕs, 
        μ  = zeros(length(ϕs)),
        Σ  = I,
        Λ  = I,
        x  = x
    )

    # Let RxInfer do all the math for you
    result = infer(
        model = model, 
        data  = (y = y,)
    )

    # Return posterior over ω
    return result.posteriors[:ω]
end

function plot_inference_results_for(; ϕs, datasets, title = "", rng = StableRNG(42))
    # Create main plot showing basis functions
    p1 = plot(
        title = "Basis functions: $(title)", 
        xlabel = "x",
        ylabel = "y",
        xlim = (-5, 5), 
        ylim = (-10, 10),
        legend = :outertopleft,
        grid = true,
        fontfamily = "Computer Modern"
    )

    # Plot basis functions in gray
    plot_ϕ!(p1, ϕs, color = :gray, alpha = 0.5, 
            labels = ["ϕ$i" for _ in 1:1, i in 1:length(ϕs)])
    
    # Add examples with random ω values
    plot_ϕ!(p1, ϕs, randn(rng, length(ϕs), 3), 
            linewidth = 2)

    # Create subplot for each dataset
    ps = map(enumerate(datasets)) do (i, dataset)
        p2 = plot(
            title = "Dataset #$(i): $(title)",
            xlabel = "x",
            ylabel = "y", 
            xlim = (-10, 10),
            ylim = (-25, 25),
            grid = true,
            fontfamily = "Computer Modern"
        )

        # Infer posterior over ω
        ωs = infer_ω(
            ϕs = ϕs, 
            x = dataset[:x_train], 
            y = dataset[:y_train]
        )

        # Plot posterior mean
        plot_ϕ!(p2, ϕs, mean(ωs),
                linewidth = 3,
                color = :green,
                labels = "Posterior mean")

        # Plot posterior samples
        plot_ϕ!(p2, ϕs, rand(ωs, 15),
                linewidth = 1,
                color = :gray,
                alpha = 0.4,
                labels = nothing)

        # Add data points
        scatter!(p2, dataset[:x_train], dataset[:y_train],
                yerror = Λ,
                label = "Training data",
                color = :royalblue,
                markersize = 4)
        scatter!(p2, dataset[:x_test], dataset[:y_test],
                yerror = Λ,
                label = "Test data", 
                color = :crimson,
                markersize = 4)

        return p2
    end

    # Combine all plots
    plot(p1, ps..., 
         size = (1000, 800),
         margin = 5Plots.mm,
         layout = (2,2))
end

# Helper function to plot basis functions
function plot_ϕ!(p, ϕs; rl = -10, rr = 10, kwargs...)
    xs = range(rl, rr, length = 200)
    ys = [ϕ(x) for x in xs, ϕ in ϕs]
    plot!(p, xs, ys; kwargs...)
end

# Helper function to plot function with given weights
function plot_ϕ!(p, ϕs, ωs; rl = -10, rr = 10, kwargs...)
    xs = range(rl, rr, length = 200)
    ys = [ϕ(x) for x in xs, ϕ in ϕs]
    yr = ys * ωs
    labels = ["Sample $i" for _ in 1:1, i in 1:size(ωs,2)]
    plot!(p, xs, yr, labels = labels; kwargs...)
end

plot_inference_results_for(
    title    = "polynomials",
    datasets = datasets,
    ϕs       = [ (x) -> x ^ i for i in 0:5 ], 
)

plot_inference_results_for(
    title    = "trigonometric sin",
    datasets = datasets,
    ϕs       = [ (x) -> sin(x / i) for i in 1:8 ], 
)

plot_inference_results_for(
    title    = "trigonometric cos",
    datasets = datasets,
    ϕs       = [ (x) -> cos(x / i) for i in 1:8 ], 
)

plot_inference_results_for(
    title    = "trigonometric sin & cos",
    datasets = datasets,
    ϕs       = [
        [ (x) -> sin(x / i) for i in 1:4 ]...,
        [ (x) -> cos(x / i) for i in 1:4 ]...,
    ], 
)

# Combine the function definition with the usage
function infer_ω_but_return_free_energy(; ϕs, x, y)
    result = infer(
        model = parametric_regression(
            ϕs = ϕs, 
            μ  = zeros(length(ϕs)),
            Σ  = I,
            Λ  = I,
            x  = x
        ), 
        data  = (y = y,),
        free_energy = true
    )
    return first(result.free_energy)
end

dfs = map(enumerate(datasets)) do (i, dataset)
    # Generate basis functions
    sin_bases = [(x) -> sin(x / i) for i in 1:8]
    cos_bases = [(x) -> cos(x / i) for i in 1:8]
    combined_bases = [
        [(x) -> sin(x / i) for i in 1:4]...,
        [(x) -> cos(x / i) for i in 1:4]...
    ]

    # Calculate free energy for each basis
    energies = [
        infer_ω_but_return_free_energy(ϕs=sin_bases, x=dataset[:x_train], y=dataset[:y_train]),
        infer_ω_but_return_free_energy(ϕs=cos_bases, x=dataset[:x_train], y=dataset[:y_train]),
        infer_ω_but_return_free_energy(ϕs=combined_bases, x=dataset[:x_train], y=dataset[:y_train])
    ]

    # Create DataFrame row
    DataFrame(
        dataset = fill(i, 3),
        fns = [:sin, :cos, :sin_cos],
        free_energy = energies
    )
end

vcat(dfs...)

plot_inference_results_for(
    title    = "switches",
    datasets = datasets,
    ϕs       = [ (x) -> sign(x - i) for i in -8:8 ], 
)

plot_inference_results_for(
    title    = "steps",
    datasets = datasets,
    ϕs       = [ (x) -> ifelse(x - i > 0, 1.0, 0.0) for i in -8:8 ], 
)

plot_inference_results_for(
    title    = "linears",
    datasets = datasets,
    ϕs       = [ (x) -> abs(x - i) for i in -8:8 ], 
)

plot_inference_results_for(
    title    = "abs exps",
    datasets = datasets,
    ϕs       = [ (x) -> exp(-abs(x - i)) for i in -8:8 ], 
)

plot_inference_results_for(
    title    = "sqrt exps",
    datasets = datasets,
    ϕs       = [ (x) -> exp(-(x - i) ^ 2) for i in -8:8 ], 
)

plot_inference_results_for(
    title    = "sigmoids",
    datasets = datasets,
    ϕs       = [ (x) -> 1 / (1 + exp(-3 * (x - i))) for i in -8:8 ], 
)

# Combine all basis functions we've explored into one powerful basis
combined_basis = vcat(
    # Polynomials (from first example)
    [ (x) -> x ^ i for i in 0:5 ],
    
    # Trigonometric functions (from second example) 
    [ (x) -> sin(i*x) for i in 1:3 ],
    [ (x) -> cos(i*x) for i in 1:3 ],
    
    # Squared exponentials (from seventh example)
    [ (x) -> exp(-(x - i)^2) for i in -8:8 ],
    
    # Sigmoids (from eighth example)
    [ (x) -> 1 / (1 + exp(-3 * (x - i))) for i in -8:8 ]
)

plot_inference_results_for(
    title    = "combined",
    datasets = datasets,
    ϕs       = combined_basis, 
)

combined_basis_ωs_all_data = infer_ω(ϕs = combined_basis, x = X, y = Y)

# Left plot - local region
p1 = plot(
    title = "Local region",
    xlabel = "x",
    ylabel = "y",
    xlim = (-10, 10),
    ylim = (-20, 20),
    grid = true
)

# Plot posterior mean
plot_ϕ!(p1, combined_basis, mean(combined_basis_ωs_all_data),
    rl = -10,
    rr = 10,
    linewidth = 3,
    color     = :green,
    labels    = "Posterior mean"
)

# Plot posterior samples (in gray)
plot_ϕ!(p1, combined_basis, rand(combined_basis_ωs_all_data, 50),
    rl = -10,
    rr = 10,
    linewidth = 1,
    color     = :gray,
    alpha     = 0.4,
    labels    = nothing
)

# Plot data points
scatter!(p1, X, Y,
    yerror     = Λ,
    label      = "Data",
    color      = :royalblue,
    markersize = 4
)

# Right plot - bigger region
p2 = plot(
    title = "Extended region",
    xlabel = "x",
    ylabel = "y",
    xlim = (-30, 30),
    ylim = (-75, 75),
    grid = true
)

# Plot posterior mean
plot_ϕ!(p2, combined_basis, mean(combined_basis_ωs_all_data),
    rl = -30,
    rr = 30,
    linewidth = 3,
    color     = :green,
    labels    = "Posterior mean"
)

# Plot posterior samples (in gray)
plot_ϕ!(p2, combined_basis, rand(combined_basis_ωs_all_data, 50),
    rl = -30,
    rr = 30,
    linewidth = 1,
    color     = :gray,
    alpha     = 0.4,
    labels    = nothing
)

# Plot data points
scatter!(p2, X, Y,
    label      = "Data", 
    color      = :royalblue,
    markersize = 2
)

p = plot(p1, p2, layout=(1,2), size=(1000,400), fontfamily = "Computer Modern")

using BenchmarkTools

@benchmark infer_ω(ϕs = $([ (x) -> x ^ i for i in 0:5 ]), x = $(datasets[1][:x_train]), y = $(datasets[1][:y_train]))

N_benchmark = 10_000
X_benchmark = range(-8, 8, length=N_benchmark)
Y_benchmark = rand(rng, MvNormalMeanCovariance(f.(X_benchmark), Λ));

@benchmark infer_ω(ϕs = $([ (x) -> x ^ i for i in 0:5 ]), x = $(X_benchmark), y = $(Y_benchmark))