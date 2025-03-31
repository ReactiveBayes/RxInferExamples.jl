# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Parameter Optimisation with Optim.jl/Parameter Optimisation with Optim.jl.ipynb
# by notebooks_to_scripts.jl at 2025-03-31T09:50:40.888
#
# Source notebook: Parameter Optimisation with Optim.jl.ipynb

using RxInfer, StableRNGs, LinearAlgebra, Plots

@model function univariate_state_space_model(y, x_prior, c, v)
    
    x0 ~ Normal(mean = mean(x_prior), variance = var(x_prior))
    x_prev = x0

    for i in eachindex(y)
        x[i] := x_prev + c
        y[i] ~ Normal(mean = x[i], variance = v)
        x_prev = x[i]
    end
end

rng    = StableRNG(42)
v      = 1.0
n      = 250
c_real = -5.0
signal = c_real .+ collect(1:n)
data   = map(x -> rand(rng, NormalMeanVariance(x, v)), signal);

# params[1] is C
# params[2] is μ1
function f(params)
    x_prior = NormalMeanVariance(params[2], 100.0)
    result = infer(
        model = univariate_state_space_model(
            x_prior = x_prior, 
            c       = params[1], 
            v       = v
        ), 
        data  = (y = data,), 
        free_energy = true
    )
    return result.free_energy[end]
end

using Optim

res = optimize(f, ones(2), GradientDescent(), Optim.Options(g_tol = 1e-3, iterations = 100, store_trace = true, show_trace = true, show_every = 10))

println("Real value vs Optimized")
println("Real:      ", [ 1.0, c_real ])
println("Optimized: ", res.minimizer)

@model function multivariate_state_space_model(y, θ, x0, Q, P)
    
    x_prior ~ MvNormal(mean = mean(x0), cov = cov(x0))
    x_prev = x_prior
    
    A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
    
    for i in eachindex(y)
        x[i] ~ MvNormal(mean = A * x_prev, covariance = Q)
        y[i] ~ MvNormal(mean = x[i], covariance = P)
        x_prev = x[i]
    end
    
end

# Generate data
function generate_rotate_ssm_data()
    rng = StableRNG(1234)

    θ = π / 8
    A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
    Q = Matrix(Diagonal(1.0 * ones(2)))
    P = Matrix(Diagonal(1.0 * ones(2)))

    n = 300

    x_prev = [ 10.0, -10.0 ]

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        
        x[i] = rand(rng, MvNormal(A * x_prev, Q))
        y[i] = rand(rng, MvNormal(x[i], Q))
        
        x_prev = x[i]
    end

    return θ, A, Q, P, n, x, y
end

θ, A, Q, P, n, x, y = generate_rotate_ssm_data();

px = plot()

px = plot!(px, getindex.(x, 1), ribbon = diag(Q)[1] .|> sqrt, fillalpha = 0.2, label = "real₁")
px = plot!(px, getindex.(x, 2), ribbon = diag(Q)[2] .|> sqrt, fillalpha = 0.2, label = "real₂")

plot(px, size = (1200, 450))

function f(params)
    x0 = MvNormalMeanCovariance(
        [ params[2], params[3] ], 
        Matrix(Diagonal(0.01 * ones(2)))
    )
    result = infer(
        model = multivariate_state_space_model(
            θ = params[1], 
            x0 = x0, 
            Q = Q, 
            P = P
        ), 
        data  = (y = y,), 
        free_energy = true
    )
    return result.free_energy[end]
end

res = optimize(f, zeros(3), LBFGS(), Optim.Options(f_tol = 1e-14, g_tol = 1e-12, show_trace = true, show_every = 10))

println("Real value vs Optimized")
println("sinθ = (", sin(θ), ", ", sin(res.minimizer[1]), ")")
println("cosθ = (", cos(θ), ", ", cos(res.minimizer[1]), ")")

x0 = MvNormalMeanCovariance([ res.minimizer[2], res.minimizer[3] ], Matrix(Diagonal(100.0 * ones(2))))

result = infer(
    model = multivariate_state_space_model(
        θ = res.minimizer[1], 
        x0 = x0, 
        Q = Q, 
        P = P
    ), 
    data  = (y = y,), 
    free_energy = true
)

xmarginals = result.posteriors[:x]

px = plot()

px = plot!(px, getindex.(x, 1), ribbon = diag(Q)[1] .|> sqrt, fillalpha = 0.2, label = "real₁")
px = plot!(px, getindex.(x, 2), ribbon = diag(Q)[2] .|> sqrt, fillalpha = 0.2, label = "real₂")
px = plot!(px, getindex.(mean.(xmarginals), 1), ribbon = getindex.(var.(xmarginals), 1) .|> sqrt, fillalpha = 0.5, label = "inf₁")
px = plot!(px, getindex.(mean.(xmarginals), 2), ribbon = getindex.(var.(xmarginals), 2) .|> sqrt, fillalpha = 0.5, label = "inf₂")

plot(px, size = (1200, 450))