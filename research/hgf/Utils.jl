module Utils

using Distributions
using Random
using StableRNGs
using Logging

export HGFParams, default_hgf_params, generate_data

Base.@kwdef struct HGFParams
    seed::Int = 42
    real_k::Float64 = 1.0
    real_w::Float64 = 0.0
    z_variance::Float64 = abs2(0.2)
    y_variance::Float64 = abs2(0.1)
    n::Int = 300
end

default_hgf_params() = HGFParams()

function generate_data(params::HGFParams)
    rng = StableRNG(params.seed)
    return generate_data(rng, params.n, params.real_k, params.real_w, params.z_variance, params.y_variance)
end

function generate_data(rng::AbstractRNG, n::Int, k::Float64, w::Float64, z_var::Float64, y_var::Float64)
    @info "Generating HGF synthetic data" n k w z_var y_var
    z_prev = 0.0
    x_prev = 0.0

    z = Vector{Float64}(undef, n)
    v = Vector{Float64}(undef, n)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)

    for i in 1:n
        z[i] = rand(rng, Normal(z_prev, sqrt(z_var)))
        v[i] = exp(k * z[i] + w)
        x[i] = rand(rng, Normal(x_prev, sqrt(v[i])))
        y[i] = rand(rng, Normal(x[i], sqrt(y_var)))

        z_prev = z[i]
        x_prev = x[i]
    end

    return z, x, y
end

end # module


