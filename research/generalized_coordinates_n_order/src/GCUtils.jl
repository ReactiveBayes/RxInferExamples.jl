module GCUtils

using LinearAlgebra, Random, Distributions, StableRNGs

export constant_acceleration_ABQ, generate_gc_car_data, free_energy_timeseries

"""
_taylor_integrator_A(dt, order)

Build the K×K upper-triangular discrete-time integrator matrix for generalized
coordinates up to given `order` (K = order). Entry A[i,j] = dt^(j-i)/(j-i)!
for j ≥ i, else 0. For order=3 this matches the standard constant-acceleration A.
"""
function _taylor_integrator_A(dt::Real, order::Int)
    K = order
    A = zeros(Float64, K, K)
    for i in 1:K
        for j in i:K
            k = j - i
            # dt^k / k!
            A[i, j] = k == 0 ? 1.0 : (dt^k) / factorial(k)
        end
    end
    return A
end

"""
constant_acceleration_ABQ(dt; order=6, σ_a=1.0, σ_v=1e-6, σ_p=1e-6, ε=1e-9)

Return (A, B, Q) for a 1D generalized-coordinates chain of integrators of size `order`.
Defaults to `order=6`. For backward compatibility, σ_p, σ_v, σ_a populate the first
three diagonal entries of Q; higher-order entries get ε by default. Ensures Q is PD.
"""
function constant_acceleration_ABQ(dt; order::Int=6, σ_a=1.0, σ_v=1e-6, σ_p=1e-6, ε=1e-9)
    K = order
    A = _taylor_integrator_A(dt, K)
    B = I(K)
    q_diag = fill(max(ε, 0.0), K)
    if K >= 1
        q_diag[1] = max(σ_p^2, ε)
    end
    if K >= 2
        q_diag[2] = max(σ_v^2, ε)
    end
    if K >= 3
        q_diag[3] = max(σ_a^2, ε)
    end
    Q = Diagonal(q_diag)
    return A, B, Q
end

"""
generate_gc_car_data(rng, n, dt; order=6, σ_a=0.5, σ_obs_pos=0.5, σ_obs_vel=NaN,
                     x0 = nothing, process_kwargs...)

Simulate trajectory under a generalized-coordinates constant-acceleration model
of dimension `order`, returning (x_seq, y_seq). If σ_obs_vel is NaN, observe only
position; else observe [position, velocity].
"""
function generate_gc_car_data(rng::AbstractRNG, n::Int, dt::Real;
                              order::Int=6, σ_a=0.5, σ_obs_pos=0.5, σ_obs_vel=NaN,
                              x0 = nothing, process_kwargs...)
    A, _, Q = constant_acceleration_ABQ(dt; order=order, σ_a=σ_a, process_kwargs...)
    K = size(A, 1)
    if x0 === nothing
        # Default initial state sized to K: take first K of [pos, vel, acc], then zeros
        base = [0.0, 1.0, 0.1]
        x0 = vcat(base[1:min(K, 3)], zeros(max(0, K - 3)))
    else
        length(x0) == K || error("x0 length $(length(x0)) must match order $K")
    end
    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    x_prev = collect(x0)
    for t in 1:n
        x[t] = rand(rng, MvNormal(A * x_prev, Matrix(Q)))
        if isnan(σ_obs_vel)
            y[t] = [x[t][1] + randn(rng) * σ_obs_pos]
        else
            y[t] = [x[t][1] + randn(rng) * σ_obs_pos,
                    x[t][2] + randn(rng) * σ_obs_vel]
        end
        x_prev = x[t]
    end
    return x, y
end

# Compute per-time Gaussian contributions approximating node-level free energy terms
# Observation term: E_q[-log p(y[t]|x[t])]
# Dynamics term: E_q[-log p(x[t]|x[t-1])], for t>=2; plus prior term at t=1
function free_energy_timeseries(y::Vector{<:AbstractVector}, xmarginals, A::AbstractMatrix, B::AbstractMatrix,
                                Q::AbstractMatrix, R::AbstractMatrix, x0_mean::AbstractVector, x0_cov::AbstractMatrix)
    n = length(y)
    dimx = length(x0_mean)
    dimy = length(first(y))

    Rinvt = inv(R)
    Qinvt = inv(Q)
    logdetR = logdet(R)
    logdetQ = logdet(Q)
    log2π = log(2π)

    μ = [mean(xmarginals[t]) for t in 1:n]
    Σ = Vector{Matrix{Float64}}(undef, n)
    for t in 1:n
        Σt = try
            cov(xmarginals[t])
        catch
            v = var(xmarginals[t])
            v isa AbstractVector ? Diagonal(v) : Matrix(v)
        end
        Σ[t] = Matrix(Σt)
    end

    obs_term = zeros(Float64, n)
    prior_term = zeros(Float64, n)
    dyn_term = zeros(Float64, n)
    # Optional: per-dimension observation terms (only sound for diagonal R)
    obs_dim_terms = Vector{Vector{Float64}}(undef, n)

    for t in 1:n
        # Observation contribution
        μy = B * μ[t]
        Σy = B * Σ[t] * B'
        resid = y[t] .- μy
        obs_term[t] = 0.5 * (dimy * log2π + logdetR + tr(Rinvt * Σy) + dot(resid, Rinvt * resid))
        # Per-dimension split only when R is diagonal (approximate otherwise)
        if isdiag(R)
            rd = diag(R)
            Σy_diag = diag(Σy)
            obs_dim_terms[t] = [0.5 * (log2π + log(rd[d]) + (Σy_diag[d] / rd[d]) + (resid[d]^2 / rd[d])) for d in 1:dimy]
        else
            obs_dim_terms[t] = fill(NaN, dimy)
        end

        if t == 1
            # Prior contribution
            Σ0 = x0_cov
            μ0 = x0_mean
            resid0 = μ[1] .- μ0
            prior_term[t] = 0.5 * (dimx * log2π + logdet(x0_cov) + tr(inv(Σ0) * Σ[1]) + dot(resid0, inv(Σ0) * resid0))
        else
            μpred = A * μ[t-1]
            Σpred = A * Σ[t-1] * A'
            residx = μ[t] .- μpred
            dyn_term[t] = 0.5 * (dimx * log2π + logdetQ + tr(Qinvt * (Σ[t] + Σpred)) + dot(residx, Qinvt * residx))
        end
    end

    total = obs_term .+ prior_term .+ dyn_term
    return (; obs_term, prior_term, dyn_term, total, obs_dim_terms)
end

end # module
