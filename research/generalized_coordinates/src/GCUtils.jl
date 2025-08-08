module GCUtils

using LinearAlgebra, Random, Distributions, StableRNGs

export constant_acceleration_ABQ, generate_gc_car_data, free_energy_timeseries

"""
constant_acceleration_ABQ(dt; σ_a=1.0, σ_v=1e-6, σ_p=1e-6, ε=1e-9)

Return (A, B, Q) for a 1D constant-acceleration model in generalized coordinates
x = [position, velocity, acceleration]. Ensures Q is positive-definite.
"""
function constant_acceleration_ABQ(dt; σ_a=1.0, σ_v=1e-6, σ_p=1e-6, ε=1e-9)
    A = [ 1.0  dt   0.5*dt^2;
          0.0  1.0  dt;
          0.0  0.0  1.0 ]
    B = I(3)
    q_diag = [max(σ_p^2, ε), max(σ_v^2, ε), max(σ_a^2, ε)]
    Q = Diagonal(q_diag)
    return A, B, Q
end

"""
generate_gc_car_data(rng, n, dt; σ_a=0.5, σ_obs_pos=0.5, σ_obs_vel=NaN,
                     x0 = [0.0, 1.0, 0.1], process_kwargs...)

Simulate trajectory under constant-acceleration, returning (x_seq, y_seq).
If σ_obs_vel is NaN, observe only position; else observe [position, velocity].
"""
function generate_gc_car_data(rng::AbstractRNG, n::Int, dt::Real;
                              σ_a=0.5, σ_obs_pos=0.5, σ_obs_vel=NaN,
                              x0 = [0.0, 1.0, 0.1], process_kwargs...)
    A, _, Q = constant_acceleration_ABQ(dt; σ_a=σ_a, process_kwargs...)
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
    dyn_term = zeros(Float64, n)

    for t in 1:n
        # Observation contribution
        μy = B * μ[t]
        Σy = B * Σ[t] * B'
        resid = y[t] .- μy
        obs_term[t] = 0.5 * (dimy * log2π + logdetR + tr(Rinvt * Σy) + dot(resid, Rinvt * resid))

        if t == 1
            # Prior contribution
            Σ0 = x0_cov
            μ0 = x0_mean
            resid0 = μ[1] .- μ0
            dyn_term[t] = 0.5 * (dimx * log2π + logdet(x0_cov) + tr(inv(Σ0) * Σ[1]) + dot(resid0, inv(Σ0) * resid0))
        else
            μpred = A * μ[t-1]
            Σpred = A * Σ[t-1] * A'
            residx = μ[t] .- μpred
            dyn_term[t] = 0.5 * (dimx * log2π + logdetQ + tr(Qinvt * (Σ[t] + Σpred)) + dot(residx, Qinvt * residx))
        end
    end

    total = obs_term .+ dyn_term
    return (; obs_term, dyn_term, total)
end

end # module
