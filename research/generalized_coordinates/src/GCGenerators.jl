module GCGenerators

using ..GCUtils
using LinearAlgebra, Random

export generate_scenario

_time_vector(n::Int, dt::Real) = collect(0:dt:(n-1)*dt)

function _observations_from_truth(rng::AbstractRNG, pos::AbstractVector, vel::AbstractVector,
                                  σ_obs_pos::Float64, σ_obs_vel::Float64)
    n = length(pos)
    y = Vector{Vector{Float64}}(undef, n)
    if isnan(σ_obs_vel)
        for i in 1:n
            y[i] = [pos[i] + randn(rng) * σ_obs_pos]
        end
    else
        for i in 1:n
            y[i] = [pos[i] + randn(rng) * σ_obs_pos,
                    vel[i] + randn(rng) * σ_obs_vel]
        end
    end
    return y
end

"""
generate_scenario(rng, cfg::GCConfig.ScenarioConfig)
Return (x_true, y, A, Q) according to cfg.
Supported generators: :constant_accel, :sinusoid, :sinusoid_mixed, :poly, :trend_plus_osc
"""
function generate_scenario(rng::AbstractRNG, cfg)
    if cfg.generator == :constant_accel
        x_true, y = GCUtils.generate_gc_car_data(rng, cfg.n, cfg.dt; σ_a=cfg.σ_a,
                                                 σ_obs_pos=cfg.σ_obs_pos, σ_obs_vel=cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    elseif cfg.generator == :sinusoid
        f = get(cfg.generator_kwargs, :freq, 0.2)
        amp = get(cfg.generator_kwargs, :amp, 5.0)
        n, dt = cfg.n, cfg.dt
        t = _time_vector(n, dt)
        pos = amp .* sin.(2π*f .* t)
        vel = 2π*f .* amp .* cos.(2π*f .* t)
        acc = -(2π*f)^2 .* amp .* sin.(2π*f .* t)
        x_true = [Float64[pos[i], vel[i], acc[i]] for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    elseif cfg.generator == :sinusoid_mixed
        n, dt = cfg.n, cfg.dt
        t = _time_vector(n, dt)
        freqs = get(cfg.generator_kwargs, :freqs, [0.1, 0.25])
        amps  = get(cfg.generator_kwargs, :amps,  [3.0, 1.5])
        pos = zeros(n); vel = zeros(n); acc = zeros(n)
        for (f, a) in zip(freqs, amps)
            pos .+= a .* sin.(2π*f .* t)
            vel .+= 2π*f .* a .* cos.(2π*f .* t)
            acc .+= -(2π*f)^2 .* a .* sin.(2π*f .* t)
        end
        x_true = [Float64[pos[i], vel[i], acc[i]] for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    elseif cfg.generator == :poly
        # Polynomial trend: p(t) = sum_{k=0}^d a_k t^k
        n, dt = cfg.n, cfg.dt
        t = _time_vector(n, dt)
        d = get(cfg.generator_kwargs, :degree, 3)
        coeffs = get(cfg.generator_kwargs, :coeffs, [randn(rng) for _ in 0:d])
        # ensure length d+1
        length(coeffs) == d+1 || (coeffs = vcat(coeffs, zeros(d+1-length(coeffs))))
        pos = zeros(n); vel = zeros(n); acc = zeros(n)
        for (k, a) in enumerate(coeffs) # k indexes 1..d+1 corresponds to power k-1
            p = k-1
            pos .+= a .* (t .^ p)
            if p >= 1
                vel .+= a * p .* (t .^ (p-1))
            end
            if p >= 2
                acc .+= a * p * (p-1) .* (t .^ (p-2))
            end
        end
        x_true = [Float64[pos[i], vel[i], acc[i]] for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    elseif cfg.generator == :trend_plus_osc
        # Polynomial trend + one sinusoid
        n, dt = cfg.n, cfg.dt
        t = _time_vector(n, dt)
        d = get(cfg.generator_kwargs, :degree, 2)
        coeffs = get(cfg.generator_kwargs, :coeffs, [0.0, 0.0, 0.01][1:(d+1)])
        f = get(cfg.generator_kwargs, :freq, 0.1)
        amp = get(cfg.generator_kwargs, :amp, 2.0)
        pos = zeros(n); vel = zeros(n); acc = zeros(n)
        for (k, a) in enumerate(coeffs)
            p = k-1
            pos .+= a .* (t .^ p)
            if p >= 1
                vel .+= a * p .* (t .^ (p-1))
            end
            if p >= 2
                acc .+= a * p * (p-1) .* (t .^ (p-2))
            end
        end
        pos .+= amp .* sin.(2π*f .* t)
        vel .+= 2π*f .* amp .* cos.(2π*f .* t)
        acc .+= -(2π*f)^2 .* amp .* sin.(2π*f .* t)
        x_true = [Float64[pos[i], vel[i], acc[i]] for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    else
        error("Unknown generator: $(cfg.generator)")
    end
end

end # module


