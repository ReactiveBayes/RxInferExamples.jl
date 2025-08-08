module GCGenerators

using ..GCUtils
using LinearAlgebra, Random

export generate_scenario

_time_vector(n::Int, dt::Real) = collect(0:dt:(n-1)*dt)

_vec3_to_K(pos::Float64, vel::Float64, acc::Float64, K::Int) = K <= 3 ? Float64[pos, vel][1:min(K,2)] |> v -> (K == 3 ? vcat(v, acc) : v) : vcat(Float64[pos, vel, acc], zeros(max(0, K-3)))

function _toK_triple(pos::Float64, vel::Float64, acc::Float64, K::Int)
    if K <= 0
        return Float64[]
    elseif K == 1
        return Float64[pos]
    elseif K == 2
        return Float64[pos, vel]
    else
        return vcat(Float64[pos, vel, acc], zeros(max(0, K-3)))
    end
end

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
Supported generators: :constant_accel, :sinusoid, :sinusoid_mixed, :poly, :trend_plus_osc, :poly_sin_mixed, :piecewise_mixed
"""
function generate_scenario(rng::AbstractRNG, cfg)
    if cfg.generator == :constant_accel
        x_true, y = GCUtils.generate_gc_car_data(rng, cfg.n, cfg.dt; order=cfg.order, σ_a=cfg.σ_a,
                                                 σ_obs_pos=cfg.σ_obs_pos, σ_obs_vel=cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; order=cfg.order, σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    elseif cfg.generator == :sinusoid
        f = get(cfg.generator_kwargs, :freq, 0.2)
        amp = get(cfg.generator_kwargs, :amp, 5.0)
        n, dt = cfg.n, cfg.dt
        t = _time_vector(n, dt)
        pos = amp .* sin.(2π*f .* t)
        vel = 2π*f .* amp .* cos.(2π*f .* t)
        acc = -(2π*f)^2 .* amp .* sin.(2π*f .* t)
        # Shape to K
        x_true = [_toK_triple(pos[i], vel[i], acc[i], cfg.order) for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; order=cfg.order, σ_a=cfg.σ_a)
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
        x_true = [_toK_triple(pos[i], vel[i], acc[i], cfg.order) for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; order=cfg.order, σ_a=cfg.σ_a)
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
        x_true = [_toK_triple(pos[i], vel[i], acc[i], cfg.order) for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; order=cfg.order, σ_a=cfg.σ_a)
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
        x_true = [_toK_triple(pos[i], vel[i], acc[i], cfg.order) for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; order=cfg.order, σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    elseif cfg.generator == :poly_sin_mixed
        # High-degree polynomial trend plus multiple sinusoids
        n, dt = cfg.n, cfg.dt
        t = _time_vector(n, dt)
        d = get(cfg.generator_kwargs, :degree, 8)
        coeffs = get(cfg.generator_kwargs, :coeffs, [0.0, 1.0, 0.0, 1e-3, 0.0, -1e-5, 0.0, 1e-7, -1e-9][1:(d+1)])
        freqs = get(cfg.generator_kwargs, :freqs, [0.05, 0.2, 0.45])
        amps  = get(cfg.generator_kwargs,  :amps,  [2.0, 1.0, 0.5])
        phases = get(cfg.generator_kwargs, :phases, zeros(length(freqs)))
        pos = zeros(n); vel = zeros(n); acc = zeros(n)
        # Polynomial contribution
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
        # Mixed sinusoids contribution
        for (f, a, φ) in zip(freqs, amps, phases)
            ω = 2π*f
            θ = ω .* t .+ φ
            pos .+= a .* sin.(θ)
            vel .+= ω .* a .* cos.(θ)
            acc .+= -(ω^2) .* a .* sin.(θ)
        end
        x_true = [_toK_triple(pos[i], vel[i], acc[i], cfg.order) for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; order=cfg.order, σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    elseif cfg.generator == :piecewise_mixed
        # Piecewise regimes with abrupt changes every `segment_length` steps
        n, dt = cfg.n, cfg.dt
        t = _time_vector(n, dt)
        segment_length = get(cfg.generator_kwargs, :segment_length, 500)
        num_segments = cld(n, segment_length)
        pos = zeros(n); vel = zeros(n); acc = zeros(n)
        default_segments = [
            Dict(:coeffs=>[0.0, 0.15, 0.0, 6e-4, 0.0, -1e-6], :freqs=>[0.05, 0.12], :amps=>[6.0, 3.5], :phases=>[0.0, π/3]),
            Dict(:coeffs=>[0.0, -0.1, 0.0, -5e-4, 0.0, 1e-6], :freqs=>[0.4, 0.8, 1.2], :amps=>[5.0, 3.0, 2.0], :phases=>[π/6, π/2, 2π/3]),
            Dict(:coeffs=>[0.0, 0.05, 0.0, 3e-4, 0.0, -8e-7], :freqs=>[0.2, 0.33, 0.5], :amps=>[4.0, 2.5, 1.5], :phases=>[π/4, 3π/4, π/8]),
            Dict(:coeffs=>[0.0, 0.0, 0.0, -2e-4, 0.0, 5e-7], :freqs=>[0.1, 0.25, 0.65], :amps=>[7.0, 3.0, 2.0], :phases=>[π/2, π/5, 7π/8]),
        ]
        segments = get(cfg.generator_kwargs, :segments, default_segments)
        # If provided fewer specs than segments, cycle; if more, truncate
        for seg_idx in 1:num_segments
            spec = segments[mod1(seg_idx, length(segments))]
            coeffs = get(spec, :coeffs, Float64[0.0])
            freqs  = get(spec, :freqs,  Float64[0.2])
            amps   = get(spec, :amps,   Float64[2.0])
            phases = get(spec, :phases, zeros(length(freqs)))
            s = (seg_idx-1)*segment_length + 1
            e = min(seg_idx*segment_length, n)
            idxs = s:e
            t_local = (0:length(idxs)-1) .* dt
            # Polynomial
            pos_seg = zeros(length(idxs)); vel_seg = zeros(length(idxs)); acc_seg = zeros(length(idxs))
            for (k, a) in enumerate(coeffs)
                p = k-1
                pos_seg .+= a .* (t_local .^ p)
                if p >= 1
                    vel_seg .+= a * p .* (t_local .^ (p-1))
                end
                if p >= 2
                    acc_seg .+= a * p * (p-1) .* (t_local .^ (p-2))
                end
            end
            # Sinusoids
            for (f, a, φ) in zip(freqs, amps, phases)
                ω = 2π*f
                θ = ω .* t_local .+ φ
                pos_seg .+= a .* sin.(θ)
                vel_seg .+= ω .* a .* cos.(θ)
                acc_seg .+= -(ω^2) .* a .* sin.(θ)
            end
            pos[idxs] .= pos_seg
            vel[idxs] .= vel_seg
            acc[idxs] .= acc_seg
        end
        x_true = [_toK_triple(pos[i], vel[i], acc[i], cfg.order) for i in 1:n]
        y = _observations_from_truth(rng, pos, vel, cfg.σ_obs_pos, cfg.σ_obs_vel)
        A, _, Qd = GCUtils.constant_acceleration_ABQ(cfg.dt; order=cfg.order, σ_a=cfg.σ_a)
        return x_true, y, A, Matrix(Qd)
    else
        error("Unknown generator: $(cfg.generator)")
    end
end

end # module


