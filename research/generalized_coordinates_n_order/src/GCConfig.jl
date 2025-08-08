module GCConfig

export ScenarioConfig, RunConfig, default_run_config

struct ScenarioConfig
    name::String
    n::Int
    dt::Float64
    order::Int
    σ_a::Float64
    σ_obs_pos::Float64
    σ_obs_vel::Float64 # use NaN for position-only
    generator::Symbol  # :constant_accel, :sinusoid, :sinusoid_mixed, :poly, :custom
    generator_kwargs::Dict{Symbol, Any}
end

ScenarioConfig(name::String, n::Int, dt::Real, σ_a::Real, σ_obs_pos::Real, σ_obs_vel::Real,
               generator::Symbol, generator_kwargs::Dict{Symbol,Any}) =
    ScenarioConfig(name, n, Float64(dt), 6, Float64(σ_a), Float64(σ_obs_pos), Float64(σ_obs_vel), generator, generator_kwargs)

# Convenience constructor accepting any dict-like kwargs (handles Dict{Symbol,Float64} etc.)
ScenarioConfig(name::String, n::Int, dt::Real, σ_a::Real, σ_obs_pos::Real, σ_obs_vel::Real,
               generator::Symbol, generator_kwargs::AbstractDict{Symbol,<:Any}) =
    ScenarioConfig(name, n, Float64(dt), 6, Float64(σ_a), Float64(σ_obs_pos), Float64(σ_obs_vel), generator, Dict{Symbol,Any}(generator_kwargs))

struct RunConfig
    iterations::Int
    autostart::Bool
    free_energy::Bool
end

default_run_config() = RunConfig(100, true, true)

end # module


