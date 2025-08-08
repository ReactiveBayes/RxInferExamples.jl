module GCConfig

export ScenarioConfig, RunConfig, default_run_config

struct ScenarioConfig
    name::String
    n::Int
    dt::Float64
    σ_a::Float64
    σ_obs_pos::Float64
    σ_obs_vel::Float64 # use NaN for position-only
    generator::Symbol  # :constant_accel, :sinusoid, :sinusoid_mixed, :poly, :custom
    generator_kwargs::Dict{Symbol, Any}
end

struct RunConfig
    iterations::Int
    autostart::Bool
    free_energy::Bool
end

default_run_config() = RunConfig(100, true, true)

end # module


