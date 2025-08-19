module InfiniteDataStreamEnv

using Random, StableRNGs, RxInfer

export Environment, getnext!, gethistory, getobservations

mutable struct Environment
    rng                   :: AbstractRNG
    current_state         :: Float64
    observation_precision :: Float64
    history               :: Vector{Float64}
    observations          :: Vector{Float64}
    
    function Environment(current_state::Float64, observation_precision::Float64; seed::Integer = 123)
        return new(StableRNG(seed), current_state, observation_precision, Float64[], Float64[])
    end
end

function getnext!(environment::Environment)
    environment.current_state = environment.current_state + 1.0
    nextstate  = 10sin(0.1 * environment.current_state)
    observation = rand(environment.rng, NormalMeanPrecision(nextstate, environment.observation_precision))
    push!(environment.history, nextstate)
    push!(environment.observations, observation)
    return observation
end

gethistory(environment::Environment) = environment.history
getobservations(environment::Environment) = environment.observations

end # module

