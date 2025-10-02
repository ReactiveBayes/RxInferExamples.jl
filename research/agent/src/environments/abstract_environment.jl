# Abstract Environment Interface
# Defines the contract that all environments must implement

include("../types.jl")
using .Main: StateVector, ActionVector, ObservationVector

"""
AbstractEnvironment{S,A,O}

Abstract type for all environments in the agent-environment framework.

Type Parameters:
- S: State dimension
- A: Action dimension  
- O: Observation dimension

Required Methods:
- `step!(env, action)`: Execute action and return observation
- `reset!(env)`: Reset environment to initial state
- `get_state(env)`: Get current internal state
- `get_observation_model_params(env)`: Get observation model parameters for agent
"""
abstract type AbstractEnvironment{S,A,O} end

"""
step!(env::AbstractEnvironment{S,A,O}, action::ActionVector{A})::ObservationVector{O}

Execute an action in the environment and return the resulting observation.

Args:
- env: The environment
- action: Action to execute

Returns:
- observation: Observed outcome of the action
"""
function step!(env::AbstractEnvironment{S,A,O}, action::ActionVector{A})::ObservationVector{O} where {S,A,O}
    error("step! not implemented for $(typeof(env))")
end

"""
reset!(env::AbstractEnvironment{S,A,O})::ObservationVector{O}

Reset the environment to its initial state.

Args:
- env: The environment

Returns:
- observation: Initial observation
"""
function reset!(env::AbstractEnvironment{S,A,O})::ObservationVector{O} where {S,A,O}
    error("reset! not implemented for $(typeof(env))")
end

"""
get_state(env::AbstractEnvironment{S,A,O})::StateVector{S}

Get the current internal state of the environment.

Args:
- env: The environment

Returns:
- state: Current state vector
"""
function get_state(env::AbstractEnvironment{S,A,O})::StateVector{S} where {S,A,O}
    error("get_state not implemented for $(typeof(env))")
end

"""
get_observation_model_params(env::AbstractEnvironment{S,A,O})

Get observation model parameters that agents can use in their generative models.

Args:
- env: The environment

Returns:
- Named tuple with observation model parameters (precision, noise covariance, etc.)
"""
function get_observation_model_params(env::AbstractEnvironment{S,A,O}) where {S,A,O}
    error("get_observation_model_params not implemented for $(typeof(env))")
end

# Export
export AbstractEnvironment, step!, reset!, get_state, get_observation_model_params

