# Abstract Agent Interface
# Defines the contract that all Active Inference agents must implement

include("../types.jl")
using .Main: StateVector, ActionVector, ObservationVector

"""
AbstractActiveInferenceAgent{S,A,O}

Abstract type for all Active Inference agents in the framework.

Type Parameters:
- S: State dimension
- A: Action dimension
- O: Observation dimension

Required Methods:
- `step!(agent, observation, action)`: Update beliefs given observation and action
- `get_action(agent)`: Get the next action to execute
- `get_predictions(agent)`: Get predicted future states
- `slide!(agent)`: Slide the planning horizon forward
- `reset!(agent)`: Reset agent to initial state
"""
abstract type AbstractActiveInferenceAgent{S,A,O} end

"""
step!(agent::AbstractActiveInferenceAgent{S,A,O}, 
      observation::ObservationVector{O},
      action::ActionVector{A})

Perform one step of Active Inference given an observation and the action that was taken.

This performs variational inference to update beliefs about states and optimal future actions.

Args:
- agent: The Active Inference agent
- observation: Observation received from environment
- action: Action that was executed
"""
function step!(agent::AbstractActiveInferenceAgent{S,A,O},
               observation::ObservationVector{O},
               action::ActionVector{A}) where {S,A,O}
    error("step! not implemented for $(typeof(agent))")
end

"""
get_action(agent::AbstractActiveInferenceAgent{S,A,O})::ActionVector{A}

Get the next action to execute based on current beliefs.

Args:
- agent: The Active Inference agent

Returns:
- action: Next action to take
"""
function get_action(agent::AbstractActiveInferenceAgent{S,A,O})::ActionVector{A} where {S,A,O}
    error("get_action not implemented for $(typeof(agent))")
end

"""
get_predictions(agent::AbstractActiveInferenceAgent{S,A,O})::Vector{StateVector{S}}

Get predicted future states over the planning horizon.

Args:
- agent: The Active Inference agent

Returns:
- predictions: Vector of predicted states for each future timestep
"""
function get_predictions(agent::AbstractActiveInferenceAgent{S,A,O})::Vector{StateVector{S}} where {S,A,O}
    error("get_predictions not implemented for $(typeof(agent))")
end

"""
slide!(agent::AbstractActiveInferenceAgent{S,A,O})

Slide the planning horizon forward by one timestep.

This updates the agent's internal state for the next timestep by shifting
control and goal priors forward and extracting updated state beliefs.

Args:
- agent: The Active Inference agent
"""
function slide!(agent::AbstractActiveInferenceAgent{S,A,O}) where {S,A,O}
    error("slide! not implemented for $(typeof(agent))")
end

"""
reset!(agent::AbstractActiveInferenceAgent{S,A,O})

Reset the agent to its initial state.

Args:
- agent: The Active Inference agent
"""
function reset!(agent::AbstractActiveInferenceAgent{S,A,O}) where {S,A,O}
    error("reset! not implemented for $(typeof(agent))")
end

# Export
export AbstractActiveInferenceAgent, step!, get_action, get_predictions, slide!, reset!

