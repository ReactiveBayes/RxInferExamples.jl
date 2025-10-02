# Simple 1D Navigation Environment
# Minimal environment for testing the interface

using LinearAlgebra

include("../types.jl")
include("abstract_environment.jl")

using .Main: StateVector, ActionVector, ObservationVector

"""
SimpleNavEnv <: AbstractEnvironment{1,1,1}

Simple 1D navigation environment with 1D state (position), 1D action (velocity), 
and 1D observation (noisy position).

State: [position]
Action: [velocity]
Observation: [position] + noise

Physics:
- position_new = position + velocity * dt
"""
mutable struct SimpleNavEnv <: AbstractEnvironment{1,1,1}
    # Current state
    current_state::Ref{StateVector{1}}
    
    # Initial state
    initial_position::Float64
    
    # Goal
    goal_position::Float64
    
    # Physics parameters
    dt::Float64  # Time step
    velocity_limit::Float64  # Maximum velocity magnitude
    
    # Observation model parameters
    observation_precision::Float64
    observation_noise_std::Float64
    
    function SimpleNavEnv(;
        initial_position::Float64 = 0.0,
        goal_position::Float64 = 1.0,
        dt::Float64 = 0.1,
        velocity_limit::Float64 = 0.5,
        observation_precision::Float64 = 1e4,
        observation_noise_std::Float64 = 0.01
    )
        # Initialize state
        initial_state = StateVector{1}([initial_position])
        state_ref = Ref(initial_state)
        
        new(
            state_ref,
            initial_position,
            goal_position,
            dt,
            velocity_limit,
            observation_precision,
            observation_noise_std
        )
    end
end

"""
step!(env::SimpleNavEnv, action::ActionVector{1})::ObservationVector{1}

Execute action in simple navigation environment.

Physics:
- position_new = position + clamp(velocity, -limit, +limit) * dt
"""
function step!(env::SimpleNavEnv, action::ActionVector{1})::ObservationVector{1}
    current_state = env.current_state[]
    position = current_state[1]
    velocity = action[1]
    
    # Clamp velocity
    velocity_clamped = clamp(velocity, -env.velocity_limit, env.velocity_limit)
    
    # Simple Euler integration
    position_new = position + velocity_clamped * env.dt
    
    # Update internal state
    env.current_state[] = StateVector{1}([position_new])
    
    # Return observation (position + small noise)
    noise = env.observation_noise_std * randn()
    observation = ObservationVector{1}([position_new + noise])
    
    return observation
end

"""
reset!(env::SimpleNavEnv)::ObservationVector{1}

Reset environment to initial state.
"""
function reset!(env::SimpleNavEnv)::ObservationVector{1}
    initial_state = StateVector{1}([env.initial_position])
    env.current_state[] = initial_state
    
    # Return initial observation
    observation = ObservationVector{1}([env.initial_position])
    return observation
end

"""
get_state(env::SimpleNavEnv)::StateVector{1}

Get current internal state.
"""
function get_state(env::SimpleNavEnv)::StateVector{1}
    return env.current_state[]
end

"""
get_observation_model_params(env::SimpleNavEnv)

Get observation model parameters for agent's generative model.
"""
function get_observation_model_params(env::SimpleNavEnv)
    return (
        observation_precision = env.observation_precision,
        observation_noise_std = env.observation_noise_std,
        dt = env.dt,
        velocity_limit = env.velocity_limit,
        goal_position = env.goal_position
    )
end

# Export
export SimpleNavEnv

