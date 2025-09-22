# World module for the Active Inference Mountain Car example
# Handles the environmental simulation and state management

@doc """
World module for mountain car environment simulation.

This module manages the environmental state and provides functions to execute
actions and observe the resulting states. It maintains the internal state
of the environment and ensures proper state transitions.
"""
module World

import ..Config: WORLD, PHYSICS
import ..Physics: next_state

@doc """
Create a world/environment instance with specified physics functions.

Args:
- Fg: Gravitational force function
- Ff: Friction force function
- Fa: Engine force function
- initial_position: Initial car position
- initial_velocity: Initial car velocity

Returns:
- execute: Function to execute an action and update state
- observe: Function to observe current state
- reset: Function to reset to initial state
"""
function create_world(; Fg::Function, Ff::Function, Fa::Function,
                     initial_position::Float64 = WORLD.initial_position,
                     initial_velocity::Float64 = WORLD.initial_velocity)

    # Initialize state variables (private to this world instance)
    state = Dict(
        :position => initial_position,
        :velocity => initial_velocity,
        :position_prev => initial_position,
        :velocity_prev => initial_velocity
    )

    @doc """
    Execute an action in the world, updating the internal state.

    Args:
    - a_t: Action to execute (engine force input)
    """
    function execute(a_t::Float64)
        # Store previous state for potential rollback
        state[:position_prev] = state[:position]
        state[:velocity_prev] = state[:velocity]

        # Compute next state using physics
        new_state = next_state([state[:position], state[:velocity]], a_t, Fa, Ff, Fg)

        # Update current state
        state[:position] = new_state[1]
        state[:velocity] = new_state[2]
    end

    @doc """
    Observe the current state of the world.

    Returns:
    - Vector containing [position, velocity]
    """
    function observe()
        return [state[:position], state[:velocity]]
    end

    @doc """
    Reset the world to initial state.
    """
    function reset()
        state[:position] = initial_position
        state[:velocity] = initial_velocity
        state[:position_prev] = initial_position
        state[:velocity_prev] = initial_velocity
    end

    @doc """
    Get current state without modifying it.
    """
    function get_state()
        return copy([state[:position], state[:velocity]])
    end

    @doc """
    Set the state to a specific value (useful for testing).
    """
    function set_state(position::Float64, velocity::Float64)
        state[:position] = position
        state[:velocity] = velocity
        state[:position_prev] = position
        state[:velocity_prev] = velocity
    end

    return (execute, observe, reset, get_state, set_state)
end

@doc """
Simulate a sequence of actions and return the resulting states.

Args:
- actions: Vector of actions to execute
- initial_position: Starting position
- initial_velocity: Starting velocity
- Fa, Ff, Fg: Physics functions

Returns:
- Vector of states after each action
"""
function simulate_trajectory(actions::Vector{Float64},
                           initial_position::Float64, initial_velocity::Float64,
                           Fa::Function, Ff::Function, Fg::Function)
    # Create a world instance
    execute, observe, reset, get_state, set_state = create_world(
        Fg = Fg, Ff = Ff, Fa = Fa,
        initial_position = initial_position,
        initial_velocity = initial_velocity
    )

    # Simulate trajectory
    states = Vector{Vector{Float64}}(undef, length(actions) + 1)
    states[1] = get_state()  # Initial state

    for (t, action) in enumerate(actions)
        execute(action)
        states[t + 1] = get_state()
    end

    return states
end

# Export public functions
export create_world, simulate_trajectory

end # module World
