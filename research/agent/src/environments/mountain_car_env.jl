# Mountain Car Environment
# Classic reinforcement learning benchmark implemented as an environment

using HypergeometricFunctions: _₂F₁
using LinearAlgebra

include("../types.jl")
include("abstract_environment.jl")

using .Main: StateVector, ActionVector, ObservationVector

"""
MountainCarEnv <: AbstractEnvironment{2,1,2}

Mountain Car environment with 2D state (position, velocity), 1D action (force), 
and 2D observations (noisy position and velocity).

The car must build momentum by oscillating to escape a valley.

State: [position, velocity]
Action: [force] ∈ [-engine_force_limit, engine_force_limit]
Observation: [position, velocity] + noise

Physics:
- Gravity: Fg(position) - pulls car down the hills
- Friction: Ff(velocity) - opposes motion
- Engine: Fa(action) - controlled force
"""
mutable struct MountainCarEnv <: AbstractEnvironment{2,1,2}
    # Current state
    current_state::Ref{StateVector{2}}
    
    # Initial state
    initial_position::Float64
    initial_velocity::Float64
    
    # Physics parameters
    engine_force_limit::Float64
    friction_coefficient::Float64
    
    # Observation model parameters
    observation_precision::Float64
    observation_noise_std::Float64
    
    # Physics functions (stored as closures)
    Fa::Function  # Engine force
    Ff::Function  # Friction force
    Fg::Function  # Gravity force
    height::Function  # Height function for visualization
    
    function MountainCarEnv(;
        initial_position::Float64 = -0.5,
        initial_velocity::Float64 = 0.0,
        engine_force_limit::Float64 = 0.04,
        friction_coefficient::Float64 = 0.1,
        observation_precision::Float64 = 1e4,
        observation_noise_std::Float64 = 0.01
    )
        # Engine force function
        Fa = (a::Real) -> engine_force_limit * tanh(a)
        
        # Friction force function
        Ff = (y_dot::Real) -> -friction_coefficient * y_dot
        
        # Gravitational force function
        Fg = (y::Real) -> begin
            if y < 0
                return -0.05*(2*y + 1)
            else
                return -0.05*((1 + 5*y^2)^(-0.5) + (y^2)*(1 + 5*y^2)^(-3/2) + (y^4)/16)
            end
        end
        
        # Height function for visualization
        height = (x::Float64) -> begin
            if x < 0
                h = x^2 + x
            else
                h = x * _₂F₁(0.5, 0.5, 1.5, -5*x^2) +
                    x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 +
                    x^5 / 80
            end
            return 0.05*h
        end
        
        # Initialize state
        initial_state = StateVector{2}([initial_position, initial_velocity])
        state_ref = Ref(initial_state)
        
        new(
            state_ref,
            initial_position,
            initial_velocity,
            engine_force_limit,
            friction_coefficient,
            observation_precision,
            observation_noise_std,
            Fa, Ff, Fg, height
        )
    end
end

"""
step!(env::MountainCarEnv, action::ActionVector{1})::ObservationVector{2}

Execute action in mountain car environment.

Physics:
- v_new = v + Fg(x) + Ff(v) + Fa(action)
- x_new = x + v_new
"""
function step!(env::MountainCarEnv, action::ActionVector{1})::ObservationVector{2}
    current_state = env.current_state[]
    x, v = current_state[1], current_state[2]
    a = action[1]
    
    # Compute next state using physics
    v_new = v + env.Fg(x) + env.Ff(v) + env.Fa(a)
    x_new = x + v_new
    
    # Update internal state
    env.current_state[] = StateVector{2}([x_new, v_new])
    
    # Return observation (state + small noise for realism)
    noise = env.observation_noise_std * randn(2)
    observation = ObservationVector{2}([x_new + noise[1], v_new + noise[2]])
    
    return observation
end

"""
reset!(env::MountainCarEnv)::ObservationVector{2}

Reset environment to initial state.
"""
function reset!(env::MountainCarEnv)::ObservationVector{2}
    initial_state = StateVector{2}([env.initial_position, env.initial_velocity])
    env.current_state[] = initial_state
    
    # Return initial observation
    observation = ObservationVector{2}([env.initial_position, env.initial_velocity])
    return observation
end

"""
get_state(env::MountainCarEnv)::StateVector{2}

Get current internal state.
"""
function get_state(env::MountainCarEnv)::StateVector{2}
    return env.current_state[]
end

"""
get_observation_model_params(env::MountainCarEnv)

Get observation model parameters for agent's generative model.
"""
function get_observation_model_params(env::MountainCarEnv)
    return (
        observation_precision = env.observation_precision,
        observation_noise_std = env.observation_noise_std,
        Fa = env.Fa,
        Ff = env.Ff,
        Fg = env.Fg,
        engine_force_limit = env.engine_force_limit
    )
end

# Export
export MountainCarEnv

