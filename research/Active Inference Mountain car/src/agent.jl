# Agent module for the Active Inference Mountain Car example
# Implements the active inference agent with probabilistic planning

@doc """
Agent module for active inference in mountain car environment.

This module implements an active inference agent that uses probabilistic planning
to determine optimal actions. The agent maintains beliefs about the environment
and plans multiple steps ahead to achieve goals.
"""
module Agent

using RxInfer
import RxInfer.ReactiveMP: getrecent, messageout
import LinearAlgebra: I, Diagonal
import ..Config: AGENT, SIMULATION, TARGET, WORLD, PHYSICS
import ..Physics: next_state

@doc """
Create an active inference agent for the mountain car problem.

Args:
- T: Planning horizon (number of steps to plan ahead)
- Fg, Fa, Ff: Physics functions
- engine_force_limit: Maximum engine force
- x_target: Target state [position, velocity]
- initial_position: Initial car position
- initial_velocity: Initial car velocity

Returns:
- compute: Function to perform inference given observation and action
- act: Function to get the next action
- slide: Function to update agent state for next timestep
- future: Function to get predicted future states
- reset: Function to reset agent to initial state
"""
function create_agent(; T::Int = SIMULATION.planning_horizon,
                      Fg::Function, Fa::Function, Ff::Function,
                      engine_force_limit::Float64 = PHYSICS.engine_force_limit,
                      x_target::Vector{Float64} = [TARGET.position, TARGET.velocity],
                      initial_position::Float64 = WORLD.initial_position,
                      initial_velocity::Float64 = WORLD.initial_velocity)

    # Numerical constants
    huge = AGENT.control_prior_variance
    tiny = AGENT.initial_state_variance

    # Initialize control priors (what actions the agent thinks it might take)
    m_u = Vector{Float64}[ [0.0] for k=1:T ]  # Mean of control prior
    V_u = Matrix{Float64}[ fill(huge, 1, 1) for k=1:T ]  # Variance of control prior

    # Initialize goal priors (where the agent wants to be)
    Sigma = AGENT.goal_prior_variance * I(2)  # Goal precision matrix
    m_x = [zeros(2) for k=1:T]  # Mean of goal prior
    V_x = [huge * I(2) for k=1:T]  # Variance of goal prior
    V_x[end] = Sigma  # Set precise goal for final timestep

    # Initialize state belief (current belief about car state)
    m_s_t_min = [initial_position, initial_velocity]
    V_s_t_min = tiny * I(2)

    # Store inference result
    result = nothing

    @doc """
    Perform inference given current action and observation.

    Args:
    - upsilon_t: Current action taken
    - y_hat_t: Current observation [position, velocity]
    """
    function compute(upsilon_t::Float64, y_hat_t::Vector{Float64})
        # Register current action with the generative model
        m_u[1] = [upsilon_t]
        V_u[1] = fill(tiny, 1, 1)  # Clamp to the action we actually took

        # Register observation with the generative model
        m_x[1] = y_hat_t
        V_x[1] = tiny * I(2)  # Clamp to the observation we actually saw

        # Prepare data for inference
        data = (
            m_u = m_u,
            V_u = V_u,
            m_x = m_x,
            V_x = V_x,
            m_s_t_min = m_s_t_min,
            V_s_t_min = V_s_t_min
        )

        # Define and run inference
        model = mountain_car_model(T = T, Fg = Fg, Fa = Fa, Ff = Ff, engine_force_limit = engine_force_limit)
        result = infer(model = model, data = data)
    end

    @doc """
    Get the next action to take based on current beliefs.
    """
    function act()
        if result !== nothing
            # Return the mode of the first future control posterior
            return mode(result.posteriors[:u][2])[1]
        else
            return 0.0  # Default action when no inference result available
        end
    end

    @doc """
    Get predicted future states.
    """
    function future()
        if result !== nothing
            return getindex.(mode.(result.posteriors[:s]), 1)
        else
            return zeros(T)
        end
    end

    @doc """
    Slide time window: update beliefs and shift planning horizon.
    """
    function slide()
        if result === nothing
            return
        end

        # Extract updated state belief from inference result
        model = RxInfer.getmodel(result.model)
        (s,) = RxInfer.getreturnval(model)
        varref = RxInfer.getvarref(model, s)
        var = RxInfer.getvariable(varref)

        # Get the updated state belief (this indexing might need adjustment based on model)
        slide_msg_idx = 3  # Model-dependent indexing
        (m_s_t_min, V_s_t_min) = mean_cov(getrecent(messageout(var[2], slide_msg_idx)))

        # Shift time window: move planning horizon one step forward
        m_u = circshift(m_u, -1)
        m_u[end] = [0.0]  # Default action for new time step
        V_u = circshift(V_u, -1)
        V_u[end] = fill(huge, 1, 1)  # Reset variance for new time step

        m_x = circshift(m_x, -1)
        m_x[end] = x_target  # Set goal for new final time step
        V_x = circshift(V_x, -1)
        V_x[end] = Sigma  # Reset goal precision for new final time step
    end

    @doc """
    Reset agent to initial state.
    """
    function reset()
        # Reset all state variables to initial values
        m_u = Vector{Float64}[ [0.0] for k=1:T ]
        V_u = Matrix{Float64}[ fill(huge, 1, 1) for k=1:T ]

        m_x = [zeros(2) for k=1:T]
        V_x = [huge * I(2) for k=1:T]
        V_x[end] = AGENT.goal_prior_variance * I(2)

        m_s_t_min = [initial_position, initial_velocity]
        V_s_t_min = tiny * I(2)

        result = nothing
    end

    return (compute, act, slide, future, reset)
end

@model function mountain_car_model(T, Fg, Fa, Ff, engine_force_limit, m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min)
"""
Generative model for the mountain car active inference agent.

This model encodes the agent's beliefs about how the environment works and
how actions influence future states. It uses nonlinear state transitions
with linearization for inference.
"""

    # Transition function modeling transition due to gravity and friction
    g = (s_t_min::AbstractVector) -> begin
        s_t = similar(s_t_min)  # Next state
        s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2])  # Update velocity
        s_t[1] = s_t_min[1] + s_t[2]  # Update position
        return s_t
    end

    # Function for modeling engine control
    h = (u::AbstractVector) -> [0.0, Fa(u[1])]

    # Inverse engine force (for linearization)
    h_inv = (delta_s_dot::AbstractVector) -> [
        atanh(clamp(delta_s_dot[2], -engine_force_limit + 1e-3, engine_force_limit - 1e-3) / engine_force_limit)
    ]

    # Internal model parameters
    Gamma = AGENT.transition_precision * I(2)  # Transition precision
    Theta = AGENT.observation_variance * I(2)  # Observation variance

    # Initial state prior
    s_t_min ~ MvNormal(mean = m_s_t_min, cov = V_s_t_min)
    s_k_min = s_t_min

    local s

    # Unroll the planning horizon
    for k in 1:T
        u[k] ~ MvNormal(mean = m_u[k], cov = V_u[k])
        u_h_k[k] ~ h(u[k]) where { meta = DeltaMeta(method = Linearization(), inverse = h_inv) }
        s_g_k[k] ~ g(s_k_min) where { meta = DeltaMeta(method = Linearization()) }
        u_s_sum[k] ~ s_g_k[k] + u_h_k[k]
        s[k] ~ MvNormal(mean = u_s_sum[k], precision = Gamma)
        x[k] ~ MvNormal(mean = s[k], cov = Theta)
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k])  # Goal prior
        s_k_min = s[k]
    end

    return (s,)
end

# Export public functions
export create_agent, mountain_car_model

end # module Agent
