# Mountain Car Active Inference Agent
# Uses real RxInfer variational inference for action selection

using RxInfer
using Distributions
using LinearAlgebra
import RxInfer.ReactiveMP: getrecent, messageout

include("../types.jl")
include("abstract_agent.jl")

using .Main: StateVector, ActionVector, ObservationVector

# ==================== RXINFER MODEL (TOP-LEVEL) ====================

# RxInfer generative model for mountain car Active Inference.
# This must be defined at top level (RxInfer v4+ requirement).
@model function mountain_car_model(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Fg, Fa, Ff, engine_force_limit)
    
    # Transition function modeling transition due to gravity and friction
    g = (s_t_min::AbstractVector) -> begin 
        s_t = similar(s_t_min) # Next state
        s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2]) # Update velocity
        s_t[1] = s_t_min[1] + s_t[2] # Update position
        return s_t
    end
    
    # Function for modeling engine control
    h = (u::AbstractVector) -> [0.0, Fa(u[1])] 
    
    # Inverse engine force, from change in state to corresponding engine force
    h_inv = (delta_s_dot::AbstractVector) -> [atanh(clamp(delta_s_dot[2], -engine_force_limit+1e-3, engine_force_limit-1e-3)/engine_force_limit)] 
    
    # Internal model parameters
    Gamma = 1e4*diageye(2) # Transition precision
    Theta = 1e-4*diageye(2) # Observation variance

    s_t_min ~ MvNormal(mean = m_s_t_min, cov = V_s_t_min)
    s_k_min = s_t_min

    local s
    
    for k in 1:T
        u[k] ~ MvNormal(mean = m_u[k], cov = V_u[k])
        u_h_k[k] ~ h(u[k]) where { meta = DeltaMeta(method = Linearization(), inverse = h_inv) }
        s_g_k[k] ~ g(s_k_min) where { meta = DeltaMeta(method = Linearization()) }
        u_s_sum[k] ~ s_g_k[k] + u_h_k[k]
        s[k] ~ MvNormal(mean = u_s_sum[k], precision = Gamma)
        x[k] ~ MvNormal(mean = s[k], cov = Theta)
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k]) # goal
        s_k_min = s[k]
    end
    
    return (s, )
end

# ==================== MOUNTAIN CAR AGENT ====================

"""
MountainCarAgent <: AbstractActiveInferenceAgent{2,1,2}

Active Inference agent for mountain car problem.

Uses RxInfer variational inference to minimize expected free energy,
planning actions that bring the car to the goal position.

Type Parameters:
- State dimension: 2 (position, velocity)
- Action dimension: 1 (force)
- Observation dimension: 2 (position, velocity)
"""
mutable struct MountainCarAgent <: AbstractActiveInferenceAgent{2,1,2}
    # Planning horizon
    horizon::Int
    
    # Goal specification
    goal_state::StateVector{2}
    
    # Current state belief
    state_belief::Ref{Tuple{Vector{Float64}, Matrix{Float64}}}  # (mean, cov)
    
    # Control priors (action beliefs over horizon)
    m_u::Vector{Vector{Float64}}
    V_u::Vector{Matrix{Float64}}
    
    # Goal priors (desired states over horizon)
    m_x::Vector{Vector{Float64}}
    V_x::Vector{Matrix{Float64}}
    
    # Environment physics (from observation model params)
    Fg::Function
    Fa::Function
    Ff::Function
    engine_force_limit::Float64
    
    # Inference result storage
    result_ref::Ref{Union{Nothing, Any}}
    
    # Diagnostics
    step_count::Int
    total_inference_time::Float64
    
    function MountainCarAgent(
        horizon::Int,
        goal_state::StateVector{2},
        initial_state::StateVector{2},
        env_params;  # From get_observation_model_params
        initial_state_precision::Float64 = 1e6
    )
        huge = 1e6
        tiny = 1 / initial_state_precision
        
        # Control priors
        Epsilon = fill(huge, 1, 1)  # Control prior variance
        m_u = Vector{Float64}[[0.0] for k=1:horizon]
        V_u = Matrix{Float64}[Epsilon for k=1:horizon]
        
        # Goal priors
        Sigma = 1e-4*diagm(ones(2))  # Goal prior variance
        x_target = Vector(goal_state)
        m_x = [zeros(2) for k=1:horizon]
        V_x = [huge*diagm(ones(2)) for k=1:horizon]
        V_x[end] = Sigma  # Set prior to reach goal at t=horizon
        
        # Initial state belief
        state_belief = Ref((Vector(initial_state), tiny * diagm(ones(2))))
        
        # Inference result storage
        result_ref = Ref{Union{Nothing, Any}}(nothing)
        
        new(
            horizon,
            goal_state,
            state_belief,
            m_u,
            V_u,
            m_x,
            V_x,
            env_params.Fg,
            env_params.Fa,
            env_params.Ff,
            env_params.engine_force_limit,
            result_ref,
            0,
            0.0
        )
    end
end

# ==================== AGENT INTERFACE IMPLEMENTATION ====================

"""
step!(agent::MountainCarAgent, observation::ObservationVector{2}, action::ActionVector{1})

Perform one step of Active Inference.
"""
function step!(agent::MountainCarAgent, 
               observation::ObservationVector{2},
               action::ActionVector{1})
    start_time = time()
    agent.step_count += 1
    
    tiny = 1e-6
    
    # Clamp current action and observation
    agent.m_u[1] = [action[1]]
    agent.V_u[1] = fill(tiny, 1, 1)
    
    agent.m_x[1] = Vector(observation)
    agent.V_x[1] = tiny * diagm(ones(2))
    
    # Prepare data for inference
    data = Dict(
        :m_u => agent.m_u,
        :V_u => agent.V_u,
        :m_x => agent.m_x,
        :V_x => agent.V_x,
        :m_s_t_min => agent.state_belief[][1],  # Get mean vector from tuple
        :V_s_t_min => agent.state_belief[][2]   # Get covariance matrix from tuple
    )
    
    # Run RxInfer variational inference
    try
        model = mountain_car_model(
            T = agent.horizon,
            Fg = agent.Fg,
            Fa = agent.Fa,
            Ff = agent.Ff,
            engine_force_limit = agent.engine_force_limit
        )
        
        agent.result_ref[] = infer(model = model, data = data)
        
    catch e
        @warn "Inference failed" exception=e
    end
    
    # Update timing
    inference_time = time() - start_time
    agent.total_inference_time += inference_time
end

"""
get_action(agent::MountainCarAgent)::ActionVector{1}

Get next action from inference result.
"""
function get_action(agent::MountainCarAgent)::ActionVector{1}
    if agent.result_ref[] === nothing
        return ActionVector{1}([0.0])
    end
    
    try
        # Get second control posterior (first future action)
        posteriors = agent.result_ref[].posteriors[:u]
        if length(posteriors) < 2
            return ActionVector{1}([0.0])
        end
        
        action_val = mode(posteriors[2])[1]
        return ActionVector{1}([action_val])
    catch e
        @warn "Failed to extract action" exception=e
        return ActionVector{1}([0.0])
    end
end

"""
get_predictions(agent::MountainCarAgent)::Vector{StateVector{2}}

Get predicted future states.
"""
function get_predictions(agent::MountainCarAgent)::Vector{StateVector{2}}
    if agent.result_ref[] === nothing
        return [StateVector{2}([0.0, 0.0]) for _ in 1:agent.horizon]
    end
    
    try
        posteriors = agent.result_ref[].posteriors[:s]
        return [StateVector{2}(mode(p)) for p in posteriors]
    catch e
        @warn "Failed to extract predictions" exception=e
        return [StateVector{2}([0.0, 0.0]) for _ in 1:agent.horizon]
    end
end

"""
slide!(agent::MountainCarAgent)

Slide planning horizon forward.
"""
function slide!(agent::MountainCarAgent)
    if agent.result_ref[] === nothing
        # Slide priors for placeholder
        huge = 1e6
        agent.m_u = circshift(agent.m_u, -1)
        agent.m_u[end] = [0.0]
        agent.V_u = circshift(agent.V_u, -1)
        agent.V_u[end] = fill(huge, 1, 1)
        
        agent.m_x = circshift(agent.m_x, -1)
        agent.m_x[end] = Vector(agent.goal_state)
        agent.V_x = circshift(agent.V_x, -1)
        agent.V_x[end] = 1e-4 * diagm(ones(2))
        return
    end
    
    try
        # Extract updated state belief
        model_obj = RxInfer.getmodel(agent.result_ref[].model)
        (s,) = RxInfer.getreturnval(model_obj)
        varref = RxInfer.getvarref(model_obj, s)
        var = RxInfer.getvariable(varref)
        
        slide_msg_idx = 3
        (m_new, V_new) = mean_cov(getrecent(messageout(var[2], slide_msg_idx)))
        
        agent.state_belief[] = (m_new, V_new)
        
        # Slide control priors
        huge = 1e6
        agent.m_u = circshift(agent.m_u, -1)
        agent.m_u[end] = [0.0]
        agent.V_u = circshift(agent.V_u, -1)
        agent.V_u[end] = fill(huge, 1, 1)
        
        # Slide goal priors
        agent.m_x = circshift(agent.m_x, -1)
        agent.m_x[end] = Vector(agent.goal_state)
        agent.V_x = circshift(agent.V_x, -1)
        agent.V_x[end] = 1e-4 * diagm(ones(2))
        
    catch e
        @warn "Failed to slide planning window" exception=e
    end
end

"""
reset!(agent::MountainCarAgent)

Reset agent to initial state.
"""
function reset!(agent::MountainCarAgent)
    tiny = 1e-6
    huge = 1e6
    
    # Reset state belief
    agent.state_belief[] = (zeros(2), tiny * diagm(ones(2)))
    
    # Reset priors
    agent.m_u = [[0.0] for _ in 1:agent.horizon]
    agent.V_u = [fill(huge, 1, 1) for _ in 1:agent.horizon]
    
    agent.m_x = [zeros(2) for _ in 1:agent.horizon]
    agent.V_x = [huge * diagm(ones(2)) for _ in 1:agent.horizon]
    agent.V_x[end] = 1e-4 * diagm(ones(2))
    agent.m_x[end] = Vector(agent.goal_state)
    
    # Reset inference
    agent.result_ref[] = nothing
    agent.step_count = 0
    agent.total_inference_time = 0.0
end

# Export
export MountainCarAgent, mountain_car_model

