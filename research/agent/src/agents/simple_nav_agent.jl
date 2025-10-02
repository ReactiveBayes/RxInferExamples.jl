# Simple 1D Navigation Active Inference Agent
# Minimal agent for testing the interface

using RxInfer
using Distributions
using LinearAlgebra
import RxInfer.ReactiveMP: getrecent, messageout

include("../types.jl")
include("abstract_agent.jl")

using .Main: StateVector, ActionVector, ObservationVector

# ==================== RXINFER MODEL (TOP-LEVEL) ====================

# RxInfer generative model for simple 1D navigation Active Inference.
# Simple linear dynamics: s_t = s_{t-1} + u_t * dt
@model function simple_nav_model(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, dt, velocity_limit)
    
    # Transition function: position += velocity * dt
    g = (s_t_min::AbstractVector) -> s_t_min  # Position stays if no control
    
    # Control function: velocity effect on position
    h = (u::AbstractVector) -> [clamp(u[1], -velocity_limit, velocity_limit) * dt]
    
    # Inverse: from position change to velocity
    h_inv = (delta_s::AbstractVector) -> [delta_s[1] / dt]
    
    # Internal model parameters
    Gamma = 1e4 * diagm(ones(1))  # Transition precision
    Theta = 1e-4 * diagm(ones(1)) # Observation variance

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

# ==================== SIMPLE NAV AGENT ====================

"""
SimpleNavAgent <: AbstractActiveInferenceAgent{1,1,1}

Active Inference agent for simple 1D navigation.

Type Parameters:
- State dimension: 1 (position)
- Action dimension: 1 (velocity)
- Observation dimension: 1 (position)
"""
mutable struct SimpleNavAgent <: AbstractActiveInferenceAgent{1,1,1}
    # Planning horizon
    horizon::Int
    
    # Goal specification
    goal_state::StateVector{1}
    
    # Current state belief
    state_belief::Ref{Tuple{Vector{Float64}, Matrix{Float64}}}  # (mean, cov)
    
    # Control priors
    m_u::Vector{Vector{Float64}}
    V_u::Vector{Matrix{Float64}}
    
    # Goal priors
    m_x::Vector{Vector{Float64}}
    V_x::Vector{Matrix{Float64}}
    
    # Environment physics
    dt::Float64
    velocity_limit::Float64
    
    # Inference result storage
    result_ref::Ref{Union{Nothing, Any}}
    
    # Diagnostics
    step_count::Int
    total_inference_time::Float64
    
    function SimpleNavAgent(
        horizon::Int,
        goal_state::StateVector{1},
        initial_state::StateVector{1},
        env_params;  # From get_observation_model_params
        initial_state_precision::Float64 = 1e6
    )
        huge = 1e6
        tiny = 1 / initial_state_precision
        
        # Control priors
        Epsilon = fill(huge, 1, 1)
        m_u = Vector{Float64}[[0.0] for k=1:horizon]
        V_u = Matrix{Float64}[Epsilon for k=1:horizon]
        
        # Goal priors
        Sigma = 1e-4 * diagm(ones(1))
        x_target = Vector(goal_state)
        m_x = [zeros(1) for k=1:horizon]
        V_x = [huge * diagm(ones(1)) for k=1:horizon]
        V_x[end] = Sigma
        
        # Initial state belief
        state_belief = Ref((Vector(initial_state), tiny * diagm(ones(1))))
        
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
            env_params.dt,
            env_params.velocity_limit,
            result_ref,
            0,
            0.0
        )
    end
end

# ==================== AGENT INTERFACE IMPLEMENTATION ====================

function step!(agent::SimpleNavAgent, 
               observation::ObservationVector{1},
               action::ActionVector{1})
    start_time = time()
    agent.step_count += 1
    
    tiny = 1e-6
    
    # Clamp current action and observation
    agent.m_u[1] = [action[1]]
    agent.V_u[1] = fill(tiny, 1, 1)
    
    agent.m_x[1] = Vector(observation)
    agent.V_x[1] = tiny * diagm(ones(1))
    
    # Prepare data for inference
    data = Dict(
        :m_u => agent.m_u,
        :V_u => agent.V_u,
        :m_x => agent.m_x,
        :V_x => agent.V_x,
        :m_s_t_min => agent.state_belief[][1],
        :V_s_t_min => agent.state_belief[][2]
    )
    
    # Run RxInfer variational inference
    try
        model = simple_nav_model(
            T = agent.horizon,
            dt = agent.dt,
            velocity_limit = agent.velocity_limit
        )
        
        agent.result_ref[] = infer(model = model, data = data)
        
    catch e
        @warn "Inference failed" exception=e
    end
    
    # Update timing
    inference_time = time() - start_time
    agent.total_inference_time += inference_time
end

function get_action(agent::SimpleNavAgent)::ActionVector{1}
    if agent.result_ref[] === nothing
        return ActionVector{1}([0.0])
    end
    
    try
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

function get_predictions(agent::SimpleNavAgent)::Vector{StateVector{1}}
    if agent.result_ref[] === nothing
        return [StateVector{1}([0.0]) for _ in 1:agent.horizon]
    end
    
    try
        posteriors = agent.result_ref[].posteriors[:s]
        return [StateVector{1}(mode(p)) for p in posteriors]
    catch e
        @warn "Failed to extract predictions" exception=e
        return [StateVector{1}([0.0]) for _ in 1:agent.horizon]
    end
end

function slide!(agent::SimpleNavAgent)
    if agent.result_ref[] === nothing
        # Slide priors
        huge = 1e6
        agent.m_u = circshift(agent.m_u, -1)
        agent.m_u[end] = [0.0]
        agent.V_u = circshift(agent.V_u, -1)
        agent.V_u[end] = fill(huge, 1, 1)
        
        agent.m_x = circshift(agent.m_x, -1)
        agent.m_x[end] = Vector(agent.goal_state)
        agent.V_x = circshift(agent.V_x, -1)
        agent.V_x[end] = 1e-4 * diagm(ones(1))
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
        agent.V_x[end] = 1e-4 * diagm(ones(1))
        
    catch e
        @warn "Failed to slide planning window" exception=e
    end
end

function reset!(agent::SimpleNavAgent)
    tiny = 1e-6
    huge = 1e6
    
    # Reset state belief
    agent.state_belief[] = (zeros(1), tiny * diagm(ones(1)))
    
    # Reset priors
    agent.m_u = [[0.0] for _ in 1:agent.horizon]
    agent.V_u = [fill(huge, 1, 1) for _ in 1:agent.horizon]
    
    agent.m_x = [zeros(1) for _ in 1:agent.horizon]
    agent.V_x = [huge * diagm(ones(1)) for _ in 1:agent.horizon]
    agent.V_x[end] = 1e-4 * diagm(ones(1))
    agent.m_x[end] = Vector(agent.goal_state)
    
    # Reset inference
    agent.result_ref[] = nothing
    agent.step_count = 0
    agent.total_inference_time = 0.0
end

# Export
export SimpleNavAgent, simple_nav_model

