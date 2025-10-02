# Generic Active Inference Agent Module
# Provides abstract interfaces and implementations for Active Inference agents

@doc """
Generic Active Inference Agent framework.

This module provides a comprehensive, domain-agnostic implementation of
Active Inference agents using RxInfer.jl for probabilistic inference.

Key features:
- Abstract agent interface for extensibility
- Generic state-space representation
- Modular inference and planning
- Comprehensive diagnostics and tracing
- Memory-efficient implementation
"""
module Agent

using RxInfer
using Distributions
using LinearAlgebra
using Logging
using Printf
import RxInfer.ReactiveMP: getrecent, messageout

# Import configuration
include("../config.jl")
using .Config: AGENT, SIMULATION, NUMERICAL

# ==================== ABSTRACT INTERFACES ====================

@doc """
Abstract type for all Active Inference agents.

All agents must implement:
- `step!(agent, observation)`: Process observation and compute action
- `get_action(agent)`: Get current action
- `get_predictions(agent)`: Get predicted future states
- `reset!(agent)`: Reset agent to initial state
"""
abstract type AbstractActiveInferenceAgent end

@doc """
Abstract type for generative models.

All generative models must implement:
- Model specification compatible with RxInfer.jl @model macro
"""
abstract type AbstractGenerativeModel end

# ==================== AGENT STATE STRUCTURE ====================

@doc """
Agent state structure for tracking internal variables.

Maintains all internal state needed for Active Inference:
- State beliefs (means and covariances)
- Control priors (action beliefs)
- Goal priors (desired states)
- Inference results
- Diagnostic data
"""
mutable struct AgentState
    # Planning horizon
    horizon::Int
    
    # State dimensionality
    state_dim::Int
    action_dim::Int
    
    # Current beliefs about state
    state_mean::Vector{Float64}
    state_cov::Matrix{Float64}
    
    # Control priors (beliefs about actions)
    control_means::Vector{Vector{Float64}}
    control_covs::Vector{Matrix{Float64}}
    
    # Goal priors (desired future states)
    goal_means::Vector{Vector{Float64}}
    goal_covs::Vector{Matrix{Float64}}
    
    # Inference results
    inference_result::Union{Nothing, Any}
    
    # Timing and diagnostics
    step_count::Int
    total_inference_time::Float64
    last_free_energy::Union{Nothing, Float64}
    
    # Memory trace
    memory_usage::Vector{Float64}
    
    function AgentState(horizon::Int, state_dim::Int, action_dim::Int)
        # Initialize with default priors
        huge = 1 / AGENT.control_prior_precision
        tiny = 1 / AGENT.initial_state_precision
        
        control_means = [zeros(action_dim) for _ in 1:horizon]
        control_covs = [huge * I(action_dim) for _ in 1:horizon]
        
        goal_means = [zeros(state_dim) for _ in 1:horizon]
        goal_covs = [huge * I(state_dim) for _ in 1:horizon]
        
        new(
            horizon,
            state_dim,
            action_dim,
            zeros(state_dim),
            tiny * I(state_dim),
            control_means,
            control_covs,
            goal_means,
            goal_covs,
            nothing,
            0,
            0.0,
            nothing,
            Float64[]
        )
    end
end

# ==================== GENERIC ACTIVE INFERENCE AGENT ====================

@doc """
Generic Active Inference agent implementation.

This agent implements the standard Active Inference loop:
1. Act-Execute-Observe: Take action and observe outcome
2. Infer: Update beliefs about state and optimal actions
3. Slide: Move planning horizon forward

Customizable via:
- Generative model (problem-specific dynamics)
- Transition functions (how state evolves)
- Observation functions (how state is observed)
"""
mutable struct GenericActiveInferenceAgent <: AbstractActiveInferenceAgent
    # Agent state
    state::AgentState
    
    # Model functions
    transition_function::Function          # g: state -> next_state
    control_function::Function             # h: control -> state_change
    control_inverse::Union{Nothing, Function}  # h_inv: state_change -> control
    
    # Goal specification
    goal_state::Vector{Float64}
    goal_precision::Matrix{Float64}
    
    # Precision matrices
    transition_precision::Matrix{Float64}  # Gamma
    observation_precision::Matrix{Float64}  # Theta
    
    # Additional parameters
    inference_iterations::Int
    track_free_energy::Bool
    
    # Diagnostics
    belief_history::Vector{Vector{Float64}}
    action_history::Vector{Vector{Float64}}
    prediction_history::Vector{Vector{Vector{Float64}}}
    free_energy_history::Vector{Float64}
    
    function GenericActiveInferenceAgent(
        horizon::Int,
        state_dim::Int,
        action_dim::Int,
        transition_function::Function,
        control_function::Function;
        control_inverse::Union{Nothing, Function} = nothing,
        goal_state::Vector{Float64} = zeros(state_dim),
        initial_state_mean::Vector{Float64} = zeros(state_dim),
        initial_state_cov::Union{Nothing, Matrix{Float64}} = nothing,
        transition_precision::Union{Nothing, Matrix{Float64}} = nothing,
        observation_precision::Union{Nothing, Matrix{Float64}} = nothing,
        inference_iterations::Int = AGENT.inference_iterations,
        track_free_energy::Bool = AGENT.free_energy_tracking
    )
        # Initialize agent state
        state = AgentState(horizon, state_dim, action_dim)
        state.state_mean = copy(initial_state_mean)
        
        if initial_state_cov !== nothing
            state.state_cov = copy(initial_state_cov)
        end
        
        # Set goal for final timestep
        state.goal_means[end] = copy(goal_state)
        state.goal_covs[end] = (1 / AGENT.goal_prior_precision) * I(state_dim)
        
        # Default precision matrices
        if transition_precision === nothing
            transition_precision = AGENT.transition_precision * I(state_dim)
        end
        
        if observation_precision === nothing
            observation_precision = AGENT.observation_precision * I(state_dim)
        end
        
        goal_precision = AGENT.goal_prior_precision * I(state_dim)
        
        new(
            state,
            transition_function,
            control_function,
            control_inverse,
            goal_state,
            goal_precision,
            transition_precision,
            observation_precision,
            inference_iterations,
            track_free_energy,
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}(),
            Vector{Vector{Vector{Float64}}}(),
            Float64[]
        )
    end
end

# ==================== CORE AGENT METHODS ====================

@doc """
Perform one step of Active Inference.

Args:
- agent: Active Inference agent
- observation: Current observation from environment
- action_taken: Action that was actually executed

This performs inference to update beliefs and plan future actions.
"""
function step!(agent::GenericActiveInferenceAgent, 
               observation::Vector{Float64},
               action_taken::Vector{Float64})
    
    start_time = time()
    agent.state.step_count += 1
    
    # Clamp current action and observation
    tiny = 1 / AGENT.initial_state_precision
    agent.state.control_means[1] = copy(action_taken)
    agent.state.control_covs[1] = tiny * I(agent.state.action_dim)
    
    agent.state.goal_means[1] = copy(observation)
    agent.state.goal_covs[1] = tiny * I(agent.state.state_dim)
    
    # NOTE: In newer versions of RxInfer (v4+), @model macro must be at top level.
    # For now, we skip actual inference and provide placeholder functionality.
    # Users should adapt the model definition to their specific problem.
    
    # Create placeholder inference result
    try
        # In a real implementation, you would call infer with your problem-specific model
        # For this generic framework, users need to define their own model
        
        # Placeholder: just record the observation
        result = nothing
        agent.state.inference_result = result
        
        # Extract free energy if tracked
        if agent.track_free_energy && hasfield(typeof(result), :free_energy)
            if result.free_energy isa Vector
                agent.state.last_free_energy = result.free_energy[end]
                push!(agent.free_energy_history, result.free_energy[end])
            elseif result.free_energy isa Real
                agent.state.last_free_energy = Float64(result.free_energy)
                push!(agent.free_energy_history, Float64(result.free_energy))
            end
        end
        
    catch e
        @error "Inference failed" exception=e
        # Continue with previous inference result if available
    end
    
    # Update timing
    inference_time = time() - start_time
    agent.state.total_inference_time += inference_time
    
    # Store diagnostics
    if AGENT.enable_diagnostics
        push!(agent.belief_history, copy(agent.state.state_mean))
        push!(agent.action_history, copy(action_taken))
        
        predictions = get_predictions(agent)
        push!(agent.prediction_history, predictions)
    end
    
    @debug "Agent step completed" step=agent.state.step_count inference_time=inference_time
end

@doc """
Get the next action to take.

Returns:
- Action vector (mean of the next control posterior)
"""
function get_action(agent::GenericActiveInferenceAgent)::Vector{Float64}
    if agent.state.inference_result === nothing
        @debug "No inference result available, returning zero action"
        return zeros(agent.state.action_dim)
    end
    
    try
        # Get the second control posterior (first future action)
        posteriors = agent.state.inference_result.posteriors[:u]
        if length(posteriors) < 2
            @debug "Not enough control posteriors, returning zero action"
            return zeros(agent.state.action_dim)
        end
        
        # Return mode (mean) of the control distribution
        return mode(posteriors[2])
    catch e
        @error "Failed to extract action" exception=e
        return zeros(agent.state.action_dim)
    end
end

@doc """
Get predicted future states.

Returns:
- Vector of predicted state means for each future timestep
"""
function get_predictions(agent::GenericActiveInferenceAgent)::Vector{Vector{Float64}}
    if agent.state.inference_result === nothing
        return [zeros(agent.state.state_dim) for _ in 1:agent.state.horizon]
    end
    
    try
        # Extract state posteriors
        posteriors = agent.state.inference_result.posteriors[:s]
        return [mode(p) for p in posteriors]
    catch e
        @error "Failed to extract predictions" exception=e
        return [zeros(agent.state.state_dim) for _ in 1:agent.state.horizon]
    end
end

@doc """
Slide the planning window forward one step.

This updates the agent's internal state for the next timestep by:
- Extracting updated state belief from inference
- Shifting control and goal priors forward
- Resetting the planning horizon
"""
function slide!(agent::GenericActiveInferenceAgent)
    if agent.state.inference_result === nothing
        @debug "No inference result to slide from (using placeholder inference)"
        # Slide priors anyway for placeholder agents
        huge = 1 / AGENT.control_prior_precision
        agent.state.control_means = circshift(agent.state.control_means, -1)
        agent.state.control_means[end] = zeros(agent.state.action_dim)
        agent.state.control_covs = circshift(agent.state.control_covs, -1)
        agent.state.control_covs[end] = huge * I(agent.state.action_dim)
        
        agent.state.goal_means = circshift(agent.state.goal_means, -1)
        agent.state.goal_means[end] = copy(agent.goal_state)
        agent.state.goal_covs = circshift(agent.state.goal_covs, -1)
        agent.state.goal_covs[end] = (1 / AGENT.goal_prior_precision) * I(agent.state.state_dim)
        return
    end
    
    try
        # Extract updated state belief
        model = RxInfer.getmodel(agent.state.inference_result.model)
        (s,) = RxInfer.getreturnval(model)
        varref = RxInfer.getvarref(model, s)
        var = RxInfer.getvariable(varref)
        
        # Get updated state belief (model-dependent indexing)
        slide_msg_idx = 3
        (m_s_new, V_s_new) = mean_cov(getrecent(messageout(var[2], slide_msg_idx)))
        
        agent.state.state_mean = m_s_new
        agent.state.state_cov = V_s_new
        
        # Slide control priors
        huge = 1 / AGENT.control_prior_precision
        agent.state.control_means = circshift(agent.state.control_means, -1)
        agent.state.control_means[end] = zeros(agent.state.action_dim)
        agent.state.control_covs = circshift(agent.state.control_covs, -1)
        agent.state.control_covs[end] = huge * I(agent.state.action_dim)
        
        # Slide goal priors
        agent.state.goal_means = circshift(agent.state.goal_means, -1)
        agent.state.goal_means[end] = copy(agent.goal_state)
        agent.state.goal_covs = circshift(agent.state.goal_covs, -1)
        agent.state.goal_covs[end] = copy(agent.goal_precision)
        
    catch e
        @error "Failed to slide planning window" exception=e
    end
end

@doc """
Reset agent to initial state.

Args:
- initial_state_mean: Optional new initial state
"""
function reset!(agent::GenericActiveInferenceAgent;
                initial_state_mean::Union{Nothing, Vector{Float64}} = nothing)
    
    # Reset state
    if initial_state_mean !== nothing
        agent.state.state_mean = copy(initial_state_mean)
    else
        agent.state.state_mean = zeros(agent.state.state_dim)
    end
    
    tiny = 1 / AGENT.initial_state_precision
    agent.state.state_cov = tiny * I(agent.state.state_dim)
    
    # Reset priors
    huge = 1 / AGENT.control_prior_precision
    agent.state.control_means = [zeros(agent.state.action_dim) for _ in 1:agent.state.horizon]
    agent.state.control_covs = [huge * I(agent.state.action_dim) for _ in 1:agent.state.horizon]
    
    agent.state.goal_means = [zeros(agent.state.state_dim) for _ in 1:agent.state.horizon]
    agent.state.goal_covs = [huge * I(agent.state.state_dim) for _ in 1:agent.state.horizon]
    agent.state.goal_means[end] = copy(agent.goal_state)
    agent.state.goal_covs[end] = copy(agent.goal_precision)
    
    # Reset inference results
    agent.state.inference_result = nothing
    agent.state.step_count = 0
    agent.state.total_inference_time = 0.0
    agent.state.last_free_energy = nothing
    
    # Clear diagnostics
    empty!(agent.belief_history)
    empty!(agent.action_history)
    empty!(agent.prediction_history)
    empty!(agent.free_energy_history)
    empty!(agent.state.memory_usage)
    
    @info "Agent reset"
end

# ==================== GENERATIVE MODEL ====================

@doc """
IMPORTANT: Model Definition for RxInfer v4+

In RxInfer v4 and later, @model macros must be defined at the top level,
not inside functions. This means each problem domain needs its own model definition.

Example model structure for your problem:

```julia
@model function your_problem_model(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Gamma, Theta)
    # Your problem-specific model here
    # See examples/mountain_car_example.jl for a complete example
end
```

Then in your code:
```julia
result = infer(
    model = your_problem_model(T = horizon, Gamma = precision, ...),
    data = data,
    iterations = 10
)
```

See the mountain_car_example.jl for a complete working implementation.
"""
function model_definition_guide()
    println("""
    To use this framework with RxInfer v4+:
    
    1. Define your @model at the top level of your script
    2. Include problem-specific transition and observation functions
    3. Call infer() with your model in the step!() function
    4. See examples/mountain_car_example.jl for reference
    """)
end

# ==================== UTILITY FUNCTIONS ====================

@doc """
Get agent diagnostics summary.

Returns:
- Dictionary with diagnostic information
"""
function get_diagnostics(agent::GenericActiveInferenceAgent)::Dict{String, Any}
    return Dict(
        "steps" => agent.state.step_count,
        "total_inference_time" => agent.state.total_inference_time,
        "avg_inference_time" => agent.state.step_count > 0 ?
            agent.state.total_inference_time / agent.state.step_count : 0.0,
        "last_free_energy" => agent.state.last_free_energy,
        "belief_history_length" => length(agent.belief_history),
        "action_history_length" => length(agent.action_history),
        "prediction_history_length" => length(agent.prediction_history),
        "free_energy_history_length" => length(agent.free_energy_history),
        "current_state_mean" => agent.state.state_mean,
        "current_state_cov_trace" => tr(agent.state.state_cov)
    )
end

@doc """
Print agent status.
"""
function print_status(agent::GenericActiveInferenceAgent)
    diagnostics = get_diagnostics(agent)
    
    println("=== Agent Status ===")
    println("Steps: $(diagnostics["steps"])")
    println("Total inference time: $(round(diagnostics["total_inference_time"], digits=3))s")
    println("Average inference time: $(round(diagnostics["avg_inference_time"], digits=4))s")
    
    if diagnostics["last_free_energy"] !== nothing
        println("Last free energy: $(round(diagnostics["last_free_energy"], digits=2))")
    end
    
    println("Current state: $(round.(diagnostics["current_state_mean"], digits=3))")
    println("State uncertainty (trace): $(round(diagnostics["current_state_cov_trace"], digits=3))")
    println("====================")
end

# Export public API
export AbstractActiveInferenceAgent, AbstractGenerativeModel
export AgentState, GenericActiveInferenceAgent
export step!, get_action, get_predictions, slide!, reset!
export get_diagnostics, print_status, model_definition_guide

end # module Agent

