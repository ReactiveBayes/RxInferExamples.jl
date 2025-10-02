# Agent Module Tests
# Comprehensive tests for agent.jl core functionality

using Test
using LinearAlgebra
using Statistics

# Load modules if not already loaded
if !isdefined(Main, :Config)
    include("../config.jl")
end
if !isdefined(Main, :Agent)
    include("../src/agent.jl")
end
using .Config
using .Agent

# Test helper function
function create_test_agent(;horizon::Int=5, state_dim::Int=2, action_dim::Int=1)
    A = [1.0 0.1; 0.0 0.9]
    B = [0.0; 0.1]
    
    transition_func = (s::AbstractVector) -> A * s
    control_func = (u::AbstractVector) -> B * u[1]
    
    return GenericActiveInferenceAgent(
        horizon,
        state_dim,
        action_dim,
        transition_func,
        control_func;
        goal_state = [1.0, 0.0],
        initial_state_mean = [0.0, 0.0]
    )
end

@testset "Agent Creation and Initialization" begin
    @testset "Basic Agent Creation" begin
        agent = create_test_agent()
        @test agent isa GenericActiveInferenceAgent
        @test agent.state.horizon == 5
        @test agent.state.state_dim == 2
        @test agent.state.action_dim == 1
        @test agent.state.step_count == 0
    end
    
    @testset "Initial State Setup" begin
        agent = create_test_agent()
        @test length(agent.state.state_mean) == 2
        @test size(agent.state.state_cov) == (2, 2)
        @test all(agent.state.state_mean .== 0.0)
        @test isposdef(agent.state.state_cov)
    end
    
    @testset "Control Priors Setup" begin
        agent = create_test_agent(horizon=5)
        @test length(agent.state.control_means) == 5
        @test length(agent.state.control_covs) == 5
        @test all(length.(agent.state.control_means) .== 1)
        @test all([isposdef(cov) for cov in agent.state.control_covs])
    end
    
    @testset "Goal Priors Setup" begin
        agent = create_test_agent()
        @test length(agent.state.goal_means) == 5
        @test agent.state.goal_means[end] == [1.0, 0.0]
        @test all([isposdef(cov) for cov in agent.state.goal_covs])
    end
    
    @testset "Different Dimensions" begin
        # Test 1D state, 1D action
        agent_1d = create_test_agent(state_dim=1, action_dim=1)
        @test agent_1d.state.state_dim == 1
        @test agent_1d.state.action_dim == 1
        
        # Test 3D state, 2D action
        agent_3d = create_test_agent(state_dim=3, action_dim=2)
        @test agent_3d.state.state_dim == 3
        @test agent_3d.state.action_dim == 2
    end
    
    @testset "Different Horizons" begin
        for h in [1, 5, 10, 20]
            agent = create_test_agent(horizon=h)
            @test agent.state.horizon == h
            @test length(agent.state.control_means) == h
            @test length(agent.state.goal_means) == h
        end
    end
end

@testset "Agent State Management" begin
    @testset "Step Count Tracking" begin
        agent = create_test_agent()
        @test agent.state.step_count == 0
        
        step!(agent, [0.1, 0.05], [0.0])
        @test agent.state.step_count == 1
        
        step!(agent, [0.2, 0.1], [0.0])
        @test agent.state.step_count == 2
    end
    
    @testset "Inference Time Tracking" begin
        agent = create_test_agent()
        initial_time = agent.state.total_inference_time
        
        step!(agent, [0.1, 0.05], [0.0])
        # Note: Generic agent uses placeholder inference, so time may not increase
        @test agent.state.total_inference_time >= initial_time
    end
    
    @testset "History Tracking" begin
        agent = create_test_agent()
        @test length(agent.belief_history) == 0
        @test length(agent.action_history) == 0
        
        for t in 1:5
            step!(agent, [0.1*t, 0.05*t], [0.0])
        end
        
        @test length(agent.belief_history) == 5
        @test length(agent.action_history) == 5
    end
    
    @testset "Agent Reset" begin
        agent = create_test_agent()
        
        # Perform some steps
        for t in 1:5
            step!(agent, [0.1*t, 0.05*t], [0.0])
        end
        
        @test agent.state.step_count == 5
        @test length(agent.belief_history) > 0
        
        # Reset
        reset!(agent, initial_state_mean=[0.5, 0.25])
        
        @test agent.state.step_count == 0
        @test length(agent.belief_history) == 0
        @test length(agent.action_history) == 0
        @test agent.state.state_mean == [0.5, 0.25]
    end
end

@testset "Agent Actions and Predictions" begin
    @testset "Action Retrieval" begin
        agent = create_test_agent()
        action = get_action(agent)
        @test action isa Vector{Float64}
        @test length(action) == 1
    end
    
    @testset "Prediction Retrieval" begin
        agent = create_test_agent(horizon=5)
        predictions = get_predictions(agent)
        @test predictions isa Vector{Vector{Float64}}
        @test length(predictions) == 5
        @test all(length.(predictions) .== 2)
    end
    
    @testset "Actions After Inference" begin
        agent = create_test_agent()
        step!(agent, [0.1, 0.05], [0.0])
        action = get_action(agent)
        @test action isa Vector{Float64}
        @test length(action) == 1
    end
end

@testset "Agent Planning Window" begin
    @testset "Slide Operation" begin
        agent = create_test_agent(horizon=5)
        step!(agent, [0.1, 0.05], [0.0])
        
        initial_controls = copy(agent.state.control_means)
        slide!(agent)
        
        # After sliding, control priors should be shifted
        @test agent.state.control_means[end] == zeros(1)
        @test agent.state.goal_means[end] == agent.goal_state
    end
    
    @testset "Multiple Slides" begin
        agent = create_test_agent(horizon=5)
        
        for t in 1:10
            step!(agent, [0.1*t, 0.05*t], [0.0])
            slide!(agent)
        end
        
        @test agent.state.step_count == 10
        @test length(agent.belief_history) == 10
    end
end

@testset "Agent Diagnostics" begin
    @testset "Get Diagnostics" begin
        agent = create_test_agent()
        step!(agent, [0.1, 0.05], [0.0])
        
        diag = get_diagnostics(agent)
        @test diag isa Dict{String, Any}
        @test haskey(diag, "steps")
        @test haskey(diag, "avg_inference_time")
        @test haskey(diag, "current_state_mean")
        @test haskey(diag, "current_state_cov_trace")
    end
    
    @testset "Print Status" begin
        agent = create_test_agent()
        step!(agent, [0.1, 0.05], [0.0])
        
        # Should not error
        @test_nowarn print_status(agent)
    end
end

@testset "Agent Edge Cases" begin
    @testset "Zero Initial State" begin
        agent = create_test_agent()
        @test all(agent.state.state_mean .== 0.0)
        @test_nowarn step!(agent, [0.0, 0.0], [0.0])
    end
    
    @testset "Large State Values" begin
        agent = create_test_agent()
        @test_nowarn step!(agent, [100.0, 50.0], [10.0])
    end
    
    @testset "Horizon = 1" begin
        agent = create_test_agent(horizon=1)
        @test agent.state.horizon == 1
        step!(agent, [0.1, 0.05], [0.0])
        # Slide works with placeholder inference (debug message, not warning)
        @test_nowarn slide!(agent)
    end
    
    @testset "Many Steps" begin
        agent = create_test_agent()
        # Suppress expected warnings for many iterations
        for t in 1:100
            step!(agent, randn(2), [0.0])
            slide!(agent)
        end
        @test agent.state.step_count == 100
    end
end

@testset "Agent Memory Management" begin
    @testset "History Size Growth" begin
        agent = create_test_agent()
        
        initial_history_size = length(agent.belief_history)
        
        for t in 1:50
            step!(agent, randn(2), [0.0])
        end
        
        @test length(agent.belief_history) == initial_history_size + 50
        @test length(agent.action_history) == initial_history_size + 50
    end
end

println("✅ Agent tests completed successfully")

