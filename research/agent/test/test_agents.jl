# Agent Tests

using Test
include("../src/types.jl")
include("../src/agents/abstract_agent.jl")
include("../src/environments/abstract_environment.jl")
include("../src/environments/mountain_car_env.jl")
include("../src/environments/simple_nav_env.jl")
include("../src/agents/mountain_car_agent.jl")
include("../src/agents/simple_nav_agent.jl")

using .Main: StateVector, ActionVector, ObservationVector

@testset "MountainCarAgent Tests" begin
    # Create environment and get params
    env = MountainCarEnv()
    env_params = get_observation_model_params(env)
    
    # Create agent
    goal_state = StateVector{2}([0.5, 0.0])
    initial_state = StateVector{2}([-0.5, 0.0])
    
    agent = MountainCarAgent(
        10,  # horizon
        goal_state,
        initial_state,
        env_params
    )
    
    @test agent isa MountainCarAgent
    @test agent isa AbstractActiveInferenceAgent{2,1,2}
    @test agent.horizon == 10
    
    # Get action (before any inference)
    action = get_action(agent)
    @test action isa ActionVector{1}
    
    # Step
    observation = ObservationVector{2}([-0.5, 0.0])
    action_taken = ActionVector{1}([0.0])
    step!(agent, observation, action_taken)
    @test agent.step_count == 1
    
    # Get predictions
    preds = get_predictions(agent)
    @test preds isa Vector{StateVector{2}}
    @test length(preds) == agent.horizon
    
    # Slide
    slide!(agent)
    @test length(agent.m_u) == agent.horizon
    
    # Reset
    reset!(agent)
    @test agent.step_count == 0
end

@testset "SimpleNavAgent Tests" begin
    # Create environment and get params
    env = SimpleNavEnv()
    env_params = get_observation_model_params(env)
    
    # Create agent
    goal_state = StateVector{1}([1.0])
    initial_state = StateVector{1}([0.0])
    
    agent = SimpleNavAgent(
        10,  # horizon
        goal_state,
        initial_state,
        env_params
    )
    
    @test agent isa SimpleNavAgent
    @test agent isa AbstractActiveInferenceAgent{1,1,1}
    @test agent.horizon == 10
    
    # Get action (before any inference)
    action = get_action(agent)
    @test action isa ActionVector{1}
    
    # Step
    observation = ObservationVector{1}([0.0])
    action_taken = ActionVector{1}([0.0])
    step!(agent, observation, action_taken)
    @test agent.step_count == 1
    
    # Get predictions
    preds = get_predictions(agent)
    @test preds isa Vector{StateVector{1}}
    @test length(preds) == agent.horizon
    
    # Slide
    slide!(agent)
    @test length(agent.m_u) == agent.horizon
    
    # Reset
    reset!(agent)
    @test agent.step_count == 0
end

@testset "Agent Interface Compliance" begin
    # Test that required methods exist
    env_mc = MountainCarEnv()
    env_mc_params = get_observation_model_params(env_mc)
    agent_mc = MountainCarAgent(
        10,
        StateVector{2}([0.5, 0.0]),
        StateVector{2}([-0.5, 0.0]),
        env_mc_params
    )
    
    env_sn = SimpleNavEnv()
    env_sn_params = get_observation_model_params(env_sn)
    agent_sn = SimpleNavAgent(
        10,
        StateVector{1}([1.0]),
        StateVector{1}([0.0]),
        env_sn_params
    )
    
    # MountainCarAgent
    @test hasmethod(step!, (MountainCarAgent, ObservationVector{2}, ActionVector{1}))
    @test hasmethod(get_action, (MountainCarAgent,))
    @test hasmethod(get_predictions, (MountainCarAgent,))
    @test hasmethod(slide!, (MountainCarAgent,))
    @test hasmethod(reset!, (MountainCarAgent,))
    
    # SimpleNavAgent
    @test hasmethod(step!, (SimpleNavAgent, ObservationVector{1}, ActionVector{1}))
    @test hasmethod(get_action, (SimpleNavAgent,))
    @test hasmethod(get_predictions, (SimpleNavAgent,))
    @test hasmethod(slide!, (SimpleNavAgent,))
    @test hasmethod(reset!, (SimpleNavAgent,))
end

