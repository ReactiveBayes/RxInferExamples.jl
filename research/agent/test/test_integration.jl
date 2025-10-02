# Integration Tests

using Test
include("../src/types.jl")
include("../src/agents/abstract_agent.jl")
include("../src/environments/abstract_environment.jl")
include("../src/agents/mountain_car_agent.jl")
include("../src/agents/simple_nav_agent.jl")
include("../src/environments/mountain_car_env.jl")
include("../src/environments/simple_nav_env.jl")
include("../src/simulation.jl")

using .Main: StateVector, ActionVector, ObservationVector

@testset "Mountain Car Integration" begin
    # Create environment
    env = MountainCarEnv(
        initial_position = -0.5,
        initial_velocity = 0.0
    )
    
    # Create agent
    env_params = get_observation_model_params(env)
    goal_state = StateVector{2}([0.5, 0.0])
    initial_state = StateVector{2}([-0.5, 0.0])
    
    agent = MountainCarAgent(
        10,
        goal_state,
        initial_state,
        env_params
    )
    
    # Run simulation
    config = SimulationConfig(
        max_steps = 10,
        enable_diagnostics = false,
        enable_logging = false,
        verbose = false
    )
    
    result = run_simulation(agent, env, config)
    
    # Verify result structure
    @test result isa SimulationResult{2,1,2}
    @test length(result.states) == 11  # Initial + 10 steps
    @test length(result.actions) == 10
    @test length(result.observations) == 11
    @test result.steps_taken == 10
    @test result.total_time > 0.0
end

@testset "Simple Nav Integration" begin
    # Create environment
    env = SimpleNavEnv(
        initial_position = 0.0,
        goal_position = 1.0
    )
    
    # Create agent
    env_params = get_observation_model_params(env)
    goal_state = StateVector{1}([1.0])
    initial_state = StateVector{1}([0.0])
    
    agent = SimpleNavAgent(
        10,
        goal_state,
        initial_state,
        env_params
    )
    
    # Run simulation
    config = SimulationConfig(
        max_steps = 10,
        enable_diagnostics = false,
        enable_logging = false,
        verbose = false
    )
    
    result = run_simulation(agent, env, config)
    
    # Verify result structure
    @test result isa SimulationResult{1,1,1}
    @test length(result.states) == 11  # Initial + 10 steps
    @test length(result.actions) == 10
    @test length(result.observations) == 11
    @test result.steps_taken == 10
    @test result.total_time > 0.0
end

@testset "Type Safety at Runtime" begin
    # These combinations should work (matching dimensions)
    
    # Mountain Car: 2D state, 1D action, 2D observation
    env_mc = MountainCarEnv()
    env_mc_params = get_observation_model_params(env_mc)
    agent_mc = MountainCarAgent(
        5,
        StateVector{2}([0.5, 0.0]),
        StateVector{2}([-0.5, 0.0]),
        env_mc_params
    )
    
    config = SimulationConfig(max_steps=5, enable_diagnostics=false, enable_logging=false, verbose=false)
    
    # This should work
    result_mc = run_simulation(agent_mc, env_mc, config)
    @test result_mc isa SimulationResult{2,1,2}
    
    # Simple Nav: 1D state, 1D action, 1D observation
    env_sn = SimpleNavEnv()
    env_sn_params = get_observation_model_params(env_sn)
    agent_sn = SimpleNavAgent(
        5,
        StateVector{1}([1.0]),
        StateVector{1}([0.0]),
        env_sn_params
    )
    
    # This should work
    result_sn = run_simulation(agent_sn, env_sn, config)
    @test result_sn isa SimulationResult{1,1,1}
end

@testset "Multi-Step Simulation" begin
    # Test longer simulation to ensure stability
    env = SimpleNavEnv()
    env_params = get_observation_model_params(env)
    
    agent = SimpleNavAgent(
        10,
        StateVector{1}([1.0]),
        StateVector{1}([0.0]),
        env_params
    )
    
    config = SimulationConfig(
        max_steps = 20,
        enable_diagnostics = true,
        enable_logging = false,
        verbose = false
    )
    
    result = run_simulation(agent, env, config)
    
    # Verify agent made progress toward goal
    initial_distance = abs(result.states[1][1] - 1.0)
    final_distance = abs(result.states[end][1] - 1.0)
    
    # Agent should have moved closer to goal (or reached it)
    @test final_distance <= initial_distance || final_distance < 0.1
    
    # Diagnostics should be collected
    @test result.diagnostics !== nothing
end

