# Environment Tests

using Test
include("../src/types.jl")
include("../src/environments/abstract_environment.jl")
include("../src/environments/mountain_car_env.jl")
include("../src/environments/simple_nav_env.jl")

using .Main: StateVector, ActionVector, ObservationVector

@testset "MountainCarEnv Tests" begin
    # Construction
    env = MountainCarEnv(
        initial_position = -0.5,
        initial_velocity = 0.0
    )
    
    @test env isa MountainCarEnv
    @test env isa AbstractEnvironment{2,1,2}
    
    # Reset
    obs = reset!(env)
    @test obs isa ObservationVector{2}
    @test obs[1] ≈ -0.5
    @test obs[2] ≈ 0.0
    
    # Get state
    state = get_state(env)
    @test state isa StateVector{2}
    @test state[1] ≈ -0.5
    @test state[2] ≈ 0.0
    
    # Step with action
    action = ActionVector{1}([0.5])
    obs = step!(env, action)
    @test obs isa ObservationVector{2}
    
    # State should have changed
    new_state = get_state(env)
    @test new_state[1] != -0.5 || new_state[2] != 0.0
    
    # Get observation model params
    params = get_observation_model_params(env)
    @test haskey(params, :observation_precision)
    @test haskey(params, :Fa)
    @test haskey(params, :Ff)
    @test haskey(params, :Fg)
    @test params.observation_precision > 0
end

@testset "SimpleNavEnv Tests" begin
    # Construction
    env = SimpleNavEnv(
        initial_position = 0.0,
        goal_position = 1.0
    )
    
    @test env isa SimpleNavEnv
    @test env isa AbstractEnvironment{1,1,1}
    
    # Reset
    obs = reset!(env)
    @test obs isa ObservationVector{1}
    @test obs[1] ≈ 0.0
    
    # Get state
    state = get_state(env)
    @test state isa StateVector{1}
    @test state[1] ≈ 0.0
    
    # Step with positive velocity
    action = ActionVector{1}([0.3])
    obs = step!(env, action)
    @test obs isa ObservationVector{1}
    
    # Position should have increased
    new_state = get_state(env)
    @test new_state[1] > 0.0
    
    # Get observation model params
    params = get_observation_model_params(env)
    @test haskey(params, :observation_precision)
    @test haskey(params, :dt)
    @test haskey(params, :velocity_limit)
    @test params.dt > 0
end

@testset "Environment Interface Compliance" begin
    # Test that required methods exist
    env_mc = MountainCarEnv()
    env_sn = SimpleNavEnv()
    
    # MountainCarEnv
    @test hasmethod(step!, (MountainCarEnv, ActionVector{1}))
    @test hasmethod(reset!, (MountainCarEnv,))
    @test hasmethod(get_state, (MountainCarEnv,))
    @test hasmethod(get_observation_model_params, (MountainCarEnv,))
    
    # SimpleNavEnv
    @test hasmethod(step!, (SimpleNavEnv, ActionVector{1}))
    @test hasmethod(reset!, (SimpleNavEnv,))
    @test hasmethod(get_state, (SimpleNavEnv,))
    @test hasmethod(get_observation_model_params, (SimpleNavEnv,))
end

@testset "Environment Physics" begin
    # Mountain car should respect physics
    env = MountainCarEnv(initial_position = -0.5, initial_velocity = 0.0)
    reset!(env)
    
    # Apply large positive force
    for i in 1:10
        action = ActionVector{1}([1.0])
        obs = step!(env, action)
    end
    
    state = get_state(env)
    # Should have moved right (positive velocity developed)
    @test state[2] > 0.0
    
    # Simple nav should integrate velocity
    env_nav = SimpleNavEnv(initial_position = 0.0)
    reset!(env_nav)
    
    # Apply constant velocity
    for i in 1:10
        action = ActionVector{1}([0.1])
        obs = step!(env_nav, action)
    end
    
    state_nav = get_state(env_nav)
    # Position should be approximately velocity * dt * steps
    expected_pos = 0.1 * 0.1 * 10  # velocity * dt * steps
    @test abs(state_nav[1] - expected_pos) < 0.1  # Allow some tolerance
end

