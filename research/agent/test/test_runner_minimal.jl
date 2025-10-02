#!/usr/bin/env julia
# Minimal test runner that avoids slow RxInfer tests

using Test
using Pkg

# Activate project
Pkg.activate(joinpath(@__DIR__, ".."))

println("="^70)
println("AGENT-ENVIRONMENT FRAMEWORK - MINIMAL TEST SUITE")
println("="^70)
println()

# Load framework once at top level to avoid reloading
include("../src/types.jl")
include("../src/environments/abstract_environment.jl")
include("../src/environments/mountain_car_env.jl")
include("../src/environments/simple_nav_env.jl")
include("../src/agents/abstract_agent.jl")

using .Main: StateVector, ActionVector, ObservationVector

@testset "Minimal Framework Tests" begin
    
    @testset "Type System" begin
        # Construction
        @test StateVector{2}([1.0, 2.0]) isa StateVector{2}
        @test ActionVector{1}([0.5]) isa ActionVector{1}
        @test ObservationVector{2}([1.1, 2.1]) isa ObservationVector{2}
        
        # Dimensions
        s = StateVector{3}([1.0, 2.0, 3.0])
        @test length(s) == 3
        @test s[1] == 1.0
        
        # Operations
        s1 = StateVector{2}([1.0, 2.0])
        s2 = StateVector{2}([3.0, 4.0])
        @test Vector(s1 + s2) == [4.0, 6.0]
        
        println("  ✓ Type system tests passed")
    end
    
    @testset "Environments (No RxInfer)" begin
        # Mountain Car
        env_mc = MountainCarEnv(initial_position = -0.5)
        @test env_mc isa AbstractEnvironment{2,1,2}
        
        obs = reset!(env_mc)
        @test obs isa ObservationVector{2}
        @test obs[1] ≈ -0.5
        
        action = ActionVector{1}([0.5])
        obs2 = step!(env_mc, action)
        @test obs2 isa ObservationVector{2}
        
        state = get_state(env_mc)
        @test state isa StateVector{2}
        
        params = get_observation_model_params(env_mc)
        @test haskey(params, :Fa)
        
        # Simple Nav
        env_sn = SimpleNavEnv(initial_position = 0.0)
        @test env_sn isa AbstractEnvironment{1,1,1}
        
        obs = reset!(env_sn)
        @test obs isa ObservationVector{1}
        
        action = ActionVector{1}([0.3])
        obs2 = step!(env_sn, action)
        @test obs2 isa ObservationVector{1}
        
        println("  ✓ Environment tests passed")
    end
    
    @testset "Agent Creation (No Inference)" begin
        # Just test that agents can be created
        include("../src/agents/mountain_car_agent.jl")
        include("../src/agents/simple_nav_agent.jl")
        
        env_mc = MountainCarEnv()
        params_mc = get_observation_model_params(env_mc)
        agent_mc = MountainCarAgent(
            5,
            StateVector{2}([0.5, 0.0]),
            StateVector{2}([-0.5, 0.0]),
            params_mc
        )
        @test agent_mc isa AbstractActiveInferenceAgent{2,1,2}
        @test agent_mc.horizon == 5
        
        env_sn = SimpleNavEnv()
        params_sn = get_observation_model_params(env_sn)
        agent_sn = SimpleNavAgent(
            5,
            StateVector{1}([1.0]),
            StateVector{1}([0.0]),
            params_sn
        )
        @test agent_sn isa AbstractActiveInferenceAgent{1,1,1}
        @test agent_sn.horizon == 5
        
        println("  ✓ Agent creation tests passed")
    end
    
end

println("\n" * "="^70)
println("✅ MINIMAL TESTS COMPLETED")
println("="^70)
println("\nNote: Full integration tests with RxInfer inference are slow.")
println("Run them with: julia test/runtests.jl")

