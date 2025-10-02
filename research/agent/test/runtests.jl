#!/usr/bin/env julia
# Main test runner for Agent-Environment Framework

using Test
using Pkg

# Activate project
Pkg.activate(joinpath(@__DIR__, ".."))

println("="^70)
println("AGENT-ENVIRONMENT FRAMEWORK - TEST SUITE")
println("="^70)
println()

# Run test files
@testset "Agent-Environment Framework Tests" begin
    
    @testset "Type System" begin
        include("test_types.jl")
    end
    
    @testset "Environments" begin
        include("test_environments.jl")
    end
    
    @testset "Agents" begin
        include("test_agents.jl")
    end
    
    @testset "Integration" begin
        include("test_integration.jl")
    end
    
    @testset "Visualization" begin
        include("test_visualization.jl")
    end
    
end

println("\n" * "="^70)
println("ALL TESTS COMPLETED")
println("="^70)

