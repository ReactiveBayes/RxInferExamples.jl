#!/usr/bin/env julia

# Simple test script for Active Inference Mountain Car example
# This script tests the basic functionality without complex module structures

import Pkg
using Test
Pkg.activate(".")

# Include and test each module individually
println("Testing Config module...")
include("config.jl")
println("✓ Config loaded successfully")

println("Testing Physics module...")
include("src/physics.jl")
# Import physics functions
using .Physics: create_physics, next_state, get_landscape_coordinates
println("✓ Physics loaded successfully")

println("Testing World module...")
include("src/world.jl")
# Import world functions
using .World: create_world, simulate_trajectory
println("✓ World loaded successfully")

println("Testing Agent module...")
include("src/agent.jl")
# Import agent functions
using .Agent: create_agent
println("✓ Agent loaded successfully")

println("Testing Visualization module...")
include("src/visualization.jl")
# Import visualization functions
using .Visualization: plot_landscape, height_at_position
println("✓ Visualization loaded successfully")

# Test basic functionality
println("\nRunning basic tests...")

# Test physics functions
Fa, Ff, Fg, height = create_physics()
@test Fa(0.0) == 0.0
@test abs(Fa(1.0)) <= 0.04 + 1e-10
println("✓ Physics functions work correctly")

# Test world creation
execute, observe, reset, get_state, set_state = create_world(
    Fg = Fg, Ff = Ff, Fa = Fa
)
initial_state = observe()
@test length(initial_state) == 2
println("✓ World creation works correctly")

# Test agent creation
compute, act, slide, future, reset_agent = create_agent(
    T = 5,
    Fa = Fa, Ff = Ff, Fg = Fg
)
initial_action = act()
@test typeof(initial_action) == Float64
println("✓ Agent creation works correctly")

println("\n🎉 All basic tests passed!")
