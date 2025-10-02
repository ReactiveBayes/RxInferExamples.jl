#!/usr/bin/env julia
# Minimal verification script for the framework

using Pkg
Pkg.activate(@__DIR__)

println("="^70)
println("FRAMEWORK VERIFICATION")
println("="^70)
println()

# Test 1: Type System
println("✓ Test 1: Loading type system...")
try
    include("src/types.jl")
    using .Main: StateVector, ActionVector, ObservationVector
    
    s = StateVector{2}([1.0, 2.0])
    a = ActionVector{1}([0.5])
    o = ObservationVector{2}([1.1, 2.1])
    
    println("  ✓ StateVector{2}: $s")
    println("  ✓ ActionVector{1}: $a")
    println("  ✓ ObservationVector{2}: $o")
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 2: Environments
println("\n✓ Test 2: Loading environments...")
try
    include("src/environments/abstract_environment.jl")
    include("src/environments/mountain_car_env.jl")
    include("src/environments/simple_nav_env.jl")

    global mc_env = MountainCarEnv()
    global sn_env = SimpleNavEnv()

    println("  ✓ MountainCarEnv created")
    println("  ✓ SimpleNavEnv created")
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 3: Agents
println("\n✓ Test 3: Loading agents...")
try
    include("src/agents/abstract_agent.jl")
    include("src/agents/mountain_car_agent.jl")
    include("src/agents/simple_nav_agent.jl")

    mc_params = get_observation_model_params(mc_env)
    global mc_agent = MountainCarAgent(
        5,
        StateVector{2}([0.5, 0.0]),
        StateVector{2}([-0.5, 0.0]),
        mc_params
    )

    sn_params = get_observation_model_params(sn_env)
    global sn_agent = SimpleNavAgent(
        5,
        StateVector{1}([1.0]),
        StateVector{1}([0.0]),
        sn_params
    )

    println("  ✓ MountainCarAgent created")
    println("  ✓ SimpleNavAgent created")
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 4: Simulation infrastructure
println("\n✓ Test 4: Loading simulation infrastructure...")
try
    # Load constants first
    include("src/constants.jl")
    # Then diagnostics and logging (which depend on constants)
    include("src/diagnostics.jl")
    include("src/logging.jl")
    # Then simulation (which depends on diagnostics/logging)
    include("src/simulation.jl")
    
    println("  ✓ Simulation infrastructure loaded")
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 5: Quick simulation run
println("\n✓ Test 5: Running quick simulation (3 steps)...")
try
    config = SimulationConfig(
        max_steps = 3,
        enable_diagnostics = false,
        enable_logging = false,
        verbose = false
    )
    
    result = run_simulation(mc_agent, mc_env, config)
    
    println("  ✓ Simulation ran $(result.steps_taken) steps")
    println("  ✓ States collected: $(length(result.states))")
    println("  ✓ Actions collected: $(length(result.actions))")
    println("  ✓ Time: $(round(result.total_time, digits=2))s")
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

println("\n" * "="^70)
println("✅ ALL VERIFICATION TESTS PASSED")
println("="^70)
println("\nFramework is ready to use!")
println("Try: julia examples/mountain_car.jl")
println("     julia examples/simple_nav.jl")
println("     julia run.jl simulate")
