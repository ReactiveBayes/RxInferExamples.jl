#!/usr/bin/env julia
# Test script to verify all enhancements work correctly

using Pkg
Pkg.activate(@__DIR__)

using Dates
using Plots  # Required for visualizations

println("="^70)
println("TESTING FRAMEWORK ENHANCEMENTS")
println("="^70)
println()

# Test 1: Load all modules
println("[1/6] Loading framework modules...")
try
    include("src/types.jl")
    include("src/environments/abstract_environment.jl")
    include("src/environments/simple_nav_env.jl")
    include("src/environments/mountain_car_env.jl")
    include("src/agents/abstract_agent.jl")
    include("src/agents/simple_nav_agent.jl")
    include("src/agents/mountain_car_agent.jl")
    include("src/diagnostics.jl")
    include("src/logging.jl")
    include("src/visualization.jl")
    include("src/simulation.jl")
    
    using .Main: StateVector, ActionVector, ObservationVector
    using .Visualization
    
    println("  ✓ All modules loaded successfully")
catch e
    println("  ✗ Module loading failed: $e")
    exit(1)
end

# Test 2: Create test directory
println("\n[2/6] Creating test output directory...")
test_dir = joinpath(@__DIR__, "outputs", "enhancement_test_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(test_dir)
println("  ✓ Created: $test_dir")

# Test 3: Run simple simulation
println("\n[3/6] Running test simulation (Simple Nav, 10 steps)...")
try
    env = SimpleNavEnv(initial_position = 0.0, goal_position = 1.0)
    env_params = get_observation_model_params(env)
    
    goal_state = StateVector{1}([1.0])
    initial_state = StateVector{1}([0.0])
    
    agent = SimpleNavAgent(10, goal_state, initial_state, env_params)
    
    config = SimulationConfig(
        max_steps = 10,
        enable_diagnostics = true,
        enable_logging = false,
        verbose = false
    )
    
    result = run_simulation(agent, env, config)
    
    println("  ✓ Simulation completed ($(result.steps_taken) steps, $(round(result.total_time, digits=3))s)")
catch e
    println("  ✗ Simulation failed: $e")
    exit(1)
end

# Test 4: Test visualization functions
println("\n[4/6] Testing visualization functions...")
try
    plots_dir = joinpath(test_dir, "plots")
    mkpath(plots_dir)
    
    # Test 1D plotting
    plot_trajectory_1d(result, plots_dir)
    println("  ✓ 1D trajectory plot created")
    
    # Test diagnostics plotting
    if result.diagnostics !== nothing
        plot_diagnostics(result.diagnostics, plots_dir)
        println("  ✓ Diagnostics plot created")
    end
    
    # Test generate_all_visualizations
    all_plots = generate_all_visualizations(result, plots_dir, 1)
    println("  ✓ Generated $(length(all_plots)) visualization(s)")
    
catch e
    println("  ✗ Visualization failed: $e")
    exit(1)
end

# Test 5: Test animation generation
println("\n[5/6] Testing animation generation...")
try
    animations_dir = joinpath(test_dir, "animations")
    mkpath(animations_dir)
    
    animate_trajectory_1d(result, animations_dir, fps=5)
    println("  ✓ 1D animation created")
    
catch e
    println("  ✗ Animation generation failed: $e")
    exit(1)
end

# Test 6: Test comprehensive output saving
println("\n[6/6] Testing comprehensive output saving...")
try
    save_simulation_outputs(
        result,
        test_dir,
        StateVector{1}([1.0]),
        generate_visualizations=true,
        generate_animations=true
    )
    
    # Verify outputs
    required_files = [
        "REPORT.md",
        "metadata.json",
        "data/trajectory.csv",
        "data/observations.csv",
        "results/summary.csv",
        "diagnostics/diagnostics.json",
        "plots/trajectory_1d.png",
        "animations/trajectory_1d.gif"
    ]
    
    all_present = true
    for file in required_files
        if !isfile(joinpath(test_dir, file))
            println("  ✗ Missing: $file")
            all_present = false
        end
    end
    
    if all_present
        println("  ✓ All expected outputs present")
    else
        println("  ✗ Some outputs missing")
        exit(1)
    end
    
catch e
    println("  ✗ Comprehensive output saving failed: $e")
    exit(1)
end

# Success summary
println("\n" * "="^70)
println("✅ ALL ENHANCEMENT TESTS PASSED")
println("="^70)
println()
println("Test outputs saved to: $test_dir")
println()
println("Verify by examining:")
println("  • REPORT.md          - Comprehensive report")
println("  • plots/             - Visualizations")
println("  • animations/        - Animated GIFs")
println("  • data/              - CSV data files")
println("  • diagnostics/       - Performance metrics")
println()
println("Framework enhancements are fully functional! 🎉")
println("="^70)

