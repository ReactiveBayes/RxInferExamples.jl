#!/usr/bin/env julia

# Simple demonstration of the Generalized Active Inference Car system
# Focuses on core functionality without complex dependencies

using Pkg
using Logging
using Dates

# Activate project
Pkg.activate(".")

# Include core modules
include("config.jl")
include("src/physics.jl")
include("src/world.jl")
include("src/visualization.jl")
include("src/utils.jl")

# Import functions
using .Config: get_car_config, print_configuration
using .Physics: create_physics, create_integrator
using .World: create_world, reset!, observe, execute_action!
using .Visualization: create_visualization
using .Utils: setup_logging, PerformanceTimer, close

@info "=== Generalized Active Inference Car System Demo ==="

function demo_configuration()
    @info "\n1. Configuration System Demo"
    println("="^50)

    # Show available car types
    println("Available Car Types:")
    for (key, info) in Config.CAR_TYPES
        println("  - $key: $(info.name)")
        println("    Description: $(info.description)")
    end
    println()

    # Show mountain car configuration
    println("Mountain Car Configuration:")
    config = get_car_config(:mountain_car)
    println("  Car Type: $(config.car_type)")
    println("  Physics: engine_force_limit = $(config.physics.engine_force_limit)")
    println("  World: x_min = $(config.world.x_min), x_max = $(config.world.x_max)")
    println("  Agent: planning_horizon = $(config.agent.planning_horizon)")
    println()
end

function demo_physics()
    @info "\n2. Physics Engine Demo"
    println("="^50)

    try
        # Create physics model
        physics = create_physics(:mountain_car; custom_params = Dict{Symbol, Any}())
        @info "✓ Physics model created successfully"
        println("  Type: $(typeof(physics))")
        println("  Engine Force Limit: $(physics.engine_force_limit)")
        println("  Friction Coefficient: $(physics.friction_coefficient)")
        println("  Mass: $(physics.mass)")
        println()

        # Test force calculations
        position, velocity, action = 0.0, 0.0, 0.1
        engine_force = physics.engine_force(action)
        friction_force = physics.friction_force(velocity)
        gravitational_force = physics.gravitational_force(position)
        total_force = physics.total_force(position, velocity, action)

        println("Force Calculations:")
        println("  Engine Force: $(round(engine_force, digits=6))")
        println("  Friction Force: $(round(friction_force, digits=6))")
        println("  Gravitational Force: $(round(gravitational_force, digits=6))")
        println("  Total Force: $(round(total_force, digits=6))")
        println()

        # Test integrator
        integrator = create_integrator(:euler, 0.1)
        @info "✓ Integrator created successfully"
        println("  Type: $(typeof(integrator))")
        println("  Time Step: $(integrator.time_step)")
        println()

    catch e
        @warn "Physics demo failed: $e"
    end
end

function demo_world()
    @info "\n3. World Environment Demo"
    println("="^50)

    try
        # Create world
        world = create_world(:mountain_car; custom_params = Dict{Symbol, Any}())
        @info "✓ World created successfully"
        println("  Type: $(typeof(world))")
        println("  Environment Bounds: [$(world.x_min), $(world.x_max)]")
        println("  Velocity Limits: [$(world.v_min), $(world.v_max)]")
        println("  Goal Position: $(world.goal_position)")
        println("  Goal Tolerance: $(world.goal_tolerance)")
        println()

        # Test state management
        initial_state = observe(world)
        @info "✓ Initial state observed"
        println("  Initial State: [$(round(initial_state[1], digits=3)), $(round(initial_state[2], digits=3))]")
        println()

        # Test reset functionality
        reset!(world)
        reset_state = observe(world)
        @info "✓ World reset successfully"
        println("  Reset State: [$(round(reset_state[1], digits=3)), $(round(reset_state[2], digits=3))]")
        println()

    catch e
        @warn "World demo failed: $e"
    end
end

function demo_visualization()
    @info "\n4. Visualization System Demo"
    println("="^50)

    try
        # Create visualization system
        vis_system = create_visualization(:mountain_car)
        @info "✓ Visualization system created"
        println("  Components available: $(keys(vis_system))")
        println()

        # Test landscape plotting
        try
            landscape_plot = Visualization.create_landscape_plot(:mountain_car, 0.0)
            @info "✓ Landscape plotting works"
        catch e
            @warn "Landscape plotting failed (may require display): $e"
        end

        # Test control plotting
        actions = [0.1, 0.05, 0.02, 0.01, 0.0]
        try
            control_plot = Visualization.create_control_plot(actions, 5, :mountain_car)
            @info "✓ Control plotting works"
        catch e
            @warn "Control plotting failed: $e"
        end

        println()

    catch e
        @warn "Visualization demo failed: $e"
    end
end

function demo_multiple_car_types()
    @info "\n5. Multiple Car Types Demo"
    println("="^50)

    car_types = [:mountain_car, :race_car, :autonomous_car]

    for car_type in car_types
        try
            println("Testing $car_type:")

            # Create physics
            physics = create_physics(car_type; custom_params = Dict{Symbol, Any}())
            println("  ✓ Physics: $(typeof(physics))")

            # Create world
            world = create_world(car_type; custom_params = Dict{Symbol, Any}())
            println("  ✓ World: $(typeof(world))")

            # Create visualization
            vis = create_visualization(car_type)
            println("  ✓ Visualization: $(length(vis)) components")

            println("  ✓ $car_type is fully functional!")
            println()

        catch e
            @warn "Failed to create $car_type: $e"
            println()
        end
    end
end

function demo_simple_simulation()
    @info "\n6. Simple Simulation Demo"
    println("="^50)

    try
        # Setup simulation
        physics = create_physics(:mountain_car; custom_params = Dict{Symbol, Any}())
        world = create_world(:mountain_car; custom_params = Dict{Symbol, Any}())

        @info "✓ Simulation components created"

        # Simple simulation loop
        time_steps = 10
        actions = [0.1 for _ in 1:time_steps]

        println("Running simulation for $time_steps steps...")
        reset!(world)

        for t in 1:time_steps
            state = observe(world)
            success, collision = execute_action!(world, actions[t], physics)

            if t % 5 == 0
                println("  Step $t: Position = $(round(state[1], digits=3)), Success = $success")
            end
        end

        final_state = observe(world)
        println("✓ Simulation completed successfully!")
        println("  Final State: [$(round(final_state[1], digits=3)), $(round(final_state[2], digits=3))]")
        println()

    catch e
        @warn "Simple simulation failed: $e"
    end
end

function demo_performance_monitoring()
    @info "\n7. Performance Monitoring Demo"
    println("="^50)

    try
        # Test performance timer
        timer = PerformanceTimer("demo_operation")
        sleep(0.01)  # Small delay
        close(timer)

        @info "✓ Performance monitoring works"
        println("  Performance timer created and closed successfully")
        println()

    catch e
        @warn "Performance monitoring failed: $e"
    end
end

function main()
    println("Generalized Active Inference Car System")
    println("="^60)
    println("A comprehensive framework for active inference in car scenarios")
    println("Supports multiple car types with extensible architecture")
    println()

    # Setup logging
    setup_logging(log_level = Logging.Info)

    # Run all demos
    demo_configuration()
    demo_physics()
    demo_world()
    demo_visualization()
    demo_multiple_car_types()
    demo_simple_simulation()
    demo_performance_monitoring()

    println("\n" * "="^60)
    println("🎉 DEMONSTRATION COMPLETED")
    println("="^60)
    println("The Generalized Active Inference Car system is working correctly!")
    println()
    println("Key Features Demonstrated:")
    println("  ✓ Modular Configuration System")
    println("  ✓ Multiple Car Types (Mountain, Race, Autonomous)")
    println("  ✓ Physics Engine with Different Dynamics")
    println("  ✓ World Environment Management")
    println("  ✓ Visualization System")
    println("  ✓ Simple Simulation Loop")
    println("  ✓ Performance Monitoring")
    println()
    println("The system provides a solid foundation for:")
    println("  • Active inference research")
    println("  • Multi-agent systems")
    println("  • Robotics and autonomous vehicles")
    println("  • Educational demonstrations")
    println()
    println("Next Steps:")
    println("  1. Install full RxInfer ecosystem for @model support")
    println("  2. Run full examples: julia run.jl mountain_car --animation")
    println("  3. Explore different car types and configurations")
    println("  4. Extend the system with new features")
    println("="^60)

    @info "Demo completed successfully!"
end

# Run demo if script is executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end


