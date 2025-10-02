# Visualization Tests

using Test
include("../src/types.jl")
include("../src/agents/abstract_agent.jl")
include("../src/environments/abstract_environment.jl")
include("../src/agents/simple_nav_agent.jl")
include("../src/agents/mountain_car_agent.jl")
include("../src/environments/simple_nav_env.jl")
include("../src/environments/mountain_car_env.jl")
include("../src/diagnostics.jl")
include("../src/simulation.jl")
include("../src/visualization.jl")

using .Main: StateVector, ActionVector, ObservationVector
using .Visualization
using Plots

@testset "Visualization Module Tests" begin
    
    @testset "1D Trajectory Plotting" begin
        # Create simple 1D result
        env = SimpleNavEnv()
        agent = SimpleNavAgent(
            5,
            StateVector{1}([1.0]),
            StateVector{1}([0.0]),
            get_observation_model_params(env)
        )
        
        config = SimulationConfig(
            max_steps = 5,
            enable_diagnostics = false,
            enable_logging = false,
            verbose = false
        )
        
        result = run_simulation(agent, env, config)
        
        # Test plotting (just verify it doesn't crash)
        output_dir = mktempdir()
        
        @test_nowarn plot_trajectory_1d(result, output_dir)
        @test isfile(joinpath(output_dir, "trajectory_1d.png"))
        
        # Cleanup
        rm(output_dir, recursive=true)
    end
    
    @testset "2D Trajectory Plotting" begin
        # Create 2D result
        env = MountainCarEnv()
        agent = MountainCarAgent(
            5,
            StateVector{2}([0.5, 0.0]),
            StateVector{2}([-0.5, 0.0]),
            get_observation_model_params(env)
        )
        
        config = SimulationConfig(
            max_steps = 5,
            enable_diagnostics = false,
            enable_logging = false,
            verbose = false
        )
        
        result = run_simulation(agent, env, config)
        
        # Test plotting
        output_dir = mktempdir()
        
        @test_nowarn plot_trajectory_2d(result, output_dir)
        @test isfile(joinpath(output_dir, "trajectory_2d.png"))
        
        @test_nowarn plot_mountain_car_landscape(result, output_dir)
        @test isfile(joinpath(output_dir, "mountain_car_landscape.png"))
        
        # Cleanup
        rm(output_dir, recursive=true)
    end
    
    @testset "Diagnostics Plotting" begin
        # Create result with diagnostics
        env = SimpleNavEnv()
        agent = SimpleNavAgent(
            5,
            StateVector{1}([1.0]),
            StateVector{1}([0.0]),
            get_observation_model_params(env)
        )
        
        config = SimulationConfig(
            max_steps = 5,
            enable_diagnostics = true,
            enable_logging = false,
            verbose = false
        )
        
        result = run_simulation(agent, env, config)
        
        # Test diagnostics plotting
        output_dir = mktempdir()
        
        @test_nowarn plot_diagnostics(result.diagnostics, output_dir)
        @test isfile(joinpath(output_dir, "diagnostics.png"))
        
        # Cleanup
        rm(output_dir, recursive=true)
    end
    
    @testset "Animation Generation (1D)" begin
        # Create simple 1D result
        env = SimpleNavEnv()
        agent = SimpleNavAgent(
            5,
            StateVector{1}([1.0]),
            StateVector{1}([0.0]),
            get_observation_model_params(env)
        )
        
        config = SimulationConfig(
            max_steps = 5,
            enable_diagnostics = false,
            enable_logging = false,
            verbose = false
        )
        
        result = run_simulation(agent, env, config)
        
        # Test animation (just verify it doesn't crash)
        output_dir = mktempdir()
        
        @test_nowarn animate_trajectory_1d(result, output_dir, fps=2)
        @test isfile(joinpath(output_dir, "trajectory_1d.gif"))
        
        # Cleanup
        rm(output_dir, recursive=true)
    end
    
    @testset "Animation Generation (2D)" begin
        # Create 2D result
        env = MountainCarEnv()
        agent = MountainCarAgent(
            5,
            StateVector{2}([0.5, 0.0]),
            StateVector{2}([-0.5, 0.0]),
            get_observation_model_params(env)
        )
        
        config = SimulationConfig(
            max_steps = 5,
            enable_diagnostics = false,
            enable_logging = false,
            verbose = false
        )
        
        result = run_simulation(agent, env, config)
        
        # Test animation
        output_dir = mktempdir()
        
        @test_nowarn animate_trajectory_2d(result, output_dir, fps=2)
        @test isfile(joinpath(output_dir, "trajectory_2d.gif"))
        
        # Cleanup
        rm(output_dir, recursive=true)
    end
    
    @testset "Comprehensive Visualization Generation" begin
        # Test the main wrapper function
        
        # 1D case
        env_1d = SimpleNavEnv()
        agent_1d = SimpleNavAgent(
            5,
            StateVector{1}([1.0]),
            StateVector{1}([0.0]),
            get_observation_model_params(env_1d)
        )
        
        config = SimulationConfig(
            max_steps = 5,
            enable_diagnostics = true,
            enable_logging = false,
            verbose = false
        )
        
        result_1d = run_simulation(agent_1d, env_1d, config)
        output_dir_1d = mktempdir()
        
        plots_created = generate_all_visualizations(result_1d, output_dir_1d, 1)
        @test length(plots_created) >= 2  # At least trajectory and diagnostics
        @test "trajectory_1d.png" in plots_created
        @test "trajectory_1d.gif" in plots_created
        
        rm(output_dir_1d, recursive=true)
        
        # 2D case
        env_2d = MountainCarEnv()
        agent_2d = MountainCarAgent(
            5,
            StateVector{2}([0.5, 0.0]),
            StateVector{2}([-0.5, 0.0]),
            get_observation_model_params(env_2d)
        )
        
        result_2d = run_simulation(agent_2d, env_2d, config)
        output_dir_2d = mktempdir()
        
        plots_created = generate_all_visualizations(result_2d, output_dir_2d, 2)
        @test length(plots_created) >= 3  # Trajectory, landscape, diagnostics
        @test "trajectory_2d.png" in plots_created
        @test "trajectory_2d.gif" in plots_created
        @test "mountain_car_landscape.png" in plots_created
        
        rm(output_dir_2d, recursive=true)
    end
    
end

