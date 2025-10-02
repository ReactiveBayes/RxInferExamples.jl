# Configuration Module Tests
# Comprehensive tests for config.jl validation and customization

using Test
using Logging

# Load Config module if not already loaded
if !isdefined(Main, :Config)
    include("../config.jl")
end
using .Config

@testset "Configuration Validation" begin
    @testset "Default Configuration Validation" begin
        issues = Config.validate_config()
        @test isempty(issues)
        @test Config.AGENT.planning_horizon > 0
        @test Config.AGENT.transition_precision > 0
        @test Config.AGENT.observation_precision > 0
        @test Config.SIMULATION.max_timesteps > 0
        @test Config.NUMERICAL.epsilon > 0
    end
    
    @testset "Agent Parameter Validation" begin
        @test Config.AGENT.planning_horizon isa Int
        @test Config.AGENT.inference_iterations > 0
        @test Config.AGENT.convergence_tolerance > 0
        @test Config.AGENT.cache_results isa Bool
        @test Config.AGENT.enable_diagnostics isa Bool
    end
    
    @testset "Simulation Parameter Validation" begin
        @test Config.SIMULATION.max_timesteps isa Int
        @test Config.SIMULATION.verbose isa Bool
        @test Config.SIMULATION.seed isa Int
        @test Config.SIMULATION.goal_tolerance > 0
    end
    
    @testset "Logging Parameter Validation" begin
        @test Config.LOGGING.enable_logging isa Bool
        @test Config.LOGGING.log_level isa LogLevel
        @test Config.LOGGING.log_to_console isa Bool
        @test Config.LOGGING.log_to_file isa Bool
        # Verify log file path is valid
        @test Config.LOGGING.log_file isa String
    end
    
    @testset "Diagnostics Parameter Validation" begin
        @test Config.DIAGNOSTICS.track_beliefs isa Bool
        @test Config.DIAGNOSTICS.track_actions isa Bool
        @test Config.DIAGNOSTICS.track_predictions isa Bool
        @test Config.DIAGNOSTICS.track_free_energy isa Bool
    end
    
    @testset "Visualization Parameter Validation" begin
        @test Config.VISUALIZATION.enable_plots isa Bool
        @test Config.VISUALIZATION.plot_size isa Tuple
        @test length(Config.VISUALIZATION.plot_size) == 2
        @test Config.VISUALIZATION.animation_fps > 0
    end
    
    @testset "Numerical Parameter Validation" begin
        @test Config.NUMERICAL.epsilon > 0
        @test Config.NUMERICAL.epsilon < 1e-6
        @test Config.NUMERICAL.tolerance > 0
        @test Config.NUMERICAL.max_iterations > 0
        @test Config.NUMERICAL.clip_values isa Bool
        @test Config.NUMERICAL.clip_range isa Tuple
    end
end

@testset "Custom Configuration Creation" begin
    @testset "Empty Custom Config" begin
        custom = Config.create_custom_config(Dict{Symbol, Any}())
        @test custom.AGENT.planning_horizon > 0
        @test custom.SIMULATION.max_timesteps > 0
    end
    
    @testset "Custom Config Merging" begin
        # Test that custom config uses defaults when no overrides provided
        custom = Config.create_custom_config(Dict{Symbol, Any}())
        @test custom.AGENT.planning_horizon == Config.AGENT.planning_horizon
        @test custom.SIMULATION.max_timesteps == Config.SIMULATION.max_timesteps
    end
end

@testset "Configuration Printing" begin
    @testset "Print Configuration Without Error" begin
        # Capture output
        original_stdout = stdout
        (read_pipe, write_pipe) = redirect_stdout()
        
        try
            Config.print_configuration()
            redirect_stdout(original_stdout)
            close(write_pipe)
            
            output = String(read(read_pipe))
            close(read_pipe)
            
            @test contains(output, "Agent Parameters")
            @test contains(output, "planning_horizon")
            @test contains(output, "Simulation Parameters")
            @test contains(output, "max_timesteps")
        catch e
            redirect_stdout(original_stdout)
            close(write_pipe)
            close(read_pipe)
            rethrow(e)
        end
    end
end

@testset "Configuration Edge Cases" begin
    @testset "Numerical Stability Parameters" begin
        @test Config.NUMERICAL.epsilon < Config.NUMERICAL.tolerance
        @test Config.NUMERICAL.clip_range[1] < Config.NUMERICAL.clip_range[2]
    end
    
    @testset "Precision Parameters Consistency" begin
        @test Config.AGENT.transition_precision > 0
        @test Config.AGENT.observation_precision > 0
        @test Config.AGENT.control_prior_precision > 0
        @test Config.AGENT.goal_prior_precision > 0
    end
    
    @testset "Path Validation" begin
        # Test that output directories are valid strings
        @test Config.OUTPUTS.base_dir isa String
        @test Config.OUTPUTS.results_dir isa String
        @test Config.OUTPUTS.logs_dir isa String
        @test Config.OUTPUTS.data_dir isa String
        @test Config.OUTPUTS.plots_dir isa String
        @test Config.OUTPUTS.animations_dir isa String
        @test Config.OUTPUTS.diagnostics_dir isa String
        @test Config.DIAGNOSTICS.diagnostics_dir isa String
        @test Config.VISUALIZATION.plots_dir isa String
        @test Config.VISUALIZATION.animations_dir isa String
    end
end

@testset "Configuration Performance" begin
    @testset "Validation Speed" begin
        # Test that validation is fast
        elapsed = @elapsed begin
            for _ in 1:1000
                Config.validate_config()
            end
        end
        @test elapsed < 1.0  # Should complete in less than 1 second
    end
    
    @testset "Configuration Access Speed" begin
        # Test that accessing config is fast
        elapsed = @elapsed begin
            for _ in 1:10000
                _ = Config.AGENT.planning_horizon
                _ = Config.SIMULATION.max_timesteps
                _ = Config.NUMERICAL.epsilon
            end
        end
        @test elapsed < 0.1  # Should be very fast
    end
end

println("✅ Configuration tests completed successfully")

