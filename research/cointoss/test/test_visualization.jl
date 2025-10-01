#!/usr/bin/env julia
# Comprehensive tests for CoinTossVisualization module

using Test
using Plots
using Distributions
using Random
using Logging

# Setup test logging
test_logger = ConsoleLogger(stderr, Logging.Info)
global_logger(test_logger)

@info "Starting CoinTossVisualization tests"

@testset "CoinTossVisualization Module Tests" begin
    
    @testset "Theme Colors - All Themes" begin
        @info "Testing theme color schemes"
        
        # Test default theme
        default_colors = get_theme_colors("default")
        @test haskey(default_colors, :background)
        @test haskey(default_colors, :prior)
        @test haskey(default_colors, :posterior)
        @test haskey(default_colors, :data)
        @test haskey(default_colors, :true_value)
        @test haskey(default_colors, :grid)
        
        # Test dark theme
        dark_colors = get_theme_colors("dark")
        @test haskey(dark_colors, :background)
        @test dark_colors[:background] == "#2b2b2b"
        
        # Test colorblind theme
        cb_colors = get_theme_colors("colorblind")
        @test haskey(cb_colors, :prior)
        @test cb_colors[:background] == "#ffffff"
        
        # Test unknown theme defaults to "default"
        unknown_colors = get_theme_colors("unknown_theme")
        @test unknown_colors == default_colors
        
        @info "Theme colors tests passed"
    end
    
    @testset "Plot Prior-Posterior - Basic" begin
        @info "Testing prior-posterior plotting"
        
        prior = Beta(2, 2)
        posterior = Beta(10, 5)
        
        p = plot_prior_posterior(prior, posterior; theta_real=0.7)
        
        @test p isa Plots.Plot
        @test length(p.series_list) >= 2  # At least prior and posterior
        
        @info "Basic prior-posterior plotting tests passed"
    end
    
    @testset "Plot Prior-Posterior - Various Configurations" begin
        @info "Testing prior-posterior with various configurations"
        
        prior = Beta(1, 1)
        posterior = Beta(20, 10)
        
        # Without true value
        p1 = plot_prior_posterior(prior, posterior)
        @test p1 isa Plots.Plot
        
        # With true value
        p2 = plot_prior_posterior(prior, posterior; theta_real=0.65)
        @test p2 isa Plots.Plot
        
        # Different themes
        p3 = plot_prior_posterior(prior, posterior; theme="dark")
        @test p3 isa Plots.Plot
        
        p4 = plot_prior_posterior(prior, posterior; theme="colorblind")
        @test p4 isa Plots.Plot
        
        # Different resolutions
        p5 = plot_prior_posterior(prior, posterior; resolution=500)
        @test p5 isa Plots.Plot
        
        p6 = plot_prior_posterior(prior, posterior; resolution=2000)
        @test p6 isa Plots.Plot
        
        @info "Various prior-posterior configurations tests passed"
    end
    
    @testset "Plot Convergence - Basic" begin
        @info "Testing convergence plotting"
        
        free_energy = [-100.0, -95.0, -92.0, -91.0, -90.5, -90.3, -90.2]
        
        p = plot_convergence(free_energy)
        
        @test p isa Plots.Plot
        @test length(p.series_list) >= 1
        
        @info "Basic convergence plotting tests passed"
    end
    
    @testset "Plot Convergence - Various Cases" begin
        @info "Testing convergence plotting with various cases"
        
        # Short trace
        fe_short = [-50.0, -45.0, -43.0]
        p1 = plot_convergence(fe_short)
        @test p1 isa Plots.Plot
        
        # Long trace
        fe_long = collect(range(-100, -90, length=50))
        p2 = plot_convergence(fe_long)
        @test p2 isa Plots.Plot
        
        # Different themes
        p3 = plot_convergence(fe_long; theme="dark")
        @test p3 isa Plots.Plot
        
        @info "Various convergence plotting tests passed"
    end
    
    @testset "Plot Data Histogram - Basic" begin
        @info "Testing data histogram plotting"
        
        data = vcat(ones(70), zeros(30))  # 70 heads, 30 tails
        
        p = plot_data_histogram(data; theta_real=0.7)
        
        @test p isa Plots.Plot
        
        @info "Basic data histogram tests passed"
    end
    
    @testset "Plot Data Histogram - Edge Cases" begin
        @info "Testing data histogram edge cases"
        
        # All heads
        data_heads = ones(100)
        p1 = plot_data_histogram(data_heads; theta_real=1.0)
        @test p1 isa Plots.Plot
        
        # All tails
        data_tails = zeros(100)
        p2 = plot_data_histogram(data_tails; theta_real=0.0)
        @test p2 isa Plots.Plot
        
        # Balanced data
        data_balanced = vcat(ones(50), zeros(50))
        p3 = plot_data_histogram(data_balanced; theta_real=0.5)
        @test p3 isa Plots.Plot
        
        # Without true value
        p4 = plot_data_histogram(data_balanced)
        @test p4 isa Plots.Plot
        
        @info "Data histogram edge cases tests passed"
    end
    
    @testset "Plot Credible Interval - Basic" begin
        @info "Testing credible interval plotting"
        
        posterior = Beta(10, 5)
        
        p = plot_credible_interval(posterior; level=0.95, theta_real=0.65)
        
        @test p isa Plots.Plot
        
        @info "Basic credible interval plotting tests passed"
    end
    
    @testset "Plot Credible Interval - Various Levels" begin
        @info "Testing credible interval with various levels"
        
        posterior = Beta(20, 10)
        
        # Different credible levels
        p1 = plot_credible_interval(posterior; level=0.90)
        @test p1 isa Plots.Plot
        
        p2 = plot_credible_interval(posterior; level=0.95)
        @test p2 isa Plots.Plot
        
        p3 = plot_credible_interval(posterior; level=0.99)
        @test p3 isa Plots.Plot
        
        # With and without true value
        p4 = plot_credible_interval(posterior; level=0.95, theta_real=0.67)
        @test p4 isa Plots.Plot
        
        p5 = plot_credible_interval(posterior; level=0.95)
        @test p5 isa Plots.Plot
        
        @info "Various credible level tests passed"
    end
    
    @testset "Plot Predictive - Basic" begin
        @info "Testing predictive plotting"
        
        posterior = Beta(10, 5)
        observed_data = ones(10)
        
        p = plot_predictive(posterior, observed_data)
        
        @test p isa Plots.Plot
        
        @info "Basic predictive plotting tests passed"
    end
    
    @testset "Plot Predictive - Various Scenarios" begin
        @info "Testing predictive plotting with various scenarios"
        
        posterior = Beta(15, 10)
        
        # Different observed data patterns
        data_heads = ones(20)
        p1 = plot_predictive(posterior, data_heads)
        @test p1 isa Plots.Plot
        
        data_tails = zeros(20)
        p2 = plot_predictive(posterior, data_tails)
        @test p2 isa Plots.Plot
        
        data_balanced = vcat(ones(10), zeros(10))
        p3 = plot_predictive(posterior, data_balanced)
        @test p3 isa Plots.Plot
        
        # Different sample sizes for prediction
        p4 = plot_predictive(posterior, data_balanced; n_samples=5000)
        @test p4 isa Plots.Plot
        
        # Different themes
        p5 = plot_predictive(posterior, data_balanced; theme="dark")
        @test p5 isa Plots.Plot
        
        @info "Various predictive plotting tests passed"
    end
    
    @testset "Comprehensive Dashboard - Basic" begin
        @info "Testing comprehensive dashboard creation"
        
        prior = Beta(2, 2)
        posterior = Beta(10, 5)
        data = ones(8)
        free_energy = [-100.0, -95.0, -92.0, -91.0, -90.5]
        
        dashboard = plot_comprehensive_dashboard(
            prior, posterior, data, free_energy;
            theta_real=0.7
        )
        
        @test dashboard isa Plots.Plot
        @test length(dashboard.subplots) >= 4  # Should have multiple subplots
        
        @info "Basic comprehensive dashboard tests passed"
    end
    
    @testset "Comprehensive Dashboard - Without Free Energy" begin
        @info "Testing dashboard without free energy"
        
        prior = Beta(2, 2)
        posterior = Beta(10, 5)
        data = ones(8)
        
        dashboard = plot_comprehensive_dashboard(
            prior, posterior, data, nothing;
            theta_real=0.7
        )
        
        @test dashboard isa Plots.Plot
        @test length(dashboard.subplots) >= 4  # Should have 4 subplots
        
        @info "Dashboard without free energy tests passed"
    end
    
    @testset "Comprehensive Dashboard - Various Themes" begin
        @info "Testing dashboard with various themes"
        
        prior = Beta(1, 1)
        posterior = Beta(20, 10)
        data = vcat(ones(15), zeros(5))
        free_energy = collect(range(-100, -90, length=10))
        
        # Default theme
        d1 = plot_comprehensive_dashboard(prior, posterior, data, free_energy)
        @test d1 isa Plots.Plot
        
        # Dark theme
        d2 = plot_comprehensive_dashboard(prior, posterior, data, free_energy; theme="dark")
        @test d2 isa Plots.Plot
        
        # Colorblind theme
        d3 = plot_comprehensive_dashboard(prior, posterior, data, free_energy; theme="colorblind")
        @test d3 isa Plots.Plot
        
        @info "Dashboard theme tests passed"
    end
    
    @testset "Animation Creation - Basic" begin
        @info "Testing animation creation"
        
        data = generate_coin_data(n=500, theta_real=0.65, seed=42).observations
        sample_sizes = [10, 50, 100, 200, 500]
        
        anim = create_inference_animation(
            data, 2.0, 2.0, sample_sizes;
            theta_real=0.65, fps=10
        )
        
        @test anim isa Plots.Animation
        
        @info "Basic animation creation tests passed"
    end
    
    @testset "Animation Creation - Various Configurations" begin
        @info "Testing animation with various configurations"
        
        data = generate_coin_data(n=300, theta_real=0.7, seed=123).observations
        
        # Different sample size sequences
        sizes1 = [20, 100, 300]
        anim1 = create_inference_animation(data, 1.0, 1.0, sizes1; theta_real=0.7)
        @test anim1 isa Plots.Animation
        
        # Many frames
        sizes2 = [10, 25, 50, 75, 100, 150, 200, 250, 300]
        anim2 = create_inference_animation(data, 2.0, 2.0, sizes2; fps=15)
        @test anim2 isa Plots.Animation
        
        # Different themes
        anim3 = create_inference_animation(data, 1.0, 1.0, sizes1; theme="dark")
        @test anim3 isa Plots.Animation
        
        # Without true value
        anim4 = create_inference_animation(data, 1.0, 1.0, sizes1)
        @test anim4 isa Plots.Animation
        
        @info "Various animation configuration tests passed"
    end
    
    @testset "Save Plot Function" begin
        @info "Testing save_plot function"
        
        # Create a simple plot
        p = plot([1, 2, 3], [1, 4, 9])
        
        # Create temp directory for test
        temp_dir = mktempdir()
        filepath = joinpath(temp_dir, "test_plot.png")
        
        # Save plot
        save_plot(p, filepath)
        
        # Check file exists
        @test isfile(filepath)
        
        # Test saving to nested directory (should create directories)
        nested_path = joinpath(temp_dir, "subdir", "nested", "plot.png")
        save_plot(p, nested_path)
        @test isfile(nested_path)
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "Save plot function tests passed"
    end
    
    @testset "Save Plot - Edge Cases" begin
        @info "Testing save plot edge cases"
        
        p = plot([1, 2, 3], [1, 2, 3])
        
        temp_dir = mktempdir()
        
        # Test different file formats
        for ext in [".png", ".pdf", ".svg"]
            filepath = joinpath(temp_dir, "plot$(ext)")
            save_plot(p, filepath)
            @test isfile(filepath)
        end
        
        # Test with existing directory
        existing_dir = joinpath(temp_dir, "existing")
        mkpath(existing_dir)
        filepath = joinpath(existing_dir, "plot.png")
        save_plot(p, filepath)
        @test isfile(filepath)
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "Save plot edge cases tests passed"
    end
    
    @testset "Plot Integration - Full Workflow" begin
        @info "Testing full visualization workflow"
        
        # Generate data and run inference
        data = generate_coin_data(n=200, theta_real=0.7, seed=999).observations
        result = run_inference(data, 2.0, 2.0; iterations=10, track_fe=true, showprogress=false)
        
        # Create all plot types
        p1 = plot_prior_posterior(result.prior, result.posterior; theta_real=0.7)
        @test p1 isa Plots.Plot
        
        p2 = plot_data_histogram(data; theta_real=0.7)
        @test p2 isa Plots.Plot
        
        p3 = plot_credible_interval(result.posterior; theta_real=0.7)
        @test p3 isa Plots.Plot
        
        p4 = plot_predictive(result.posterior, data)
        @test p4 isa Plots.Plot
        
        p5 = plot_convergence(result.free_energy)
        @test p5 isa Plots.Plot
        
        # Create dashboard
        dashboard = plot_comprehensive_dashboard(
            result.prior, result.posterior, data, result.free_energy;
            theta_real=0.7
        )
        @test dashboard isa Plots.Plot
        
        @info "Full visualization workflow tests passed"
    end
    
    @testset "Visualization Consistency" begin
        @info "Testing visualization consistency across themes"
        
        prior = Beta(2, 2)
        posterior = Beta(10, 5)
        data = ones(8)
        
        themes = ["default", "dark", "colorblind"]
        
        for theme in themes
            p = plot_prior_posterior(prior, posterior; theta_real=0.7, theme=theme)
            @test p isa Plots.Plot
            
            # All plots should be creatable with all themes
            @test_nowarn plot_data_histogram(data; theme=theme)
            @test_nowarn plot_credible_interval(posterior; theme=theme)
            @test_nowarn plot_predictive(posterior, data; theme=theme)
        end
        
        @info "Visualization consistency tests passed"
    end
    
end

@info "All CoinTossVisualization tests completed successfully"

