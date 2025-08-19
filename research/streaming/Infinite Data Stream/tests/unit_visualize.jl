using Test

@testset "visualize" begin
    # Explicitly load only what's needed to avoid module conflicts
    include(joinpath(@__DIR__, "..", "utils.jl"))
    using .InfiniteDataStreamUtils
    InfiniteDataStreamUtils.load_modules!()
    using .InfiniteDataStreamViz
    
    println("Testing plot_estimates...")
    μ = collect(1.0:10.0)
    σ2 = fill(0.5, 10)
    hist = sin.(range(0, step=0.1, length=10)) .* 10
    obs = hist .+ 0.1
    p = plot_estimates(μ, σ2, hist, obs; upto=10)
    @test !isnothing(p)
    
    println("Testing animate_estimates...")
    # Skip animation tests in CI/headless environments
    if get(ENV, "SKIP_ANIMATION_TESTS", "0") == "1"
        println("  Skipping animation tests (SKIP_ANIMATION_TESTS=1)")
        @test true # Dummy test to avoid empty testset
    else
        # Use minimal data and large stride to avoid hanging
        println("  Creating minimal animation with 2 frames...")
        anim = animate_estimates(μ[1:5], σ2[1:5], hist[1:5], obs[1:5]; stride=4)
        @test !isnothing(anim)
        
        println("Testing animate_free_energy...")
        println("  Creating minimal FE animation with 2 frames...")
        anim_fe = animate_free_energy([1.0, 2.0, 3.0]; stride=2)
        @test !isnothing(anim_fe)
    end
    
    println("Visualize tests completed successfully")
end


