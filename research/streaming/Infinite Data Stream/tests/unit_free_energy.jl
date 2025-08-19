using Test
using DelimitedFiles

project_root = normpath(joinpath(@__DIR__, ".."))

@testset "free energy per-timestep (static)" begin
    println("Testing free energy per-timestep computation...")
    # Create a minimal test environment
    include(joinpath(@__DIR__, "..", "utils.jl"))
    using .InfiniteDataStreamUtils
    InfiniteDataStreamUtils.load_modules!()
    
    println("  Loading modules...")
    using .InfiniteDataStreamModel
    using .InfiniteDataStreamEnv
    using .InfiniteDataStreamUpdates
    using .InfiniteDataStreamStreams
    using RxInfer # Explicitly import RxInfer
    
    # Create a small test environment
    println("  Creating test environment...")
    env = Environment(0.0, 0.1)
    n_test = 15 # Smaller test size
    for _ in 1:n_test
        getnext!(env)
    end
    
    # Get observations and create datastream
    observations = getobservations(env)
    
    # Compute FE for a few timesteps
    println("  Computing FE for $(length(1:5:n_test)) timesteps...")
    fe_series = Float64[]
    for t in 1:5:n_test
        println("    Processing timestep $t/$n_test")
        ds_t = to_namedtuple_stream(observations[1:t])
        au = make_autoupdates()
        init = make_initialization()
        eng_t = infer(
            model          = kalman_filter(),
            constraints    = filter_constraints(),
            datastream     = ds_t,
            autoupdates    = au,
            returnvars     = (:x_current,),
            initialization = init,
            iterations     = 2, # Fewer iterations
            free_energy    = true,
            keephistory    = 1,
            autostart      = true,
        )
        push!(fe_series, eng_t.free_energy_history[end])
    end
    
    @test length(fe_series) == length(1:5:n_test)
    @test all(isfinite, fe_series)
    println("Free energy tests completed successfully")
end


