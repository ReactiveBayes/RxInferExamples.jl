using Test

@testset "model and updates" begin
    # Explicitly load only what's needed to avoid module conflicts
    include(joinpath(@__DIR__, "..", "utils.jl"))
    using .InfiniteDataStreamUtils
    InfiniteDataStreamUtils.load_modules!()
    using .InfiniteDataStreamModel
    using .InfiniteDataStreamUpdates
    using .InfiniteDataStreamStreams
    
    println("Testing autoupdates and initialization...")
    au = make_autoupdates()
    init = make_initialization()
    @test !isnothing(au)
    @test !isnothing(init)
    
    # simple smoke: build infer with tiny stream
    println("Testing inference with tiny stream...")
    using RxInfer
    data = [0.1, 0.2, 0.3]
    ds = to_namedtuple_stream(data)
    eng = infer(
        model          = kalman_filter(),
        constraints    = filter_constraints(),
        datastream     = ds,
        autoupdates    = au,
        returnvars     = (:x_current, ),
        keephistory    = 100,
        historyvars    = (x_current = KeepLast(),),
        initialization = init,
        iterations     = 2,
        autostart      = true,
        free_energy    = true,
    )
    @test eng.history !== nothing && haskey(eng.history, :x_current)
    @test !isempty(eng.free_energy_history)
    println("Model and updates tests completed successfully")
end


