module InfiniteDataStreamUtils

export load_modules!

function load_modules!()
    # Load implementation files into Main so they are available as Main.InfiniteDataStream*
    thisdir = @__DIR__
    Base.include(Main, joinpath(thisdir, "model.jl"))
    Base.include(Main, joinpath(thisdir, "environment.jl"))
    Base.include(Main, joinpath(thisdir, "updates.jl"))
    Base.include(Main, joinpath(thisdir, "streams.jl"))
    Base.include(Main, joinpath(thisdir, "visualize.jl"))
end

end # module

