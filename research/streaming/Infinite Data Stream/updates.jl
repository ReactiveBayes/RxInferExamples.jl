module InfiniteDataStreamUpdates

using RxInfer

export make_autoupdates, make_initialization

function make_autoupdates()
    return @autoupdates begin
        x_prev_mean, x_prev_var = mean_var(q(x_current))
        τ_shape = shape(q(τ))
        τ_rate = rate(q(τ))
    end
end

function make_initialization(; x0_mean::Float64 = 0.0, x0_var::Float64 = 1e3, τ_shape0::Float64 = 1.0, τ_rate0::Float64 = 1.0)
    return @initialization begin
        q(x_current) = NormalMeanVariance(x0_mean, x0_var)
        q(τ) = GammaShapeRate(τ_shape0, τ_rate0)
    end
end

end # module

