module Run

using RxInfer
using Logging
using Statistics

import ..Utils: HGFParams
import ..Model: hgf, hgfconstraints, hgfmeta, hgf_smoothing, hgfconstraints_smoothing, hgfmeta_smoothing

export run_filter, run_smoothing, run_hgf

function run_filter(y, real_k::Real, real_w::Real, z_var::Real, y_var::Real)
    @info "Running HGF filtering" k = real_k w = real_w
    autoupdates = @autoupdates begin
        z_prev_mean, z_prev_var = mean_var(q(z_next))
        x_prev_mean, x_prev_var = mean_var(q(x_next))
    end

    init = @initialization begin
        q(x_next) = NormalMeanVariance(0.0, 5.0)
        q(z_next) = NormalMeanVariance(0.0, 5.0)
    end

    result = infer(
        model          = hgf(κ = real_k, ω = real_w, z_variance = z_var, y_variance = y_var),
        constraints    = hgfconstraints(),
        meta           = hgfmeta(),
        data           = (y = y,),
        autoupdates    = autoupdates,
        keephistory    = length(y),
        historyvars    = (x_next = KeepLast(), z_next = KeepLast()),
        initialization = init,
        iterations     = 20,
        free_energy    = true,
    )
    return result
end

function run_smoothing(y, z_var::Real, y_var::Real)
    @info "Running HGF smoothing"
    init = @initialization function hgf_init_smoothing()
        q(x) = NormalMeanVariance(0.0, 5.0)
        q(z) = NormalMeanVariance(0.0, 5.0)
        q(κ) = NormalMeanVariance(1.5, 1.0)
        q(ω) = NormalMeanVariance(0.0, 0.05)
    end

    result = infer(
        model = hgf_smoothing(z_variance = z_var, y_variance = y_var),
        data = (y = y,),
        meta = hgfmeta_smoothing(),
        constraints = hgfconstraints_smoothing(),
        initialization = hgf_init_smoothing(),
        iterations = 50,
        options = (limit_stack_depth = 100,),
        returnvars = (x = KeepLast(), z = KeepLast(), ω = KeepLast(), κ = KeepLast()),
        free_energy = true,
    )
    return result
end

function run_hgf(params::HGFParams)
    return params # placeholder to keep cohesive API when used as a script
end

end # module


