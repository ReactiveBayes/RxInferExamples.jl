module Model

using RxInfer
using Distributions
using Logging

export hgf, hgfconstraints, hgfmeta, hgf_smoothing, hgfconstraints_smoothing, hgfmeta_smoothing

@model function hgf(y, κ, ω, z_variance, y_variance, z_prev_mean, z_prev_var, x_prev_mean, x_prev_var)
    z_prev ~ Normal(mean = z_prev_mean, variance = z_prev_var)
    x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
    z_next ~ Normal(mean = z_prev, variance = z_variance)
    x_next ~ GCV(x_prev, z_next, κ, ω)
    y ~ Normal(mean = x_next, variance = y_variance)
end

@constraints function hgfconstraints()
    q(x_next, x_prev, z_next) = q(x_next)q(x_prev)q(z_next)
end

@meta function hgfmeta()
    GCV() -> GCVMetadata(GaussHermiteCubature(31))
end

@model function hgf_smoothing(y, z_variance, y_variance)
    z_prev ~ Normal(mean = 0.0, variance = 5.0)
    x_prev ~ Normal(mean = 0.0, variance = 5.0)
    κ ~ Normal(mean = 1.5, variance = 1.0)
    ω ~ Normal(mean = 0.0, variance = 0.05)
    for i in eachindex(y)
        z[i] ~ Normal(mean = z_prev, variance = z_variance)
        x[i] ~ GCV(x_prev, z[i], κ, ω)
        y[i] ~ Normal(mean = x[i], variance = y_variance)
        z_prev = z[i]
        x_prev = x[i]
    end
end

@constraints function hgfconstraints_smoothing()
    q(x_prev, x, z, κ, ω) = q(x_prev, x)q(z)q(κ)q(ω)
end

@meta function hgfmeta_smoothing()
    GCV() -> GCVMetadata(GaussHermiteCubature(31))
end

end # module


