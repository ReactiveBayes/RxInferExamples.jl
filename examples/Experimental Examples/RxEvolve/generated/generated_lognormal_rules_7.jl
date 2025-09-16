using RxInfer
using ReactiveMP
using GraphPPL
using Distributions
using Statistics

const LOG2PI = log(2 * pi)
const _EPS   = eps(Float64)

# Digamma approximation for positive arguments
function _digamma_pos(x::Real)
    xx = float(x)
    res = 0.0
    while xx < 6.0
        res -= 1.0 / xx
        xx += 1.0
    end
    inv = 1.0 / xx
    inv2 = inv * inv
    return res + log(xx) - 0.5 * inv - inv2 * (1/12 - inv2 * (1/120 - inv2 * (1/252)))
end

# Safe mean and variance extraction for μ
function _mean_var_mu(q)
    m = mean(q)
    v = try
        var(q)
    catch
        zero(float(m))
    end
    return m, v
end

# E[log(out)]
function _elog_out(q)
    if q isa Distributions.LogNormal
        return q.μ
    elseif q isa ReactiveMP.PointMass
        return log(max(mean(q), _EPS))
    else
        return log(max(mean(q), _EPS))
    end
end

# Var[log(out)]
function _vlog_out(q)
    if q isa Distributions.LogNormal
        return q.σ^2
    elseif q isa ReactiveMP.PointMass
        return 0.0
    else
        return 0.0
    end
end

# E[log(σ)]
function _elog_sigma(q)
    if q isa Distributions.InverseGamma
        α = Distributions.shape(q)
        β = Distributions.scale(q)
        return log(β) - _digamma_pos(α)
    elseif q isa ReactiveMP.PointMass
        return log(max(mean(q), _EPS))
    else
        return log(max(mean(q), _EPS))
    end
end

# E[1/σ^2] for σ ~ InverseGamma(α, β)
function _e_inv_sigma2(q)
    if q isa Distributions.InverseGamma
        α = Distributions.shape(q)
        β = Distributions.scale(q)
        return (α * (α + 1.0)) / (β^2)
    elseif q isa ReactiveMP.PointMass
        s = max(mean(q), _EPS)
        return 1.0 / (s^2)
    else
        s = max(mean(q), _EPS)
        return 1.0 / (s^2)
    end
end

ReactiveMP.@node LogNormal Stochastic [out, (μ, aliases = [logmean]), (σ, aliases = [logscale])]

ReactiveMP.@functional LogNormal(out, μ, σ) = Distributions.LogNormal(μ, σ)

ReactiveMP.@average_energy LogNormal (q_out::Any, q_μ::Any, q_σ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    Einvσ2    = _e_inv_sigma2(q_σ)
    Elogσ     = _elog_sigma(q_σ)
    # E[-log p(out | μ, σ)] = log σ + 0.5 log(2π) + E[log out] + 0.5 * E[1/σ^2] * E[(log out - μ)^2]
    return 0.5 * LOG2PI + Elogσ + lout_mean + 0.5 * Einvσ2 * (lout_var + μ_var + (lout_mean - μ_mean)^2)
end

# Message to out: use means of μ and σ directly (standard parameterization)
ReactiveMP.@rule LogNormal(:out, Marginalisation) (q_μ::Any, q_σ::Any) = Distributions.LogNormal(mean(q_μ), mean(q_σ))

# Message to μ (logmean) -> NormalMeanVariance
ReactiveMP.@rule LogNormal(:μ, Marginalisation) (q_out::Any, q_σ::Any) = begin
    lout_mean = _elog_out(q_out)
    varμ = inv(_e_inv_sigma2(q_σ))
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

# Message to σ (logscale) -> InverseGamma with mean equal to expected σ
ReactiveMP.@rule LogNormal(:σ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    D = lout_var + μ_var + (lout_mean - μ_mean)^2  # E[(log out - μ)^2]
    s = sqrt(max(D, 0.0) + _EPS)                   # target E[σ]
    α = convert(typeof(s), 3.0)
    β = (α - one(α)) * s                           # ensures mean σ = β / (α - 1) = s
    Distributions.InverseGamma(α, β)
end
