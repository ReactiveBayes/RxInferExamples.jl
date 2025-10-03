using RxInfer
using ReactiveMP
using Distributions
using Statistics

const LOG2PI = log(2π)
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
        0.0
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
        try
            return mean(log, q)
        catch
            return log(max(mean(q), _EPS))
        end
    end
end

# Var[log(out)]
function _vlog_out(q)
    if q isa Distributions.LogNormal
        return q.σ^2
    elseif q isa ReactiveMP.PointMass
        return 0.0
    else
        try
            return var(log, q)
        catch
            return 0.0
        end
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

@node LogNormal Stochastic [out, (μ, aliases = [logmean]), (σ, aliases = [logscale])]

@functional LogNormal(out, μ, σ) = Distributions.LogNormal(μ, σ)

@average_energy LogNormal (q_out::Any, q_μ::Any, q_σ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    Einvσ2    = _e_inv_sigma2(q_σ)
    Elogσ     = _elog_sigma(q_σ)
    return 0.5 * LOG2PI + Elogσ + lout_mean + 0.5 * Einvσ2 * (lout_var + μ_var + (lout_mean - μ_mean)^2)
end

# Message to out: use means of μ and σ directly (standard parameterization)
@rule LogNormal(:out, Marginalisation) (q_μ::Any, q_σ::Any) = Distributions.LogNormal(mean(q_μ), mean(q_σ))

# Message to μ (logmean) -> NormalMeanVariance
@rule LogNormal(:μ, Marginalisation) (q_out::Any, q_σ::Any) = begin
    lout_mean = _elog_out(q_out)
    varμ = 1.0 / _e_inv_sigma2(q_σ)
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

@rule LogNormal(:μ, Marginalisation) (m_out::ReactiveMP.PointMass, q_σ::Any) = begin
    lout_mean = log(max(mean(m_out), _EPS))
    varμ = 1.0 / _e_inv_sigma2(q_σ)
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

# Message to σ (logscale) -> InverseGamma with mean equal to expected σ
@rule LogNormal(:σ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    D = lout_var + μ_var + (lout_mean - μ_mean)^2
    s = sqrt(max(D, 0.0) + _EPS)  # target mean σ
    α = 3.0
    β = (α - 1.0) * s             # ensures mean σ = β / (α - 1) = s
    Distributions.InverseGamma(α, β)
end

@rule LogNormal(:σ, Marginalisation) (m_out::ReactiveMP.PointMass, q_μ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = log(max(mean(m_out), _EPS))
    lout_var  = 0.0
    D = lout_var + μ_var + (lout_mean - μ_mean)^2
    s = sqrt(max(D, 0.0) + _EPS)
    α = 3.0
    β = (α - 1.0) * s
    Distributions.InverseGamma(α, β)
end
