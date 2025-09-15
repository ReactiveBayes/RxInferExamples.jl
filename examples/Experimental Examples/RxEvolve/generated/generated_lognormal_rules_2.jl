using RxInfer
using ReactiveMP
using Distributions
using Statistics

const LOG2PI = log(2*pi)

# Digamma approximation (for positive arguments)
function _digamma(x::Real)
    xx = float(x)
    result = 0.0
    while xx < 6.0
        result -= 1.0/xx
        xx += 1.0
    end
    inv = 1.0/xx
    inv2 = inv*inv
    result + log(xx) - 0.5*inv - (1/12)*inv2 + (1/120)*inv2^2 - (1/252)*inv2^3
end

# Safe mean/var for mu
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
        return log(max(mean(q), eps(Float64)))
    else
        try
            return mean(log, q)
        catch
            return log(max(mean(q), eps(Float64)))
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
        return log(β) - _digamma(α)
    elseif q isa ReactiveMP.PointMass
        return log(max(mean(q), eps(Float64)))
    else
        try
            return mean(log, q)
        catch
            return log(max(mean(q), eps(Float64)))
        end
    end
end

# E[1/σ^2]
function _e_inv_sigma2(q)
    if q isa Distributions.InverseGamma
        α = Distributions.shape(q)
        β = Distributions.scale(q)
        return (α * (α + 1.0)) / (β^2)
    elseif q isa ReactiveMP.PointMass
        s = mean(q)
        return 1.0 / (s^2)
    else
        m = mean(q)
        return 1.0 / (m^2 + eps(Float64))
    end
end

@node LogNormal Stochastic [out, (μ, aliases = [logmean]), (σ, aliases = [logscale])]

@average_energy LogNormal (q_out::Any, q_μ::Any, q_σ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    Einvσ2    = _e_inv_sigma2(q_σ)
    Elogσ     = _elog_sigma(q_σ)
    return 0.5 * LOG2PI + Elogσ + lout_mean + 0.5 * Einvσ2 * (lout_var + μ_var + (lout_mean - μ_mean)^2)
end

# Message to out
@rule LogNormal(:out, Marginalisation) (q_μ::ReactiveMP.PointMass, q_σ::ReactiveMP.PointMass) = begin
    Distributions.LogNormal(mean(q_μ), mean(q_σ))
end

@rule LogNormal(:out, Marginalisation) (q_μ::Any, q_σ::Any) = begin
    Distributions.LogNormal(mean(q_μ), mean(q_σ))
end

# Message to μ (logmean) -> NormalMeanVariance
@rule LogNormal(:μ, Marginalisation) (m_out::ReactiveMP.PointMass, q_σ::Any) = begin
    lout_mean = log(max(mean(m_out), eps(Float64)))
    varμ = 1.0 / _e_inv_sigma2(q_σ)
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

@rule LogNormal(:μ, Marginalisation) (q_out::Any, q_σ::Any) = begin
    lout_mean = _elog_out(q_out)
    varμ = 1.0 / _e_inv_sigma2(q_σ)
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

# Message to σ (logscale) -> InverseGamma (moment-matched)
@rule LogNormal(:σ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    D = lout_var + μ_var + (lout_mean - μ_mean)^2
    α = 1.5
    s = sqrt(max(D, 0.0) + eps(Float64))
    β = (α - 1.0) * s
    Distributions.InverseGamma(α, β)
end

@rule LogNormal(:σ, Marginalisation) (m_out::ReactiveMP.PointMass, q_μ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = log(max(mean(m_out), eps(Float64)))
    lout_var  = 0.0
    D = lout_var + μ_var + (lout_mean - μ_mean)^2
    α = 1.5
    s = sqrt(max(D, 0.0) + eps(Float64))
    β = (α - 1.0) * s
    Distributions.InverseGamma(α, β)
end
