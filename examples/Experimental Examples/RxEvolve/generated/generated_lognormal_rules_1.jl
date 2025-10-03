using RxInfer
using ReactiveMP
using Distributions
using Statistics
using StatsFuns

const LOG2PI = log(2*pi)

# Expected value of log(X)
function _elog_out(q)
    try
        return mean(log, q)
    catch
        if q isa Distributions.LogNormal
            return q.μ
        elseif q isa ReactiveMP.PointMass
            return log(mean(q))
        else
            return log(mean(q))
        end
    end
end

# Variance of log(X)
function _vlog_out(q)
    try
        return var(log, q)
    catch
        if q isa Distributions.LogNormal
            return q.σ^2
        elseif q isa ReactiveMP.PointMass
            return 0.0
        else
            return 0.0
        end
    end
end

# Expected log of σ
function _elog_sigma(q)
    if q isa Distributions.InverseGamma
        α = Distributions.shape(q)
        β = Distributions.scale(q)
        return log(β) - StatsFuns.digamma(α)
    elseif q isa ReactiveMP.PointMass
        return log(mean(q))
    else
        return log(mean(q))
    end
end

# Expected value of 1/σ^2
function _e_inv_sigma2(q)
    if q isa Distributions.InverseGamma
        α = Distributions.shape(q)
        β = Distributions.scale(q)
        return (α * (α + 1)) / (β^2)
    elseif q isa ReactiveMP.PointMass
        s = mean(q)
        return 1 / (s^2)
    else
        m = mean(q)
        return 1 / (m^2 + eps(Float64))
    end
end

# Safe mean and variance for μ
function _mean_var_mu(q)
    return (mean(q), try var(q) catch; 0.0 end)
end

@node LogNormal Stochastic [out, (μ, aliases = [logmean]), (σ, aliases = [logscale])]

@average_energy LogNormal (q_out::Any, q_μ::Any, q_σ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    Einvσ2    = _e_inv_sigma2(q_σ)
    return 0.5 * LOG2PI + _elog_sigma(q_σ) + lout_mean + 0.5 * Einvσ2 * (lout_var + μ_var + (lout_mean - μ_mean)^2)
end

# Message to out (uses means of parameters)
@rule LogNormal(:out, Marginalisation) (q_μ::ReactiveMP.PointMass, q_σ::ReactiveMP.PointMass) = begin
    Distributions.LogNormal(mean(q_μ), mean(q_σ))
end

@rule LogNormal(:out, Marginalisation) (q_μ::Any, q_σ::Any) = begin
    Distributions.LogNormal(mean(q_μ), mean(q_σ))
end

# Message to μ (logmean) -> NormalMeanVariance
@rule LogNormal(:μ, Marginalisation) (m_out::ReactiveMP.PointMass, q_σ::Any) = begin
    lout_mean = log(mean(m_out))
    Einvσ2    = _e_inv_sigma2(q_σ)
    varμ = 1 / Einvσ2
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

@rule LogNormal(:μ, Marginalisation) (q_out::Any, q_σ::Any) = begin
    lout_mean = _elog_out(q_out)
    Einvσ2    = _e_inv_sigma2(q_σ)
    varμ = 1 / Einvσ2
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

# Message to σ (logscale) -> InverseGamma
@rule LogNormal(:σ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    D = lout_var + μ_var + (lout_mean - μ_mean)^2
    α = 1.5
    β = (α - 1.0) * sqrt(max(D, 0.0) + eps(Float64))
    Distributions.InverseGamma(α, β)
end
