using RxInfer
using ReactiveMP
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

# Safe mean and variance extraction
function _mean_var_mu(q)
    m = mean(q)
    v = try
        var(q)
    catch
        zero(float(m))
    end
    return m, v
end

# Approximate E[log(out)]
function _elog_out(q)
    if q isa Distributions.LogNormal
        return q.μ
    elseif q isa ReactiveMP.PointMass
        return log(max(mean(q), _EPS))
    else
        m = try
            mean(q)
        catch
            return log(_EPS)
        end
        v = try
            var(q)
        catch
            0.0
        end
        mpos = max(m, _EPS)
        s2 = log(1 + max(v, 0.0) / (mpos * mpos))
        return log(mpos) - 0.5 * s2
    end
end

# Approximate Var[log(out)]
function _vlog_out(q)
    if q isa Distributions.LogNormal
        return q.σ^2
    elseif q isa ReactiveMP.PointMass
        return 0.0
    else
        m = try
            mean(q)
        catch
            return 0.0
        end
        v = try
            var(q)
        catch
            0.0
        end
        mpos = max(m, _EPS)
        return log(1 + max(v, 0.0) / (mpos * mpos))
    end
end

# E[log(σ)] for σ ~ InverseGamma(α, β)
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

@average_energy LogNormal (q_out::Any, q_μ::Any, q_σ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    Einvσ2    = _e_inv_sigma2(q_σ)
    Elogσ     = _elog_sigma(q_σ)
    # E[-log p(out | μ, σ)] = log σ + 0.5 log(2π) + E[log out] + 0.5 * E[1/σ^2] * E[(log out - μ)^2]
    return 0.5 * LOG2PI + Elogσ + lout_mean + 0.5 * Einvσ2 * (lout_var + μ_var + (lout_mean - μ_mean)^2)
end

# Message to out: use means of μ and σ directly (standard parameterization)
@rule LogNormal(:out, Marginalisation) (q_μ::Any, q_σ::Any) = Distributions.LogNormal(mean(q_μ), mean(q_σ))

# Message to μ (logmean) -> NormalMeanVariance
@rule LogNormal(:μ, Marginalisation) (q_out::Any, q_σ::Any) = begin
    lout_mean = _elog_out(q_out)
    varμ = inv(max(_e_inv_sigma2(q_σ), _EPS))
    ReactiveMP.NormalMeanVariance(lout_mean, varμ)
end

# Message to σ (logscale) -> InverseGamma
# Use target E[σ] ≈ sqrt(E[(log out - μ)^2]), and set a relatively large α to align E[1/σ^2] ≈ 1 / E[(log out - μ)^2]
@rule LogNormal(:σ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    μ_mean, μ_var = _mean_var_mu(q_μ)
    lout_mean = _elog_out(q_out)
    lout_var  = _vlog_out(q_out)
    D = lout_var + μ_var + (lout_mean - μ_mean)^2
    s = sqrt(max(D, 0.0) + _EPS)                # target E[σ]
    α = convert(typeof(s), 30.0)                # large α makes E[1/σ^2] ≈ 1 / s^2
    β = (α - one(α)) * s                        # ensures mean σ = β / (α - 1) = s
    Distributions.InverseGamma(α, β)
end
