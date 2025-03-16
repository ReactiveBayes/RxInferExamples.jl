using ReactiveMP
using ExpectationApproximations, ExponentialFamily, BayesBase
import ReactiveMP: getmethod, DeltaMeta
ReactiveMP.is_delta_node_compatible(::GenUnscented) = Val(true)

ReactiveMP.deltafn_rule_layout(::DeltaFnNode, ::GenUnscented, inverse::Nothing)        = ReactiveMP.DeltaFnDefaultRuleLayout()
ReactiveMP.deltafn_rule_layout(::DeltaFnNode, ::GenUnscented, inverse::Function)        = ReactiveMP.DeltaFnDefaultKnownInverseRuleLayout()

# # unknown inverse
@rule ReactiveMP.DeltaFn((:in, k), Marginalisation) (q_ins::JointNormal, m_in::NormalDistributionsFamily, meta::DeltaMeta{M, I}) where {M <: GenUnscented, I <: Nothing} = begin
    # Divide marginal on inx by forward message
    ξ_inx, Λ_inx       = weightedmean_precision(component(q_ins, k))
    ξ_fw_inx, Λ_fw_inx = weightedmean_precision(m_in)
    ξ_bw_inx = ξ_inx - ξ_fw_inx
    Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations
    return convert(promote_variate_type(typeof(ξ_inx), NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
end

# known inverse, single input
@rule DeltaFn((:in, _), Marginalisation) (m_out::NormalDistributionsFamily, m_ins::Nothing, meta::DeltaMeta{M, I}) where {M <: GenUnscented, I <: Function} = begin
    return approximate(getmethod(meta), getnodefn(meta, Val(:in)), (m_out,))
end

@rule DeltaFn((:in, k), Marginalisation) (
    m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M, I}
) where {N, M <: GenUnscented, L, I <: NTuple{L, Function}} = begin
    return approximate(getmethod(meta), getnodefn(meta, Val(:in), k), (m_out, m_ins...))
end



@marginalrule ReactiveMP.DeltaFn(:ins) (m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: GenUnscented} = begin
    # Approximate joint inbounds
    statistics = mean_cov.(m_ins)
    μs_fw_in = first.(statistics)
    Σs_fw_in = last.(statistics)
    sizes = size.(m_ins)
    (μ_tilde, Σ_tilde,_, _, C_tilde) = ExpectationApproximations.unscented_statistics(getmethod(meta),m_ins, getnodefn(meta, Val(:out)), Val(true))

    joint              = convert(JointNormal, μs_fw_in, Σs_fw_in)
    (μ_fw_in, Σ_fw_in) = mean_cov(joint)
    ds                 = ExponentialFamily.dimensionalities(joint)

    # Apply the RTS smoother
    (μ_bw_out, Σ_bw_out) = mean_cov(m_out)

    (μ_in, Σ_in)         = ReactiveMP.smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)

    dist = convert(promote_variate_type(typeof(μ_in), NormalMeanVariance), μ_in, Σ_in)
    return JointNormal(dist, sizes)
end


@rule ReactiveMP.DeltaFn(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: GenUnscented} = begin
    res = approximate(getmethod(meta), getnodefn(meta, Val(:out)), m_ins)
    return res
end