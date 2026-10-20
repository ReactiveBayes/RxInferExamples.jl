using RxInfer, BayesBase

# `CVIProjection` leaves its message towards an input unevaluated: the projected marginal of the
# input divided by the message that arrived from it, a `DivisionOf`. Where that message meets the
# Gaussian messages on the input's other edges, their product is Gaussian, and this form
# constraint computes it. In weighted-mean/precision form each factor adds its parameters and a
# division subtracts its denominator's, so only the product as a whole needs to be proper.
struct CallClosedProd <: AbstractFormConstraint end

ReactiveMP.default_prod_constraint(::CallClosedProd) = GenericProd()
ReactiveMP.default_form_check_strategy(::CallClosedProd) = FormConstraintCheckLast()

const DivisionOf = Base.get_extension(DeltaMessagePassingRules, :DeltaMessagePassingRulesProjectionExt).DivisionOf

gaussian_parameters(d::NormalDistributionsFamily) = weightedmean_precision(d)
gaussian_parameters(d::DivisionOf) = gaussian_parameters(d.numerator) .- gaussian_parameters(d.denumerator)
gaussian_parameters(d::BayesBase.ProductOf) = gaussian_parameters(BayesBase.getleft(d)) .+ gaussian_parameters(BayesBase.getright(d))

gaussian(ξ::Real, w::Real) = NormalWeightedMeanPrecision(ξ, w)
gaussian(ξ::AbstractVector, W::AbstractMatrix) = MvNormalWeightedMeanPrecision(ξ, W)

ReactiveMP.constrain_form(::CallClosedProd, distribution::Distribution) = distribution
ReactiveMP.constrain_form(::CallClosedProd, product::Union{BayesBase.ProductOf, DivisionOf}) = gaussian(gaussian_parameters(product)...)
