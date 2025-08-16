using ReactiveMP, BayesBase

struct CallClosedProd <: AbstractFormConstraint end

ReactiveMP.default_prod_constraint(::CallClosedProd) = ClosedProd()
ReactiveMP.default_form_check_strategy(::CallClosedProd) = FormConstraintCheckLast()

function ReactiveMP.constrain_form(::CallClosedProd, product_of::BayesBase.ProductOf) 
    return BayesBase.prod(ClosedProd(), product_of)
end

function ReactiveMP.constrain_form(::CallClosedProd, distribution::Distribution) 
    return distribution
end

ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)


function ReactiveMP.constrain_form(::CallClosedProd, distribution::ext.DivisionOf{A, B}) where {A <: GaussianDistributionsFamily, B <: GaussianDistributionsFamily}
    ef_num = convert(ExponentialFamily.ExponentialFamilyDistribution, distribution.numerator)
    ef_den = convert(ExponentialFamily.ExponentialFamilyDistribution, distribution.denumerator)
    result = getnaturalparameters(ef_num) - getnaturalparameters(ef_den)
    ef_typetag = ExponentialFamily.exponential_family_typetag(ef_num)
    ef_result = ExponentialFamily.ExponentialFamilyDistribution(ef_typetag, result)
    return convert(Distribution, ef_result)
end
