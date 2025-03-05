using ExponentialFamily, RxInfer, BayesBase
import ReactiveMP: AbstractFactorNode, NodeInterface, IndexedNodeInterface, FactorNodeActivationOptions, Marginalisation, Deterministic, PredefinedNodeFunctionalForm,FunctionalDependencies, collect_functional_dependencies, activate!, functional_dependencies, collect_latest_messages, collect_latest_marginals
import ExponentialFamily: getnaturalparameters, exponential_family_typetag
export MagicMixture, MagicMixtureNode

# Mixture Functional Form
struct MagicMixture{N} end

ReactiveMP.as_node_symbol(::Type{<:MagicMixture}) = :MagicMixture

interfaces(::Type{<:MagicMixture}) = Val((:out, :switch, :inputs))
alias_interface(::Type{<:MagicMixture}, ::Int64, name::Symbol) = name
is_predefined_node(::Type{<:MagicMixture}) = PredefinedNodeFunctionalForm()
sdtype(::Type{<:MagicMixture}) = Deterministic()
collect_factorisation(::Type{<:MagicMixture}, factorization) = MagicMixtureNodeFactorisation()

struct MagicMixtureNodeFactorisation end

struct MagicMixtureNode{N} <: AbstractFactorNode
    """
        MagicMixtureNode{N}

    A factor node that represents a magic mixture model with N components that under the hood performs Bayesian model comparison.

    # Interfaces
    - `:out`: The output interface representing the magic mixture distribution
    - `:switch`: The switch interface representing the selector variable
    - `:inputs`: The inputs interface representing the input distributions
    """     

    out    :: NodeInterface
    switch :: NodeInterface
    inputs :: NTuple{N, IndexedNodeInterface}
end 

functionalform(factornode::MagicMixtureNode{N}) where {N} = MagicMixture{N}
getinterfaces(factornode::MagicMixtureNode) = (factornode.out, factornode.switch, factornode.inputs...)
sdtype(factornode::MagicMixtureNode) = Deterministic()

interfaceindices(factornode::MagicMixtureNode, iname::Symbol)                       = (interfaceindex(factornode, iname),)
interfaceindices(factornode::MagicMixtureNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

function interfaceindex(factornode::MagicMixtureNode, iname::Symbol)
    if iname === :out
        return 1
    elseif iname === :switch
        return 2
    elseif iname === :inputs
        return 3
    end
end

function factornode(::Type{<:MagicMixture}, interfaces, factorization)
    outinterface = interfaces[findfirst(((name, variable),) -> name == :out, interfaces)]
    switchinterface = interfaces[findfirst(((name, variable),) -> name == :switch, interfaces)]
    inputinterfaces = filter(((name, variable),) -> name == :inputs, interfaces)
    N = length(inputinterfaces)
    return MagicMixtureNode(NodeInterface(outinterface...), NodeInterface(switchinterface...), ntuple(i -> IndexedNodeInterface(i, NodeInterface(inputinterfaces[i]...)), N))
    
end

struct MagicMixtureNodeInboundInterfaces end

getinboundinterfaces(::MagicMixtureNode) = MagicMixtureNodeInboundInterfaces()
clustername(::MagicMixtureNodeInboundInterfaces) = :switch_inputs


struct MagicMixtureNodeFunctionalDependencies <: FunctionalDependencies end

collect_functional_dependencies(::MagicMixtureNode, ::Nothing) = MagicMixtureNodeFunctionalDependencies()
collect_functional_dependencies(::MagicMixtureNode, ::MagicMixtureNodeFunctionalDependencies) = MagicMixtureNodeFunctionalDependencies()
collect_functional_dependencies(::MagicMixtureNode, ::Any) =
    error("The functional dependencies for MagicMixtureNode must be either `Nothing` or `MagicMixtureNodeFunctionalDependencies`")

function activate!(factornode::MagicMixtureNode, options::FactorNodeActivationOptions)
    dependencies = collect_functional_dependencies(factornode, getdependecies(options))
    return activate!(dependencies, factornode, options)
end


function functional_dependencies(::MagicMixtureNodeFunctionalDependencies, factornode::MagicMixtureNode{N}, interface, iindex::Int) where {N}
    message_dependencies = if iindex === 1
        # output depends on input messages:
        (factornode.inputs, )
    elseif iindex === 2
        # switch depends on:
        (factornode.out, factornode.inputs)
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.out, )
    else
        error("Bad index in functional_dependencies for MixtureNode")
    end

    marginal_dependencies = if iindex === 1
        # output depends on:
        (factornode.switch, )
    elseif iindex === 2
        # switch depends on:
        ( )
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.switch,)
    else
        error("Bad index in functional_dependencies for MagicMixtureNode")
    end

    return message_dependencies, marginal_dependencies
end


function collect_latest_messages(::MagicMixtureNodeFunctionalDependencies, factornode::MagicMixtureNode{N}, messages::Tuple{NodeInterface}) where {N}
    outputinterface = messages[1]

    msgs_names = Val{(name(outputinterface),)}()
    msgs_observable = combineLatestUpdates((messagein(outputinterface),), PushNew())
    return msgs_names, msgs_observable
end

function collect_latest_marginals(::MagicMixtureNodeFunctionalDependencies, factornode::MagicMixtureNode{N}, marginals::Tuple{NodeInterface}) where {N}
    switchinterface = marginals[1]

    marginal_names = Val{(name(switchinterface),)}()
    marginal_observable = combineLatestUpdates((
        getmarginal(getvariable(switchinterface), IncludeAll()),
    ), PushNew())

    return marginal_names, marginal_observable
end

function collect_latest_marginals(::MagicMixtureNodeFunctionalDependencies, factornode::MagicMixtureNode{N}, marginals::NTuple{N,IndexedNodeInterface}) where {N}
    inputsinterfaces = marginals
    
    marginal_names = Val{(name(first(inputsinterfaces)),)}()
    marginal_observable = combineLatest(map(input -> getmarginal(getvariable(input), IncludeAll()), inputsinterfaces), PushNew()) |> map_to((ManyOf(map(input -> getmarginal(getvariable(input), IncludeAll()), inputsinterfaces)),))
    
    return marginal_names, marginal_observable
end

function collect_latest_messages(::MagicMixtureNodeFunctionalDependencies, factornode::MagicMixtureNode{N}, messages::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}}) where {N}
    output_or_switch_interface = messages[1]
    inputsinterfaces = messages[2]

    msgs_names = Val{(name(output_or_switch_interface), name(inputsinterfaces[1]))}()
    msgs_observable =
        combineLatest(
            (messagein(output_or_switch_interface), combineLatest(map(input -> messagein(input), inputsinterfaces), PushNew())),
            PushNew()
        ) |> map_to((
            messagein(output_or_switch_interface), 
            ManyOf(map(input -> messagein(input), inputsinterfaces))
        ))
    
    return msgs_names, msgs_observable
end

function collect_latest_messages(::MagicMixtureNodeFunctionalDependencies, factornode::MagicMixtureNode{N}, messages::Tuple{NTuple{N,IndexedNodeInterface}}) where {N}
    inputsinterfaces = messages[1]
    
    msgs_names = Val{(name(first(inputsinterfaces)),)}()
    msgs_observable = combineLatest(map(input -> messagein(input), inputsinterfaces), PushNew()) |> map_to((ManyOf(map(input -> messagein(input), inputsinterfaces)),))
    
    return msgs_names, msgs_observable
end


marginalrule(fform::Type{<:MagicMixture}, on::Val{:switch_inputs}, mnames::Any, messages::Any, qnames::Nothing, marginals::Nothing, meta::Nothing, __node::Any) = begin
    m_out = getdata(messages[1])
    m_switch = getdata(messages[2])
    m_inputs = getdata.(messages[3:end])


    return FactorizedJoint((m_inputs..., m_switch))
end

@rule MagicMixture(:out, Marginalisation) (q_switch::Any, m_inputs::ManyOf{N, Any}) where {N} = begin
    return MixtureDistribution(collect(m_inputs), probvec(q_switch))
end


@rule MagicMixture(:switch, Marginalisation) (m_out::Any, m_inputs::ManyOf{N, Any}) where {N} = begin
    logscales = map(input -> compute_logscale(prod(GenericProd(),m_out,input), m_out, input), m_inputs)
    p = softmax(collect(logscales))
    return Multinomial(1, p)
end


@rule MagicMixture((:inputs, k), Marginalisation) (m_out::Any, q_switch::Any,) = begin
    z = probvec(q_switch)[k]
    ef_out = convert(ExponentialFamilyDistribution, m_out)
    η      = getnaturalparameters(ef_out)
    ef_opt = ExponentialFamilyDistribution(exponential_family_typetag(ef_out), η * z)

    return convert(Distribution, ef_opt)
end


@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::MixtureDistribution, meta::Any) = begin 
    comps = BayesBase.components(m_in)
    new_components = similar(comps)
    @inbounds for (i,component) in enumerate(comps)
        new_components[i] = @call_rule typeof(*)(:out, Marginalisation) (m_A = m_A, m_in = component, meta = meta)
    end
    dist = MixtureDistribution(new_components, BayesBase.weights(m_in))
    return dist
end

@rule typeof(dot)(:out, Marginalisation) (m_in1::MixtureDistribution, m_in2::PointMass, meta::Any) = begin 
    comps = BayesBase.components(m_in1)
    new_components = []
    @inbounds for (i, component) in enumerate(comps)
        push!(new_components, @call_rule typeof(dot)(:out, Marginalisation) (m_in1 = component, m_in2 = m_in2, meta = meta))
    end
    
    mixture = MixtureDistribution(new_components, BayesBase.weights(m_in1))

    return mixture
end


function BayesBase.mean(mixture::MixtureDistribution)
    component_means = mean.(BayesBase.components(mixture))
    component_weights = softmax(BayesBase.weights(mixture))
    return mapreduce((m,w) -> w*m, +, component_means, component_weights)
end

function BayesBase.precision(mixture::MixtureDistribution)
    component_precisions = precision.(BayesBase.components(mixture))
    component_weights = softmax(BayesBase.weights(mixture))
    return mapreduce((m,w) -> w*m, +, component_precisions, component_weights)
end

function BayesBase.cov(mixture::MixtureDistribution)
    component_cov = cov.(BayesBase.components(mixture))
    component_weights = BayesBase.weights(mixture)
    return mapreduce((m,w) -> w*m, +, component_cov, component_weights)
end

function BayesBase.var(mixture::MixtureDistribution)
    cov = BayesBase.cov(mixture)
    return diag(cov)
end

@rule typeof(dot)(:in1, Marginalisation) (m_out::MixtureDistribution, m_in2::PointMass, meta::Any) = begin 
    comps = BayesBase.components(m_out)
    weights = BayesBase.weights(m_out)
    new_comps = []
    for (comp, weight) in zip(comps, weights)
        new_comp = @call_rule typeof(dot)(:in1, Marginalisation) (m_out = comp, m_in2 = m_in2, meta = meta)
        push!(new_comps, new_comp)
    end
    return MixtureDistribution(new_comps, weights)
end

function BayesBase.prod(::BayesBase.UnspecifiedProd, left::GaussianDistributionsFamily, right::MixtureDistribution)
    comps = BayesBase.components(right)
    weights = BayesBase.weights(right)
    new_comps = []
    for comp in comps
        new_comp = prod(GenericProd(),left, comp)
        push!(new_comps, new_comp)
    end
    
    return MixtureDistribution(new_comps, weights)
end

BayesBase.prod(::BayesBase.UnspecifiedProd, left::MixtureDistribution, right::GaussianDistributionsFamily) = prod(GenericProd(),right, left)
BayesBase.paramfloattype(::MixtureDistribution) = Float64


import ExponentialFamily.LogExpFunctions: logsumexp
function BayesBase.prod(::GenericProd, left::Categorical, right::Multinomial)
    @assert right.n == 1
    right_cat = Categorical(right.p)

    p = prod(GenericProd(), left, right_cat).p 
    return Multinomial(1, p)

end



BayesBase.prod(::GenericProd, left::Multinomial, right::Categorical) = prod(GenericProd(), right, left)
function BayesBase.prod(::GenericProd, left::Multinomial, right::Multinomial) 
    @assert left.n == right.n

    p = left.p .* right.p
    p = p ./ sum(p)
    return Multinomial(left.n, p)
end

BayesBase.prod(::BayesBase.UnspecifiedProd, left::Multinomial, right::Multinomial) = prod(GenericProd(), left, right)


function BayesBase.compute_logscale(dist1::Multinomial, dist2::Multinomial, dist3::Multinomial) 
    logp1 = log.(dist1.p) - log(dist1.p[end])
    logp2 = log.(dist2.p) - log(dist2.p[end])
    logp3 = log.(dist3.p) - log(dist3.p[end])
    return logsumexp(logp1) - logsumexp(logp2) - logsumexp(logp3)
end

BayesBase.compute_logscale(d1::ExponentialFamily.WishartFast, d2::ExponentialFamily.WishartFast, d3::ExponentialFamily.WishartFast) = begin
    return logpartition(convert(ExponentialFamilyDistribution, d1)) - logpartition(convert(ExponentialFamilyDistribution, d2)) - logpartition(convert(ExponentialFamilyDistribution, d3))
end


ExponentialFamily.probvec(d::Multinomial) = d.p

@rule ContinuousTransition(:W, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_a::MixtureDistribution, meta::Any) = begin 
    q_a_normal = convert(promote_variate_type(typeof(mean(q_a)), NormalMeanPrecision), mean(q_a), precision(q_a))
    return @call_rule ContinuousTransition(:W, Marginalisation) (q_y_x = q_y_x, q_a = q_a_normal, meta = meta)
end

@rule ContinuousTransition(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_a::MixtureDistribution, q_W::Any, meta::Any) = begin 
    q_a_normal = convert(promote_variate_type(typeof(mean(q_a)), NormalMeanPrecision), mean(q_a), precision(q_a))
    return @call_rule ContinuousTransition(:y, Marginalisation) (m_x = m_x, q_a = q_a_normal, q_W = q_W, meta = meta)
end


@rule ContinuousTransition(:a, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_a::MixtureDistribution, q_W::Any, meta::Any) = begin 
    q_a_normal = convert(promote_variate_type(typeof(mean(q_a)), NormalMeanPrecision), mean(q_a), precision(q_a))
    return @call_rule ContinuousTransition(:a, Marginalisation) (q_y_x = q_y_x, q_a = q_a_normal, q_W = q_W, meta = meta)
end

@rule ContinuousTransition(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily , q_a::MixtureDistribution, q_W::Any, meta::Any) = begin 
    q_a_normal = convert(promote_variate_type(typeof(mean(q_a)), NormalMeanPrecision), mean(q_a), precision(q_a))
    return @call_rule ContinuousTransition(:x, Marginalisation) (m_y = m_y, q_a = q_a_normal, q_W = q_W, meta = meta)
end

@rule ContinuousTransition(:y, Marginalisation) (m_x::MixtureDistribution, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::Any) = begin 
    m_x_normal = convert(promote_variate_type(typeof(mean(m_x)), NormalMeanPrecision), mean(m_x), precision(m_x))
    return @call_rule ContinuousTransition(:y, Marginalisation) (m_x = m_x_normal, q_a = q_a, q_W = q_W, meta = meta)
end


@marginalrule ContinuousTransition(:y_x) (m_y::MultivariateNormalDistributionsFamily, m_x::MixtureDistribution, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::Any) = begin 
    m_x_normal = convert(promote_variate_type(typeof(mean(m_x)), NormalMeanPrecision), mean(m_x), precision(m_x))
    return @call_marginalrule ContinuousTransition(:y_x) (m_y = m_y, m_x = m_x_normal, q_a = q_a, q_W = q_W, meta = meta)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::MixtureDistribution, ) = begin 
    return @call_rule typeof(+)(:out, Marginalisation) (m_in1 = m_in1, m_in2 = convert(promote_variate_type(typeof(mean(m_in2)), NormalMeanPrecision), mean(m_in2), precision(m_in2)))
end

@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::MixtureDistribution, ) = begin 
    return @call_rule typeof(+)(:in1, Marginalisation) (m_out = convert(promote_variate_type(typeof(mean(m_out)), NormalMeanPrecision), mean(m_out), precision(m_out)), m_in2 = convert(promote_variate_type(typeof(mean(m_in2)), NormalMeanPrecision), mean(m_in2), precision(m_in2)))
end



Base.length(d::MixtureDistribution) = length(d.components)
Base.ndims(d::MixtureDistribution) = first(size(first(d.components)))

ExponentialFamily.probvec(d::Multinomial) = d.p

BayesBase.entropy(d::MixtureDistribution) = mapreduce((c,w) -> w * BayesBase.entropy(c), +, d.components, d.weights)

BayesBase.mean(f::F, itr::MixtureDistribution) where {F} = mapreduce((c,w) -> w * mean(f, c), +, itr.components, itr.weights)