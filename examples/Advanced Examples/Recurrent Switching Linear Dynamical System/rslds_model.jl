include("extensions.jl")
import ExponentialFamily: softmax


"""
    RSLDSHyperparameters{T}

Structure containing hyperparameters for the Recurrent Switching Linear Dynamical System (RSLDS) model.

# Fields
- `a_w::T = 2.0`: Shape parameter for the Gamma prior on precision parameter w (when n_switches=1)
- `b_w::T = 2.0`: Rate parameter for the Gamma prior on precision parameter w (when n_switches=1)
- `Ψ_w::Matrix{T}`: Scale matrix for the Wishart prior on precision matrix w (when n_switches>1)
- `Ψ_R::Union{Matrix{T}, T}`: Scale matrix/parameter for the Wishart/Gamma prior on observation precision
- `ν_R::T`: Degrees of freedom for the Wishart prior on observation precision
- `α::Matrix{T}`: Parameter matrix for the Dirichlet prior on transition probabilities
- `C::Matrix{T}`: Observation matrix mapping latent states to observations
"""
Base.@kwdef struct RSLDSHyperparameters{T} 
   a_w::T = 2.0
   b_w::T = 2.0
   Ψ_w::Matrix{T}
   Ψ_R::Union{Matrix{T}, T}
   ν_R::T
   α::Matrix{T} 
   C::Matrix{T}
end

"""
    get_hyperparameters(hyperparameters::RSLDSHyperparameters)

Extract all hyperparameters from the RSLDSHyperparameters structure.

# Arguments
- `hyperparameters::RSLDSHyperparameters`: Structure containing the hyperparameters

# Returns
A tuple containing all hyperparameters in the order: a_w, b_w, Ψ_w, Ψ_R, ν_R, α, C
"""
function get_hyperparameters(hyperparameters::RSLDSHyperparameters)
    return hyperparameters.a_w, hyperparameters.b_w, hyperparameters.Ψ_w, hyperparameters.Ψ_R, hyperparameters.ν_R, hyperparameters.α, hyperparameters.C
end

"""
    default_hyperparameters(n_switches, obs_dim, dim_latent)

Create a default set of hyperparameters for the RSLDS model.

# Arguments
- `n_switches`: Number of switching states in the model
- `obs_dim`: Dimension of the observation space
- `dim_latent`: Dimension of the latent state space

# Returns
An RSLDSHyperparameters structure with default values
"""
function default_hyperparameters(n_switches, obs_dim, dim_latent)
    return RSLDSHyperparameters(
        a_w = 2.0,
        b_w = 2.0,
        Ψ_w = diageye(n_switches),
        Ψ_R = diageye(obs_dim),
        ν_R = obs_dim + 2.0,
        α = ones(n_switches+1, n_switches+1),
        C = diageye(obs_dim,dim_latent)
    )
end


@model function rslds_model_learning(obs,n_obs,n_switches, dim_latent, η, Ψ, hyperparameters, learn_observation_covariance)
    local H,A,Λ,u
    transformation  = (x) -> reshape(x, (dim_latent, dim_latent))
    transformation2 = (x) -> reshape(x, (n_switches, dim_latent))
    ##Hyperparameters
    a_w, b_w, Ψ_w, Ψ_R,ν_R, α, C = get_hyperparameters(hyperparameters)
    ## Priors on the parameters 
    if n_switches == 1
        w ~ GammaShapeRate(a_w, b_w)
    else
        w ~ Wishart(n_switches+2,Ψ_w)
    end 

    if learn_observation_covariance
        if n_obs == 1
            R ~ GammaShapeRate(ν_R, Ψ_R)
        else
            R ~ Wishart(ν_R, Ψ_R) 
        end
    else
        R = Ψ_R
    end
    
    for k in 1:n_switches+1
        H[k] ~ MvNormalMeanCovariance(zeros(dim_latent^2), diageye(dim_latent^2))
        Λ[k] ~ Wishart(dim_latent+2, diageye(dim_latent))
    end
    P ~ DirichletCollection(α)
    ϕ ~ MvNormalMeanCovariance(zeros(dim_latent*n_switches), diageye(dim_latent*n_switches))
    ## States Initialisation 
    x[1] ~ MvNormalMeanCovariance(zeros(dim_latent), diageye(dim_latent))
    for t in eachindex(obs)  
        ## Recurrent Layer
        if n_switches == 1
            u[t] ~ softdot(ϕ, x[t], w)
        else
            u[t] ~ ContinuousTransition(x[t], ϕ, w) where {meta = CTMeta(transformation2)}
        end     
        s[t] ~ MultinomialPolya(1, u[t]) where {dependencies = RequireMessageFunctionalDependencies(ψ = convert(promote_variate_type(typeof(η), NormalWeightedMeanPrecision), η, Ψ))}   
        s[t+1] ~ DiscreteTransition(s[t], P)
        ##Transition Layer
        A[t] := Gate(switch=s[t+1], inputs=H)
        B[t] := Gate(switch=s[t+1], inputs=Λ)
        x[t+1] ~ ContinuousTransition(x[t], A[t], B[t]) where {meta = CTMeta(transformation)}
        ## Observation Layer
        obs[t] ~ MvNormalMeanPrecision(C*x[t+1], R)
    end
end

@constraints  function rslds_learning_constraints(learn_observation_covariance)
    if learn_observation_covariance
        q(x,s,u,ϕ,w,P,H,A,Λ,B,R) = q(x,u)q(A)q(s)q(ϕ)q(w)q(P)q(H)q(Λ)q(B)q(R)
    else
        q(x,s,u,ϕ,w,P,H,A,Λ,B) = q(x,u)q(A)q(s)q(ϕ)q(w)q(P)q(H)q(Λ)q(B)
    end
end

@initialization function rslds_learning_initmarginals(n_switches, dim_latent, obs_dim, learn_observation_covariance)    
    q(x) = vague(MvNormalWeightedMeanPrecision, dim_latent)
    q(s) = Multinomial(1,softmax(randn(n_switches+1)))
    q(ϕ) = vague(MvNormalWeightedMeanPrecision, dim_latent*(n_switches))
    if n_switches == 1
        q(w) = vague(GammaShapeRate)
    else
        q(w) = vague(Wishart, n_switches)   
    end
    q(A) = vague(MvNormalWeightedMeanPrecision, dim_latent^2)
    q(P) = DirichletCollection(ones(n_switches+1,n_switches+1))
    q(Λ) = vague(Wishart, dim_latent)
    q(H) = vague(MvNormalWeightedMeanPrecision, dim_latent^2)
    q(B) = vague(Wishart, dim_latent)
    if learn_observation_covariance
        if obs_dim == 1
            q(R) = vague(GammaShapeRate)
        else
            q(R) = Wishart(obs_dim+2, diageye(obs_dim))
        end
    end
end;



"""
    fit_rslds(data, n_switches, dim_latent, n_obs; kwargs...)

Fit a Recurrent Switching Linear Dynamical System (RSLDS) model to the provided data.

# Arguments
- `data`: Time series observation data
- `n_switches`: Number of switching states in the model. Note: The user provides the total number of states,
  but internally we use (n_switches-1) because the MultinomialPolya distribution adds an extra dimension
  to represent the recurrent influence on state transitions.
- `dim_latent`: Dimension of the latent state space
- `n_obs`: Dimension of the observation space

# Keyword Arguments
- `iterations::Int = 60`: Number of inference iterations
- `η = nothing`: Mean parameter for the functional dependency in MultinomialPolya
- `Ψ = nothing`: Precision parameter for the functional dependency in MultinomialPolya
- `hyperparameters = nothing`: Custom hyperparameters for the model
- `progress::Bool = false`: Whether to show progress during inference
- `learn_observation_covariance::Bool = false`: Whether to learn the observation covariance

# Returns
The result of the inference procedure
"""
function fit_rslds(data, n_switches, dim_latent, n_obs; iterations = 60, η = nothing, Ψ = nothing, hyperparameters = nothing, progress = false, learn_observation_covariance = false)
    @assert n_switches > 1 "n_switches must be greater than 1"
    # We subtract 1 from n_switches because the MultinomialPolya distribution
    # internally adds an extra dimension to represent the recurrent influence
    # on state transitions. This convention allows the model to maintain the
    # correct dimensionality while incorporating the recurrent dynamics.
    n_switches = n_switches - 1

    if hyperparameters === nothing
        hyperparameters = default_hyperparameters(n_switches, length(data[1]), dim_latent)
    end

    if η === nothing
        if n_switches == 1
            η = 0.0
        else
            η = zeros(n_switches)
        end
    end
    if Ψ === nothing
        if n_switches == 1
            Ψ = 0.0001
        else
            Ψ = 0.0001*diageye(n_switches)
        end
    end
    model = rslds_model_learning(n_obs = n_obs, n_switches = n_switches, dim_latent = dim_latent, η = η, Ψ = Ψ, hyperparameters = hyperparameters, learn_observation_covariance = learn_observation_covariance)
    constraints = rslds_learning_constraints(learn_observation_covariance)
    initmarginals = rslds_learning_initmarginals(n_switches, dim_latent, n_obs, learn_observation_covariance)
    
    return infer(model = model, data = (obs=data, ), constraints = constraints, initialization = initmarginals, iterations = iterations,
    showprogress = progress,
    returnvars = KeepEach(),
    free_energy = true,
    options = (limit_stack_depth = 100,)
    )
end

# 

function states_to_categorical(states)
    return [argmax(states[t].p) for t in 1:length(states)]
end
