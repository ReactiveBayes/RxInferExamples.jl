import ExponentialFamily: softmax

function create_P_matrix(n_states)
    P = zeros(n_states, n_states)
    for i in 1:n_states
        P[i,:] = 0.5 * ones(n_states)
        P[i,i] = 1.0
    end
    return P
end

Base.@kwdef struct RSLDSHyperparameters{T} 
   a_w::T = 2.0
   b_w::T = 2.0
   Ψ_w::Matrix{T}
   Ψ_R::Matrix{T}
   ν_R::T
   α::Matrix{T} 
   C::Matrix{T}
end


function get_hyperparameters(hyperparameters::RSLDSHyperparameters)
    return hyperparameters.a_w, hyperparameters.b_w, hyperparameters.Ψ_w, hyperparameters.Ψ_R, hyperparameters.ν_R, hyperparameters.α, hyperparameters.C
end

function default_hyperparameters(n_states, obs_dim, n_latent)
    return RSLDSHyperparameters(
        a_w = 2.0,
        b_w = 2.0,
        Ψ_w = diageye(n_states),
        Ψ_R = diageye(obs_dim),
        ν_R = obs_dim + 2.0,
        α = ones(n_states+1, n_states+1),
        C = diageye(obs_dim,n_latent)
    )
end

@model function rslds_model_learning(obs,n_states, n_latent, η, Ψ, hyperparameters)
    local H,A,Λ,u
    transformation  = (x) -> reshape(x, (n_latent, n_latent))
    transformation2 = (x) -> reshape(x, (n_states, n_latent))
    ##Hyperparameters
    a_w, b_w, Ψ_w, Ψ_R,ν_R, α, C = get_hyperparameters(hyperparameters)
    ## Priors on the parameters 
    if n_states == 1
        w ~ GammaShapeRate(a_w, b_w)
    else
        w ~ Wishart(n_states+2,Ψ_w)
    end 
    
    for k in 1:n_states+1
        H[k] ~ MvNormalMeanCovariance(0.01*ones(n_latent^2), diageye(n_latent^2))
        Λ[k] ~ Wishart(n_latent+2, diageye(n_latent))
    end
    P ~ DirichletCollection(α)
    ϕ ~ MvNormalMeanCovariance(zeros(n_latent*n_states), diageye(n_latent*n_states))
    ## States Initialisation 
    x[1] ~ MvNormalMeanCovariance(zeros(n_latent), diageye(n_latent))
    # s[1] ~ Categorical(ones(n_states+1)/(n_states+1))
    for t in eachindex(obs)  
        ## Recurrent Layer
        if n_states == 1
            u[t] ~ softdot(ϕ, x[t], w)
        else
            u[t] ~ ContinuousTransition(x[t], ϕ, w) where {meta = CTMeta(transformation2)}
        end     
        s[t] ~ MultinomialPolya(1, u[t]) where {dependencies = RequireMessageFunctionalDependencies(ψ = convert(promote_variate_type(typeof(η), NormalWeightedMeanPrecision), η, Ψ))}   
        s[t+1] ~ DiscreteTransition(s[t], P)
        ##Transition Layer
        A[t] ~ MagicMixture(switch=s[t+1], inputs=H)
        B[t] ~ MagicMixture(switch=s[t+1], inputs=Λ)
        x[t+1] ~ ContinuousTransition(x[t], A[t], B[t]) where {meta = CTMeta(transformation)}
        ## Observation Layer
        obs[t] ~ MvNormalMeanCovariance(C*x[t+1], Ψ_R)
    end
end

@constraints  function rslds_learning_constraints()
    q(x,s,u,ϕ,w,P,H,A,Λ,B)=q(x,u)q(A)q(s)q(ϕ)q(w)q(P)q(H)q(Λ)q(B)
end

@initialization function rslds_learning_initmarginals(n_states, n_latent, obs_dim)    
    q(x) = vague(MvNormalWeightedMeanPrecision, n_latent)
    q(s) = Multinomial(1,softmax(randn(n_states+1)))
    q(ϕ) = vague(MvNormalWeightedMeanPrecision, n_latent*(n_states))
    if n_states == 1
        q(w) = vague(GammaShapeRate)
    else
        q(w) = vague(Wishart, n_states)   
    end
    q(A) = vague(MvNormalWeightedMeanPrecision, n_latent^2)
    q(P) = DirichletCollection(ones(n_states+1,n_states+1))
    q(Λ) = vague(Wishart, n_latent)
    q(H) = vague(MvNormalWeightedMeanPrecision, n_latent^2)
    q(B) = vague(Wishart, n_latent)
    # q(R) = vague(InverseWishart, obs_dim)
end;



function fit_rslds(data,n_states,n_latent;iterations = 60, η = nothing, Ψ = nothing, hyperparameters  = nothing, progress = false)
  
    n_states = n_states - 1

    if hyperparameters === nothing
        hyperparameters = default_hyperparameters(n_states, length(data[1]),n_latent)
    end

    if η === nothing
        if n_states == 1
            η = 0.0
        else
            η = zeros(n_states)
        end
    end
    if Ψ === nothing
        if n_states == 1
            Ψ = 1.0
        else
            Ψ = diageye(n_states)
        end
    end
    model = rslds_model_learning( n_states = n_states, n_latent = n_latent, η = η, Ψ = Ψ, hyperparameters = hyperparameters)
    constraints = rslds_learning_constraints()
    initmarginals = rslds_learning_initmarginals(n_states, n_latent, length(data[1]))
    
    init_result = infer(model = model,data = (obs =data, ), constraints = constraints, initialization = initmarginals,iterations = iterations,
    showprogress = progress,
    returnvars = KeepEach(),
    free_energy = true,
    # addons = (AddonMemory(),),
    # postprocess = NoopPostprocess(),
    # callbacks = (on_marginal_update = onmarginalupdate, ),
    options = (limit_stack_depth = 100,)
    )

end

# 

function states_to_categorical(states)
    return [argmax(states[t].p) for t in 1:length(states)]
end
