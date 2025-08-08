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