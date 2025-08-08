@model function binomial_model(prior_xi, prior_precision, n_trials, X, y) 
    β ~ MvNormalWeightedMeanPrecision(prior_xi, prior_precision)
    for i in eachindex(y)
        y[i] ~ BinomialPolya(X[i], n_trials[i], β) where {
            dependencies = RequireMessageFunctionalDependencies(β = MvNormalWeightedMeanPrecision(prior_xi, prior_precision))
        }
    end