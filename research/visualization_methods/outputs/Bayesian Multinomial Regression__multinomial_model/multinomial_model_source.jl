@model function multinomial_model(obs, N, ξ_ψ, W_ψ)
    ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
    obs .~ MultinomialPolya(N, ψ) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ))}
end