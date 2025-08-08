@model function multinomial_regression(obs, N, X, ϕ, ξβ, Wβ)
    β ~ MvNormalWeightedMeanPrecision(ξβ, Wβ)
    for i in eachindex(obs)
        Ψ[i] := ϕ(X[i])*β
        obs[i] ~ MultinomialPolya(N, Ψ[i]) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(zeros(length(obs[i])-1), diageye(length(obs[i])-1)))}
    end