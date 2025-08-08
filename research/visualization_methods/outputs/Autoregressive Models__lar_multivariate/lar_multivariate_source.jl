@model function lar_multivariate(y, order, γ)
    # `c` is a unit vector of size `order` with first element equal to 1
    c = ReactiveMP.ar_unit(Multivariate, order)
    
    τ  ~ Gamma(α = 1.0, β = 1.0)
    θ  ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    
    x_prev = x0
    
    for i in eachindex(y)
 
        x[i] ~ AR(x_prev, θ, τ) 
        y[i] ~ Normal(mean = dot(c, x[i]), precision = γ)
        
        x_prev = x[i]
    end