@model function beta_model_john(y)

    # specify John's prior model over θ
    θ ~ Beta(7.0, 2.0)

    # create likelihood models
    y .~ Bernoulli(θ)
    
end