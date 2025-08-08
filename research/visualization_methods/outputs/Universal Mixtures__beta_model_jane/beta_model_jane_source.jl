@model function beta_model_jane(y)

    # specify Jane's prior model over θ
    θ ~ Beta(2.0, 7.0)

    # create likelihood models
    y .~ Bernoulli(θ)
    
end