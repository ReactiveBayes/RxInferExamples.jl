@model function beta_model_mary(y)

    # specify John's and Jane's prior models over θ
    θ_jane ~ Beta(2.0, 7.0)
    θ_john ~ Beta(7.0, 2.0)

    # specify initial guess as to who is right
    john_is_right ~ Bernoulli(0.7) 

    # specify mixture prior Distribution
    θ ~ Mixture(switch = john_is_right, inputs = [θ_jane, θ_john])

    # create likelihood models
    y .~ Bernoulli(θ)
    
end