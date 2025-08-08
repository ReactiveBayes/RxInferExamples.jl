@model function normal_square_model(y)
    # describe prior on latent state, we set an arbitrary prior 
    # in a positive domain
    x ~ Normal(mean = 5, precision = 1e-3)
    # transform latent state
    mean := f(x)
    # observation model
    y .~ Normal(mean = mean, precision = 0.1)
end