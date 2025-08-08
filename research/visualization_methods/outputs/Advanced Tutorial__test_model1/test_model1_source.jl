@model function test_model1(s_mean, s_precision, y)
    
    # the `tilde` operator creates a functional dependency
    # between variables in our model and can be read as 
    # `sampled from` or `is modeled by`
    s ~ Normal(mean = s_mean, precision = s_precision)
    y ~ Normal(mean = s, precision = 1.0)
    
    # It is possible to return something from the model specification (including variables and nodes)
    return "Hello world"
end