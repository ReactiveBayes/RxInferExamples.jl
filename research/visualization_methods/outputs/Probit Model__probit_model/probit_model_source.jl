@model function probit_model(y, prior_x)
    
    # specify uninformative prior
    x_prev ~ prior_x
    
    # create model 
    for k in eachindex(y)
        x[k] ~ Normal(mean = x_prev + 0.1, precision = 100)
        y[k] ~ Probit(x[k]) where {
            # Probit node by default uses RequireMessage pipeline with vague(NormalMeanPrecision) message as initial value for `in` edge
            # To change initial value user may specify it manually, like. Changes to the initial message may improve stability in some situations
            dependencies = RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0.0, 0.01))
        }
        x_prev = x[k]
    end