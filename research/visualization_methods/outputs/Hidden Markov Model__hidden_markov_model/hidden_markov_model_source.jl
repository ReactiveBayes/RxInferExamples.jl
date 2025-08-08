@model function hidden_markov_model(x)
    
    A ~ DirichletCollection(ones(3,3))
    B ~ DirichletCollection([ 10.0 1.0 1.0; 
                                            1.0 10.0 1.0; 
                                            1.0 1.0 10.0 ])
    
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    
    s_prev = s_0
    
    for t in eachindex(x)
        s[t] ~ DiscreteTransition(s_prev, A) 
        x[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end