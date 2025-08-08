@model function regulator_model(T, m_u, v_u, x_t, lo, hi, epsilon, atol)
    
    # Loop over horizon
    x_k_last = x_t
    for k = 1:T
        u[k] ~ NormalMeanVariance(m_u[k], v_u[k]) # Control prior
        x[k] ~ x_k_last + u[k] # Transition model
        x[k] ~ ChanceConstraint(lo, hi, epsilon, atol) where { # Simultaneous constraint on state
            dependencies = RequireMessageFunctionalDependencies(out = NormalWeightedMeanPrecision(0, 0.01))} # Predefine inbound message to break circular dependency
        x_k_last = x[k]
    end