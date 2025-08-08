@model function room_model(temp_obs, α, noise_T)
    # Set prior for initial temperature
    T0 ~ Normal(mean = temp_obs[1], variance = 5.0)
    
    # Initial value
    T_prev = T0
    
    for i in eachindex(temp_obs)
        # Simple linear model (similar to univariate example)
        T[i] := T_prev + α
        
        # Observation with Gaussian noise
        temp_obs[i] ~ Normal(mean = T[i], variance = noise_T^2)
        
        # Update previous value for next iteration
        T_prev = T[i]
    end