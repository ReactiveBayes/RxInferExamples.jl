@model function drone_model_3d(drone, environment, initial_state, goal, horizon, dt)
    # Extract properties
    g = get_gravity(environment)
    m = drone.mass
    
    # Initial state prior
    s[1] ~ MvNormal(mean = initial_state, covariance = 1e-5 * I)
    
    for i in 1:horizon
        # Prior on motor actions (mean compensates for gravity)
        hover_force = m * g / 4
        u[i] ~ MvNormal(μ = [hover_force, hover_force, hover_force, hover_force], Σ = diageye(4))
        
        # State transition
        s[i + 1] ~ MvNormal(
            μ = state_transition_3d(s[i], u[i], drone, environment, dt),
            Σ = 1e-10 * I
        )
    end