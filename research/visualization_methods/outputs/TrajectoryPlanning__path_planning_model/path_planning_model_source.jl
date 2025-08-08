@model function path_planning_model(environment, agents, goals, nr_steps)

    # Model's parameters are fixed, refer to the original 
    # paper's implementation for more details about these parameters
    local dt = 1
    local A  = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
    local B  = [0 0; dt 0; 0 0; 0 dt]
    local C  = [1 0 0 0; 0 0 1 0]
    local γ  = 1

    local control
    local state
    local path   
    
    # Extract radiuses of each agent in a separate collection
    local rs = map((a) -> a.radius, agents)

    # Model is fixed for 4 agents
    for k in 1:4

        # Prior on state, the state structure is 4 dimensional, where
        # [ x_position, x_velocity, y_position, y_velocity ]
        state[k, 1] ~ MvNormal(mean = zeros(4), covariance = 1e2I)

        for t in 1:nr_steps

            # Prior on controls
            control[k, t] ~ MvNormal(mean = zeros(2), covariance = 1e-1I)

            # State transition
            state[k, t+1] ~ A * state[k, t] + B * control[k, t]

            # Path model, the path structure is 2 dimensional, where 
            # [ x_position, y_position ]
            path[k, t] ~ C * state[k, t+1]

            # Environmental distance
            zσ2[k, t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
            z[k, t]   ~ g(environment, rs[k], path[k, t])
            
            # Halfspase priors were defined previousle in this experiment
            z[k, t] ~ Halfspace(0, zσ2[k, t], γ)

        end