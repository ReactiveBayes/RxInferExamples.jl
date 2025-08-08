@model function path_planning_model(environment, agents, goals, nr_steps; model_params=nothing)

    # Model's parameters are fixed, refer to the original 
    # paper's implementation for more details about these parameters
    if model_params === nothing
        # Default parameters if no configuration is provided
        local dt = 1
        local A  = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
        local B  = [0 0; dt 0; 0 0; 0 dt]
        local C  = [1 0 0 0; 0 0 1 0]
        local γ  = 1
        local initial_state_variance = 1e2
        local control_variance = 1e-1
        local goal_constraint_variance = 1e-5
        local gamma_shape = 3/2
        local gamma_scale = γ^2/2
    else
        # Use parameters from configuration
        local dt = model_params.dt
        local A  = model_params.A
        local B  = model_params.B
        local C  = model_params.C
        local γ  = model_params.gamma
        local initial_state_variance = model_params.initial_state_variance
        local control_variance = model_params.control_variance
        local goal_constraint_variance = model_params.goal_constraint_variance
        local gamma_shape = model_params.gamma_shape
        local gamma_scale = model_params.gamma_scale_factor * γ^2
    end