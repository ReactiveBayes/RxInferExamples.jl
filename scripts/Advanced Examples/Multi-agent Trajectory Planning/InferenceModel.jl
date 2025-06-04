module InferenceModel

using LinearAlgebra
using RxInfer
using StableRNGs
using ..Environment
using ..HalfspaceNode
using ..DistanceFunctions
using ..ConfigLoader

export path_planning, path_planning_model, path_planning_constraints

# For more details about the model, please refer to the original paper
# Keep RxInfer.jl @model syntax carefully in mind
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

    local control
    local state
    local path   
    
    # Extract radiuses of each agent in a separate collection
    local rs = map((a) -> a.radius, agents)

    # Model is fixed for 4 agents
    for k in 1:4

        # Prior on state, the state structure is 4 dimensional, where
        # [ x_position, x_velocity, y_position, y_velocity ]
        state[k, 1] ~ MvNormal(mean = zeros(4), covariance = initial_state_variance * I)

        for t in 1:nr_steps

            # Prior on controls
            control[k, t] ~ MvNormal(mean = zeros(2), covariance = control_variance * I)

            # State transition
            state[k, t+1] ~ A * state[k, t] + B * control[k, t]

            # Path model, the path structure is 2 dimensional, where 
            # [ x_position, y_position ]
            path[k, t] ~ C * state[k, t+1]

            # Environmental distance
            zσ2[k, t] ~ GammaShapeRate(gamma_shape, gamma_scale)
            z[k, t]   ~ g(environment, rs[k], path[k, t])
            
            # Halfspace priors were defined previously in this experiment
            z[k, t] ~ Halfspace(0, zσ2[k, t], γ)

        end

        # goal priors (indexing reverse due to definition)
        goals[1, k] ~ MvNormal(mean = state[k, 1], covariance = goal_constraint_variance * I)
        goals[2, k] ~ MvNormal(mean = state[k, nr_steps+1], covariance = goal_constraint_variance * I)

    end

    for t = 1:nr_steps

        # observation constraint
        dσ2[t] ~ GammaShapeRate(gamma_shape, gamma_scale)
        d[t] ~ h(environment, rs, path[1, t], path[2, t], path[3, t], path[4, t])
        d[t] ~ Halfspace(0, dσ2[t], γ)

    end

end

@constraints function path_planning_constraints()
    # Mean-field variational constraints on the parameters
    q(d, dσ2) = q(d)q(dσ2)
    q(z, zσ2) = q(z)q(zσ2)
end

function path_planning(; environment, agents, nr_iterations = 350, nr_steps = 40, seed = 42, save_intermediates = false, intermediate_steps = 10, output_dir = nothing, model_params = nothing)
    println("Starting path planning inference...")
    # Fixed number of agents
    nr_agents = 4

    # Form goals compatible with the model
    goals = hcat(
        map(agents) do agent
            return [
                [ agent.initial_position[1], 0, agent.initial_position[2], 0 ],
                [ agent.target_position[1], 0, agent.target_position[2], 0 ]
            ]
        end...
    )
    
    rng = StableRNG(seed)
    
    # Initialize variables, more details about initialization 
    # can be found in the original paper
    init = @initialization begin

        q(dσ2) = repeat([PointMass(1)], nr_steps)
        q(zσ2) = repeat([PointMass(1)], nr_agents, nr_steps)
        q(control) = repeat([PointMass(0)], nr_steps)

        μ(state) = MvNormalMeanCovariance(randn(rng, 4), 100I)
        μ(path) = MvNormalMeanCovariance(randn(rng, 2), 100I)

    end

    # Define approximation methods for the non-linear functions used in the model
    # `Linearization` is a simple and fast approximation method, but it is not
    # the most accurate one. For more details about the approximation methods,
    # please refer to the RxInfer documentation
    door_meta = @meta begin 
        h() -> Linearization()
        g() -> Linearization()
    end

    # Configure callback to save intermediate results if requested
    if save_intermediates && output_dir !== nothing
        callback = on_iteration((i, pv, elbo) -> begin
            if i % intermediate_steps == 0
                intermediate_dir = joinpath(output_dir, "intermediates")
                mkpath(intermediate_dir)
                paths = mean.(pv[:path])
                open(joinpath(intermediate_dir, "paths_iteration_$(i).csv"), "w") do io
                    println(io, "agent,step,x,y")
                    for k in 1:nr_agents
                        for t in 1:nr_steps
                            println(io, "$(k),$(t),$(paths[k, t][1]),$(paths[k, t][2])")
                        end
                    end
                end
                open(joinpath(intermediate_dir, "elbo.csv"), "a") do io
                    println(io, "$(i),$(elbo)")
                end
            end
            return false
        end)
        
        callback_opts = (callbacks = callback,)
    else
        callback_opts = ()
    end

    println("Running inference with $(nr_iterations) iterations...")
    results = infer(
        model 			= path_planning_model(environment = environment, agents = agents, nr_steps = nr_steps, model_params = model_params),
        data  			= (goals = goals, ),
        initialization  = init,
        constraints 	= path_planning_constraints(),
        meta 			= door_meta,
        iterations 		= nr_iterations,
        returnvars 		= KeepLast(), 
        options         = (limit_stack_depth = 300, callback_opts...)
    )
    println("Inference completed.")
    return results
end

end # module 