# This file was automatically generated from examples/Basic Examples/POMDP Control/POMDP Control.ipynb
# by notebooks_to_scripts.jl at 2025-03-14T05:52:02.131
#
# Source notebook: POMDP Control.ipynb

using RxInfer
using Distributions
using Plots
using Random
using ProgressMeter

using RxEnvironments
using Plots

struct WindyGridWorld{N}
    wind::NTuple{N,Int}
    agents::Vector
    goal::Tuple{Int,Int}
end

mutable struct WindyGridWorldAgent
    position::Tuple{Int,Int}
end



RxEnvironments.update!(env::WindyGridWorld, dt) = nothing # The environment has no "internal" updating process over time

function RxEnvironments.receive!(env::WindyGridWorld{N}, agent::WindyGridWorldAgent, action::Tuple{Int,Int}) where {N}
    if action[1] != 0
        @assert action[2] == 0 "Only one of the two actions can be non-zero"
    elseif action[2] != 0
        @assert action[1] == 0 "Only one of the two actions can be non-zero"
    end
    new_position = (agent.position[1] + action[1], agent.position[2] + action[2] + env.wind[agent.position[1]])
    if all(elem -> 0 < elem < N, new_position)
        agent.position = new_position
    end
end

function RxEnvironments.what_to_send(env::WindyGridWorld, agent::WindyGridWorldAgent)
    return agent.position
end

function RxEnvironments.what_to_send(agent::WindyGridWorldAgent, env::WindyGridWorld)
    return agent.position
end

function RxEnvironments.add_to_state!(env::WindyGridWorld, agent::WindyGridWorldAgent)
    push!(env.agents, agent)
end

function reset_env!(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    for agent in env.agents
        agent.position = (1, 1)
    end
    for subscriber in RxEnvironments.subscribers(environment)
        send!(subscriber, environment, (1, 1))
    end
end

function plot_environment(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    p1 = scatter([env.goal[1]], [env.goal[2]], color=:blue, label="Goal", xlims=(0, 6), ylims=(0, 6))
    for agent in env.agents
        p1 = scatter!([agent.position[1]], [agent.position[2]], color=:red, label="Agent")
    end
    return p1
end

env = RxEnvironment(WindyGridWorld((0, 1, 1, 1, 0), [], (4, 3)))
agent = add!(env, WindyGridWorldAgent((1, 1)))
plot_environment(env)

@model function pomdp_model(p_A, p_B, p_goal, p_control, previous_control, p_previous_state, current_y, future_y, T, m_A, m_B)
    # Instantiate all model parameters with priors
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state
    
    # Paremeter inference
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)

    prev_state = current_state
    # Inference-as-planning
    for t in 1:T
        controls[t] ~ p_control
        s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
        future_y[t] ~ DiscreteTransition(s[t], m_A)
        prev_state = s[t]
    end
    # Goal prior initialization
    s[end] ~ p_goal
end

init = @initialization begin
    q(A) = DirichletCollection(diageye(25) .+ 0.1)
    q(B) = DirichletCollection(ones(25, 25, 4))
end

constraints = @constraints begin
    q(previous_state, previous_control, current_state, B) = q(previous_state, previous_control, current_state)q(B)
    q(current_state, current_y, A) = q(current_state, current_y)q(A)
    q(current_state, s, controls, B) = q(current_state, s, controls), q(B)
    q(s, future_y, A) = q(s, future_y), q(A)
end

p_A = DirichletCollection(diageye(25) .+ 0.1)
p_B = DirichletCollection(ones(25, 25, 4))

function grid_location_to_index(pos::Tuple{Int, Int})
    return (pos[2] - 1) * 5 + pos[1]
end

function index_to_grid_location(index::Int)
    return (index % 5, index ÷ 5 + 1,)
end

function index_to_one_hot(index::Int)
    return [i == index ? 1.0 : 0.0 for i in 1:25]
end

goal = Categorical(index_to_one_hot(grid_location_to_index((4, 3))))


# Number of times to run the experiment
n_experiments = 100
# Number of steps in each experiment
T = 4
observations = keep(Any)
# Subscribe the agent to receive observations
RxEnvironments.subscribe_to_observations!(agent, observations)
successes = []


@showprogress for i in 1:n_experiments
    # Reset environment to initial state and initialize state belief to starting position (1,1)
    reset_env!(env)
    p_s = Categorical(index_to_one_hot(grid_location_to_index((1, 1))))
    # Initialize previous action as "down", as this is neutral from the starting position
    policy = [Categorical([0.0, 0.0, 1.0, 0.0])]
    prev_u = [0.0, 0.0, 1.0, 0.0]
    # Run for T-1 steps in each experiment
    for t in 1:T

         # Convert policy to actual movement in environment
         current_action = mode(first(policy))
         if current_action == 1
             send!(env, agent, (0, 1))  # Move up 
             prev_u = [1.0, 0.0, 0.0, 0.0]
         elseif current_action == 2
             send!(env, agent, (1, 0))  # Move right
             prev_u = [0.0, 1.0, 0.0, 0.0]
         elseif current_action == 3
             send!(env, agent, (0, -1))  # Move down
             prev_u = [0.0, 0.0, 1.0, 0.0]
         elseif current_action == 4
             send!(env, agent, (-1, 0))  # Move left
             prev_u = [0.0, 0.0, 0.0, 1.0]
         end

        # Get last observation and convert to one-hot encoding
        last_observation = index_to_one_hot(grid_location_to_index(RxEnvironments.data(last(observations))))
        
        # Perform inference using the POMDP model
        inference_result = infer(
            model = pomdp_model(
                p_A = p_A,  # prior on observation model parameters
                p_B = p_B,  # prior on transition model parameters
                T = max(T - t, 1),  # remaining time steps
                p_previous_state = p_s,  # posterior belief on previous state
                p_goal = goal,  # prior on goal state
                p_control = vague(Categorical, 4),  # prior over controls
                m_A = mean(p_A),
                m_B = mean(p_B)
            ),
            # Provide data for inference
            data = (
                previous_control = UnfactorizedData(prev_u),
                current_y = UnfactorizedData(last_observation),
                future_y = UnfactorizedData(fill(missing, max(T - t, 1)))
            ),
            constraints = constraints,
            initialization = init,
            iterations = 10
        )
        
        # Update beliefs based on inference results
        p_s = last(inference_result.posteriors[:current_state])  # Update state belief
        policy = last(inference_result.posteriors[:controls])  # Get policy

        # Update model parameters globally for the entire notebook
        global p_A = last(inference_result.posteriors[:A])  # Update observation model
        global p_B = last(inference_result.posteriors[:B])  # Update transition model

        if RxEnvironments.data(last(observations)) == (4, 3)
            break
        end
    end
    if RxEnvironments.data(last(observations)) == (4, 3)
        push!(successes, true)
    else
        push!(successes, false)
    end
end



mean(successes)

plot_environment(env)