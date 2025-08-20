using RxInfer
using Distributions
using RxEnvironments

# utils and model are included into the module namespace by POMDPControl.jl
const _grid_location_to_index = grid_location_to_index
const _index_to_one_hot = index_to_one_hot
const _pomdp_model = pomdp_model
const _build_pomdp = build_pomdp

function run_pomdp_experiments(; n_experiments::Int=100, T::Int=4, goal_pos::Tuple{Int,Int}=(4,3))
    env = RxEnvironment(WindyGridWorld((0, 1, 1, 1, 0), [], goal_pos))
    agent = add!(env, WindyGridWorldAgent((1, 1)))
    observations = keep(Any)
    RxEnvironments.subscribe_to_observations!(agent, observations)

    p_A, p_B, init_loc, constraints_loc = _build_pomdp()
    goal = Categorical(_index_to_one_hot(_grid_location_to_index(goal_pos)))

    successes = Bool[]

    for i in 1:n_experiments
        reset_env!(env)
        p_s = Categorical(_index_to_one_hot(_grid_location_to_index((1, 1))))
        policy = [Categorical([0.0, 0.0, 1.0, 0.0])]
        prev_u = [0.0, 0.0, 1.0, 0.0]

        for t in 1:T
            current_action = mode(first(policy))
            if current_action == 1
                send!(env, agent, (0, 1))
                prev_u = [1.0, 0.0, 0.0, 0.0]
            elseif current_action == 2
                send!(env, agent, (1, 0))
                prev_u = [0.0, 1.0, 0.0, 0.0]
            elseif current_action == 3
                send!(env, agent, (0, -1))
                prev_u = [0.0, 0.0, 1.0, 0.0]
            elseif current_action == 4
                send!(env, agent, (-1, 0))
                prev_u = [0.0, 0.0, 0.0, 1.0]
            end

            last_observation = _index_to_one_hot(_grid_location_to_index(RxEnvironments.data(last(observations))))

            inference_result = infer(
                model = _pomdp_model(
                    p_A = p_A,
                    p_B = p_B,
                    T = max(T - t, 1),
                    p_previous_state = p_s,
                    p_goal = goal,
                    p_control = vague(Categorical, 4),
                    m_A = mean(p_A),
                    m_B = mean(p_B)
                ),
                data = (
                    previous_control = prev_u,
                    current_y = last_observation,
                    future_y = UnfactorizedData(fill(missing, max(T - t, 1)))
                ),
                constraints = constraints_loc,
                initialization = init_loc,
                iterations = 10
            )

            p_s = last(inference_result.posteriors[:current_state])
            policy = last(inference_result.posteriors[:controls])

            p_A = last(inference_result.posteriors[:A])
            p_B = last(inference_result.posteriors[:B])

            if RxEnvironments.data(last(observations)) == goal_pos
                break
            end
        end

        push!(successes, RxEnvironments.data(last(observations)) == goal_pos)
    end

    return (; env, agent, successes, p_A, p_B)
end


