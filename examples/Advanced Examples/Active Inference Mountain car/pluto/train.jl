using RxInfer
using JLD2
# We are going to use some private functionality from ReactiveMP,
# in the future we should expose a proper API for this
import RxInfer.ReactiveMP: getrecent, messageout
include("utils.jl")

engine_force_limit   = 0.04
friction_coefficient = 0.1

Fa, Ff, Fg, height = create_physics(
    engine_force_limit = engine_force_limit,
    friction_coefficient = friction_coefficient
);
initial_position = -0.5
initial_velocity = 0.0

x_target = [0.5, 0.0]

valley_x = range(-2, 2, length=400)
valley_y = [ height(xs) for xs in valley_x ]

@model function mountain_car(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Fg, Fa, Ff, engine_force_limit)
    
    # Transition function modeling transition due to gravity and friction
    g = (s_t_min::AbstractVector) -> begin 
        s_t = similar(s_t_min) # Next state
        s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2]) # Update velocity
        s_t[1] = s_t_min[1] + s_t[2] # Update position
        return s_t
    end
    
    # Function for modeling engine control
    h = (u::AbstractVector) -> [0.0, Fa(u[1])] 
    
    # Inverse engine force, from change in state to corresponding engine force
    h_inv = (delta_s_dot::AbstractVector) -> [atanh(clamp(delta_s_dot[2], -engine_force_limit+1e-3, engine_force_limit-1e-3)/engine_force_limit)] 
    
    # Internal model perameters
    Gamma = 1e4*diageye(2) # Transition precision
    Theta = 1e-4*diageye(2) # Observation variance

    s_t_min ~ MvNormal(mean = m_s_t_min, cov = V_s_t_min)
    s_k_min = s_t_min

    local s
    
    for k in 1:T
        u[k] ~ MvNormal(mean = m_u[k], cov = V_u[k])
        u_h_k[k] ~ h(u[k]) where { meta = DeltaMeta(method = Linearization(), inverse = h_inv) }
        s_g_k[k] ~ g(s_k_min) where { meta = DeltaMeta(method = Linearization()) }
        u_s_sum[k] ~ s_g_k[k] + u_h_k[k]
        s[k] ~ MvNormal(mean = u_s_sum[k], precision = Gamma)
        x[k] ~ MvNormal(mean = s[k], cov = Theta)
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k]) # x[1] is the current observation (u[1] is action leading to that observation). x[end] === x[T] is the goal
        s_k_min = s[k]
    end
    
    return (s, )
end

function create_agent(;T = 20, Fg, Fa, Ff, engine_force_limit, x_target, initial_position, initial_velocity)
    huge = 1e6
    tiny = 1e-6
    Epsilon = fill(huge, 1, 1)                # Control prior variance
    m_u = Vector{Float64}[ [ 0.0] for k=1:T ] # Set control priors
    V_u = Matrix{Float64}[ Epsilon for k=1:T ]

    Sigma    = 1e-4*diageye(2) # Goal prior variance
    m_x      = [zeros(2) for k=1:T]
    V_x      = [huge*diageye(2) for k=1:T]
    V_x[end] = Sigma # Set prior to reach goal at t=T

    # Set initial brain state prior
    m_s_t_min = [initial_position, initial_velocity]
    V_s_t_min = tiny * diageye(2)

    # Set current inference results
    result = nothing

    # The `infer` function is the heart of the agent
    # It calls the `RxInfer.inference` function to perform Bayesian inference by message passing
    compute = (upsilon_t::Float64, y_hat_t::Vector{Float64}) -> begin
        m_u[1] = [ upsilon_t ] # Register action with the generative model
        V_u[1] = fill(tiny, 1, 1) # Clamp control prior to performed action

        m_x[1] = y_hat_t # Register observation with the generative model
        V_x[1] = tiny*diageye(2) # Clamp goal prior to observation

        data = Dict(:m_u       => m_u,
                    :V_u       => V_u,
                    :m_x       => m_x,
                    :V_x       => V_x,
                    :m_s_t_min => m_s_t_min,
                    :V_s_t_min => V_s_t_min)

        model  = mountain_car(T = T, Fg = Fg, Fa = Fa, Ff = Ff, engine_force_limit = engine_force_limit)
        result = infer(model = model, data = data)
    end

    # The `act` function returns the inferred best possible action
    act = () -> begin
        if result !== nothing
            return mode(result.posteriors[:u][2])[1]
        else
            return 0.0 # Without inference result we return some 'random' action
        end
    end

    # The `future` function returns the inferred future states
    future = () -> begin
        if result !== nothing
            return getindex.(mode.(result.posteriors[:s]), 1)
        else
            return zeros(T)
        end
    end

    # The `slide` function modifies the `(m_s_t_min, V_s_t_min)` for the next step
    # and shifts (or slides) the array of future goals `(m_x, V_x)` and inferred actions `(m_u, V_u)`
    slide = () -> begin

        model  = RxInfer.getmodel(result.model)
        (s, )  = RxInfer.getreturnval(model)
        varref = RxInfer.getvarref(model, s)
        var    = RxInfer.getvariable(varref)

        slide_msg_idx = 3 # This index is model dependend
        (m_s_t_min, V_s_t_min) = mean_cov(getrecent(messageout(var[2], slide_msg_idx)))

        m_u = circshift(m_u, -1)
        m_u[end] = [0.0]
        V_u = circshift(V_u, -1)
        V_u[end] = Epsilon

        m_x = circshift(m_x, -1)
        m_x[end] = x_target
        V_x = circshift(V_x, -1)
        V_x[end] = Sigma
    end

    return (compute, act, slide, future)
end

(execute_ai, observe_ai) = create_world(
    Fg = Fg, Ff = Ff, Fa = Fa,
    initial_position = initial_position,
    initial_velocity = initial_velocity
) # Let there be a world

T_ai = 50

(compute_ai, act_ai, slide_ai, future_ai) = create_agent(; # Let there be an agent
    T  = T_ai,
    Fa = Fa,
    Fg = Fg,
    Ff = Ff,
    engine_force_limit = engine_force_limit,
    x_target = x_target,
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

N_ai = 100


# Step through experimental protocol
agent_a = zeros(N_ai) # agent action

var_dims = Dict(
    :s => 2, # latent states
    :u => 1, # controls
    :x => 2, # observations
    :u_h_k => 2, # engine controls i.e. velocities
    :u_s_sum => 2, # no comment...
    :s_g_k => 2, # next state... no comment...
)

agent_f = zeros(T_ai, N_ai) # predicted future for each timestep in planning horizon
agent_x = zeros(var_dims[:x], N_ai)

means = Dict{Symbol, Array}()
covs  = Dict{Symbol, Array}()

for (sym, D) in var_dims
    means[sym] = zeros(D, T_ai, N_ai)
    covs[sym]  = zeros(D, D, T_ai, N_ai)
end

for t=1:N_ai
    agent_a[t] = act_ai()               # Invoke an action from the agent
    agent_f[:, t] = future_ai()            # Fetch the predicted future states
    execute_ai(agent_a[t])              # The action influences hidden external states
    agent_x[:, t] = observe_ai()           # Observe the current environmental outcome (update p)
    results = compute_ai(agent_a[t], agent_x[:, t]) # Infer beliefs from current model state (update q)
    for sym in keys(var_dims)
        if haskey(results.posteriors, sym)
            means[sym][:, :, t] .= hcat(mean.(results.posteriors[sym])...)
            covs[sym][:, :, :, t] .= stack(cov.(results.posteriors[sym]); dims = 3)
        end
    end
    slide_ai()                          # Prepare for next iteration
end

mean_pairs = Dict(Symbol("$(k)_means") => v for (k, v) in means)
cov_pairs  = Dict(Symbol("$(k)_covs") => v for (k, v) in covs)

DIR_RESULTS = "results"
mkpath(DIR_RESULTS)
jldsave("$DIR_RESULTS/data.jld2";
    agent_a, agent_f, agent_x,
    engine_force_limit, friction_coefficient,
    initial_position, initial_velocity,
    x_target, valley_x, valley_y,
    mean_pairs..., cov_pairs...,
    var_dims
)
