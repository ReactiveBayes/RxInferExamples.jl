using RxInfer
using Distributions

@model function pomdp_model(p_A, p_B, p_goal, p_control, previous_control, p_previous_state, current_y, future_y, T, m_A, m_B)
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state

    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)

    prev_state = current_state
    for t in 1:T
        controls[t] ~ p_control
        s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
        future_y[t] ~ DiscreteTransition(s[t], m_A)
        prev_state = s[t]
    end
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

function build_pomdp()
    p_A = DirichletCollection(diageye(25) .+ 0.1)
    p_B = DirichletCollection(ones(25, 25, 4))
    return p_A, p_B, init, constraints
end


