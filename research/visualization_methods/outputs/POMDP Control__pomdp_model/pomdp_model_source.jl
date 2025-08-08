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