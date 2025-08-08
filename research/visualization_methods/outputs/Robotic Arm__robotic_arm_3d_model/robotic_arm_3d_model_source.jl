@model function robotic_arm_3d_model(arm, environment, initial_state, goal, horizon, dt)
    # Extract properties
    g = get_gravity(environment)
    num_links, _, link_masses, _ = get_properties(arm)
    
    # Initial state prior
    s[1] ~ MvNormal(mean = initial_state, covariance = 1e-5 * I)
    
    for i in 1:horizon
        # Prior on torques - compensate for gravity at each joint
        # For 3D arm: first joint (yaw) not affected by gravity, 
        # pitch joints affected based on angle
        gravity_compensation = zeros(2*num_links)
        for j in 1:num_links
            if j > 1  # Skip first joint (base yaw)
                gravity_compensation[2*j-1] = link_masses[j] * g * 0.5  # Pitch compensation
            end