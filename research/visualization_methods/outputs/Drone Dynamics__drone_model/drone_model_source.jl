@model function drone_model(drone, environment, initial_state, goal, horizon, dt)	

	# extract environment properties
	g = get_gravity(environment)

	# extract drone properties
	m = get_mass(drone)

	# initial state prior
	s[1] ~ MvNormal(mean = initial_state, covariance = 1e-5 * I)

	for i in 1:horizon

		# prior on actions (mean compensates for gravity)
		u[i] ~ MvNormal(μ = [m * g / 2, m * g / 2], Σ = diageye(2))

		# state transition
		s[i + 1] ~ MvNormal(
            μ = state_transition(s[i], u[i], drone, environment, dt), 
			Σ = 1e-10 * I
		)
	end