@model function vestibular_perception_model(y, As, Q, B, R)
    # This model represents how the brain might process proprioceptive signals
    # y: proprioceptive measurements
    # x: perceptual states (conscious awareness of posture)
    # As: dynamically learned transition matrices
    # Q: process noise in perceptual system
    # B: observation matrix
    # R: proprioceptive measurement noise
    
    # Prior beliefs about initial postural state
    x_prior_mean = ones(Float32, 3)
    x_prior_cov  = Matrix(Diagonal(ones(Float32, 3)))
    
    # Initial perceptual state and measurement
    x[1] ~ MvNormal(mean = x_prior_mean, cov = x_prior_cov)
    y[1] ~ MvNormal(mean = B * x[1], cov = R)
    
    # Perceptual dynamics and measurements over time
    for i in 2:length(y)
        # Perceptual state evolution (conscious awareness)
        x[i] ~ MvNormal(mean = As[i - 1] * x[i - 1], cov = Q) 
        # Proprioceptive measurement given perceptual state
        y[i] ~ MvNormal(mean = B * x[i], cov = R)
    end