
function generate_switching_data(T, A1, A2, c, Q, R, x_0)
    # Initialize arrays to store states and observations
    x = zeros(2, T)  # State matrix: 2 dimensions × T timesteps
    y = zeros(2, T)  # Observation matrix: 2 dimensions × T timesteps
    
    # Set initial state
    x[:,1] = x_0
    
    # Generate state transitions and observations
    for t in 2:T
        # Switch dynamics multiple times through the sequence
        if t < T/3 || (t >= T/2 && t < 3T/4)
            x[:,t] = A2 * x[:,t-1] + rand(MvNormal(zeros(2), Q))  # First regime
        else
            x[:,t] = A1 * x[:,t-1] + rand(MvNormal(zeros(2), Q))  # Second regime
        end
        
        # Generate observation from current state
        y[:,t] = c * x[:,t] + rand(MvNormal(zeros(2), R))
    end

    return x, y
end
        

# System parameters
T = 1000  # Time horizon
θ = π / 15  # Rotation angle

# Define system matrices
A1 = [cos(θ) -sin(θ); sin(θ) cos(θ)]    # Rotation matrix
A2 = [0.3 -0.1; -0.1 0.5]         
c = [0.6 0.0; 0.0 0.2]                   # Observation/distortion matrix

# Noise parameters
Q = [1.0 0.0; 0.0 1.0]                   # State noise covariance
R = 0.1 * [1.0 0.0; 0.0 1.0]            # Observation noise variance
x_0 = [0.0, 0.0]                         # Initial state vector

# Generate synthetic data
x, y = generate_switching_data(T, A1, A2, c, Q, R, x_0)
y = [y[:,i] for i in 1:T]
x = [x[:,i] for i in 1:T]