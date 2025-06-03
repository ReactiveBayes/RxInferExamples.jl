# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Structural Dynamics with Augmented Kalman Filter/Structural Dynamics with Augmented Kalman Filter.ipynb
# by notebooks_to_scripts.jl at 2025-06-03T10:14:29.254
#
# Source notebook: Structural Dynamics with Augmented Kalman Filter.ipynb

using LinearAlgebra, Statistics, Random, Plots

# define a data structure for the structural model environment
struct StructuralModelData
    t::Union{Nothing, Any}
    ndof::Union{Nothing, Int64}
    nf::Union{Nothing, Int64}
    N_data::Union{Nothing, Int64}
    y_meas::Union{Nothing, Vector{Vector{Float64}}}
    A_aug::Union{Nothing, Matrix{Float64}}
    G_aug::Union{Nothing, Matrix{Float64}}
    G_aug_fullfield::Union{Nothing, Matrix{Float64}}
    Q_akf::Union{Nothing, Matrix{Float64}}
    R::Union{Nothing, LinearAlgebra.Diagonal{Float64, Vector{Float64}}}
    x_real::Union{Nothing, Matrix{Float64}}
    y_real::Union{Nothing, Matrix{Float64}}
    p_real::Union{Nothing, Matrix{Float64}}
end

# define the structural system matrices
struct StructuralMatrices
    M::Union{Nothing, Matrix{Float64}}
    K::Union{Nothing, Matrix{Float64}}
    C::Union{Nothing, Matrix{Float64}}
end


M = I(4)


K = [
    2  -1   0    0;
   -1   2  -1    0;
    0  -1   2   -1;
    0   0  -1    1
] * 1e3

C = [
    2   -1    0    0;
   -1    2   -1    0;
    0   -1    2   -1;
    0    0   -1    1
]

StructuralModel = StructuralMatrices(M, K, C);

# function to construct the state space model
function construct_ssm(StructuralModel,dt, ndof, nf)
    # unpack the structural model
    M = StructuralModel.M
    K = StructuralModel.K
    C = StructuralModel.C
    
    
    Sp = zeros(ndof, nf)
    Sp[4, 1] = 1

    Z = zeros(ndof, ndof)
    Id = I(ndof)

    A_continuous = [Z Id;
                    -(M \ K) -(M \ C)]
    B_continuous = [Z; Id \ M] * Sp

    A = exp(dt * A_continuous)
    B = (A - I(2*ndof)) * A_continuous \ B_continuous

    return A, B, Sp
end

# function to generate random input noise
function generate_input(N_data::Int, nf::Int; input_mu::Float64, input_std::Float64)
    Random.seed!(42)
    p_real = input_mu .+ randn(N_data, nf) .* input_std
    return p_real
end

# function to generate the measurements and noise
function generate_measurements(ndof, na, nv, nd, N_data, x_real, y_real, p_real, StructuralModel, Sp)
    # unpack the structural model
    M = StructuralModel.M
    K = StructuralModel.K
    C = StructuralModel.C
    
    Sa = zeros(na, ndof)            # selection matrix
    Sa[1, 1] = 1                    # acceleration at node 1
    Sa[2, 4] = 1                    # acceleration at node 4
    G = Sa * [-(M \ K) -(M \ C)] 
    J = Sa * (I \ M) * Sp

    ry = Statistics.var(y_real[2*ndof+1, :], ) * (0.1^2)        # simulate noise as 1% RMS of the noise-free acceleration response

    nm = na + nv + nd

    R = I(nm) .* ry

    y_meas = zeros(nm, N_data)
    y_noise = sqrt(ry) .* randn(nm, N_data)

    # reconstruct the measurements
    y_meas = Vector{Vector{Float64}}(undef, N_data)
    for i in 1:N_data
        y_meas[i] = G * x_real[:, i] + J * p_real[i, :] + y_noise[:, i]
    end

    return y_meas, G, J, R
end

# function to simulate the structural response
function simulate_response(A, B, StructuralModel, Sp, nf, ndof, N_data)
    # unpack the structural model
    M = StructuralModel.M
    K = StructuralModel.K
    C = StructuralModel.C
    
    p_real = generate_input(N_data, nf, input_mu = 0.0, input_std = 0.05)

    Z = zeros(ndof, ndof)
    Id = I(ndof)
    
    G_full = [
            Id Z;
            Z Id;
            -(M \ K) -(M \ C)
            ]

    J_full = [
        Z;
        Z;
        Id \ M
    ] * Sp
    
    # preallocate matrices
    x_real = zeros(2 * ndof, N_data)
    y_real = zeros(3 * ndof, N_data)

    for i in 2:N_data
        x_real[:, i] = A * x_real[:, i-1] + B * p_real[i-1, :]
        y_real[:, i] = G_full * x_real[:, i-1] + J_full * p_real[i-1, :]
    end

    return x_real, y_real, p_real, G_full, J_full
end 

# function to construct the augmented model
function construct_augmented_model(A, B, G, J, G_full, J_full, nf, ndof)
    Z_aug = zeros(nf, 2*ndof)
    A_aug = [
        A B;
        Z_aug I(nf)
        ]
    G_aug = [G J]

    G_aug_fullfield = [G_full J_full]                               # full-field augmented matrix

    Qp_aug = I(nf) * 1e-2                                           # assumed known or pre-callibrated
    
    Qx_aug = zeros(2*ndof, 2*ndof)
    Qx_aug[(ndof+1):end, (ndof+1):end] = I(ndof) * 1e-1             # assumed known or pre-callibrated

    Q_akf = [
        Qx_aug Z_aug';
        Z_aug Qp_aug
    ]

    return A_aug, G_aug, Q_akf, G_aug_fullfield
end

function get_structural_model(StructuralModel, simulation_time, dt)

    # intialize
    ndof = size(StructuralModel.M)[1]                               # number of degrees of freedom
    nf = 1                                                          # number of inputs
    na, nv, nd = 2, 0, 0                                            # number of oberved accelerations, velocities, and displacements
    N_data = Int(simulation_time / dt) + 1
    t = range(0, stop=simulation_time, length=N_data)

    # construct state-space model from structural matrices
    A, B, Sp = construct_ssm(StructuralModel, dt, ndof, nf)

    # Generate input and simulate response
    x_real, y_real, p_real, G_full, J_full = simulate_response(A, B, StructuralModel, Sp, nf, ndof, N_data)

    # Generate measurements
    y_meas, G, J, R = generate_measurements(ndof, na, nv, nd, N_data, x_real, y_real, p_real, StructuralModel, Sp)

    # Construct augmented model
    A_aug, G_aug, Q_akf, G_aug_fullfield = construct_augmented_model(A, B, G, J, G_full, J_full, nf, ndof)

    return StructuralModelData(t, ndof, nf, N_data, y_meas, A_aug, G_aug, G_aug_fullfield, Q_akf, R, x_real, y_real, p_real)
end

simulation_time = 5.0
dt = 0.001

model_data = get_structural_model(StructuralModel, simulation_time, dt);

using RxInfer

@model function smoother_model(y, x0, A, G, Q, R)

    x_prior ~ x0
    x_prev = x_prior  # initialize previous state with x_prior

    for i in 1:length(y)
        x[i] ~ MvNormal(mean = A * x_prev, cov = Q)
        y[i] ~ MvNormal(mean = G * x[i], cov = R)
        x_prev = x[i]
    end

end

# RxInfer returns the result in its own structure. 
# Here we wrap the results in a different struct for the example's convenience
struct InferenceResults
    state_marginals
    y_full_means
    y_full_stds
    p_means
    p_stds
end

function run_smoother(model_data)
    # unpack the model data
    t               = model_data.t;
    N_data          = model_data.N_data
    A_aug           = model_data.A_aug;
    G_aug           = model_data.G_aug;
    G_aug_fullfield = model_data.G_aug_fullfield;
    Q_akf           = model_data.Q_akf;
    R               = model_data.R;
    y_meas          = model_data.y_meas;
    
    # initialize the state - required when doing smoothing
    x0 = MvNormalMeanCovariance(zeros(size(A_aug, 1)), Q_akf);

    # define the smoother engine
    function smoother_engine(y_meas, A, G, Q, R)
        # run the akf smoother
        result_smoother = infer(
            model   = smoother_model(x0 = x0, A = A, G = G, Q = Q, R = R),
            data    = (y = y_meas,),
            options = (limit_stack_depth = 500, ) # This setting is required for large models
        )

        # return posteriors as this inference task returns the results as posteriors
        # because inference is done over the full graph
        return result_smoother.posteriors[:x]
    end

    # get the marginals of x
    state_marginals = smoother_engine(y_meas, A_aug, G_aug, Q_akf, R)
    
    # reconstructing the full-field response:
    # use helper function to reconstruct the full-field response
    y_full_means, y_full_stds = reconstruct_full_field(state_marginals, G_aug_fullfield, N_data)

    # extract the estimated input (input modeled as an augmentation state)
    p_results_means = getindex.(mean.(state_marginals), length(state_marginals[1]))
    p_results_stds = getindex.(std.(state_marginals), length(state_marginals[1]))
    
    return InferenceResults(state_marginals, y_full_means, y_full_stds, p_results_means, p_results_stds)
end

# helper function to reconstruct the full field response from the state posteriors
function reconstruct_full_field(
    x_marginals,
    G_aug_fullfield,
    N_data::Int
)
    
    # preallocate the full field response
    y_means = Vector{Vector{Float64}}(undef, N_data)        # vector of vectors
    y_stds = Vector{Vector{Float64}}(undef, N_data)

    # reconstruct the full-field response using G_aug_fullfield
    for i in 1:N_data
        # extract the mean and covariance of the state posterior
        state_mean = mean(x_marginals[i])       # each index is a vector
        state_cov = cov(x_marginals[i])

        # project mean and covariance onto the full-field response space
        y_means[i] = G_aug_fullfield * state_mean
        y_stds[i] = sqrt.(diag(G_aug_fullfield * state_cov * G_aug_fullfield'))
    end

    return y_means, y_stds

end

# run the smoother
smoother_results = run_smoother(model_data);

# helper function
function plot_with_uncertainty(
    t,
    true_values,
    estimated_means,
    estimated_uncertainties,
    ylabel_text,
    title_text,
    label_suffix="";
    plot_size = (700,300),
   
)
    # plot true values
    plt = plot(
        t,
        true_values,
        label="true ($label_suffix)",
        lw=2,
        color=:blue,
        size=plot_size,
        left_margin = 5Plots.mm,
        top_margin = 5Plots.mm,  
        bottom_margin = 5Plots.mm  
    )

    # plot estimated values with uncertainty ribbon
    plot!(
        plt,
        t,
        estimated_means,
        ribbon=estimated_uncertainties,
        fillalpha=0.3,
        label="estimated ($label_suffix)",
        lw=2,
        color=:orange,
        linestyle=:dash
    )

    # add labels and title
    xlabel!("time (s)")
    ylabel!(ylabel_text)
    title!(title_text)
    
    return plt
end

# select some DOFs to plot
ndof = size(StructuralModel.M)[1]

display_state_dof    = 4                # dof 1:4 displacements, dof 5:8 velocities
display_response_dof = 2*ndof + 1       # dof 1:4 displacements, dof 5:8 velocities, dof 9:12 accelerations
display_input_dof    = 1                # the only one really

# plot the states
state_plot = plot_with_uncertainty(
    model_data.t,
    model_data.x_real[display_state_dof, :],
    getindex.(mean.(smoother_results.state_marginals), display_state_dof),
    getindex.(std.(smoother_results.state_marginals), display_state_dof),
    "state value",
    "state estimate (dof $(display_state_dof))",
    "state dof $(display_state_dof)"
);

# plot the responses
response_plot = plot_with_uncertainty(
    model_data.t,
    model_data.y_real[display_response_dof, :],
    getindex.(smoother_results.y_full_means, display_response_dof),
    getindex.(smoother_results.y_full_stds, display_response_dof),
    "response value",
    "reconstructed response (dof $(display_response_dof))",
    "response dof $(display_response_dof)"
);

# plot the inputs
input_plot = plot_with_uncertainty(
    model_data.t,
    model_data.p_real[:, display_input_dof],
    smoother_results.p_means,
    smoother_results.p_stds,
    "force value",
    "input estimate (applied at dof $(display_input_dof))",
    "input force $(display_input_dof)"
);

display(state_plot)
display(response_plot)
display(input_plot)
