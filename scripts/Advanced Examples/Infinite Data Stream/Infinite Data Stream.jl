# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Infinite Data Stream/Infinite Data Stream.ipynb
# by notebooks_to_scripts.jl at 2025-04-21T06:26:04.817
#
# Source notebook: Infinite Data Stream.ipynb
# Refactored for terminal execution, modularity, and documentation.

using RxInfer, Plots, Random, StableRNGs
using Logging, Dates, Printf
using Statistics # Added for std() function

# Setup basic console logger
global_logger(ConsoleLogger(stderr, Logging.Info))

# Apply a plot theme
theme(:default)

# == Environment Simulation ==

"""
    Environment(current_state, observation_precision; seed=123)

Represents the simulated environment state.

# Fields
- `rng`: Random number generator instance.
- `current_state`: The current internal state value used for generating the signal.
- `observation_precision`: The precision of the noise added to the observation.
- `history`: A vector storing the history of the true hidden states.
- `observations`: A vector storing the history of noisy observations.
"""
mutable struct Environment
    rng                   :: AbstractRNG
    current_state         :: Float64
    observation_precision :: Float64
    history               :: Vector{Float64}
    observations          :: Vector{Float64}
end

"""
    Environment(current_state::Real, observation_precision::Real; seed::Int = 123)

Constructs an `Environment` object.
"""
function Environment(current_state::Real, observation_precision::Real; seed::Int = 123)
     return Environment(StableRNG(seed), Float64(current_state), Float64(observation_precision), Float64[], Float64[])
end

"""
    getnext!(environment::Environment)

Advances the environment one time step:
1. Updates the internal state.
2. Generates the next true hidden state (sine wave).
3. Generates a noisy observation based on the true state.
4. Stores the true state and observation in the environment's history.

# Returns
- The generated noisy observation for the current step.
"""
function getnext!(environment::Environment)
    environment.current_state = environment.current_state + 1.0
    # The underlying hidden state follows a sine wave
    nextstate  = 10sin(0.1 * environment.current_state) 
    # Generate observation with Gaussian noise
    observation = rand(environment.rng, NormalMeanPrecision(nextstate, environment.observation_precision))
    push!(environment.history, nextstate)
    push!(environment.observations, observation)
    return observation
end

"""
    gethistory(environment::Environment) -> Vector{Float64}

Returns the history of true hidden states recorded in the environment.
"""
gethistory(environment::Environment) = environment.history

"""
    getobservations(environment::Environment) -> Vector{Float64}

Returns the history of noisy observations recorded in the environment.
"""
getobservations(environment::Environment) = environment.observations

"""
    simulate_environment_data!(environment::Environment, n_steps::Int)

Runs the environment simulation for a specified number of steps.

# Arguments
- `environment`: The `Environment` object to simulate.
- `n_steps`: The number of simulation steps to perform.
"""
function simulate_environment_data!(environment::Environment, n_steps::Int)
    @info "Generating $(n_steps) data points..."
    for _ in 1:n_steps
        getnext!(environment)
    end
    @info "Data generation complete."
end


# == RxInfer Model Definition ==

#=
    kalman_filter(x_prev_mean, x_prev_var, τ_shape, τ_rate, y)

Defines the probabilistic model for a Kalman filter using RxInfer's `@model` macro.
It models a random walk for the hidden state `x_current` based on the previous state `x_prev`,
and an observation `y` which depends on the current state `x_current` and an unknown precision `τ`.
The precision `τ` itself is given a Gamma prior.

# Arguments
- `x_prev_mean`, `x_prev_var`: Mean and variance of the prior belief about the previous state `x_prev`.
- `τ_shape`, `τ_rate`: Shape and rate parameters for the Gamma prior on the observation precision `τ`.
- `y`: The observed data point at the current time step.
=#
@model function kalman_filter(x_prev_mean, x_prev_var, τ_shape, τ_rate, y)
    # Prior belief about the previous state
    x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
    
    # Prior belief about the observation precision
    τ ~ Gamma(shape = τ_shape, rate = τ_rate)

    # State transition model: Random walk with fixed precision (variance = 1.0)
    x_current ~ Normal(mean = x_prev, precision = 1.0)
    
    # Observation model: Observation `y` depends on the current state `x_current` and precision `τ`
    y ~ Normal(mean = x_current, precision = τ)    
end

#=
    filter_constraints()

Defines the mean-field factorization constraints for the variational distribution `q` 
using RxInfer's `@constraints` macro.
It assumes that the joint posterior `q(x_prev, x_current, τ)` factorizes as `q(x_prev, x_current) * q(τ)`.
=#
@constraints function filter_constraints()
    q(x_prev, x_current, τ) = q(x_prev, x_current)q(τ)
end

"""
    create_autoupdates()

Defines the update rules for the priors based on the posteriors from the previous time step,
using RxInfer's `@autoupdates` macro.
The mean and variance of the prior `x_prev` for the next step are updated using the posterior `q(x_current)` from the current step.
The shape and rate of the prior `τ` for the next step are updated using the posterior `q(τ)` from the current step.
"""
function create_autoupdates()
    return @autoupdates begin 
        # Update prior for x_prev based on posterior of x_current
        x_prev_mean, x_prev_var = mean_var(q(x_current))
        # Update prior for τ based on posterior of τ
        τ_shape = shape(q(τ))
        τ_rate = rate(q(τ))
    end
end

"""
    create_initialization()

Defines the initial priors/variational distributions for the inference engine,
using RxInfer's `@initialization` macro.
Initializes `q(x_current)` as a broad Gaussian and `q(τ)` as a standard Gamma distribution.
"""
function create_initialization()
    return @initialization begin
        q(x_current) = NormalMeanVariance(0.0, 1e3) # Vague prior for the initial state
        q(τ) = GammaShapeRate(1.0, 1.0)            # Standard Gamma prior for precision
    end
end


# == Inference Execution ==

"""
    run_static_inference(observations::Vector{Float64}; iterations=10)

Runs the Kalman filter inference on a static (complete) dataset.

# Arguments
- `observations`: A vector of observations.
- `iterations`: The number of variational message passing iterations per data point.

# Returns
- The RxInfer inference engine object containing results and history.
"""
function run_static_inference(observations::Vector{Float64}; iterations::Int=10)
    
    # Convert observations to RxInfer's expected datastream format
    datastream = from(observations) |> map(NamedTuple{(:y,), Tuple{Float64}}, (d) -> (y = d, ))

    engine = infer(
        model          = kalman_filter(),
        constraints    = filter_constraints(),
        datastream     = datastream,
        autoupdates    = create_autoupdates(),
        initialization = create_initialization(),
        returnvars     = (:x_current, :τ), # Return posteriors for state and precision
        keephistory    = length(observations), # Keep history for all steps
        historyvars    = (x_current = KeepLast(), τ = KeepLast()), # Specify what to store
        iterations     = iterations,
        free_energy    = true,          # Calculate Bethe Free Energy
        autostart      = true           # Start inference immediately
    )
    
    return engine
end

"""
    run_realtime_inference(datastream, posterior_callback; iterations=10)

Sets up and starts the Kalman filter inference for a real-time data stream.

# Arguments
- `datastream`: An RxInfer compatible observable representing the stream of observations.
- `posterior_callback`: A function to be called with the posterior of `x_current` after each update.
- `iterations`: The number of variational message passing iterations per data point.

# Returns
- The RxInfer inference engine object.
"""
function run_realtime_inference(datastream, posterior_callback; iterations::Int=10)

    engine = infer(
        model         = kalman_filter(),
        constraints   = filter_constraints(),
        datastream    = datastream,
        autoupdates   = create_autoupdates(),
        initialization = create_initialization(),
        returnvars    = (:x_current, ), # Only need state posterior for the callback
        iterations    = iterations,
        autostart     = false,          # Do not start automatically, requires `RxInfer.start(engine)`
    )
    
    # Subscribe the callback function to the posterior stream of x_current
    qsubscription = subscribe!(engine.posteriors[:x_current], posterior_callback)
    
    # Manually start the engine
    RxInfer.start(engine)
    
    return engine
end


# == Visualization Functions ==

"""
    generate_environment_animation(env_history::Vector{Float64}, env_observations::Vector{Float64}, filename::String)

Generates and saves a GIF animation showing the evolution of the environment's hidden state and observations.
"""
function generate_environment_animation(env_history::Vector{Float64}, env_observations::Vector{Float64}, filename::String)
    n_frames = length(env_observations)
    animation = @animate for i in 1:n_frames
        p = plot(size = (1000, 300), title = "Environment Simulation (Step $i/$n_frames)")
        plot!(p, 1:i, env_history[1:i], label = "Hidden signal", lw=2)
        scatter!(p, 1:i, env_observations[1:i], ms = 4, alpha = 0.7, label = "Observation")
    end
    gif(animation, filename, fps = 24, show_msg = false)
    @info "Saved environment simulation animation to: $(filename)"
end

"""
    plot_static_inference_results(result, history::Vector{Float64}, observations::Vector{Float64}, true_precision::Float64, output_dir::String)

Plots the results of the static inference:
1. Estimated vs true state and observations.
2. Estimation error (Estimated Mean - True State).
3. Estimated observation precision (τ) vs true value.
4. Bethe Free Energy.
Saves the combined plot to a file.
"""
function plot_static_inference_results(result, history::Vector{Float64}, observations::Vector{Float64}, true_precision::Float64, output_dir::String)
    n = length(observations)
    timestamps = 1:n
    
    # Extract results
    estimated_states = result.history[:x_current]
    estimated_taus = result.history[:τ]
    free_energy = result.free_energy_history
    
    # Calculate means and standard deviations
    state_means = mean.(estimated_states)
    state_stds = std.(estimated_states)
    tau_means = mean.(estimated_taus)
    tau_stds = std.(estimated_taus)
    estimation_error = state_means .- history
    
    # --- Create Subplots ---
    
    # 1. State Estimation Plot
    p1 = plot(timestamps, state_means, ribbon = state_stds, label = "Estimation (Mean ± 1 SD)", 
              xlabel="Time step", ylabel="State Value", title="State Estimation vs True State", legend=:bottomright)
    plot!(p1, timestamps, history, label = "True states", lw=2, color=:black, linestyle=:dash)
    scatter!(p1, timestamps, observations, ms = 2, label = "Observations", alpha=0.6, markerstrokewidth=0)

    # 2. Estimation Error Plot
    p2 = plot(timestamps, estimation_error, label="Estimation Error", 
              xlabel="Time step", ylabel="Error (Est - True)", title="State Estimation Error")
    hline!(p2, [0], label=nothing, color=:black, linestyle=:dash)

    # 3. Tau (Precision) Estimation Plot
    p3 = plot(timestamps, tau_means, ribbon = tau_stds, label = "Estimation (Mean ± 1 SD)", 
              xlabel="Time step", ylabel="Precision (τ)", title="Observation Precision Estimation", legend=:topright)
    hline!(p3, [true_precision], label="True Precision", color=:red, linestyle=:dash, lw=2)

    # 4. Free Energy Plot
    # Plot FE against iteration number (length of history), not timestamp, as FE is usually calculated per iteration for the last data point
    fe_iterations = 1:length(free_energy)
    p4 = plot(fe_iterations, free_energy, label = nothing, 
              xlabel="Iteration (last data point)", ylabel="Free Energy", title="Bethe Free Energy Convergence", legend=false)

    # --- Combine and Save ---
    combined_plot = plot(p1, p2, p3, p4, layout = (4, 1), size = (1000, 1200)) # Adjusted size for 4 plots
    results_filename = joinpath(output_dir, "infinite-data-stream-static-summary.png")
    savefig(combined_plot, results_filename)
    @info "Saved combined static inference results plot to: $(results_filename)"

    # Remove old separate saving logic
    # results_filename = joinpath(output_dir, "infinite-data-stream-static-results.png")
    # savefig(final_plot, results_filename)
    # @info "Saved static inference results plot to: $(results_filename)"
    # if !isempty(result.free_energy_history)
    #     fe_plot = plot(...) 
    #     fe_filename = joinpath(output_dir, "infinite-data-stream-static-fe.png")
    #     savefig(fe_plot, fe_filename)
    #     @info "Saved static inference free energy plot to: $(fe_filename)"
    # else
    #     @warn "Free energy history is empty, skipping plot."
    # end
end

"""
    generate_static_state_animation(result, history::Vector{Float64}, observations::Vector{Float64}, filename::String)

Generates and saves a GIF animation showing the evolution of the state estimation during static inference.
"""
function generate_static_state_animation(result, history::Vector{Float64}, observations::Vector{Float64}, filename::String)
    n = length(observations)
    estimated_states = result.history[:x_current]
    state_means = mean.(estimated_states)
    state_stds = std.(estimated_states)
    
    animation = @animate for i in 1:n
        p = plot(1:i, state_means[1:i], ribbon = state_stds[1:i], label = "Estimation (Mean ± 1 SD)", 
                 xlabel="Time step", ylabel="State Value", title="Static State Estimation (Step $i/$n)", legend=:bottomright,
                 xlims=(1, n), ylims=(minimum(history)-2, maximum(history)+2)) # Adjust ylims as needed
        plot!(p, 1:i, history[1:i], label = "True states", lw=2, color=:black, linestyle=:dash)
        scatter!(p, 1:i, observations[1:i], ms = 2, label = "Observations", alpha=0.6, markerstrokewidth=0)
    end
    
    gif(animation, filename, fps = 24, show_msg = false)
    @info "Saved static state estimation animation to: $(filename)"
end

"""
    generate_static_tau_animation(result, true_precision::Float64, filename::String)

Generates and saves a GIF animation showing the evolution of the Tau (precision) estimation during static inference.
"""
function generate_static_tau_animation(result, true_precision::Float64, filename::String)
    estimated_taus = result.history[:τ]
    n = length(estimated_taus)
    tau_means = mean.(estimated_taus)
    tau_stds = std.(estimated_taus)
    
    # Determine reasonable y-limits, avoiding extremes from initial steps if necessary
    ymin = minimum(tau_means[max(1, n-100):n]) - 2*maximum(tau_stds[max(1, n-100):n]) # Look at last 100 steps
    ymax = maximum(tau_means[max(1, n-100):n]) + 2*maximum(tau_stds[max(1, n-100):n])
    ymin = max(0.01, ymin) # Precision must be positive
    ymax = max(ymin + 0.1, ymax) # Ensure some range

    animation = @animate for i in 1:n
        p = plot(1:i, tau_means[1:i], ribbon = tau_stds[1:i], label = "Estimation (Mean ± 1 SD)", 
                 xlabel="Time step", ylabel="Precision (τ)", title="Static Precision Estimation (Step $i/$n)", legend=:topright,
                 xlims=(1, n), ylims=(ymin, ymax))
        hline!(p, [true_precision], label="True Precision", color=:red, linestyle=:dash, lw=2)
    end
    
    gif(animation, filename, fps = 24, show_msg = false)
    @info "Saved static Tau estimation animation to: $(filename)"
end

"""
    create_realtime_plot_callback(environment::Environment, output_dir::String)

Creates a callback function suitable for `run_realtime_inference`. 
This callback plots the current state of inference (posterior estimate vs true state vs observations) 
and saves the plot to a file in the specified `output_dir`.
"""
function create_realtime_plot_callback(environment::Environment, output_dir::String)
    posteriors = [] # Stores the history of posteriors within the closure
    plot_count = Ref(0) # Mutable counter for naming plots

    # Ensure the output directory for these plots exists
    mkpath(output_dir) 

    # Define and return the callback function
    return (q_current) -> begin 
        try
            push!(posteriors, q_current)
            plot_count[] += 1
            current_step = plot_count[]

            # Get current state of the environment (up to the current step)
            current_history = gethistory(environment)[1:current_step]
            current_observations = getobservations(environment)[1:current_step]

            # Create the plot
            p = plot(mean.(posteriors), ribbon = var.(posteriors), label = "Estimation (Mean ± 1 SD)")
            plot!(p, 1:current_step, current_history, label = "True states", lw=2)    
            scatter!(p, 1:current_step, current_observations, ms = 2, label = "Observations", alpha=0.7)
            plot!(p, size = (1000, 300), legend = :bottomright, title="Real-time Inference (Step $(current_step))", xlabel="Time step")

            # Save the plot
            plot_filename = @sprintf("realtime_step_%04d.png", current_step)
            savefig(p, joinpath(output_dir, plot_filename)) 
        catch e
            @error "Error during plotting callback at step $(plot_count[])" exception=(e, catch_backtrace())
        end
    end
end


# == Main Execution Logic ==

function main()
    @info "Starting Infinite Data Stream script..."
    timestamp_str = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    base_output_dir = joinpath(@__DIR__, "output", timestamp_str)
    mkpath(base_output_dir)
    @info "Output will be saved to subdirectories within: $(base_output_dir)"

    # --- Parameters ---
    initial_state         = 0.0
    observation_precision = 0.1 # This is the TRUE precision τ
    n_static              = 300 # Number of points for static analysis
    n_env_animation       = 100 # Number of points for environment animation
    n_realtime            = 100 # Number of points for real-time simulation (if IJulia)
    inference_iterations  = 10

    # --- Environment Simulation & Animation ---
    @info "Simulating environment for animation..."
    anim_env = Environment(initial_state, observation_precision)
    simulate_environment_data!(anim_env, n_env_animation)
    generate_environment_animation(
        gethistory(anim_env), 
        getobservations(anim_env), 
        joinpath(base_output_dir, "environment_simulation.gif")
    )

    # --- Static Inference ---
    @info "Setting up static inference environment..."
    static_env = Environment(initial_state, observation_precision)
    simulate_environment_data!(static_env, n_static)
    
    @info "Running static inference..."
    static_results = run_static_inference(getobservations(static_env); iterations=inference_iterations)
    @info "Static inference completed."

    # --- Static Results Visualization ---
    @info "Plotting static inference results summary..."
    plot_static_inference_results(
        static_results, 
        gethistory(static_env), 
        getobservations(static_env), 
        observation_precision, # Pass the true precision for comparison
        base_output_dir
    )
    
    @info "Generating static inference animations..."
    # Generate State Estimation Animation
    generate_static_state_animation(
        static_results,
        gethistory(static_env),
        getobservations(static_env),
        joinpath(base_output_dir, "static_state_estimation_animation.gif")
    )
    # Generate Tau Estimation Animation
    generate_static_tau_animation(
        static_results,
        observation_precision,
        joinpath(base_output_dir, "static_tau_estimation_animation.gif")
    )

    # --- Real-time Inference (IJulia only) ---
    # This part uses RxJulia's timer and relies on an interactive environment.
    # It is skipped during normal terminal execution.
    if isdefined(Main, :IJulia) && Main.IJulia.inited
        @info "Running real-time inference example (IJulia detected)..."
        realtime_output_dir = joinpath(base_output_dir, "realtime_plots")
        # Note: Environment state needs to be managed carefully for the callback
        # We create a new environment instance just for this part
        realtime_env = Environment(initial_state, observation_precision); 
        
        # Create the datastream using RxJulia timer
        timegen_ms = 41 
        @info "Setting up $(n_realtime) data points stream for real-time inference (interval: $(timegen_ms)ms)..."
        # map -> getnext! populates the environment's history/observations as data flows
        observations_stream = timer(timegen_ms, timegen_ms) |> 
                              map(Float64, (_) -> getnext!(realtime_env)) |> 
                              take(n_realtime) 
        
        # Convert to the format RxInfer expects
        datastream = observations_stream |> map(NamedTuple{(:y,), Tuple{Float64}}, (d) -> (y = d, ))

        # Create the plotting callback
        plotting_callback = create_realtime_plot_callback(realtime_env, realtime_output_dir)
        
        # Setup and start the inference
        @info "Starting real-time inference engine. Plots will be saved to: $(realtime_output_dir)"
        realtime_engine = run_realtime_inference(datastream, plotting_callback; iterations=inference_iterations)
        
        # Note: The engine runs asynchronously. In a script, we might need to wait 
        # or handle its lifecycle explicitly if we needed to do something *after* it finishes.
        # For now, we just let it run (typical in interactive sessions).
        # Consider adding `wait(realtime_engine)` or similar if needed downstream.
        
    else
        @info "Skipping real-time plotting example (requires IJulia environment)."
    end

    @info "Infinite Data Stream script finished."
end

# Execute the main function
main()

# Removed old function definitions and execution blocks (comments deleted for clarity)