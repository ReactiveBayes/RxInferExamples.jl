module LVAVisualization

using Plots, DelimitedFiles, Printf, Dates, Statistics, LinearAlgebra, StatsBase, JSON, Distributions, SpecialFunctions
using Statistics: mean, median, std, quantile, cor
using LinearAlgebra: diag, tr, norm, svd
using Plots: Animation, @animate, gif
using Distributions: Normal
using SpecialFunctions: erfinv

# Define log_message function used in this module
function log_message(message; level="INFO")
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    if level == "INFO"
        println("[$timestamp] [INFO] $message")
    elseif level == "WARNING"
        println("[$timestamp] [WARNING] $message")
    elseif level == "ERROR"
        println("[$timestamp] [ERROR] $message")
    elseif level == "DEBUG"
        println("[$timestamp] [DEBUG] $message")
    end
end

# Function to create standard plot theme for consistent styling
function set_custom_theme()
    theme(:wong)  # Use a colorblind-friendly theme
    default(
        fontfamily="Computer Modern",
        linewidth=2,
        framestyle=:box,
        label=nothing,
        grid=false,
        palette=:darktest,
        legendfontsize=8,
        tickfontsize=8,
        guidefontsize=10,
        titlefontsize=12,
        margin=5Plots.mm,
        size=(800, 600)
    )
end

# Function to create basic visualizations
function create_visualizations(mresult, true_data, observations, n_samples, n_missing, test_rmse_by_process, output_dir)
    log_message("Generating basic visualization plots")
    
    set_custom_theme()
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    log_message("Output directory created at: $output_dir")
    
    # Calculate average RMSE for title
    avg_test_rmse = mean(test_rmse_by_process)
    
    combined_plot = plot(layout = (3, 1), size = (800, 900), legend = :topleft)
    
    # Plotting options
    marker_alpha = 0.7 
    marker_size = 5  
    ribbon_alpha = 0.3
    observation_color = :green
    
    # Define the training range indices
    train_indices = 1:(n_samples - n_missing)
    # Extract observations for the training range
    train_observations = observations[train_indices]
    
    # Indices to plot
    plot_indices = [5, 10, 15]
    
    for (plot_idx, index) in enumerate(plot_indices)
        log_message("Generating plot for process $index", level="DEBUG")
        
        # Plot predictions with uncertainty
        plot!(combined_plot[plot_idx], 
              getindex.(mean.(mresult.predictions[:y][end]), index), 
              ribbon = getindex.(diag.(cov.(mresult.predictions[:y][end])), index), 
              fillalpha=ribbon_alpha, 
              label = "Inferred $(index)",
              linewidth=2)
        
        # Plot true data
        plot!(combined_plot[plot_idx], 
              getindex.(true_data, index), 
              label = "True $(index)",
              linewidth=2,
              linestyle=:dash)
        
        # Plot observations (training data)
        scatter!(combined_plot[plot_idx], 
                 train_indices, 
                 getindex.(train_observations, index), 
                 label = "Observations $(index)", 
                 marker=:xcross, 
                 markeralpha=marker_alpha, 
                 markersize=marker_size, 
                 color=observation_color)
        
        # Add training/test split line
        vline!(combined_plot[plot_idx], 
               [n_samples-n_missing], 
               label=(plot_idx==1 ? "Training/Test split" : ""), 
               linestyle=:dash, 
               color=:black)
        
        # Add RMSE to the title
        plot!(combined_plot[plot_idx], 
              title = @sprintf("LVAR Process %d (Test RMSE: %.4f)", index, test_rmse_by_process[index]))
    end
    
    # Set title for first subplot to serve as combined title
    plot!(combined_plot[1], 
         title = @sprintf("Latent Vector Autoregressive Model Predictions (Average Test RMSE: %.4f)", avg_test_rmse))
    
    # Save plot to file
    plot_filename = joinpath(output_dir, "lvar_predictions.png")
    savefig(combined_plot, plot_filename)
    log_message("Basic prediction plot saved to: $plot_filename")
    
    return plot_filename
end

# Function to create heatmap of prediction errors across all processes
function create_error_heatmap(mresult, true_data, n_samples, n_missing, n_ar_processes, output_dir)
    log_message("Generating error heatmap")
    
    # Extract predictions and true values
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    actual_values = getindex.(true_data, :)
    
    # Calculate errors for all time points and processes
    errors = zeros(n_samples, n_ar_processes)
    for t in 1:n_samples
        for p in 1:n_ar_processes
            # For missing data points in training, use actual observation
            if t <= n_samples
                errors[t, p] = predicted_means[t][p] - actual_values[t][p]
            end
        end
    end
    
    # Highlight test period with a different color scheme
    test_indices = (n_samples-n_missing+1):n_samples
    
    # Create heatmap
    p = heatmap(1:n_ar_processes, 1:n_samples, errors, 
                color=:RdBu, 
                aspect_ratio=:auto,
                xlabel="Process Index",
                ylabel="Time Step",
                title="Prediction Error Heatmap",
                colorbar_title="Error (Pred - True)",
                size=(800, 600))
                
    # Add a horizontal line to separate train and test regions
    hline!([n_samples-n_missing+0.5], color=:black, linewidth=2, label="Train-Test Split")
    
    # Save figure
    heatmap_filename = joinpath(output_dir, "error_heatmap.png")
    savefig(p, heatmap_filename)
    log_message("Error heatmap saved to: $heatmap_filename")
    
    return heatmap_filename
end

# Create correlation plots between processes
function create_correlation_plots(mresult, true_data, n_ar_processes, output_dir)
    log_message("Generating correlation plots")
    
    # Extract predicted means
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    
    # Calculate correlation matrix of predictions
    pred_data = zeros(length(predicted_means), n_ar_processes)
    for t in 1:length(predicted_means)
        for p in 1:n_ar_processes
            pred_data[t, p] = predicted_means[t][p]
        end
    end
    
    # Calculate correlation matrices
    pred_corr = cor(pred_data)
    
    # Create true data matrix
    true_matrix = zeros(length(true_data), n_ar_processes)
    for t in 1:length(true_data)
        for p in 1:n_ar_processes
            true_matrix[t, p] = true_data[t][p]
        end
    end
    true_corr = cor(true_matrix)
    
    # Create correlation heatmaps
    p1 = heatmap(pred_corr, 
                title="Predicted Process Correlations", 
                xlabel="Process Index", 
                ylabel="Process Index",
                color=:viridis,
                aspect_ratio=:equal,
                clim=(-1,1),
                size=(600, 600))
                
    p2 = heatmap(true_corr, 
                title="True Process Correlations", 
                xlabel="Process Index", 
                ylabel="Process Index",
                color=:viridis,
                aspect_ratio=:equal,
                clim=(-1,1),
                size=(600, 600))
                
    # Create correlation difference heatmap
    p3 = heatmap(pred_corr - true_corr, 
                title="Correlation Difference (Pred - True)", 
                xlabel="Process Index", 
                ylabel="Process Index",
                color=:RdBu,
                aspect_ratio=:equal,
                clim=(-0.5,0.5),
                size=(600, 600))
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(1,3), size=(1800, 600))
    
    # Save figure
    corr_filename = joinpath(output_dir, "correlation_plots.png")
    savefig(p, corr_filename)
    log_message("Correlation plots saved to: $corr_filename")
    
    return corr_filename
end

# Create uncertainty visualization
function create_uncertainty_plots(mresult, true_data, n_samples, n_missing, output_dir)
    log_message("Generating uncertainty visualization")
    
    # Calculate the uncertainty (standard deviation) across all processes and time steps
    uncertainties = [sqrt.(diag(cov.(mresult.predictions[:y][end])[t])) for t in 1:n_samples]
    
    # Flatten for plotting
    all_uncertainties = vcat(uncertainties...)
    
    # Create histogram of uncertainties
    p1 = histogram(all_uncertainties, 
                  bins=30, 
                  title="Distribution of Prediction Uncertainties",
                  xlabel="Standard Deviation",
                  ylabel="Count",
                  legend=false,
                  alpha=0.7,
                  size=(600, 400))
    
    # Calculate prediction errors for test set
    test_indices = (n_samples-n_missing+1):n_samples
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    actual_values = getindex.(true_data, :)
    
    # Collect errors and corresponding uncertainties
    test_errors = Float64[]
    test_uncertainties = Float64[]
    
    for t in test_indices
        for p in 1:length(predicted_means[t])
            err = predicted_means[t][p] - actual_values[t][p]
            unc = sqrt(diag(cov.(mresult.predictions[:y][end])[t])[p])
            push!(test_errors, err)
            push!(test_uncertainties, unc)
        end
    end
    
    # Scatter plot of errors vs uncertainties
    p2 = scatter(test_uncertainties, abs.(test_errors),
                xlabel="Predicted Uncertainty (Std)",
                ylabel="Absolute Error",
                title="Uncertainty vs. Error (Test Set)",
                legend=false,
                alpha=0.4,
                size=(600, 400))
    
    # Add a trend line
    if length(test_uncertainties) > 1
        # Simple linear regression
        X = [ones(length(test_uncertainties)) test_uncertainties]
        β = X \ abs.(test_errors)
        
        # Plot regression line
        x_range = range(minimum(test_uncertainties), maximum(test_uncertainties), length=100)
        y_pred = [β[1] + β[2]*x for x in x_range]
        plot!(p2, x_range, y_pred, linewidth=2, color=:red, label="Trend")
        
        # Add correlation coefficient
        corr_val = cor(test_uncertainties, abs.(test_errors))
        annotate!(p2, maximum(test_uncertainties)*0.8, maximum(abs.(test_errors))*0.9, 
                 text(@sprintf("Correlation: %.3f", corr_val), 10))
    end
    
    # Combine plots
    p = plot(p1, p2, layout=(1,2), size=(1200, 400))
    
    # Save figure
    uncertainty_filename = joinpath(output_dir, "uncertainty_analysis.png")
    savefig(p, uncertainty_filename)
    log_message("Uncertainty plots saved to: $uncertainty_filename")
    
    return uncertainty_filename
end

# Create temporal evolution of model parameters
function create_parameter_evolution_plot(mresult, n_ar_processes, output_dir)
    log_message("Generating parameter evolution plot")
    
    # Check if parameter history is available
    if !haskey(mresult, :history) || !haskey(mresult.history, :θ) || !haskey(mresult.history, :γ) || !haskey(mresult.history, :τ)
        # If no parameter history, create a dummy plot with a note
        p = plot(title="Parameter Evolution During Inference",
                xlabel="Iteration",
                ylabel="Mean Parameter Value",
                legend=false,
                size=(800, 500))
                
        # Add text annotation explaining lack of parameter history
        annotate!(p, 0.5, 0.5, text("Parameter history not available\nEnable history tracking in the inference call with 'historyvars'", 12, :center))
        
        # Save figure
        param_filename = joinpath(output_dir, "parameter_evolution.png")
        savefig(p, param_filename)
        log_message("Parameter evolution plot saved to: $param_filename (dummy version - history not available)")
        
        return param_filename
    end
    
    # Get the parameter histories
    θ_history = mresult.history[:θ]
    γ_history = mresult.history[:γ]
    τ_history = mresult.history[:τ]
    
    # Number of iterations tracked
    n_iterations = length(θ_history)
    x_axis = 1:n_iterations
    
    # Create subplot for precision parameters (γ)
    log_message("Processing γ parameter evolution", level="DEBUG")
    p1 = plot(title="AR Process Precision (γ) Evolution",
            xlabel="Iteration",
            ylabel="Mean γ Value",
            legend=(:topright, 6),
            size=(800, 300))
    
    # Select a subset of AR processes to visualize to avoid overcrowding
    process_indices = n_ar_processes > 5 ? round.(Int, range(1, n_ar_processes, length=5)) : 1:n_ar_processes
    
    for k in process_indices
        if k <= length(γ_history[1])
            # Extract the mean of γ across iterations for this process
            γ_means = [mean(γ_history[i][k]) for i in 1:n_iterations]
            plot!(p1, x_axis, γ_means, label="Process $k", linewidth=2)
        end
    end
    
    # Create subplot for AR coefficients (θ)
    log_message("Processing θ parameter evolution", level="DEBUG")
    p2 = plot(title="AR Coefficient (θ) Evolution",
            xlabel="Iteration",
            ylabel="Mean θ Value",
            legend=(:topright, 6),
            size=(800, 300))
    
    # Select a single process to visualize its coefficients
    selected_process = 1
    if length(θ_history[1]) >= selected_process
        # Get the order of the AR process
        ar_order = length(mean(θ_history[1][selected_process]))
        
        # Extract each coefficient's evolution
        for coef_idx in 1:min(ar_order, 5)
            θ_means = [mean(θ_history[i][selected_process])[coef_idx] for i in 1:n_iterations]
            plot!(p2, x_axis, θ_means, label="θ$coef_idx (Proc $selected_process)", linewidth=2)
        end
    end
    
    # Create subplot for observation precision (τ)
    log_message("Processing τ parameter evolution", level="DEBUG")
    p3 = plot(title="Observation Precision (τ) Evolution",
            xlabel="Iteration",
            ylabel="Mean τ Value",
            legend=false,
            size=(800, 300))
    
    # Extract the mean of τ across iterations
    τ_means = [mean(τ_history[i]) for i in 1:n_iterations]
    plot!(p3, x_axis, τ_means, linewidth=2, color=:purple)
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    
    # Save figure
    param_filename = joinpath(output_dir, "parameter_evolution.png")
    savefig(p, param_filename)
    log_message("Parameter evolution plot saved to: $param_filename")
    
    return param_filename
end

# Create animation of predictions over time
function create_prediction_animation(mresult, true_data, observations, n_samples, n_missing, test_rmse_by_process, output_dir)
    log_message("Generating prediction animation")
    
    set_custom_theme()
    
    # Prepare data
    train_indices = 1:(n_samples - n_missing)
    train_observations = observations[train_indices]
    
    # Select a few processes to visualize
    n_ar_processes = length(test_rmse_by_process)
    plot_indices = [5, 10, 15]
    
    # Create animation
    anim = @animate for t in 1:n_samples
        p = plot(layout=(length(plot_indices),1), size=(800, 600),
                 legend=:topright, link=:x)
        
        # Time range to show: a window around current time
        window_size = 30
        start_t = max(1, t - window_size)
        end_t = min(n_samples, t + 5)
        time_range = start_t:end_t
        
        for (i, idx) in enumerate(plot_indices)
            # Plot true data
            plot!(p[i], time_range, getindex.(true_data[time_range], idx), 
                 label="True", linewidth=2, linestyle=:dash, color=:blue)
            
            # Plot observations up to current time
            obs_range = start_t:min(t, n_samples-n_missing)
            if !isempty(obs_range)
                scatter!(p[i], obs_range, getindex.(observations[obs_range], idx), 
                        label="Observed", marker=:circle, markersize=3, color=:green)
            end
            
            # Plot predictions up to current time with uncertainty
            if t > 1
                pred_range = start_t:t
                pred_means = getindex.(mean.(mresult.predictions[:y][end][pred_range]), idx)
                
                # Calculate standard deviations element-wise
                pred_stds = Float64[]
                for time_idx in pred_range
                    cov_matrix = cov(mresult.predictions[:y][end][time_idx])
                    std_val = sqrt(cov_matrix[idx, idx])
                    push!(pred_stds, std_val)
                end
                
                plot!(p[i], pred_range, pred_means, 
                     ribbon=pred_stds, fillalpha=0.3, 
                     label="Predicted", linewidth=2, color=:red)
            end
            
            # Add vertical line for current time point
            vline!(p[i], [t], color=:black, linestyle=:dot, linewidth=1, label=nothing)
            
            # Add vertical line for train-test split
            if n_samples-n_missing >= start_t && n_samples-n_missing <= end_t
                vline!(p[i], [n_samples-n_missing], color=:black, linestyle=:dash, linewidth=1, 
                      label="Train-Test Split")
            end
            
            # Set title for subplot
            title!(p[i], "Process $idx")
            
            # Only show x-axis label on the bottom plot
            if i == length(plot_indices)
                xlabel!(p[i], "Time Step")
            end
            ylabel!(p[i], "Value")
        end
        
        # Add main title
        title = @sprintf("LVAR Prediction (t=%d/%d)", t, n_samples)
        plot!(title=title)
    end
    
    # Save animation
    animation_filename = joinpath(output_dir, "prediction_animation.gif")
    gif(anim, animation_filename, fps=5)
    log_message("Prediction animation saved to: $animation_filename")
    
    return animation_filename
end

# Create model complexity analysis plots
function create_complexity_analysis(mresult, true_data, n_ar_processes, output_dir)
    log_message("Generating model complexity analysis")
    
    # Extract predicted means and variances
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    predicted_covs = cov.(mresult.predictions[:y][end])
    
    # Calculate total variance per time step
    total_variances = [tr(cov) for cov in predicted_covs]
    
    # Perform SVD on the predicted data matrix to analyze latent structure
    pred_matrix = zeros(length(predicted_means), n_ar_processes)
    for t in 1:length(predicted_means)
        for p in 1:n_ar_processes
            pred_matrix[t, p] = predicted_means[t][p]
        end
    end
    
    # Compute SVD
    svd_result = svd(pred_matrix)
    
    # Calculate cumulative variance explained
    singular_values = svd_result.S
    total_variance = sum(singular_values.^2)
    explained_variance = cumsum(singular_values.^2) ./ total_variance
    
    # Create plot of singular values
    p1 = bar(singular_values[1:min(20, length(singular_values))],
            xlabel="Component",
            ylabel="Singular Value",
            title="Top Singular Values",
            legend=false,
            alpha=0.7,
            size=(600, 400))
            
    # Create plot of cumulative variance explained
    p2 = plot(explained_variance[1:min(20, length(explained_variance))],
             xlabel="Number of Components",
             ylabel="Cumulative Explained Variance",
             title="Variance Explained by Components",
             legend=false,
             marker=:circle,
             linewidth=2,
             size=(600, 400))
    
    # Draw reference line at 95% explained variance
    hline!(p2, [0.95], linestyle=:dash, color=:red, label="95% Explained")
    
    # Combine plots
    p = plot(p1, p2, layout=(1,2), size=(1200, 400))
    
    # Save figure
    complexity_filename = joinpath(output_dir, "complexity_analysis.png")
    savefig(p, complexity_filename)
    log_message("Complexity analysis plots saved to: $complexity_filename")
    
    return complexity_filename
end

# Create residual analysis plots
function create_residual_analysis(mresult, true_data, n_samples, n_missing, output_dir)
    log_message("Generating residual analysis plots")
    
    # Calculate residuals for test set
    test_indices = (n_samples-n_missing+1):n_samples
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    actual_values = getindex.(true_data, :)
    
    # Collect all residuals
    residuals = Float64[]
    for t in test_indices
        for p in 1:length(predicted_means[t])
            res = predicted_means[t][p] - actual_values[t][p]
            push!(residuals, res)
        end
    end
    
    # Create histogram of residuals
    p1 = histogram(residuals, 
                  bins=30, 
                  title="Distribution of Residuals (Test Set)",
                  xlabel="Residual (Pred - True)",
                  ylabel="Count",
                  legend=false,
                  alpha=0.7,
                  size=(600, 400))
    
    # Add a normal distribution fit
    μ = mean(residuals)
    σ = std(residuals)
    x_range = range(minimum(residuals), maximum(residuals), length=100)
    y_normal = [exp(-(x-μ)^2/(2σ^2))/(σ*sqrt(2π)) * length(residuals) * (maximum(residuals)-minimum(residuals))/30 for x in x_range]
    plot!(p1, x_range, y_normal, linewidth=2, color=:red, label="Normal Fit")
    
    # Calculate empirical quantiles
    sort_residuals = sort(residuals)
    emp_quantiles = Float64[]
    for p in range(0.01, 0.99, length=100)
        idx = max(1, min(length(sort_residuals), round(Int, p * length(sort_residuals))))
        push!(emp_quantiles, sort_residuals[idx])
    end
    
    # Simple approximation for theoretical normal quantiles
    # This approximation avoids using erfinv which might not be available
    theo_quantiles = Float64[]
    for p in range(0.01, 0.99, length=100)
        # Simple approximation for normal quantiles
        z = if p < 0.5
            -sqrt(-2 * log(p))
        else
            sqrt(-2 * log(1 - p))
        end
        push!(theo_quantiles, μ + z * σ)
    end
    
    # Create QQ plot
    p2 = scatter(theo_quantiles, emp_quantiles,
                xlabel="Theoretical Quantiles",
                ylabel="Sample Quantiles",
                title="Q-Q Plot of Residuals",
                legend=false,
                size=(600, 400))
    
    # Add reference line
    min_q = min(minimum(theo_quantiles), minimum(emp_quantiles))
    max_q = max(maximum(theo_quantiles), maximum(emp_quantiles))
    plot!(p2, [min_q, max_q], [min_q, max_q], color=:red, linestyle=:dash)
    
    # Combine plots
    p = plot(p1, p2, layout=(1,2), size=(1200, 400))
    
    # Save figure
    residual_filename = joinpath(output_dir, "residual_analysis.png")
    savefig(p, residual_filename)
    log_message("Residual analysis plots saved to: $residual_filename")
    
    return residual_filename
end

# Function to export prediction data to CSV
function export_prediction_data(true_data, mresult, n_samples, plot_indices, output_dir)
    # Prepare data for CSV export
    log_message("Saving prediction results to CSV")
    results_data = Array{Float64}(undef, n_samples, 3*length(plot_indices) + 1)
    results_data[:, 1] = 1:n_samples  # Time index
    
    for (i, idx) in enumerate(plot_indices)
        # True values
        results_data[:, 3*i-1] = getindex.(true_data, idx)
        
        # Predicted means
        results_data[:, 3*i] = getindex.(mean.(mresult.predictions[:y][end]), idx)
        
        # Prediction standard deviations
        results_data[:, 3*i+1] = sqrt.(getindex.(diag.(cov.(mresult.predictions[:y][end])), idx))
    end
    
    # Create header
    header = ["time_index"]
    for idx in plot_indices
        push!(header, "true_$idx", "pred_mean_$idx", "pred_std_$idx")
    end
    
    # Write to CSV
    csv_filename = joinpath(output_dir, "lvar_predictions.csv")
    open(csv_filename, "w") do io
        writedlm(io, [header], ',')
        writedlm(io, results_data, ',')
    end
    log_message("Results saved to: $csv_filename")
    
    return csv_filename
end

# Export detailed statistics to JSON
function export_detailed_statistics(mresult, true_data, test_rmse_by_process, n_samples, n_missing, output_dir)
    log_message("Exporting detailed statistics")
    
    # Calculate additional statistics
    test_indices = (n_samples-n_missing+1):n_samples
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    actual_values = getindex.(true_data, :)
    
    # Calculate overall statistics
    n_ar_processes = length(test_rmse_by_process)
    avg_rmse = mean(test_rmse_by_process)
    median_rmse = median(test_rmse_by_process)
    min_rmse = minimum(test_rmse_by_process)
    max_rmse = maximum(test_rmse_by_process)
    rmse_std = std(test_rmse_by_process)
    
    # Calculate average uncertainty (prediction standard deviation)
    avg_uncertainty = mean([mean(sqrt.(diag(cov.(mresult.predictions[:y][end])[t]))) for t in test_indices])
    
    # Prepare statistics in JSON format
    stats = Dict(
        "overall_statistics" => Dict(
            "avg_rmse" => avg_rmse,
            "median_rmse" => median_rmse,
            "min_rmse" => min_rmse,
            "max_rmse" => max_rmse,
            "rmse_std" => rmse_std,
            "avg_uncertainty" => avg_uncertainty
        ),
        "process_statistics" => Dict(
            "rmse_by_process" => test_rmse_by_process
        ),
        "model_parameters" => Dict(
            "n_processes" => n_ar_processes,
            "n_samples" => n_samples,
            "test_samples" => n_missing,
            "training_samples" => n_samples - n_missing
        )
    )
    
    # Write to JSON file
    json_filename = joinpath(output_dir, "model_statistics.json")
    open(json_filename, "w") do io
        JSON.print(io, stats, 4) # Pretty print with 4-space indentation
    end
    log_message("Detailed statistics saved to: $json_filename")
    
    return json_filename
end

# Combined function to handle all enhanced visualization and data export
function visualize_and_export(mresult, true_data, observations, n_samples, n_missing, test_rmse_by_process)
    # Create output directory with timestamp
    output_dir = joinpath(dirname(@__FILE__), "results", Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS"))
    mkpath(output_dir)
    
    n_ar_processes = length(test_rmse_by_process)
    plot_indices = [5, 10, 15]
    
    # Initialize results dictionary
    results = Dict(
        "output_dir" => output_dir
    )
    
    # Generate basic plots
    log_message("Generating all visualizations...")
    
    # Use try-catch blocks to handle potential errors in each visualization
    
    # Basic prediction plots
    try
        results["basic_plot"] = create_visualizations(mresult, true_data, observations, n_samples, n_missing, test_rmse_by_process, output_dir)
    catch e
        log_message("Error generating basic visualizations: $e", level="ERROR")
    end
    
    # Error heatmap
    try
        results["error_heatmap"] = create_error_heatmap(mresult, true_data, n_samples, n_missing, n_ar_processes, output_dir)
    catch e
        log_message("Error generating error heatmap: $e", level="ERROR")
    end
    
    # Correlation plots
    try
        results["correlation_plots"] = create_correlation_plots(mresult, true_data, n_ar_processes, output_dir)
    catch e
        log_message("Error generating correlation plots: $e", level="ERROR")
    end
    
    # Uncertainty plots
    try
        results["uncertainty_plots"] = create_uncertainty_plots(mresult, true_data, n_samples, n_missing, output_dir)
    catch e
        log_message("Error generating uncertainty plots: $e", level="ERROR")
    end
    
    # Parameter evolution (may not be available in all result objects)
    try
        results["parameter_plot"] = create_parameter_evolution_plot(mresult, n_ar_processes, output_dir)
    catch e
        log_message("Error generating parameter evolution plot: $e", level="ERROR")
    end
    
    # Model complexity analysis
    try
        results["complexity_plot"] = create_complexity_analysis(mresult, true_data, n_ar_processes, output_dir)
    catch e
        log_message("Error generating complexity analysis: $e", level="ERROR")
    end
    
    # Residual analysis
    try
        results["residual_plot"] = create_residual_analysis(mresult, true_data, n_samples, n_missing, output_dir)
    catch e
        log_message("Error generating residual analysis: $e", level="ERROR")
    end
    
    # Prediction animation
    try
        results["animation_file"] = create_prediction_animation(mresult, true_data, observations, n_samples, n_missing, test_rmse_by_process, output_dir)
    catch e
        log_message("Error generating prediction animation: $e", level="ERROR")
    end
    
    # Export data to CSV
    try
        results["csv_file"] = export_prediction_data(true_data, mresult, n_samples, plot_indices, output_dir)
    catch e
        log_message("Error exporting prediction data: $e", level="ERROR")
    end
    
    # Export detailed statistics as JSON
    try
        results["stats_file"] = export_detailed_statistics(mresult, true_data, test_rmse_by_process, n_samples, n_missing, output_dir)
    catch e
        log_message("Error exporting detailed statistics: $e", level="ERROR")
    end
    
    log_message("All visualizations and data exports completed")
    
    return results
end

end # module 