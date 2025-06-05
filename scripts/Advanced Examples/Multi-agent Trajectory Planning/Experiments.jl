module Experiments

using ..Environment
using ..Models
using ..Visualizations

export execute_and_save_animation, run_all_experiments

function execute_and_save_animation(environment, agents; gifname = "result.gif", kwargs...)
    println("Planning paths for environment with $(length(environment.obstacles)) obstacles...")
    
    # Check if a logger is provided in kwargs
    logger = get(kwargs, :logger, nothing)
    
    # Extract the output directory
    output_dir = dirname(gifname)
    output_dir = (output_dir == "") ? "." : output_dir
    
    # Run the inference
    result = path_planning(environment = environment, agents = agents; kwargs...)
    
    # Extract path means for visualization
    paths = mean.(result.posteriors[:path])
    
    # Create animation and save it
    animate_paths(environment, agents, paths; filename = gifname)
    
    # Check for ELBO tracking
    elbo_tracked = false
    elbo_values = Float64[]
    
    # Extract ELBO values if available
    if hasfield(typeof(result), :diagnostics) && haskey(result.diagnostics, :elbo)
        elbo_values = result.diagnostics[:elbo]
        
        if !isempty(elbo_values)
            elbo_tracked = true
            
            # Log success in ELBO tracking
            log_msg = "ELBO tracking successful. Collected $(length(elbo_values)) values."
            if logger !== nothing
                log_message(log_msg, logger)
            else
                println(log_msg)
            end
            
            # Create ELBO convergence plot
            log_msg = "Generating ELBO convergence plot..."
            if logger !== nothing
                log_message(log_msg, logger)
            else
                println(log_msg)
            end
            
            plot_elbo_convergence(elbo_values, filename = joinpath(output_dir, "convergence.png"))
            
            # Save raw ELBO data
            metrics_file = joinpath(output_dir, "convergence_metrics.csv")
            open(metrics_file, "w") do f
                for (i, val) in enumerate(elbo_values)
                    println(f, "$i,$val")
                end
            end
            
            log_msg = "Saved convergence metrics to $metrics_file"
            if logger !== nothing
                log_message(log_msg, logger)
            else
                println(log_msg)
            end
            
            # Log convergence quality
            if length(elbo_values) >= 2
                initial_elbo = elbo_values[1]
                final_elbo = elbo_values[end]
                improvement = final_elbo - initial_elbo
                
                log_msg = "Inference converged with ELBO improvement of $improvement (from $initial_elbo to $final_elbo)"
                if logger !== nothing
                    log_message(log_msg, logger)
                else
                    println(log_msg)
                end
            end
        end
    end
    
    # If ELBO wasn't tracked successfully, log the issue
    if !elbo_tracked
        log_msg = "ELBO tracking was not successful. This could be due to:"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "  - Callback mechanism not properly configured"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "  - RxInfer.jl not exposing free_energy field in metadata"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "  - ELBO values not being stored correctly"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "Generating placeholder convergence plot..."
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        # Generate a placeholder convergence plot
        using Plots
        p = plot(
            [0, 100], [0, 0], 
            linewidth=0, 
            xlabel="Iteration", 
            ylabel="ELBO", 
            title="Convergence of Inference", 
            legend=false, 
            size=(800, 400),
            annotations=[(50, 0.5, Plots.text("Data Not Found", :red, 14))]
        )
        savefig(p, joinpath(output_dir, "convergence.png"))
        
        log_msg = "Placeholder convergence plot saved to $(joinpath(output_dir, "convergence.png"))"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
    end
    
    return paths
end

function run_all_experiments()
    # Create environments
    door_environment = create_door_environment()
    wall_environment = create_wall_environment()
    combined_environment = create_combined_environment()
    
    # Create agents
    agents = create_standard_agents()
    
    println("Running experiments for door environment...")
    execute_and_save_animation(door_environment, agents; seed = 42, gifname = "door_42.gif")

    println("Running experiments with different seed...")
    execute_and_save_animation(door_environment, agents; seed = 123, gifname = "door_123.gif")

    println("Running experiments for wall environment...")
    execute_and_save_animation(wall_environment, agents; seed = 42, gifname = "wall_42.gif")

    println("Running experiments with different seed...")
    execute_and_save_animation(wall_environment, agents; seed = 123, gifname = "wall_123.gif")

    println("Running experiments for combined environment...")
    execute_and_save_animation(combined_environment, agents; seed = 42, gifname = "combined_42.gif")

    println("Running final experiment...")
    execute_and_save_animation(combined_environment, agents; seed = 123, gifname = "combined_123.gif")

    println("All experiments completed successfully.")
end

end # module 