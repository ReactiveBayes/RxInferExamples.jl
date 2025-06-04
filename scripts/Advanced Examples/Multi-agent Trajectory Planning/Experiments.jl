module Experiments

using ..Environment
using ..Models
using ..Visualizations

export execute_and_save_animation, run_all_experiments

function execute_and_save_animation(environment, agents; gifname = "result.gif", kwargs...)
    println("Planning paths for environment with $(length(environment.obstacles)) obstacles...")
    result = path_planning(environment = environment, agents = agents; kwargs...)
    paths = mean.(result.posteriors[:path])
    
    # Create animation and save it
    animate_paths(environment, agents, paths; filename = gifname)
    
    # Extract ELBO values if available
    if hasfield(typeof(result), :diagnostics) && haskey(result.diagnostics, :elbo)
        elbo_values = result.diagnostics[:elbo]
        if !isempty(elbo_values)
            # Get output directory from gifname
            output_dir = dirname(gifname)
            output_dir = (output_dir == "") ? "." : output_dir
            
            # Create ELBO convergence plot
            println("Generating ELBO convergence plot...")
            plot_elbo_convergence(elbo_values, filename = joinpath(output_dir, "convergence.png"))
            
            # Save raw ELBO data
            open(joinpath(output_dir, "convergence_metrics.csv"), "w") do f
                for (i, val) in enumerate(elbo_values)
                    println(f, "$i,$val")
                end
            end
            println("Saved convergence metrics to $(joinpath(output_dir, "convergence_metrics.csv"))")
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