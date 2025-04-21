using Plots, Printf, LinearAlgebra # Ensure necessary plotting and calculation packages are available

# Function to plot posterior bar charts for basic scenarios (Clouded, Rain, Sprinkler)
function plot_posterior_bars_basic(posteriors, title_suffix, output_path)
    clouded_posterior = last(posteriors[:clouded])
    rain_posterior = last(posteriors[:rain])
    sprinkler_posterior = last(posteriors[:sprinkler])

    p1 = bar(clouded_posterior.p,
        xticks=(1:2, ["Not Clouded", "Clouded"]),
        ylabel="Probability",
        title=@sprintf("P(Clouded | %s)", title_suffix),
        titlefontsize=10,
        legend=false)

    p2 = bar(rain_posterior.p,
        xticks=(1:2, ["No Rain", "Rain"]),
        ylabel="Probability", 
        title=@sprintf("P(Rain | %s)", title_suffix),
        titlefontsize=10,
        legend=false)

    p3 = bar(sprinkler_posterior.p,
        xticks=(1:2, ["Off", "On"]),
        ylabel="Probability",
        title=@sprintf("P(Sprinkler | %s)", title_suffix), 
        titlefontsize=10,
        legend=false)

    plot(p1, p2, p3, layout=(1,3), size=(900,300))
    savefig(output_path)
    @info "Saved posterior plot to $(output_path)"
end

# Function to plot posterior bar charts for extended scenarios (flexible variables)
function plot_posterior_bars_extended(posteriors, var_symbols, labels, title_suffix, output_path; layout_dims=(1, -1), size_dims=(300 * length(var_symbols), 300))
    plots_list = []
    for (i, var_sym) in enumerate(var_symbols)
        posterior = last(posteriors[var_sym])
        p = bar(posterior.p,
            xticks=(1:length(labels[i]), labels[i]),
            ylabel="Probability",
            title=@sprintf("P(%s | %s)", string(var_sym), title_suffix),
            titlefontsize=8,
            legend=false)
        push!(plots_list, p)
    end
    
    # Adjust layout: default is horizontal row
    final_layout = layout_dims == (1, -1) ? (1, length(plots_list)) : layout_dims
    
    plot(plots_list..., layout=final_layout, size=size_dims)
    savefig(output_path)
    @info "Saved posterior plot to $(output_path)"
end


# Function to plot the learned CPTs as heatmaps
function plot_learned_cpts(cpt_cr, cpt_cs, cpt_srw, output_path)
    # Plot CPT for P(Rain | Cloudy)
    p1 = heatmap(cpt_cr, 
            title="Learned P(Rain | Cloudy)", 
            xlabel="Cloudy", ylabel="Rain",
            xticks=(1:2, ["False", "True"]), yticks=(1:2, ["False", "True"]),
            xrotation=45, clims=(0, 1), seriescolor=:viridis,
            left_margin=10Plots.mm, bottom_margin=10Plots.mm, aspect_ratio=:equal)

    # Plot CPT for P(Sprinkler | Cloudy)
    p2 = heatmap(cpt_cs,
            title="Learned P(Sprinkler | Cloudy)", xlabel="Cloudy",
            xticks=(1:2, ["False", "True"]), yticks=(1:2, ["False", "True"]),
            xrotation=45, clims=(0, 1), seriescolor=:viridis,
            bottom_margin=10Plots.mm, aspect_ratio=:equal)

    # Plot CPT for P(Wet Grass | Sprinkler, Rain) - False slice
    p3 = heatmap(cpt_srw[1, :, :], # Index 1 for WetGrass=False
            title="Learned P(Wet=False | Sprinkler, Rain)", xlabel="Rain", ylabel="Sprinkler",
            xticks=(1:2, ["False", "True"]), yticks=(1:2, ["False", "True"]),
            xrotation=45, clims=(0, 1), seriescolor=:viridis,
            bottom_margin=10Plots.mm, aspect_ratio=:equal)
            
    # Plot CPT for P(Wet Grass | Sprinkler, Rain) - True slice
    p4 = heatmap(cpt_srw[2, :, :], # Index 2 for WetGrass=True
            title="Learned P(Wet=True | Sprinkler, Rain)", xlabel="Rain",
            xticks=(1:2, ["False", "True"]), yticks=(1:2, ["False", "True"]),
            xrotation=45, clims=(0, 1), seriescolor=:viridis,
            bottom_margin=15Plots.mm, aspect_ratio=:equal)

    plot(p1, p2, p3, p4, layout=(1,4), size=(1700,305)) 
    savefig(output_path)
    @info "Saved learned CPT plot to $(output_path)"
end

# Helper for softmax
# Ensure numerical stability by subtracting max logit
function stable_softmax(logits; dims)
    max_logit = maximum(logits, dims=dims)
    exp_logits = exp.(logits .- max_logit)
    sum_exp_logits = sum(exp_logits, dims=dims)
    return exp_logits ./ sum_exp_logits
end

# Function to unpack logit parameters into CPT probability matrices/arrays
function logits_to_cpts(logit_params)
    # Parameter vector structure: [logit_cr_cf, logit_cr_ct, logit_cs_cf, logit_cs_ct, logit_srw_sf_rf, logit_srw_st_rf, logit_srw_sf_rt, logit_srw_st_rt]
    # Each logit corresponds to P(Var=True | Condition)
    
    sigmoid = (x) -> 1.0 / (1.0 + exp(-x))

    # P(Rain | Cloudy) : 2x2, needs 2 logits
    p_true_cr_f = sigmoid(logit_params[1]) # P(Rain=T | Cloudy=F)
    p_true_cr_t = sigmoid(logit_params[2]) # P(Rain=T | Cloudy=T)
    # Construct correctly for DiscreteTransition: [P(Out=F|In=F) P(Out=F|In=T); P(Out=T|In=F) P(Out=T|In=T)]
    cpt_cr = [1.0-p_true_cr_f  1.0-p_true_cr_t;  # P(Rain=F | Cloudy=F/T)
              p_true_cr_f      p_true_cr_t]      # P(Rain=T | Cloudy=F/T)

    # P(Sprinkler | Cloudy) : 2x2, needs 2 logits
    p_true_cs_f = sigmoid(logit_params[3]) # P(Sprinkler=T | Cloudy=F)
    p_true_cs_t = sigmoid(logit_params[4]) # P(Sprinkler=T | Cloudy=T)
    # Construct correctly for DiscreteTransition
    cpt_cs = [1.0-p_true_cs_f  1.0-p_true_cs_t;  # P(Sprinkler=F | Cloudy=F/T)
              p_true_cs_f      p_true_cs_t]      # P(Sprinkler=T | Cloudy=F/T)

    # P(Wet | Sprinkler, Rain) : 2x2x2, needs 4 logits
    p_true_srw_ff = sigmoid(logit_params[5]) # P(Wet=T | S=F, R=F)
    p_true_srw_tf = sigmoid(logit_params[6]) # P(Wet=T | S=T, R=F)
    p_true_srw_ft = sigmoid(logit_params[7]) # P(Wet=T | S=F, R=T)
    p_true_srw_tt = sigmoid(logit_params[8]) # P(Wet=T | S=T, R=T)
    
    cpt_srw = Array{Float64}(undef, 2, 2, 2) # Shape (Wet, Sprinkler, Rain)
    # Dim 3 = Rain (1:F, 2:T), Dim 2 = Sprinkler (1:F, 2:T), Dim 1 = Wet (1:F, 2:T)
    cpt_srw[2, 1, 1] = p_true_srw_ff # P(Wet=T | S=F, R=F)
    cpt_srw[2, 2, 1] = p_true_srw_tf # P(Wet=T | S=T, R=F)
    cpt_srw[2, 1, 2] = p_true_srw_ft # P(Wet=T | S=F, R=T)
    cpt_srw[2, 2, 2] = p_true_srw_tt # P(Wet=T | S=T, R=T)
    
    cpt_srw[1, :, :] = 1.0 .- cpt_srw[2, :, :] # P(Wet=False | S, R)

    return cpt_cr, cpt_cs, cpt_srw
end 