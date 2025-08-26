module Plotting

using Plots
import ..Config: CONFIG

function make_plots(results, config)
    n_iterations = config.n_iterations

    animation = @animate for i in 1:n_iterations
        initial_means = [0.0, 10.0]
        initial_vars = [1e2, 1e2]
        posterior_means = [mean.(results.posteriors[:m][i])...]
        posterior_vars = inv.([mean.(results.posteriors[:w][i])...])

        x = -10:0.01:20

        plt = plot(
            title="RxLLM: Sentiment Clustering",
            xlabel="Sentiment Spectrum",
            ylabel="Density",
            size=(800, 500),
            dpi=300,
            background_color=:white,
            titlefontsize=14,
            legendfontsize=11
        )

        plot!(plt, x, pdf.(Normal(posterior_means[1], sqrt(posterior_vars[1])), x),
            fillalpha=0.4, fillrange=0, fillcolor=:red,
            linewidth=3, linecolor=:darkred,
            label="Negative Sentiment")

        plot!(plt, x, pdf.(Normal(posterior_means[2], sqrt(posterior_vars[2])), x),
            fillalpha=0.4, fillrange=0, fillcolor=:blue,
            linewidth=3, linecolor=:darkblue,
            label="Positive Sentiment")

        plot!(plt, x, pdf.(Normal(initial_means[1], sqrt(initial_vars[1])), x),
            linewidth=1, linestyle=:dash, linecolor=:gray, alpha=0.6,
            label="Initial Prior")

        plot!(plt, x, pdf.(Normal(initial_means[2], sqrt(initial_vars[2])), x),
            linewidth=1, linestyle=:dash, linecolor=:gray, alpha=0.6,
            label="")

        cluster_probs = probvec.(results.posteriors[:z][i])
        plt2 = bar(1:length(cluster_probs), [p[1] for p in cluster_probs],
            title="Positive Sentiment Probability", ylabel="P(Positive)", xlabel="Data Point")

        plot(plt, plt2)
    end

    gif(animation, "inference_process.gif", fps=1, show_msg=false)
end

end # module

export make_plots


