### Visualization

Defined in `visualize.jl`:
- `plot_hidden_and_obs(history, observations)`: quick look at the synthetic signal and noisy observations.
- `plot_estimates(μ, σ2, history, observations; upto)`: ribbon plot of posterior mean and variance, plus ground truth and observations.
- `save_gif(anim, path; fps=24)`: helper wrapper around `Plots.gif`.

Outputs written by runs:
- Static: `static_inference.gif` and `static_free_energy.png`.
- Realtime: `realtime_summary.txt` currently records completion timestamp (extendable to snapshots).

