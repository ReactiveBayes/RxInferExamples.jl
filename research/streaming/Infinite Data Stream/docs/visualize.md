### Visualization

Defined in `visualize.jl`:
- `plot_hidden_and_obs(history, observations)`: quick look at the synthetic signal and noisy observations.
- `plot_estimates(μ, σ2, history, observations; upto)`: ribbon plot of posterior mean and variance, plus ground truth and observations.
- `save_gif(anim, path; fps=24)`: helper wrapper around `Plots.gif`.

Outputs written by runs:
- Static: `static_inference.png/gif`, `static_free_energy.csv/png/gif`, composed GIF, posteriors and τ CSV/PNG, truth/obs CSV, and stepwise metrics CSV.
- Realtime: `realtime_inference.png/gif`, `realtime_free_energy.csv/png/gif` (live stream if available, else strict online per-step FE; else offline), composed GIF, posteriors and τ CSV/PNG, truth/obs CSV trimmed to matched length, stepwise metrics CSV, and a summary text file.

