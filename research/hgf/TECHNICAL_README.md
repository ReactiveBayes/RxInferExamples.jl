## HGF — Technical Notes

This document summarizes the probabilistic model, the Gaussian Controlled Variance (GCV) transition, and the mean-field variational setup used in the HGF scripts.

### Generative model
We consider a two-layer hierarchical state-space model with states `z` (upper) and `x` (lower), and observations `y`:

1) State dynamics for `z`:
   - z₀ ∼ Normal(0, 5)
   - zₖ ∼ Normal(zₖ₋₁, σ²_z)

2) Controlled-variance step for `x` via GCV:
   - x₀ ∼ Normal(0, 5)
   - xₖ ∼ Normal(xₖ₋₁, exp(κ zₖ + ω))

3) Observation model:
   - yₖ ∼ Normal(xₖ, σ²_y)

Here κ and ω are parameters governing the log-variance of the `x` transition. The implementation uses RxInfer's `GCV(x_prev, z, κ, ω)` node, with Gauss–Hermite cubature for moment evaluations of the non-linear mapping exp(κ z + ω).

### Filtering model (`Model.hgf`)
Single-step model with known κ, ω and provided priors for `z_prev`, `x_prev`:

    z_prev ~ Normal(z_prev_mean, z_prev_var)
    x_prev ~ Normal(x_prev_mean, x_prev_var)
    z_next ~ Normal(z_prev, σ²_z)
    x_next ~ GCV(x_prev, z_next, κ, ω)
    y ~ Normal(x_next, σ²_y)

With structured factorization constraint:

    q(x_next, x_prev, z_next) = q(x_next) q(x_prev) q(z_next)

Gauss–Hermite cubature points: 31 by default in `hgfmeta()`.

### Smoothing model (`Model.hgf_smoothing`)
Joint model over full sequence with priors on κ, ω:

    z₀ ~ Normal(0, 5)
    x₀ ~ Normal(0, 5)
    κ  ~ Normal(1.5, 1.0)
    ω  ~ Normal(0.0, 0.05)
    for k in 1:n
        zₖ ~ Normal(zₖ₋₁, σ²_z)
        xₖ ~ GCV(xₖ₋₁, zₖ, κ, ω)
        yₖ ~ Normal(xₖ, σ²_y)
    end

Constraint:

    q(x_prev, x, z, κ, ω) = q(x_prev, x) q(z) q(κ) q(ω)

Metadata uses the same GCV Gauss–Hermite setup as in filtering.

### Variational objectives
Inference uses RxInfer's variational message passing with Bethe free energy F. With mean-field factorizations above and conjugate sub-structures for Gaussian nodes, updates follow standard Gaussian moment matching. For the non-linear GCV node, expectations of functions involving exp(κ z + ω) are approximated via Gauss–Hermite cubature with specified number of points. The free energy is monitored to assess convergence.

### Outputs and diagnostics
The runner stores:
- Hidden state trajectories (`ẑ`, `x̂`) with uncertainty ribbons
- Bethe free energy sequences for filtering/smoothing
- Posteriors for κ, ω (smoothing)
- Diagnostics: residuals, ACF, QQ, coverage, variance trajectories, and animations

Refer to `Viz.jl` for plotting and animation routines.


