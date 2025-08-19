## Technical README: Generalized Coordinates and Variational Inference (LaTeX Markdown)

This document provides a mathematical and computational explanation of the generalized-coordinates constant-acceleration model, its variational inversion with RxInfer, and connections to the Free Energy Principle (FEP), generalized filtering, Hierarchical Gaussian Filter (HGF), and message passing. Equations are formatted in LaTeX-style markdown.

### 1. State-Space Model in Generalized Coordinates (n-order)

We model a 1D generalized-coordinates chain of length K with state
\[ x_t = \begin{bmatrix} p_t \\ v_t \\ a_t \\ x^{(3)}_t \\ \vdots \\ x^{(K-1)}_t \end{bmatrix} \in \mathbb{R}^K. \]

Discrete-time dynamics (Euler) with time-step \(\Delta t\):
\[
\begin{aligned}
 x_t &= A x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q), \\
 y_t &= B x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R),
\end{aligned}
\]
with
\[
A_{ij} = \begin{cases}
  \dfrac{(\Delta t)^{j-i}}{(j-i)!}, & j \ge i, \\
  0, & j < i
\end{cases}, \quad Q = \operatorname{diag}(\sigma_p^2, \sigma_v^2, \sigma_a^2, \varepsilon, \ldots, \varepsilon).
\]
Observation matrix \(B\) is either `1×K` with \(B_{1,1}=1\) (position only) or `2×K` with \(B_{1,1}=1, B_{2,2}=1\) (position+velocity). \(R\) is taken diagonal with respective observation variances in the provided scripts.

The initial prior is \(x_1 \sim \mathcal{N}(x_0^{\text{mean}}, x_0^{\text{cov}})\).

#### Configuring order K
- In code, choose `K` via `order`.
  - Model matrices: `GCUtils.constant_acceleration_ABQ(dt; order=K, ...)`.
  - Data generation: `GCUtils.generate_gc_car_data(...; order=K, ...)`.
  - Scripts: set `order = K` in `run_gc_car.jl`, or per-scenario in `run_gc_suite.jl`.
- The dashboard and plots render all K levels automatically (position, velocity, acceleration, and higher derivatives as `x^(3)`, …).

### 2. Variational Inference with Mean-Field Constraint

We approximate the posterior with a mean-field family
\[ q(x_{1:T}, y_{1:T}) = q(x_{1:T})\, q(y_{1:T}). \]
In RxInfer this is implemented via a constraint `q(x,y) = q(x) q(y)` and exact linear-Gaussian messages under that constraint. The inference is performed with history collection to obtain per-time marginals \(q(x_t) = \mathcal{N}(\mu_t, \Sigma_t)\).

### 3. Free Energy: Exact (Model-Level) vs Approximate (Node-Level)

RxInfer reports the (negative) variational free energy per iteration. For linear-Gaussian models, this relates to the evidence lower bound (ELBO):
\[ F = \mathbb{E}_q[\ln q(x)] - \mathbb{E}_q[\ln p(y,x)]. \]

For analysis and visualization, we also compute a per-time Gaussian approximation of node-level contributions:
\[
\begin{aligned}
F_t^{\text{obs}} &= \tfrac{1}{2}\Big( d_y \ln 2\pi + \ln|R| + \operatorname{tr}(R^{-1} \Sigma_{y,t}) + (y_t - B\mu_t)^\top R^{-1} (y_t - B\mu_t) \Big), \\
F_1^{\text{prior}} &= \tfrac{1}{2}\Big( d_x \ln 2\pi + \ln|\Sigma_0| + \operatorname{tr}(\Sigma_0^{-1} \Sigma_{1}) + (\mu_1 - \mu_0)^\top \Sigma_0^{-1} (\mu_1 - \mu_0) \Big), \\
F_t^{\text{dyn}} &= \tfrac{1}{2}\Big( d_x \ln 2\pi + \ln|Q| + \operatorname{tr}(Q^{-1} (\Sigma_t + A\Sigma_{t-1}A^\top)) + (\mu_t - A\mu_{t-1})^\top Q^{-1} (\mu_t - A\mu_{t-1}) \Big),\ t\ge2,
\end{aligned}
\]
where \(\Sigma_{y,t} = B\Sigma_t B^\top\), and \(d_x, d_y\) are state and observation dimensions. We plot
\[ F_t^{\text{total}} = F_t^{\text{obs}} + \begin{cases} F_1^{\text{prior}}, & t=1,\\ F_t^{\text{dyn}}, & t\ge 2.\end{cases} \]

This decomposition is useful for diagnostics but is not the exact ELBO that RxInfer optimizes; it is consistent under Gaussian assumptions and helps assess fit quality over time. The `rxinfer_free_energy.csv` contains the exact per-iteration ELBO reported by RxInfer.

### 4. Connections to Generalized Filtering and FEP

Generalized filtering (GF) performs continuous-time variational inversion in generalized coordinates \(\tilde{x} = [x, \dot{x}, \ddot{x}, \ldots]^\top\) under a Laplace assumption. The recognition dynamics follow
\[ \dot{\tilde{\mu}} = D\tilde{\mu} - \nabla_{\tilde{\mu}} F, \]
where \(D\) is a differential operator and \(F\) the free energy functional. In discrete time with linear-Gaussian assumptions, the updates reduce to Kalman-like prediction-correction with precision weighting. Our `A` encodes a finite-difference version of \(D\) for constant-acceleration models.

The Free Energy Principle (FEP) frames inference and action as minimizing (expected) variational free energy. In our setting, the sensory mapping is \(y_t = Bx_t + \varepsilon_t\) and the dynamics prior is \(x_t = Ax_{t-1} + w_t\). Minimization yields posteriors consistent with prediction-error minimization weighted by precisions \(R^{-1}, Q^{-1}\).

### 5. Relation to HGF and Variational Message Passing

- HGF models hierarchical Gaussian random walks with precision-weighted updates. Our single-level linear-Gaussian setup corresponds to the base level of HGF, where precisions induce adaptive learning rates analogous to Kalman gains.
- Variational Message Passing (VMP) on Gaussian graphical models produces Gaussian messages parameterized by natural parameters. RxInfer implements fast Gaussian message passing; under the mean-field constraint `q(x,y)=q(x)q(y)`, the updates match the standard linear-Gaussian conjugate forms.

### 6. Implementation Notes

- `GCUtils.constant_acceleration_ABQ` builds the K×K Taylor-integrator `A` and diagonal `Q`, and safeguards positive definiteness by thresholding variances with a small epsilon. The first three diagonal entries correspond to `(σ_p^2, σ_v^2, σ_a^2)`; higher orders default to `ε`.
- `run_gc_car.jl` collects full posterior history and writes diagnostics including `rxinfer_free_energy.csv` (exact ELBO per iteration with deltas) and a Gaussian per-time FE decomposition (`gc_free_energy_timeseries.csv`, split into observation, prior, and dynamics). When `R` is diagonal, it additionally writes `gc_free_energy_obs_dim_terms.csv` with per-dimension observation contributions.
- `GCViz` renders a K-order dashboard (`summary_dashboard_all`) stacking all state panels (1..K) and the free-energy panel; other PPC plots (residuals, QQ/ACF, standardized-residuals time series, coverage, RMSE) adapt to K. It also includes a finite-difference consistency check of generalized coordinates (`plot_derivative_consistency`).

### 7. Reproducibility

- The scripts use `StableRNGs` for deterministic simulations.
- Set `ENV["JULIA_PKG_PRECOMPILE_AUTO"]= "0"` to avoid latency from precompilation during iterative runs.

### 8. How to Extend

- Add jerk (third derivative) or higher by increasing `order = K` and, if needed, tuning the corresponding diagonal entries of `Q` in `constant_acceleration_ABQ`.
- Introduce control inputs via an input matrix `G` and a known control sequence `u_t` with `x_t = A x_{t-1} + G u_t + w_t` (not implemented in this module but straightforward to add).
- Replace position-only observation with nonlinear sensors by changing `B x_t` to `h(x_t)` and using appropriate nodes in RxInfer; then reassess PPC plots.

### 9. References and Further Reading

- Friston et al., Generalized Filtering (GF)
- FEP expositions connecting message passing, path integrals, and Bayesian mechanics
- HGF literature on precision-weighted updates
- Standard references on Kalman filtering and Gaussian graphical models



### 10. Suite and Meta-Analysis

- Batch runs (`run_gc_suite.jl`) evaluate orders `K=1..8` across scenario generators in `GCGenerators` (constant acceleration, sinusoids, polynomials, piecewise mixed). Each run writes plots and CSVs per scenario under `outputs/order_K/<scenario>/`.
- Meta-analysis (`run_meta_analysis.jl`) scans suite outputs and produces CSV tables and heatmaps in `outputs/meta_analysis/` summarizing RMSE, 95% coverage, mean correlation, and free-energy diagnostics across orders and scenarios.

### 11. Practical Notes

- Initialization uses a broad Gaussian prior on `x_1` and `@initialization` to seed `q(x)` in RxInfer; this stabilizes early iterations for higher orders.
- Scripts force a non-interactive GR backend (`ENV["GKSwstype"] = "100"`) to ensure plots render in headless setups.
- For deeper models or long sequences, consider adjusting `iterations`, `keephistory`, and `limit_stack_depth` in run configs to balance speed and memory.