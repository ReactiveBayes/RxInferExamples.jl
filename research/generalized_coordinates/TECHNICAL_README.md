## Technical README: Generalized Coordinates and Variational Inference (LaTeX Markdown)

This document provides a mathematical and computational explanation of the generalized-coordinates constant-acceleration model, its variational inversion with RxInfer, and connections to the Free Energy Principle (FEP), generalized filtering, Hierarchical Gaussian Filter (HGF), and message passing. Equations are formatted in LaTeX-style markdown.

### 1. State-Space Model in Generalized Coordinates

We model a 1D constant-acceleration system with state
\[ x_t = \begin{bmatrix} p_t \\ v_t \\ a_t \end{bmatrix} \in \mathbb{R}^3. \]

Discrete-time dynamics (Euler) with time-step \(\Delta t\):
\[
\begin{aligned}
 x_t &= A x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q), \\
 y_t &= B x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R),
\end{aligned}
\]
with
\[
A = \begin{bmatrix}
 1 & \Delta t & \tfrac{1}{2}\Delta t^2 \\
 0 & 1 & \Delta t \\
 0 & 0 & 1
\end{bmatrix}, \quad Q = \operatorname{diag}(\sigma_p^2, \sigma_v^2, \sigma_a^2).
\]
Observation matrix \(B\) is either \([1\ 0\ 0]\) (position only) or \(\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}\) (position+velocity). \(R\) is diagonal with respective observation variances.

The initial prior is \(x_1 \sim \mathcal{N}(x_0^{\text{mean}}, x_0^{\text{cov}})\).

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

This decomposition is useful for diagnostics but is not the exact ELBO that RxInfer optimizes; it is consistent under Gaussian assumptions and helps assess fit quality over time.

### 4. Connections to Generalized Filtering and FEP

Generalized filtering (GF) performs continuous-time variational inversion in generalized coordinates \(\tilde{x} = [x, \dot{x}, \ddot{x}, \ldots]^\top\) under a Laplace assumption. The recognition dynamics follow
\[ \dot{\tilde{\mu}} = D\tilde{\mu} - \nabla_{\tilde{\mu}} F, \]
where \(D\) is a differential operator and \(F\) the free energy functional. In discrete time with linear-Gaussian assumptions, the updates reduce to Kalman-like prediction-correction with precision weighting. Our `A` encodes a finite-difference version of \(D\) for constant-acceleration models.

The Free Energy Principle (FEP) frames inference and action as minimizing (expected) variational free energy. In our setting, the sensory mapping is \(y_t = Bx_t + \varepsilon_t\) and the dynamics prior is \(x_t = Ax_{t-1} + w_t\). Minimization yields posteriors consistent with prediction-error minimization weighted by precisions \(R^{-1}, Q^{-1}\).

### 5. Relation to HGF and Variational Message Passing

- HGF models hierarchical Gaussian random walks with precision-weighted updates. Our single-level linear-Gaussian setup corresponds to the base level of HGF, where precisions induce adaptive learning rates analogous to Kalman gains.
- Variational Message Passing (VMP) on Gaussian graphical models produces Gaussian messages parameterized by natural parameters. RxInfer implements fast Gaussian message passing; under the mean-field constraint `q(x,y)=q(x)q(y)`, the updates match the standard linear-Gaussian conjugate forms.

### 6. Implementation Notes

- `GCUtils.constant_acceleration_ABQ` safeguards positive definiteness of `Q` by thresholding variances with a small epsilon.
 - `run_gc_car.jl` collects full posterior history and writes diagnostics including `rxinfer_free_energy.csv` (exact ELBO per iteration with deltas) and a Gaussian per-time FE decomposition (`gc_free_energy_timeseries.csv`, split into observation, prior, and dynamics). When `R` is diagonal, it additionally writes `gc_free_energy_obs_dim_terms.csv` with per-dimension observation contributions.
- `GCViz` provides posterior predictive checks: standardized residuals, QQ, ACF, coverage, and RMSE; these are essential for validating the Gaussian assumptions and model misspecification.

### 7. Reproducibility

- The scripts use `StableRNGs` for deterministic simulations.
- Set `ENV["JULIA_PKG_PRECOMPILE_AUTO"]= "0"` to avoid latency from precompilation during iterative runs.

### 8. How to Extend

- Add jerk (third derivative) by extending the state to 4D and augmenting `A`, `Q`, and `B` accordingly.
- Introduce control inputs via an input matrix `G` and a known control sequence `u_t` with `x_t = A x_{t-1} + G u_t + w_t`.
- Replace position-only observation with nonlinear sensors by changing `B x_t` to `h(x_t)` and using appropriate nodes in RxInfer; then reassess PPC plots.

### 9. References and Further Reading

- Friston et al., Generalized Filtering (GF)
- FEP expositions connecting message passing, path integrals, and Bayesian mechanics
- HGF literature on precision-weighted updates
- Standard references on Kalman filtering and Gaussian graphical models


