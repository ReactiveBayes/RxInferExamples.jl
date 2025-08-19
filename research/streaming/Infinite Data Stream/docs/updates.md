### Variational Updates and Initialization

Defined in `updates.jl`:
- `make_autoupdates()`: sets the prior parameters from the last posterior — `x_prev_mean,var` from `q(x_current)` and `τ` parameters from `q(τ)`.
- `make_initialization(...)`: stable defaults for `q(x_current)` and `q(τ)` as in the notebook.

Rationale:
- Mirrors the notebook’s `@autoupdates` and `@initialization` blocks for consistent convergence and free-energy traces.

