### Infinite Data Stream: Modular Overview

This folder refactors the working example from `examples/Advanced Examples/Infinite Data Stream/Infinite Data Stream.ipynb` into composable modules. It preserves the algorithmic behavior (Kalman-like random walk with unknown observation precision) while making each concern (model, data generation, streaming, updates, visualization) independently testable and reusable.

Key goals:
- Keep the original logic intact and verifiable against the notebook
- Single-responsibility files with narrow interfaces
- Unified `run.jl` that exercises both static and realtime modes and writes timestamped artifacts under `output/`

Reading order:
1. `project.md` — environment, dependencies, entry points
2. `model.md` — probabilistic model and constraints
3. `environment.md` — synthetic environment and observations
4. `streams.md` — static and realtime Rocket streams
5. `updates.md` — autoupdates and initialization
6. `visualize.md` — plotting and artifact writing
7. `run.md` — orchestration of static and realtime runs

Validation parity with the notebook:
- The same state evolution (`10 * sin(0.1 * t)`) and NormalMeanPrecision observation model
- The same variational structure and updates
- The same plotting semantics for hidden state, observations, and estimated posterior mean/variance

