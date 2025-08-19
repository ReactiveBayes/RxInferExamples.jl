### Unified Research Runner

Run setup, notebook→script conversion, and research examples from a single entrypoint.

#### Quickstart

```bash
research/run_research/run.sh --incremental
# or overwrite all conversions
research/run_research/run.sh --overwrite
```

Defaults come from `research/run_research/run_config.yaml`. Override via CLI or interactive prompts:

```bash
research/run_research/run.sh --interactive
research/run_research/run.sh --config /abs/path/custom.yaml --incremental
```

#### What it does

- Runs `support/setup.jl` with the chosen flags (build examples/docs, quiet/force)
- Runs `support/notebooks_to_scripts.jl` (incremental or overwrite)
- Executes example entry points:
  - `research/generalized_coordinates/run_gc_car.jl`
  - `research/generalized_coordinates/run_gc_suite.jl`
  - `research/generalized_coordinates_n_order/run_gc_car.jl`
  - `research/generalized_coordinates_n_order/run_gc_suite.jl`
  - `research/hgf/run_hgf.jl`

#### Config (run_config.yaml)

- `mode`: `incremental` | `overwrite`
- `build.examples`: 0 (build) | 1 (skip)
- `build.docs`: 0 (build) | 1 (skip)
- `quiet`: 0 | 1
- `force`: 0 | 1
- `runs.*`: enable/disable each run (`gc_single`, `gc_suite`, `gcn_single`, `gcn_suite`, `hgf`)


