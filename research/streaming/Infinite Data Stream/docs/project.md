### Project and Entry Points

Files:
- `Project.toml` / `Manifest.toml`: self-contained environment (`RxInfer`, `Rocket`, `Plots`, `StableRNGs`).
- `README.md`: quick-start and structure.
- `run.jl`: unified driver. Creates `output/<timestamp>/` and runs both static and realtime.
- `run_static.jl` / `run_realtime.jl`: standalone entry points for focused runs.

How to run:
```julia
# one-shot end-to-end
julia --project=. run.jl

# or, load modules and run static only
include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
include("run_static.jl")
```

Outputs:
- `output/<ts>/static_inference.gif`
- `output/<ts>/static_free_energy.png`
- `output/<ts>/realtime_summary.txt`

