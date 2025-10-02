# Outputs Directory

This directory contains all simulation outputs organized in timestamped run-specific subdirectories.

## Directory Structure

Each simulation run creates a unique timestamped directory:

```
outputs/
├── logs/                                    # Top-level logs directory
│   └── agent.log                           # Main log file (if enabled)
│
└── {agent}_{environment}_{YYYYMMDD_HHMMSS}/ # Run-specific directory
    ├── REPORT.md                            # Comprehensive markdown report
    ├── metadata.json                        # Run configuration and metadata
    │
    ├── plots/                               # Static visualizations (PNG)
    │   ├── trajectory_1d.png               # 1D trajectory (position, actions)
    │   ├── trajectory_2d.png               # 2D trajectory (4-panel)
    │   ├── mountain_car_landscape.png      # Landscape with trajectory (2D only)
    │   └── diagnostics.png                 # Performance metrics
    │
    ├── animations/                          # Animated visualizations (GIF)
    │   ├── trajectory_1d.gif               # 1D animated trajectory
    │   └── trajectory_2d.gif               # 2D animated trajectory (4-panel)
    │
    ├── data/                                # Raw simulation data (CSV)
    │   ├── trajectory.csv                  # Complete state trajectory
    │   └── observations.csv                # Observation sequence
    │
    ├── diagnostics/                         # Performance metrics (JSON)
    │   ├── diagnostics.json                # Comprehensive diagnostics
    │   └── performance.json                # Detailed performance breakdown
    │
    ├── results/                             # Summary statistics (CSV)
    │   └── summary.csv                     # High-level metrics
    │
    └── logs/                                # Run-specific logs (if enabled)
        └── run.log                          # Timestamped log for this run
```

## Naming Convention

**Directory Format:** `{agent}_{environment}_{timestamp}/`

**Examples:**
- `mountaincar_mountaincar_20251002_133145/` - Config-driven mountain car
- `mountaincar_explicit_20251002_133053/` - Explicit mountain car example
- `simplenav_explicit_20251002_124927/` - Simple navigation example
- `quick_test_20251002_133008/` - Quick visualization test

## File Types

### Static Visualizations (PNG)
- **High-quality plots** suitable for publication
- **Typical size:** 50-200 KB per plot
- **Formats:** Position, velocity, phase space, actions, diagnostics

### Animations (GIF)
- **Animated trajectories** showing real-time evolution
- **Typical size:** 500 KB - 2 MB
- **Frame rate:** 10 fps (configurable)

### Data Files (CSV)
- **Raw trajectory data** with all states and actions
- **Observation sequences** for analysis
- **Summary statistics** with metrics

### Diagnostics (JSON)
- **Comprehensive performance metrics**
- **Memory usage statistics**
- **Inference timing data**

### Reports (Markdown)
- **Human-readable summary** of entire run
- **All metrics and statistics** in one place
- **File inventory** and verification

## Usage

### Viewing Outputs

```bash
# List all runs
ls outputs/

# View specific run
cd outputs/mountaincar_20251002_133145/

# Read report
cat REPORT.md

# View plots (on Mac)
open plots/*.png
open animations/*.gif

# Load data for analysis
julia -e 'using CSV, DataFrames; df = CSV.read("data/trajectory.csv", DataFrame)'
```

### Analyzing Data

```julia
using CSV, DataFrames, JSON, Plots

# Load trajectory
traj = CSV.read("outputs/myrun/data/trajectory.csv", DataFrame)

# Load diagnostics
diag = JSON.parsefile("outputs/myrun/diagnostics/diagnostics.json")

# Load metadata
meta = JSON.parsefile("outputs/myrun/metadata.json")

# Plot
plot(traj.step, traj.position, label="Position")
```

### Comparing Runs

```julia
# Load multiple runs
run1 = CSV.read("outputs/run1/data/trajectory.csv", DataFrame)
run2 = CSV.read("outputs/run2/data/trajectory.csv", DataFrame)

# Compare
plot(run1.step, run1.position, label="Run 1")
plot!(run2.step, run2.position, label="Run 2")
```

## Disk Usage

**Typical run (100 steps):**
- Plots: 150-400 KB
- Animations: 500 KB - 2 MB
- Data: 20-100 KB
- Diagnostics: 5-10 KB
- **Total:** 2-5 MB per run

## Cleanup

To remove old runs:

```bash
# Remove specific run
rm -rf outputs/old_run_20251001_120000/

# Remove all runs (keep logs)
rm -rf outputs/*/

# Remove everything including logs
rm -rf outputs/*
```

## Notes

- **Never overwrites:** Each run gets unique timestamped directory
- **Complete outputs:** Data, plots, animations, diagnostics, reports
- **Self-documenting:** Each run includes REPORT.md with all details
- **Analysis-ready:** CSV and JSON formats for easy processing

---

**Framework Version:** 0.1.1  
**Last Updated:** October 2, 2025

