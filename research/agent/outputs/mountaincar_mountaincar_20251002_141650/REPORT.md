# Simulation Report

**Generated:** 2025-10-02T14:18:03.514

---

## Configuration

- **State Dimension:** 2
- **Action Dimension:** 1
- **Observation Dimension:** 2
- **Steps:** 250
- **Total Time:** 13.065s
- **Avg Time/Step:** 0.0523s

- **Goal State:** [0.5, 0.0]
- **Final State:** [0.6608411398847073, 0.0050889805236232905]

---

## Results

| Metric | Value |
|--------|-------|
| steps_taken | 250.0 |
| total_time | 13.06491 |
| avg_time_per_step | 0.05226 |
| final_position | 0.660841 |
| final_velocity | 0.005089 |
| goal_position | 0.5 |
| goal_velocity | 0.0 |
| distance_to_goal_position | 0.160841 |
| distance_to_goal_velocity | 0.005089 |
| goal_reached | 0.0 |

---

## Diagnostics

### Memory Usage

- Peak: 6127.9 MB
- Average: 5682.52 MB
- Growth: 834.55 MB
- GC Time: 1.081s

### Performance

**inference:**
- Count: 250
- Total Time: 12.688s
- Avg Time: 0.0508s
- Min/Max: 0.0029s / 11.7044s

---

## Outputs

### Data Files
- `data/trajectory.csv` - Full state trajectory
- `data/observations.csv` - Observation sequence

### Results
- `results/summary.csv` - Summary statistics

### Diagnostics
- `diagnostics/diagnostics.json` - Comprehensive diagnostics
- `diagnostics/performance.json` - Performance metrics

### Visualizations
- `plots/trajectory_2d.png` - Trajectory plot
- `plots/mountain_car_landscape.png` - Landscape visualization
- `plots/diagnostics.png` - Diagnostics plots

### Animations
- `animations/trajectory_2d.gif` - Animated trajectory

---

**Framework:** Generic Agent-Environment Framework v0.1.0
