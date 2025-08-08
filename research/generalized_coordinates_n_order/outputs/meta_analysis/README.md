# Meta-analysis Summary

This folder aggregates metrics across orders and scenarios.

Tables:
- rmse_dim1.csv
- coverage95_dim1.csv
- rmse_mean.csv
- coverage95_mean.csv
- fe_iter_final.csv
- fe_time_total_sum.csv
- fe_time_final.csv

Heatmaps:
- heatmap_rmse_dim1.png
- heatmap_coverage_dim1.png
- heatmap_rmse_mean.png
- heatmap_coverage_mean.png
- heatmap_fe_iter_final.png
- heatmap_fe_time_total_sum.png
- heatmap_fe_time_final.png

Notes:
- Dim 1 corresponds to position. Mean metrics are averaged across available state dimensions.
- Correlation metrics require saved per-time posterior means and true states; if desired, extend the suite to dump those inputs and add correlation here.
