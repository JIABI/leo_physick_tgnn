# Final paper data

Start with `results_summary.csv` for the final numeric results and `case_manifest.csv` for their experiment settings. The data directory contains one set of final records; paper table views are in `../results/tables/` and figures in `../results/figures/`.

- `raw/`: executed association, request, and rate arrays.
- `episode_results.csv`: episode-level counts and metrics.
- `raw_manifest.csv` and `episode_manifest.csv`: mappings from cases to arrays and episode indices.
- `calibration/`: the geometric calibration and fixed-retention diagnostic inputs.
- `figure_points/`: final plot coordinates, means, and sample SDs.
- `validation/`: the validation comparisons used in the manuscript.

See [the data definitions](../docs/results.md) for units, aggregation, and association encoding. Recompute from the repository root with:

```bash
cox-physick reproduce --full --output outputs/reproduced
```
