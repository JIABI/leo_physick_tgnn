# Final manuscript results

The tables and figures below use the single authoritative summary in `../data/results_summary.csv`.

| File | Manuscript content |
|---|---|
| `tables/table_2_nominal.csv` | Table II: eight nominal controllers |
| `tables/table_3_horizon.csv` | Table III: H=800 comparisons |
| `tables/table_4_ablations.csv` | Table IV: component and ephemeris comparisons |
| `tables/table_5_hysteresis.csv` | Table V: test-time hysteresis |
| `tables/table_6_parameters.csv` | Table VI: message-module parameter counts |
| `tables/cost_components.csv` | Cost decomposition at K=100 and K=500 |
| `tables/attempts_and_outages.csv` | Attempt rate, executed handovers, and outage partition rates |
| `tables/validation_costs.csv` | TGN validation/test comparison with greedy |
| `figures/figure_3_density.*` | Figure 3: user-density shift |
| `figures/figure_4_kappa.*` | Figure 4: independent κ diagnostic |
| `figures/figure_5_weights.*` | Figure 5: cost-weight sensitivity |

CSV values retain full precision. Manuscript tables round these values for presentation. The 49 reported mean±SD rows appear once in Tables II–V; reused nominal comparisons reference Table II. `tables/index.csv` maps each export to its source.

Figures are available in PDF, SVG, and PNG. Their exact coordinates and error bars are in `../data/figure_points/`. The nominal Table I configuration is specified by `../configs/paper.yaml` and the manuscript.

Regenerate from the repository root:

```bash
cox-physick tables --output results/tables
cox-physick plots --output results/figures
```

See `verification.json` for the delivered consistency checks.
