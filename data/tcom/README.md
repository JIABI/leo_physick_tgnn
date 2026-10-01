# Current TCOM results

Start with `results_summary.csv`, `case_manifest.csv`, `raw_manifest.csv` and `episode_manifest.csv`. All paths in those manifests are relative to this directory. The complete 315-file execution archive is in `raw/`; all pre-existing source tables and diagnostic records are preserved in `archive/`.

See `../../docs/TCOM_RESULTS.md` for metric definitions, uncertainty units, the finite calibration inputs and recomputation commands. `checks_full/results_audit.json` and `checks_calibration/calibration_audit.json` record the independently recomputed checks. Both pass without numerical discrepancies.
