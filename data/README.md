# Synthetic TLE fixture

`synthetic_leo_500.tle` is a deterministic software-test fixture, not a public
Starlink snapshot and not manuscript source data. Regenerate it from the
repository root with:

```bash
python scripts/make_synthetic_tle.py \
  --N 500 \
  --out data/synthetic_leo_500.tle \
  --alt_km 550 \
  --inc_deg 53 \
  --planes 72 \
  --ecc 0.0001 \
  --seed 7 \
  --epoch_utc 2025-01-01T00:00:00Z
```

The smoke workflow uses the kinematic simulator and does not require this TLE
fixture. `configs/data/starlink_like.yaml` uses it only to exercise the
Skyfield/SGP4 input path.
