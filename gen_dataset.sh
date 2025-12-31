python scripts/make_synthetic_tle.py --N 500 --out data/synthetic_leo_500.tle

python scripts/gen_data.py \
  --cfg configs/data/starlink_like.yaml \
  --out data/starlink_like_500.pt \
  --episodes 1