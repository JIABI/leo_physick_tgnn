# generate rollout .json files
python scripts/export_rollout_metrics.py \
  --cfg configs/default.yaml \
  --Hs 20,50,100,200 \
  --split train \
  --which last \
  --methods kan \
  --device cuda

#python scripts/plot_system_metrics.py \
#  --mlp runs/tgn_1000_mlp_multi/rollout_metrics.json \
#  --kan runs/tgn_1000_kan_multi/rollout_metrics.json \
#  --physick runs/tgn_500_physick_multi/rollout_metrics.json \
#  --out_dir runs/plots_500_multi \
#  --combined --dt 1.0 --target_label "Beam-load prediction"




# plot all evaluation metrics
python scripts/plot_system_metrics.py \
  --mlp runs/tgn_500_mlp_multi/rollout_metrics.json \
  --kan runs/tgn_500_kan_multi/rollout_metrics.json \
  --physick runs/tgn_500_physick_multi/rollout_metrics.json \
  --out_dir runs/plots/tgn_500_multi

