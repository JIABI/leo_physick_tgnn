# UAV result artifact contract

`cli/evaluate.py` writes one immutable shard per training-run/held-out-episode
pair and one aggregate bundle per experiment task.

Core identity fields are `run_seed`, `run_index`,
`local_episode_index`, `episode_seed`, `paired_episode_id`,
`exogenous_sequence_id`, checkpoint SHA-256, model fingerprint, protocol
fingerprint, perturbation fingerprint, ablation name and paired fingerprint.
The exogenous sequence identifier equals the seed that keys every stochastic
simulator stream; the paired episode identifier is shared across methods for
the same run-local held-out episode. Each condition stores the full
proposal/execution trace and both simulator and controller-facing descriptors.

Per-episode endpoint rows include service failure, ABA reassociation rate,
user-first/time-second tail mission service, residual-energy tail diagnostics,
active-station load CV, Top-1 agreement, Kendall correlation, near-tie support
and flip rate, and native oracle-score regret. Undefined denominators are
serialized as `null`, never as zero; aggregate tables record omitted support.

Primary uncertainty is the paired hierarchical bootstrap with training run as
the top-level unit and matched held-out episode resampling inside selected
runs. The configured paper analysis uses 10,000 draws. The flat paired episode
bootstrap is retained only as a sensitivity output.
