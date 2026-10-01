# Models, controllers, and training

Cox-PhysiCK provides rule-based, temporal-graph, and adapted DQN controllers. All methods below use the same `Observation → requested slots → joint execution → StepResult` interface. Satellite and beam identifiers address graph memories and break exact ties; they are not numeric neural features.

## Message models

The two node types have separate shared 128-dimensional GRU cells. One edge message is computed from pre-update user and beam memories, then mean-aggregated at both endpoints. Both node types update synchronously once per epoch. Nodes without edges retain their memories. The residual head uses updated memories; it has dimensions `262 → 128 → 64 → 1`, SiLU hidden activations, and a zero-initialized last layer. The auxiliary load head is `262 → 64 → 1` with SiLU and does not affect decisions.

| Module | Mapping | Trainable parameters, excluding shared memories/readouts |
|---|---|---:|
| TGN-MLP / no kernel bank | 262 → 213 → 128, SiLU | 83,411 |
| TGN-KAN message | 262 → 32 → 128 | 112,480 |
| KAN coefficient head | 262 → 32 → 10 | 78,378 |
| Full PhysiCK message | KAN head + ten 128 × 4 lifts | 83,498 |
| MLP coefficient head | 262 → 287 → 10, SiLU | 78,361 |

Each KAN connection has eight cubic B-spline coefficients and a SiLU base coefficient, with one output-node bias. Five grid intervals span `[-1, 1]`. Open-clamped knots make the cubic spline branch saturate at the boundaries; the SiLU branch uses the unsaturated input. The grid stays fixed. A temperature-one softmax produces ten mixture weights.

Physical coordinates are `(P0, log1p(rate × 60)/log(11), log1p(RST/60)/log(21))`, clipped to `[0,1]`. The first eight scalar kernels are the tensor products of `(1-u,u)` across those coordinates. The last two are the candidate-window zero-entry probability and broadcast normalized load. Each scalar multiplies its learned `128 × 4` lift of `[1, observed linear SINR, normalized broadcast load, handover cost]`. Physical bank inputs are not replaced by standardized learned inputs.

The six neural edge features use population moments from valid candidate edges of the first analytic-policy training collection. Moments are accumulated in FP64, then frozen before optimization and retained for all later rounds, validation, and tests. A constant training feature has unit scale. GRU memories and the physical bank retain their own values.

## Descriptor interventions

| Ledger ID | Implementation |
|---|---|
| `full` | Cox common-window descriptors and candidate-RST kernel |
| `mlp213` | TGN-MLP; also the no-kernel ablation, sharing the same trained model |
| `kan_generic` | Generic KAN message in the same recurrent scaffold |
| `no_triplet` | Masks `P0`, entry rate, and RST in learned inputs and physical coordinates; zero-entry kernel is one |
| `no_cox_rst` | Masks only `P0` and entry rate; retains RST; zero-entry kernel is one |
| `eph_physick` | Exact one-second future geometric entry counts replace Cox probabilities/rates in the respective common and candidate windows |
| `mlp_coeff` | Physical bank retained; replaces KAN coefficient head by the parameter-matched MLP |

The ephemeris variant uses the same current feasibility retention, excludes currently visible satellites from future entries, and applies no κ scaling to known entries. Environment construction must use `descriptor_mode: ephemeris`; training and checkpoint loading set this mode automatically. Every learned intervention is trained separately, except the explicitly shared TGN-MLP/no-kernel identity.

## Rule controllers

- **Load-aware greedy:** analytic utility, nominal hysteresis 0.05, no TTT. Exact ties prefer the current pair, then satellite and beam IDs.
- **Cox-only:** the same utility minus the candidate's own RST-window zero-entry probability, coefficient one, with the same hysteresis.
- **Max-SINR+TTT:** a candidate needs at least a 3 dB advantage over the feasible source for two consecutive epochs; losing source feasibility bypasses the waiting rule.
- **Max-RST:** longest feasible RST, using the common exact-tie rule.

No feasible candidate yields an empty request. No rule controller updates occupancy while selecting requests. The environment handles all requests together.

## Residual training

Default training is four rounds, each with 80 complete 800-second episodes and 5,000 AdamW updates. The first collection uses the analytic policy; later collections use the preceding round's learned policy. Uniform feasible-candidate exploration is 0.20, 0.15, 0.10, and 0.05. Each round's collected episodes are stored independently and its updates sample that round's data.

An update samples four within-episode 64-step sequences. Memories start at zero for each sampled sequence; the first 32 steps warm up memory without gradients and the next 32 contribute the recurrent loss. Requested nonempty edges fit `-executed_cost - requested_analytic_utility`; this target keeps the final outcome even when admission rejects or falls back. All valid graph edges additionally predict their satellite's executed normalized occupancy with weight 0.1. Both MSE terms pool their actual valid elements over the entire minibatch, excluding padding and empty requests.

The learning rate warms up linearly for 500 updates to 0.0003, then decays by a cosine schedule to 0.00003 by update 20,000. Weight decay is 0.00001. Every 1,000 positive updates, ten complete fixed validation episodes determine mean executed cost; the minimum selects the model. Step zero is excluded. Updates, collection files, normalization, optimizer state, RNG states, and validation costs are saved. Resume requires the original collected files and effective configuration.

## Adapted LEO-MADRL

A single online network, target network, and replay buffer are shared across all users within a training seed. The network is `73 → 128 → 128 → 9`, with ReLU hidden layers. Each of eight candidate slots contains six edge descriptors, current-source flag, slot-exists flag, and feasibility flag. The last input is the empty-source flag. Continuous features use moments collected during the initial training replay warmup and then freeze; flags remain 0/1 and missing slots remain zero.

The ninth action means empty association and is allowed only if all candidates are infeasible. Missing/infeasible actions are masked in exploration, greedy selection, and target maximization. Exact greedy ties use the same source/ID preference in collection and evaluation. There is no utility hysteresis or TTT on DQN actions.

A system epoch freezes observations, selects every request, executes joint admission and links, stores one transition per user, then performs one shared update once 10,000 user transitions have been collected. Replay capacity is 100,000; batch size is 128; discount is 0.99. The target is the direct target-network masked maximum. Huber loss has threshold one and detached weights of one for nonnegative TD errors and 0.2 for negative TD errors. Adam uses learning rate 0.0001, betas `(0.9,0.999)`, epsilon `1e-8`, no weight decay, and global gradient-norm clipping at one. Target synchronization and validation occur every 1,000 optimizer updates.

Training lasts 256,000 system epochs. Epsilon decreases linearly from one to 0.05 over the first 102,400 epochs, then stays fixed. Evaluation uses zero exploration. Resume files are saved at complete-episode boundaries and include replay, optimizer, RNGs, and episode counters. This is the manuscript's adapted parameter-sharing hysteretic DQN; no claim is made that ordinary minibatch updates implement the cited paper's complete successive mechanism.

## Splits, results, and execution

Training seeds determine training trajectories. Validation uses environment seed 10001 and episode IDs 0–9; test execution uses seed 20001 and episode IDs 0–29 by default. These exogenous splits are shared across methods. All saved evaluation configurations include the actual method, environment, weights, and source checkpoint identity. Density/horizon/hysteresis changes are explicit evaluation overrides; training normalization stays frozen.

Use the repository CLI for `train`, `evaluate`, `plan`, and `run-plan`. Training writes `effective_config.yaml`, collected rollouts, `training_history.json`, `latest.pt`, and `selected.pt`. Evaluation returns time-first association, proposal, rate, switching cost, occupancy, attempted/failed handover, and cost arrays. `record_diagnostics=True` also retains execution checks, candidate bands, visibility, S/I/N, and outage-state fields supplied by the environment. An explicitly constructed environment can export its geometry, beam plan, and weather records after evaluation.

The supplied final results are read by the result-reduction and plotting tools. New training and evaluation outputs are written under `runs/`.
