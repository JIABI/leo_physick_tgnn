# NTN protocol parameter audit

`configs/paper_protocol.yaml` separates manuscript-reported values from choices
that cannot be recovered uniquely from the manuscript package.  The latter are
valid engineering starting points, not claims about the lost original runs.

## Directly reported and retained

| Group | Values | Assessment |
|---|---|---|
| Scale | 1,000 satellites, 128 users | Matches the testbed summary. |
| Timeline | 1 ms PHY, 100 ms control, 200 train / 600 eval steps | Matches the Methods and SI. |
| Candidates | 10 degree visibility, Top-6, smaller-id tie break | Matches the protocol. |
| Execution | capacity 40, gamma gate -5 dB, load gate 0.95 | Matches the simulator table. |
| Flow | arrival 0.8, service 1.0, feedback 0.25, EMA 0.9 | Matches the SI numeric table. |
| Controller | weights 1.0/0.4/0.6, margin 1/6, dwell 10 | Matches the fixed score rule. |
| Cox | beta 0.8/0.9/0.2, 10 s look-ahead, 128 trapezoids | Matches the SI table. |
| Model | width 128, 2 graph layers, 16 kernels, signed-L1 radius 1 | Matches the architecture table. |
| Runtime-control capacity classes | BigMLP 2.1M, EdgeAttn 2.1M, TempTrans 2.4M | Directly reported in the SI runtime table to one decimal place; these are rounded parameter classes, not complete architecture specifications. |
| Training | AdamW, 5e-4, 1e-4, dropout 0.1, clip 1, batch 32, 50 epochs | Matches the training table. |
| Statistics | 5 runs, 2,000 train episodes, 30 eval episodes, 10,000 paired bootstrap resamples | Matches the reporting protocol. |
| A3 | offsets 1/3/5/7 dB; TTT 1/3/5/8; selected 3/3 | Numeric grid matches the manuscript; its TTT unit is ambiguous between the table and prose. |
| Strong baselines | LTT-R rollout H=10 and about 12M parameters; DA-GWM about 8.5-8.7M | Matches the fairness table. |
| Snapshot score search | gamma `{0.5, 0.75, 1.0, 1.25, 1.5}`; feasibility/load `{0.2, 0.4, 0.6, 0.8}` | Matches the SI search grid. |
| Snapshot selected cells | MLP `(1.0, 0.6, 0.8)`; PhysiCK `(1.0, 0.4, 0.8)` | Retained as the reported selected weights for the two primary factorial cells. |
| UAV shared-service axes | nominal 24 UAVs, 6 stations, 2 active slots, FIFO capacity 8, Top-3 candidates, density and capacity-compression stress axes | Retained in the isolated UAV protocol and its data/model/training/evaluation contracts. |

## Explicit estimates and reasonableness

| Choice | Configured value | Assessment |
|---|---:|---|
| Five integer seeds | 17, 29, 43, 61, 79 | Statistically harmless if frozen before runs, but cannot recreate the lost seed pairing. |
| Validation set | 200 episodes | A conventional 10% of the reported 2,000 training episodes; does not alter the stated 2,000/30 train/test counts. |
| VLEO phases/dynamics | deterministic 350 km kinematic shell | Suitable for exercising the protocol code; not a replacement for the missing TLE snapshot or public-trace claim. |
| Flow coupling `Phi` | global mean load | Monotone and shared-resource coupled, but one of several plausible functions. It must be treated as an estimated simulator choice. |
| Cox baseline hazard | 0.1 per second | Gives an order-one 10 s baseline integrated hazard. Numerically stable and interpretable, but not recoverable from the manuscript. |
| Cox scales | 10 dB and 90 degrees | Keeps covariates near order one and limits exponential overflow. Reasonable normalization, not an original value. |
| D0/new-edge prior | gamma -3 dB, intensity 0.1, flow 0.5 | Conservative and non-semantic-zero; prevents missing predictions from being ranked as risk-free. It materially affects early epochs and must be reported. |
| Snapshot D0/new-edge prior | gamma -3 dB, feasibility margin 0, admitted load 0.5; oracle D0 admitted load 0 | Explicit no-teacher initialization values needed by the staged Snapshot stream. They are executable estimates, not recovered paper parameters. |
| Loss weights | 0.01/1/1/0.1 (+0.2 DA term) | Compensates for raw gamma being measured in dB while other targets are bounded/log-scaled. Final weights should be selected on validation data only. |
| Node-feature injection | 0.1 after every temporal update | Makes the pre-existing TGN update constant explicit and keeps it identical in S4/Mamba2/Conformer swaps; the manuscript does not report it. |
| Scheduled sampling | linear 0 to 0.5 by epoch 25 | A moderate exposure schedule. The manuscript plots `p_ss^max` but does not disclose the selected schedule. |
| CHO | 0.1-0.3 s preparation, 0.5-2.0 s validity | Covers plausible control-plane timing at 100 ms epochs; must be labelled an estimated grid. |
| Load-aware greedy | 2-8 dB/unit-flow penalty; 1-5 dB hysteresis | Brackets a useful scale relative to the A3 offsets; should be tuned only on validation episodes. |
| A3 TTT interpretation | 1/3/5/8 control steps | The code records this choice and also supports explicit seconds-to-step conversion. Do not treat the chosen unit as recovered. |
| S4/Mamba2/Conformer depth | four temporal blocks over the shared width-128 Paper PhysiCK message stream | Depth is an explicit estimate. Width, two PhysiCK message layers and the Intensity--Flow readout remain the reported shared contract rather than an independent width-256 graph model. |
| BigMLP replacement internals | hidden width 2,400 inside each two-linear-layer message MLP | The two 128-d graph-message layers, GRU memory and readout remain shared; only message-MLP width changes. The resolved model has 2,082,308 trainable parameters and therefore rounds to the SI table's 2.1M class. The original hidden width is not reported. |
| EdgeAttn replacement internals | attention width 1,792 and 8 heads in each of two graph-message layers | The 128-d message output, GRU memory and readout remain shared; only the message operator becomes GAT-style. The resolved model has 2,103,812 trainable parameters and therefore rounds to 2.1M. The original attention width/head count is not reported. |
| TempTrans replacement internals | 128-d temporal model, 5 Transformer layers, 8 heads, context 200, FFN 1,408 | The two standard MLP graph-message layers and 128-d readout remain shared; only GRU temporal mixing is replaced. The resolved model has 2,370,564 trainable parameters and therefore rounds to 2.4M. The original Transformer depth, heads, context and FFN width are not reported. |
| Runtime-table target provenance | mandatory targets 2.1M/2.1M/2.4M with a 50,000-parameter rounding tolerance | The factory validates the resolved count and stores target, tolerance, provenance, architecture and replacement scope. Targets come from the SI runtime table, not from the current reconstructed PhysiCK implementation. The manuscript itself is numerically inconsistent: that runtime table lists TGN--PhysiCK as 2.1M, whereas the main/fairness table lists Intensity--Flow + PhysiCK as 3.4M; these runtime controls therefore follow the table in which they are defined. |
| Snapshot feasibility margin | minimum normalized current-gamma/current-occupancy slack | Provides a signed memoryless Snapshot feasibility proxy. It is deliberately separate from the simulator's authoritative gamma/EMA-flow execution gate, and the manuscript does not uniquely specify how the two Snapshot slacks were scalarized. |
| Snapshot LTT-R/DA-GWM score weights | `(1.0, 0.6, 0.8)` in the executable config | The manuscript does not report selected values for these opt-in cells; they are labelled estimates and must not be conflated with the reported MLP/PhysiCK selections. |
| Snapshot memory semantics | admitted load is current executed admitted-count/capacity; margin uses that occupancy and current gamma | Implements the manuscript's memoryless Snapshot contrast: neither descriptor reads integrated intensity, `L_t`, `Phi` or the EMA update. The simulator feasibility mask remains authoritative only for execution and an optional hard mask. |
| UAV unreported dynamics | bounded region, speed/turn, energy, arrival/service and motion-noise values in `EstimatedDynamicsParameters` | Sufficient to define the executable second-platform pipeline and explicitly labelled estimates. They cannot support the reported historical numbers unless the original values are recovered or prospectively reselected. |

## Author values still preferable

If recovered, the original TLE snapshot/phases, `Phi(L)`, link-proxy
normalization, Cox baseline/scales, initializer, multi-task weights, exact run
seeds, CHO validity rule, load-aware conversion, Snapshot LTT-R/DA-GWM selected
weights, and UAV dynamics should replace the estimates before claiming numerical
reproduction. Changing any of them must produce a new protocol fingerprint and
a new result directory.
