#!/usr/bin/env python3
"""Profile NTN, Snapshot, or UAV inference on a formal held-out episode.

The mandatory main row is the manuscript's staged descriptor-autoregressive
model condition: predict ``D_(t+1)``, select the current action, commit it in
the simulator, and carry staged descriptors/new-edge initialization into the
next decision epoch. ``--include-model-predict-step`` adds a fixed-input
``predict_step`` microbenchmark as a separately labelled diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from leo_pg.runtime_profile import (
    RuntimeProfileConfig,
    build_runtime_report,
    checkpoint_identity,
    file_sha256,
    profile_operation,
    write_runtime_report,
)
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device


SI_NTN_SNAPSHOT_HORIZON_STEPS = 600
UAV_EVALUATION_HORIZON_STEPS = 120


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return copy.deepcopy(dict(value))


def _episode_records(episode: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    records = episode.get("steps", episode.get("records"))
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("formal episode requires a steps/records sequence")
    if not records or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("formal episode records must be non-empty mappings")
    return records  # type: ignore[return-value]


def _record(records: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any]:
    if type(index) is not int or not 0 <= index < len(records):
        raise IndexError(
            f"record index {index} is outside the episode range [0,{len(records)})"
        )
    return records[index]


def _checkpoint_metadata(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        value = payload.get(key)
        if value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value
    return result


def _entry_provenance(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        return {}
    allowed = (
        "episode_index",
        "episode_id",
        "seed",
        "run_seed",
        "local_episode_index",
        "relative_path",
        "size_bytes",
        "sha256",
        "protocol_fingerprint",
    )
    return {key: entry[key] for key in allowed if key in entry}


def _formal_input_provenance(
    *,
    data_path: Path,
    dataset_kind: Any,
    episode: Mapping[str, Any],
    episode_index: int,
    record_index: int,
    entry: Mapping[str, Any] | None,
    evaluation_horizon_steps: int,
) -> dict[str, Any]:
    return {
        "source_kind": "formal_heldout_episode_shard",
        "synthetic": False,
        "dataset_index_file": data_path.name,
        "dataset_index_sha256": file_sha256(data_path),
        "dataset_kind": dataset_kind,
        "split": "test",
        "split_position": episode_index,
        "episode_index": int(episode.get("episode_id", episode_index)),
        "episode_seed": int(episode["seed"]),
        "record_index_for_model_predict_step": record_index,
        "full_decision_epoch_seed": int(episode["seed"]),
        "full_decision_epoch_stream": "model_recursive_staged_descriptors",
        "dataset_episode_horizon_steps": int(episode["horizon_steps"]),
        "evaluation_horizon_steps": evaluation_horizon_steps,
        "dataset_role_for_full_decision": "seed_and_provenance_only",
        "shard": _entry_provenance(entry),
    }


def _evaluation_horizon(root: Mapping[str, Any], platform_name: str) -> int:
    if platform_name in {"ntn", "snapshot"}:
        evaluation = _mapping(root, "paper_evaluation")
        value = evaluation.get("horizon_steps")
        if type(value) is not int or value != SI_NTN_SNAPSHOT_HORIZON_STEPS:
            raise ValueError(
                "paper_evaluation.horizon_steps must be 600 for the SI runtime protocol"
            )
        return value
    pipeline = _mapping(root, "uav_pipeline")
    evaluation = _mapping(pipeline, "evaluation")
    value = evaluation.get("horizon_steps")
    if type(value) is not int or value != UAV_EVALUATION_HORIZON_STEPS:
        raise ValueError(
            "uav_pipeline.evaluation.horizon_steps must be 120 for the formal protocol"
        )
    return value


def _release_profile_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


class _NTNDecisionSession:
    def __init__(
        self,
        *,
        root: Mapping[str, Any],
        model: torch.nn.Module,
        episode: Mapping[str, Any],
        evaluation_horizon_steps: int,
        device: torch.device,
    ) -> None:
        from leo_pg.control.policy import FixedRankPolicy
        from leo_pg.eval.closed_loop import ConstantPolicyStreamInitializer

        self.root = copy.deepcopy(dict(root))
        self.root["seed"] = int(episode["seed"])
        protocol = self.root.setdefault("paper_protocol", {})
        if not isinstance(protocol, dict):
            raise TypeError("paper_protocol must be a mapping")
        protocol["horizon_steps"] = evaluation_horizon_steps
        self.model = model
        self.device = device
        self.initializer = ConstantPolicyStreamInitializer.from_config(self.root)
        self.policy_type = FixedRankPolicy
        self.needs_reset = True
        self.environment: Any = None
        self.observation: Any = None
        self.policy: Any = None
        self.state: Any = None
        self.reset()

    def reset(self) -> None:
        from leo_pg.eval.substitution import cold_start_policy_stream
        from leo_pg.sim.paper_environment import PaperAlignedLEOEnv

        environment = PaperAlignedLEOEnv(self.root, device=self.device)
        observation = environment.reset_control()
        initial_mask = torch.ones(
            observation.edge_count,
            dtype=torch.bool,
            device=observation.candidate_edge_ids.device,
        )
        self.environment = environment
        self.observation = cold_start_policy_stream(
            observation,
            self.initializer.initialize(observation, initial_mask),
        )
        self.policy = self.policy_type(environment.fixed_policy_config())
        self.state = None
        self.needs_reset = False

    def prepare(self) -> None:
        if self.needs_reset:
            self.reset()

    def step(self) -> None:
        from leo_pg.eval.substitution import carry_policy_stream, persistent_candidate_mask

        observation = self.observation
        output, self.state = self.model.predict_step(
            observation.as_model_step(), self.state, self.device
        )
        prediction = output.policy_descriptors
        action = self.policy.select_action(observation)
        next_observation, _execution, done = self.environment.step_action(action)
        if done:
            self.needs_reset = True
            return
        if next_observation is None or prediction is None:
            raise RuntimeError("NTN staged prediction/next observation is missing")
        persistent = persistent_candidate_mask(observation, next_observation)
        initialized = (
            self.initializer.initialize(next_observation, ~persistent)
            if bool((~persistent).any().item())
            else None
        )
        self.observation = carry_policy_stream(
            observation, prediction, next_observation, initialized
        )


class _SnapshotDecisionSession:
    def __init__(
        self,
        *,
        root: Mapping[str, Any],
        model: torch.nn.Module,
        episode: Mapping[str, Any],
        pipeline: Any,
        evaluation_horizon_steps: int,
        device: torch.device,
    ) -> None:
        from leo_pg.paper.snapshot import SnapshotFixedRankController

        self.root = copy.deepcopy(dict(root))
        self.root["seed"] = int(episode["seed"])
        protocol = self.root.setdefault("paper_protocol", {})
        if not isinstance(protocol, dict):
            raise TypeError("paper_protocol must be a mapping")
        protocol["horizon_steps"] = evaluation_horizon_steps
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.controller = SnapshotFixedRankController(pipeline.policy_config)
        self.environment: Any = None
        self.observation: Any = None
        self.control: Any = None
        self.state: Any = None
        self.needs_reset = True
        self.reset()

    def reset(self) -> None:
        from leo_pg.paper.snapshot import simulator_snapshot_output
        from leo_pg.paper.snapshot_data import (
            SnapshotInitializationRequest,
            resolve_initial_admitted_load,
            snapshot_control_from_observation,
        )
        from leo_pg.sim.paper_environment import PaperAlignedLEOEnv

        environment = PaperAlignedLEOEnv(self.root, device=self.device)
        observation = environment.reset_control()
        oracle_load = resolve_initial_admitted_load(
            self.pipeline.initial_oracle_admitted_load,
            satellite_count=environment.S,
            dtype=environment.flow.dtype,
            device=environment.device,
        )
        oracle_snapshot = simulator_snapshot_output(
            observation, oracle_load, self.pipeline.margin_config
        )
        request = SnapshotInitializationRequest.from_observation(observation)
        all_edges = torch.ones(
            request.edge_count, dtype=torch.bool, device=request.device
        )
        initial = self.pipeline.initializer.initialize(request, all_edges)
        self.environment = environment
        self.observation = observation
        self.control = snapshot_control_from_observation(
            observation,
            initial,
            self.pipeline.feature_config,
            initialized_edge=all_edges,
        )
        self.state = None
        self.needs_reset = False

    def prepare(self) -> None:
        if self.needs_reset:
            self.reset()

    def step(self) -> None:
        from leo_pg.paper.snapshot import simulator_snapshot_output
        from leo_pg.paper.snapshot_data import (
            carry_snapshot_stream,
            instantaneous_load_after_execution,
            snapshot_control_from_observation,
        )

        control = self.control
        prediction, self.state = self.model.predict_step(
            control.as_model_step(), self.state, self.device
        )
        action = self.controller.select_action(control, control.descriptors)
        next_observation, execution, done = self.environment.step_action(action)
        if done:
            self.needs_reset = True
            return
        if next_observation is None or prediction is None:
            raise RuntimeError("Snapshot staged prediction/next observation is missing")
        oracle_load = instantaneous_load_after_execution(self.environment, execution)
        oracle_snapshot = simulator_snapshot_output(
            next_observation, oracle_load, self.pipeline.margin_config
        )
        next_template = snapshot_control_from_observation(
            next_observation, oracle_snapshot, self.pipeline.feature_config
        )
        self.control = carry_snapshot_stream(
            control, prediction, next_template, self.pipeline.initializer
        )
        self.observation = next_observation


class _UAVDecisionSession:
    def __init__(
        self,
        *,
        root: Mapping[str, Any],
        model: torch.nn.Module,
        episode: Mapping[str, Any],
        evaluation_horizon_steps: int,
        device: torch.device,
    ) -> None:
        from dataclasses import replace

        from leo_pg.uav_shared.config import resolve_uav_shared_config
        from leo_pg.uav_shared.evaluation import ConstantUAVPolicyStreamInitializer
        from leo_pg.uav_shared.policy import UAVFixedRankPolicy, UAVFixedRankPolicyConfig

        resolved = resolve_uav_shared_config(root)
        self.protocol = replace(resolved.protocol, episode_seed=int(episode["seed"]))
        if evaluation_horizon_steps != self.protocol.horizon_steps:
            raise ValueError(
                "UAV evaluation horizon differs from the resolved formal protocol"
            )
        pipeline = _mapping(root, "uav_pipeline")
        initializer_cfg = _mapping(pipeline, "initializer")
        self.initializer = ConstantUAVPolicyStreamInitializer(
            eta=float(initializer_cfg["eta"]),
            intensity=float(initializer_cfg["intensity"]),
            station_flow=float(initializer_cfg["station_flow"]),
        )
        self.policy = UAVFixedRankPolicy(UAVFixedRankPolicyConfig.from_source(root))
        self.model = model
        self.device = device
        self.environment: Any = None
        self.oracle_observation: Any = None
        self.model_observation: Any = None
        self.state: Any = None
        self.needs_reset = True
        self.reset()

    def reset(self) -> None:
        from leo_pg.uav_shared.environment import UAVSharedServiceEnv

        environment = UAVSharedServiceEnv(self.protocol, device=self.device)
        oracle = environment.reset_control()
        all_edges = torch.ones(
            oracle.edge_count,
            dtype=torch.bool,
            device=oracle.candidate_edge_ids.device,
        )
        d0 = self.initializer.initialize(oracle, all_edges)
        self.environment = environment
        self.oracle_observation = oracle
        self.model_observation = oracle.with_policy_descriptors(d0)
        self.state = None
        self.needs_reset = False

    def prepare(self) -> None:
        if self.needs_reset:
            self.reset()

    def step(self) -> None:
        from leo_pg.uav_shared.evaluation import carry_model_policy_stream

        oracle = self.oracle_observation
        model_observation = self.model_observation
        output, self.state = self.model.predict_step(
            model_observation, self.state
        )
        prediction = output.to_policy_descriptors()
        action = self.policy.select_action(
            model_observation, model_observation.policy_descriptors
        )
        next_oracle, _execution, done = self.environment.step_action(action)
        if done:
            self.needs_reset = True
            return
        if next_oracle is None or prediction is None:
            raise RuntimeError("UAV staged prediction/next observation is missing")
        self.model_observation, _initialized = carry_model_policy_stream(
            model_observation, prediction, next_oracle, self.initializer
        )
        self.oracle_observation = next_oracle


def _load_ntn(
    args: argparse.Namespace,
    root: Mapping[str, Any],
    device: torch.device,
    evaluation_horizon_steps: int,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], Any, int | None]:
    from leo_pg.paper.dataset import PaperEpisodeDataset
    from leo_pg.paper.evaluation import load_frozen_model
    from leo_pg.paper.models import normalize_paper_method
    from leo_pg.paper.snapshot_models import SNAPSHOT_METHODS

    method = normalize_paper_method(args.method)
    if method in SNAPSHOT_METHODS:
        raise ValueError("Snapshot methods require --platform snapshot")
    model, manifest = load_frozen_model(
        root,
        args.ckpt,
        method=method,
        device="cpu",
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    model.to(device).eval()
    dataset = PaperEpisodeDataset(args.data, split="test")
    episode = dataset[args.episode_index]
    records = _episode_records(episode)
    _record(records, args.record_index)

    def predict_factory() -> Any:
        from leo_pg.paper.training import observation_from_record

        observation = observation_from_record(
            _record(records, args.record_index), device
        )
        model_step = observation.as_model_step()

        def predict() -> None:
            model.predict_step(model_step, None, device)

        return predict

    def session_factory() -> _NTNDecisionSession:
        return _NTNDecisionSession(
            root=root,
            model=model,
            episode=episode,
            evaluation_horizon_steps=evaluation_horizon_steps,
            device=device,
        )

    entry = None
    if dataset.storage_layout != "monolithic_v1":
        entry = dataset.payload["episode_shards"][dataset.indices[args.episode_index]]
    provenance = _formal_input_provenance(
        data_path=Path(args.data).expanduser().resolve(),
        dataset_kind=dataset.payload.get("dataset_kind"),
        episode=episode,
        episode_index=args.episode_index,
        record_index=args.record_index,
        entry=entry,
        evaluation_horizon_steps=evaluation_horizon_steps,
    )
    identity = checkpoint_identity(args.ckpt)
    identity["metadata"] = manifest.get("metadata", {})
    identity["allow_legacy_checkpoint"] = bool(args.allow_legacy_checkpoint)
    declared = (
        manifest.get("metadata", {})
        .get("training", {})
        .get("trainable_parameters")
    )
    return model, identity, provenance, (predict_factory, session_factory), declared


def _load_snapshot(
    args: argparse.Namespace,
    root: Mapping[str, Any],
    device: torch.device,
    evaluation_horizon_steps: int,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], Any, int | None]:
    from leo_pg.paper.snapshot_config import resolve_snapshot_pipeline
    from leo_pg.paper.snapshot_data import (
        SNAPSHOT_DATASET_KIND,
        SnapshotEpisodeDataset,
        snapshot_control_from_record,
    )
    from leo_pg.paper.snapshot_models import build_snapshot_model
    from leo_pg.train.checkpoint import load_ckpt

    pipeline = resolve_snapshot_pipeline(root, args.method)
    model = build_snapshot_model(pipeline.resolved_model_config, pipeline.method)
    declared = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    checkpoint = load_ckpt(
        str(Path(args.ckpt).expanduser().resolve()),
        model,
        map_location="cpu",
        strict=True,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    if checkpoint.get("paper_method") != pipeline.method:
        raise ValueError("Snapshot checkpoint paper_method differs from --method")
    if checkpoint.get("paper_interface") != "snapshot_v1":
        raise ValueError("checkpoint is not a Snapshot checkpoint")
    saved_fingerprint = checkpoint.get("snapshot_pipeline_fingerprint")
    if saved_fingerprint != pipeline.fingerprint:
        raise ValueError("Snapshot checkpoint pipeline differs from --cfg")
    model.to(device).eval()
    dataset = SnapshotEpisodeDataset(args.data, split="test")
    if dataset.payload.get("dataset_kind") != SNAPSHOT_DATASET_KIND:
        raise ValueError("Snapshot profiler requires a formal Snapshot dataset")
    if dataset.payload.get("snapshot_method") != pipeline.method:
        raise ValueError("Snapshot dataset method differs from --method")
    if dataset.payload.get("snapshot_pipeline_fingerprint") != pipeline.fingerprint:
        raise ValueError("Snapshot dataset pipeline differs from --cfg")
    episode = dataset[args.episode_index]
    records = _episode_records(episode)
    _record(records, args.record_index)

    def predict_factory() -> Any:
        control = snapshot_control_from_record(
            _record(records, args.record_index), device
        )
        model_step = control.as_model_step()

        def predict() -> None:
            model.predict_step(model_step, None, device)

        return predict

    def session_factory() -> _SnapshotDecisionSession:
        return _SnapshotDecisionSession(
            root=root,
            model=model,
            episode=episode,
            pipeline=pipeline,
            evaluation_horizon_steps=evaluation_horizon_steps,
            device=device,
        )

    entry = dataset.entries[args.episode_index]
    provenance = _formal_input_provenance(
        data_path=Path(args.data).expanduser().resolve(),
        dataset_kind=dataset.payload.get("dataset_kind"),
        episode=episode,
        episode_index=args.episode_index,
        record_index=args.record_index,
        entry=entry,
        evaluation_horizon_steps=evaluation_horizon_steps,
    )
    identity = checkpoint_identity(args.ckpt)
    identity["metadata"] = _checkpoint_metadata(
        checkpoint,
        (
            "checkpoint_schema_version",
            "paper_method",
            "paper_interface",
            "epoch",
            "global_step",
            "snapshot_pipeline_fingerprint",
        ),
    )
    identity["allow_legacy_checkpoint"] = bool(args.allow_legacy_checkpoint)
    del checkpoint
    return model, identity, provenance, (predict_factory, session_factory), declared


def _load_uav(
    args: argparse.Namespace,
    root: Mapping[str, Any],
    device: torch.device,
    evaluation_horizon_steps: int,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], Any, int | None]:
    from leo_pg.uav_shared.config import resolve_uav_shared_config
    from leo_pg.uav_shared.data import UAVEpisodeDataset, load_uav_dataset
    from leo_pg.uav_shared.model import UAV_MODEL_METHOD, load_uav_model_checkpoint

    method = args.method or UAV_MODEL_METHOD
    index = load_uav_dataset(args.data)
    run_seeds = sorted(
        {
            int(entry["run_seed"])
            for entry in index["episode_shards"]
            if entry.get("split") == "test" and "run_seed" in entry
        }
    )
    run_seed = args.run_seed
    if run_seed is None:
        if len(run_seeds) != 1:
            raise ValueError(
                "multi-run UAV data requires --run-seed to match one checkpoint"
            )
        run_seed = run_seeds[0]
    dataset = UAVEpisodeDataset(args.data, "test", run_seed=run_seed)
    model = load_uav_model_checkpoint(
        root, args.ckpt, method=method, device="cpu", strict=True
    )
    raw_metadata = getattr(model, "checkpoint_metadata", {})
    training_run_seed = raw_metadata.get("training_run_seed")
    if training_run_seed is not None and int(training_run_seed) != run_seed:
        raise ValueError("UAV checkpoint training_run_seed differs from --run-seed")
    resolved = resolve_uav_shared_config(root)
    episode = dataset[args.episode_index]
    records = _episode_records(episode)
    _record(records, args.record_index)

    def predict_factory() -> Any:
        from leo_pg.uav_shared.training import UAVRecordAdapter

        adapter = UAVRecordAdapter(resolved.protocol, device)
        graph = adapter.graph(_record(records, args.record_index))

        def predict() -> None:
            model.predict_step(graph, None)

        return predict

    def session_factory() -> _UAVDecisionSession:
        return _UAVDecisionSession(
            root=root,
            model=model,
            episode=episode,
            evaluation_horizon_steps=evaluation_horizon_steps,
            device=device,
        )

    entry = dataset.entries[args.episode_index]
    provenance = _formal_input_provenance(
        data_path=Path(args.data).expanduser().resolve(),
        dataset_kind=index.get("dataset_kind"),
        episode=episode,
        episode_index=args.episode_index,
        record_index=args.record_index,
        entry=entry,
        evaluation_horizon_steps=evaluation_horizon_steps,
    )
    provenance["run_seed"] = run_seed
    identity = checkpoint_identity(args.ckpt)
    scalar_metadata = _checkpoint_metadata(
        raw_metadata,
        (
            "uav_trainer_schema_version",
            "uav_model_contract_version",
            "epoch",
            "global_step",
            "best_epoch",
            "best_validation",
            "training_run_seed",
            "uav_model_fingerprint",
        ),
    )
    identity["metadata"] = scalar_metadata
    # The shared loader restores on CPU with weights_only=True.  Retain only
    # scalar provenance before moving model parameters; optimizer/RNG payloads
    # are neither transferred to CUDA nor kept alive by the model.
    model.checkpoint_metadata = scalar_metadata
    del raw_metadata
    model.to(device).eval()
    declared = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return model, identity, provenance, (predict_factory, session_factory), declared


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("ntn", "snapshot", "uav"), required=True)
    parser.add_argument("--cfg", required=True, help="Protocol YAML")
    parser.add_argument("--method", default=None, help="Checkpoint model method")
    parser.add_argument("--ckpt", required=True, help="Strictly loaded checkpoint file")
    parser.add_argument("--data", required=True, help="Formal sharded dataset index")
    parser.add_argument("--episode-index", type=int, default=0, help="Position in test split")
    parser.add_argument("--record-index", type=int, default=0, help="Fixed model-input record")
    parser.add_argument("--run-seed", type=int, default=None, help="Required for multi-run UAV data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Full-decision samples; defaults to the formal evaluation horizon",
    )
    parser.add_argument(
        "--include-model-predict-step",
        action="store_true",
        help="Also emit the fixed-graph model-only diagnostic after the main profile",
    )
    parser.add_argument(
        "--include-full-decision-epoch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args(argv)
    if args.episode_index < 0 or args.record_index < 0:
        parser.error("episode-index and record-index must be non-negative")
    if args.run_seed is not None and args.run_seed < 0:
        parser.error("run-seed must be non-negative")
    if args.platform != "uav" and not args.method:
        parser.error("--method is required for NTN and Snapshot checkpoints")

    root = load_cfg(args.cfg)
    device = get_device(args.device, strict=True)
    evaluation_horizon_steps = _evaluation_horizon(root, args.platform)
    repeats = (
        evaluation_horizon_steps if args.repeats is None else args.repeats
    )
    config = RuntimeProfileConfig(warmup_steps=args.warmup, repeats=repeats)
    loader = {"ntn": _load_ntn, "snapshot": _load_snapshot, "uav": _load_uav}[
        args.platform
    ]
    model, checkpoint, provenance, factories, declared = loader(
        args, root, device, evaluation_horizon_steps
    )
    predict_factory, session_factory = factories
    _release_profile_memory(device)
    session = session_factory()
    full_profile = profile_operation(
        "full_decision_epoch",
        session.step,
        device=device,
        config=config,
        prepare_each=session.prepare,
        reset_after_warmup=session.reset,
        timing_contract=(
            "descriptor-autoregressive model condition; staged D_(t+1) "
            "prediction + fixed-policy action + simulator commit + stable-edge "
            "carry/new-edge initialization; episode reset excluded from timing; "
            "terminal prediction computed then discarded for uniform per-step timing"
        ),
    )
    full_profile.update(
        {
            "batch_size": 1,
            "profile_role": "si_primary_decision_time",
            "evaluation_horizon_steps": evaluation_horizon_steps,
            "first_measured_epoch": 0,
            "complete_evaluation_sequence": repeats == evaluation_horizon_steps,
            "uniform_model_prediction_each_sample": True,
            "terminal_prediction_contract": (
                "computed before the terminal action and discarded after commit; "
                "it cannot affect any action or state"
            ),
            "sequence_reset_contract": (
                "reset after warmup and after a completed evaluation episode; "
                "all resets occur outside timed samples"
            ),
        }
    )
    profiles = [full_profile]
    del session
    _release_profile_memory(device)
    if args.include_model_predict_step:
        predict = predict_factory()
        model_profile = profile_operation(
            "model_predict_step",
            predict,
            device=device,
            config=config,
            timing_contract=(
                "optional diagnostic fixed formal test graph; predict_step only; "
                "fresh recurrent state; no target, policy, simulator, or optimizer"
            ),
        )
        model_profile["profile_role"] = "optional_model_only_diagnostic"
        profiles.append(model_profile)
        del predict
        _release_profile_memory(device)
    method = args.method or "uav_recurrent_gnn"
    config_path = Path(args.cfg).expanduser().resolve()
    report = build_runtime_report(
        platform_name=args.platform,
        method=method,
        model=model,
        checkpoint=checkpoint,
        input_provenance=provenance,
        profiles=profiles,
        device=device,
        configuration={
            "file_name": config_path.name,
            "sha256": file_sha256(config_path),
        },
        evaluation_horizon_steps=evaluation_horizon_steps,
        batch_size=1,
        declared_trainable_parameters=declared,
    )
    json_path, csv_path = write_runtime_report(
        report, json_path=args.json_out, csv_path=args.csv_out
    )
    print(f"runtime profile JSON: {json_path}")
    print(f"runtime profile CSV: {csv_path}")


if __name__ == "__main__":
    main()
