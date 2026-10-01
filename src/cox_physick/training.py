"""Executed-utility residual training and parameter-sharing hysteretic DQN.

Published result records are not used as training samples. Training here runs
fresh physical environments, stores its own trajectories, and records selection
costs on a fixed validation split. No test data enter model fitting/selection.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from .config import Config
from .models import (TGNModel, DQNNetwork, canonical_method, RESIDUAL_METHODS,
                     transformed_descriptors, dqn_features, action_mask, hysteretic_huber)
from .controllers import (RuleController, NeuralController, DQNController,
                          explore_requests, make_controller, rank_slots)


def _environment(cfg, seed, episode_id):
    from .environment import Environment
    return Environment(cfg, seed=seed, episode_id=episode_id)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state(rng):
    return {"numpy_generator": rng.bit_generator.state, "python": random.getstate(),
            "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state, rng):
    rng.bit_generator.state = state["numpy_generator"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load(path, device):
    # Only load checkpoints produced locally by this release or explicitly
    # trusted by the caller; torch checkpoints can contain Python objects.
    return torch.load(path, map_location=device, weights_only=False)


def _configured(cfg, method, overrides):
    cfg = deepcopy(cfg)
    if overrides:
        cfg.training.update(overrides)
    if canonical_method(method) == "ephemeris":
        cfg.environment["descriptor_mode"] = "ephemeris"
    return cfg


def evaluate_episode(controller, cfg, seed: int, episode_id: int,
                     steps: int | None = None, environment=None,
                     record_diagnostics: bool = False) -> dict[str, Any]:
    """Return complete raw arrays from a single executed rollout.

    Associations and requests encode satellite*n_beams+beam, with -1 empty.
    Arrays are time-first. ``steps`` chooses a prefix of the configured episode;
    it never changes the geometry or stochastic trace underlying that episode.
    """
    env = environment if environment is not None else _environment(cfg, seed, episode_id)
    obs = env.reset()
    controller.reset()
    records = {name: [] for name in ("association", "requests", "rates", "c_exec", "occupancy", "attempts", "failures", "cost", "subband")}
    diagnostics = {}
    n_beams = cfg.environment["n_beams"]
    max_steps = cfg.environment["horizon"] if steps is None else min(steps, cfg.environment["horizon"])
    for _ in range(max_steps):
        if obs is None:
            break
        slots = controller.select(obs)
        valid = slots >= 0
        requests = np.full(len(slots), -1, dtype=np.int64)
        rows = np.flatnonzero(valid)
        requests[rows] = obs.sat_id[rows, slots[rows]] * n_beams + obs.beam_id[rows, slots[rows]]
        result = env.step(slots)
        executed = np.where(result.executed_sat >= 0,
                            result.executed_sat * n_beams + result.executed_beam, -1)
        for key, value in {
            "association": executed, "requests": requests, "rates": result.rate_mbps,
            "c_exec": result.c_exec, "occupancy": result.occupancy, "attempts": result.attempt,
            "failures": result.failure, "cost": result.cost, "subband": result.executed_subband,
        }.items():
            records[key].append(np.asarray(value).copy())
        if record_diagnostics:
            for key, value in result.info.items():
                if isinstance(value, (np.ndarray, int, float, bool, np.generic)):
                    diagnostics.setdefault(key, []).append(np.asarray(value).copy())
        obs = result.next_observation
    if not records["association"]:
        raise ValueError("Empty evaluation episode")
    out = {key: np.stack(value) for key, value in records.items()}
    out.update({"diagnostic_" + key: np.stack(value) for key, value in diagnostics.items()})
    out["proposal"] = out["requests"]
    out["rate_mbps"] = out["rates"]
    out.update(mean_cost=float(out["cost"].mean()), environment_seed=int(seed), episode_id=int(episode_id))
    return out


def _validation_cost(model, cfg, method, device):
    seed = int(cfg.training.get("validation_environment_seed", 10001))
    episodes = int(cfg.training["validation_episodes"])
    controller = make_controller(method, cfg, model, device)
    return float(np.mean([evaluate_episode(controller, cfg, seed, episode)["mean_cost"]
                          for episode in range(episodes)]))


def _collect_residual_episode(cfg, model, controller, seed, episode_id, exploration, rng,
                              destination, fit_normalizer=False):
    env = _environment(cfg, seed, episode_id)
    obs = env.reset()
    controller.reset()
    store = {key: [] for key in ("z", "survival", "sat_id", "beam_id", "request_slot", "target_residual", "target_load")}
    while obs is not None:
        z, survival = transformed_descriptors(obs, model.method)
        if fit_normalizer:
            model.normalizer.update(z, obs.sat_id >= 0)
        requests = explore_requests(obs, controller.select(obs), exploration, rng)
        result = env.step(requests)
        target_residual = np.zeros(len(requests), dtype=np.float32)
        valid = requests >= 0
        rows = np.flatnonzero(valid)
        target_residual[rows] = -result.cost[rows] - obs.analytic[rows, requests[rows]]
        target_load = np.asarray(result.occupancy)[np.maximum(obs.sat_id, 0)] / cfg.environment["capacity"]
        target_load = np.where(obs.sat_id >= 0, target_load, 0.)
        values = dict(z=z.astype(np.float32), survival=survival.astype(np.float32),
                      sat_id=obs.sat_id.astype(np.int32), beam_id=obs.beam_id.astype(np.int16),
                      request_slot=requests.astype(np.int16), target_residual=target_residual,
                      target_load=target_load.astype(np.float32))
        for key in store:
            store[key].append(values[key])
        obs = result.next_observation
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **{key: np.stack(value) for key, value in store.items()})


class SequenceStore:
    """Lazy episodic storage; sequence draws never cross an episode boundary."""
    def __init__(self, paths, max_cached=2):
        self.paths = [Path(p) for p in paths]
        self.max_cached = max_cached
        self.cache = OrderedDict()
        if not self.paths:
            raise ValueError("No collected training episodes")

    def _episode(self, index):
        path = self.paths[index]
        if path not in self.cache:
            with np.load(path, allow_pickle=False) as record:
                self.cache[path] = {key: record[key] for key in record.files}
            while len(self.cache) > self.max_cached:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path]

    def sample(self, rng, length):
        data = self._episode(int(rng.integers(len(self.paths))))
        n = len(data["z"])
        if n < length:
            raise ValueError(f"Training episode length {n} shorter than sequence length {length}")
        start = int(rng.integers(n - length + 1))
        return {key: value[start:start + length] for key, value in data.items()}


def residual_sequence_loss(model, sequences, cfg, device):
    """Global MSE over requested edges and valid graph edges in the batch."""
    burn = int(cfg.training["burn_in"])
    request_sum = next(model.parameters()).sum() * 0.
    load_sum = request_sum
    request_count = load_count = 0
    for data in sequences:
        tensors = {key: torch.as_tensor(value, device=device) for key, value in data.items()}
        state = model.initial_state(data["z"].shape[1], cfg.environment["n_satellites"])
        for t in range(len(data["z"])):
            args = (tensors["z"][t].float(), tensors["survival"][t].float(),
                    tensors["sat_id"][t].long(), tensors["beam_id"][t].long(), state)
            if t < burn:
                with torch.no_grad():
                    _, _, state = model(*args)
                state = state.detach()
                continue
            prediction, load, state = model(*args)
            slots = tensors["request_slot"][t].long()
            users = (slots >= 0).nonzero().flatten()
            if len(users):
                target = tensors["target_residual"][t, users].float()
                request_sum = request_sum + F.mse_loss(prediction[users, slots[users]], target, reduction="sum")
                request_count += len(users)
            valid = tensors["sat_id"][t] >= 0
            if valid.any():
                load_sum = load_sum + F.mse_loss(load[valid], tensors["target_load"][t][valid].float(), reduction="sum")
                load_count += int(valid.sum())
    return request_sum / max(1, request_count) + cfg.training["load_loss_weight"] * load_sum / max(1, load_count)


def _residual_lr(update, total, cfg):
    peak, final = cfg.training["learning_rate"], cfg.training["final_learning_rate"]
    warmup = int(cfg.training["warmup_updates"])
    if warmup and update <= warmup:
        return peak * update / warmup
    fraction = min(1., max(0., (update - warmup) / max(1, total - warmup)))
    return final + (peak - final) * .5 * (1. + math.cos(math.pi * fraction))


def train_residual(cfg: Config, method: str, seed: int, output,
                   device="cpu", overrides: dict | None = None, resume=None) -> Path:
    """Four rounds of collected executed-utility regression, with full resume.

    ``overrides`` can reduce workload for an integration run. Its effective
    configuration is saved, so a short test cannot masquerade as paper training.
    """
    method = canonical_method(method)
    if method not in RESIDUAL_METHODS:
        raise ValueError(method)
    cfg = _configured(cfg, method, overrides)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(output / "effective_config.yaml")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = TGNModel(method, cfg.environment["n_beams"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training["learning_rate"], weight_decay=cfg.training["weight_decay"])
    total = int(cfg.training["rounds"] * cfg.training["updates_per_round"])
    interval = int(cfg.training["selection_interval"])
    if interval < 1 or total < 1 or cfg.training["burn_in"] >= cfg.training["sequence_length"]:
        raise ValueError("Invalid update/selection/sequence configuration")
    update = 0
    best_cost = math.inf
    selected_update = None
    round_start = 0
    inner_start = 0
    history = []
    resume_paths = None
    if resume is not None:
        checkpoint = _load(resume, device)
        if checkpoint["kind"] != "residual" or checkpoint["method"] != method:
            raise ValueError("Checkpoint model identity mismatch")
        if checkpoint["config"] != cfg.to_dict():
            raise ValueError("Resume configuration differs; evaluate separately or resume the original configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        update = checkpoint["update"]
        best_cost = checkpoint["best_validation_cost"]
        selected_update = checkpoint["selected_update"]
        round_start, inner_start = checkpoint["round_index"], checkpoint["update_in_round"]
        history = checkpoint["history"]
        resume_paths = checkpoint["dataset_paths"]
        _restore_rng(checkpoint["rng"], rng)
    episodes = int(cfg.training["episodes_per_round"])
    explorations = cfg.training["exploration"]
    for round_index in range(round_start, int(cfg.training["rounds"])):
        paths = [output / "rollouts" / f"round_{round_index + 1}" / f"episode_{episode:03d}.npz" for episode in range(episodes)]
        continuing = resume is not None and round_index == round_start and resume_paths is not None
        if continuing:
            paths = [output / p for p in resume_paths]
            if not all(p.exists() for p in paths):
                raise FileNotFoundError("Resume requires the original collected trajectory files")
        else:
            policy = RuleController("greedy", cfg) if round_index == 0 else NeuralController(model, cfg, device)
            for episode, path in enumerate(paths):
                _collect_residual_episode(cfg, model, policy, seed, round_index * episodes + episode,
                                          explorations[min(round_index, len(explorations) - 1)], rng,
                                          path, fit_normalizer=(round_index == 0))
            if round_index == 0:
                model.normalizer.freeze()
            inner_start = 0
        store = SequenceStore(paths)
        for inner in range(inner_start, int(cfg.training["updates_per_round"])):
            update += 1
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = _residual_lr(update, total, cfg)
            sequences = [store.sample(rng, int(cfg.training["sequence_length"]))
                         for _ in range(int(cfg.training["batch_sequences"]))]
            optimizer.zero_grad(set_to_none=True)
            loss = residual_sequence_loss(model, sequences, cfg, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at update {update}")
            loss.backward()
            # Residual training does not borrow the DQN's clipping setting.
            optimizer.step()
            row = dict(update=update, round=round_index + 1, loss=float(loss.detach()), lr=optimizer.param_groups[0]["lr"])
            evaluate = update % interval == 0
            if evaluate:
                cost = _validation_cost(model, cfg, method, device)
                row["validation_cost"] = cost
                if cost < best_cost:
                    best_cost, selected_update = cost, update
                    _save(output / "selected.pt", dict(format_version=1, kind="residual", method=method,
                          model=model.state_dict(), config=cfg.to_dict(), seed=seed,
                          update=update, validation_cost=cost))
            history.append(row)
            if evaluate or inner + 1 == cfg.training["updates_per_round"]:
                _save(output / "latest.pt", dict(format_version=1, kind="residual", method=method,
                      model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg.to_dict(), seed=seed,
                      update=update, round_index=round_index, update_in_round=inner + 1,
                      best_validation_cost=best_cost, selected_update=selected_update, history=history,
                      dataset_paths=[str(p.relative_to(output)) for p in paths], rng=_rng_state(rng)))
                (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        inner_start = 0
        resume_paths = None
        resume = None
    if not (output / "selected.pt").exists():
        raise RuntimeError("No positive-update model was selected")
    return output / "selected.pt"


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.states = np.zeros((capacity, 73), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.next_states = np.zeros((capacity, 73), np.float32)
        self.next_masks = np.zeros((capacity, 9), bool)
        self.done = np.zeros(capacity, bool)
        self.position = self.size = 0

    def add_batch(self, states, actions, rewards, next_states, next_masks, done):
        n = len(states)
        for i in range(n):
            p = self.position
            self.states[p] = states[i]
            self.actions[p] = actions[i]
            self.rewards[p] = rewards[i]
            self.next_states[p] = next_states[i]
            self.next_masks[p] = next_masks[i]
            self.done[p] = done[i]
            self.position = (p + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, size, rng, device):
        if size > self.size:
            raise ValueError("Replay batch exceeds collected transitions")
        index = rng.choice(self.size, size, replace=False)
        return {key: torch.as_tensor(getattr(self, key)[index], device=device)
                for key in ("states", "actions", "rewards", "next_states", "next_masks", "done")}

    def state_dict(self):
        return {key: getattr(self, key) for key in ("capacity", "position", "size", "states", "actions", "rewards", "next_states", "next_masks", "done")}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity changed")
        for key, value in state.items():
            setattr(self, key, value)


def dqn_update(online, target, optimizer, replay, rng, device, batch_size=128, discount=.99):
    batch = replay.sample(batch_size, rng, device)
    prediction = online(batch["states"]).gather(1, batch["actions"][:, None]).squeeze(1)
    with torch.no_grad():
        next_values = target(batch["next_states"]).masked_fill(~batch["next_masks"], -torch.inf).max(-1).values
        next_values = torch.where(batch["done"], 0., next_values)
        if not torch.isfinite(next_values).all():
            raise ValueError("Nonterminal transition has no legal next action")
        value = batch["rewards"] + discount * next_values
    loss = hysteretic_huber(prediction, value)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 1.)
    optimizer.step()
    return float(loss.detach())


def train_dqn(cfg: Config, seed: int, output, device="cpu",
              overrides: dict | None = None, resume=None) -> Path:
    """Shared 73→128→128→9 hysteretic DQN; one update per system epoch.

    Resume snapshots are written at complete-episode boundaries and preserve
    replay, RNGs, optimizer, selected model, and environment-episode counters.
    """
    cfg = _configured(cfg, "madrl", overrides)
    options = dict(epochs=256000, anneal_epochs=102400, replay_capacity=100000,
                   replay_warmup=10000, batch_size=128, target_interval=1000,
                   learning_rate=1e-4, discount=.99)
    options.update(cfg.training.get("dqn", {}))
    # Flat dqn_* overrides are convenient for CLI integration runs.
    for key in options:
        options[key] = cfg.training.get("dqn_" + key, options[key])
    if options["replay_warmup"] < options["batch_size"]:
        raise ValueError("DQN warmup must cover at least one minibatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(output / "effective_config.yaml")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    online, target = DQNNetwork().to(device), DQNNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=options["learning_rate"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    replay = ReplayBuffer(options["replay_capacity"])
    epoch = updates = collected = episode_id = 0
    best_cost = math.inf
    selected_update = None
    history = []
    if resume is not None:
        checkpoint = _load(resume, device)
        if checkpoint["kind"] != "dqn" or checkpoint["config"] != cfg.to_dict():
            raise ValueError("DQN resume configuration mismatch")
        online.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        replay.load_state_dict(checkpoint["replay"])
        epoch, updates, collected, episode_id = (checkpoint[k] for k in ("epoch", "updates", "collected", "next_episode_id"))
        best_cost, selected_update, history = (checkpoint[k] for k in ("best_validation_cost", "selected_update", "history"))
        _restore_rng(checkpoint["rng"], rng)
    while epoch < int(options["epochs"]):
        env = _environment(cfg, seed, episode_id)
        obs = env.reset()
        while obs is not None and epoch < int(options["epochs"]):
            online.eval()
            features = dqn_features(obs)
            if not bool(online.normalizer.frozen):
                slots = features[:, :72].reshape(-1, 8, 9)
                online.normalizer.update(slots[..., :6], slots[..., 7] > 0)
            epsilon = 1. - .95 * min(1., epoch / max(1, options["anneal_epochs"]))
            mask = action_mask(obs)
            with torch.no_grad():
                q = online(torch.as_tensor(features, device=device)).cpu().numpy()
            slots = rank_slots(obs, q[:, :8])
            actions = np.where(slots >= 0, slots, 8)
            for k in range(len(actions)):
                if rng.random() < epsilon:
                    actions[k] = rng.choice(np.flatnonzero(mask[k]))
            result = env.step(np.where(actions == 8, -1, actions))
            next_obs = result.next_observation
            terminal = next_obs is None or epoch + 1 >= int(options["epochs"])
            if terminal:
                next_features = np.zeros_like(features)
                next_masks = np.zeros_like(mask)
                next_masks[:, -1] = True
            else:
                next_features = dqn_features(next_obs)
                next_masks = action_mask(next_obs)
            replay.add_batch(features, actions, np.asarray(result.reward), next_features, next_masks,
                             np.full(len(actions), terminal, bool))
            collected += len(actions)
            epoch += 1
            if collected >= options["replay_warmup"] and replay.size >= options["batch_size"]:
                if not bool(online.normalizer.frozen):
                    online.normalizer.freeze()
                    target.load_state_dict(online.state_dict())
                online.train()
                loss = dqn_update(online, target, optimizer, replay, rng, device,
                                  options["batch_size"], options["discount"])
                updates += 1
                row = dict(epoch=epoch, update=updates, loss=loss, epsilon=epsilon)
                evaluate = updates % options["target_interval"] == 0
                if evaluate:
                    target.load_state_dict(online.state_dict())
                    cost = _validation_cost(online, cfg, "madrl", device)
                    row["validation_cost"] = cost
                    if cost < best_cost:
                        best_cost, selected_update = cost, updates
                        _save(output / "selected.pt", dict(format_version=1, kind="dqn", method="madrl", model=online.state_dict(),
                              config=cfg.to_dict(), seed=seed, update=updates, epoch=epoch, validation_cost=cost))
                history.append(row)
            obs = next_obs
        episode_id += 1
        _save(output / "latest.pt", dict(format_version=1, kind="dqn", method="madrl", model=online.state_dict(),
              target=target.state_dict(), optimizer=optimizer.state_dict(), replay=replay.state_dict(), config=cfg.to_dict(),
              options=options, seed=seed, epoch=epoch, updates=updates, collected=collected, next_episode_id=episode_id,
              best_validation_cost=best_cost, selected_update=selected_update, history=history, rng=_rng_state(rng)))
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not (output / "selected.pt").exists():
        raise RuntimeError("Budget ended before DQN optimization; no trained model exists")
    return output / "selected.pt"


def load_controller(path, cfg: Config | None = None, device="cpu"):
    """Load for evaluation; explicit cfg can change density/horizon/hysteresis."""
    saved = _load(path, device)
    saved_cfg = Config(**saved["config"])
    cfg = deepcopy(saved_cfg if cfg is None else cfg)
    method = saved["method"]
    if cfg.environment["n_beams"] != saved_cfg.environment["n_beams"]:
        raise ValueError("Beam-node identity convention cannot change during checkpoint evaluation")
    if method == "ephemeris":
        cfg.environment["descriptor_mode"] = "ephemeris"
    if saved["kind"] == "residual":
        model = TGNModel(method, cfg.environment["n_beams"])
    elif saved["kind"] == "dqn":
        model = DQNNetwork()
    else:
        raise ValueError("Unknown checkpoint kind")
    model.load_state_dict(saved["model"])
    if not bool(model.normalizer.frozen):
        raise ValueError("Checkpoint lacks frozen training normalization")
    controller = make_controller(method, cfg, model, device)
    controller.training_seed = int(saved["seed"])
    controller.checkpoint_path = str(Path(path).resolve())
    return controller
