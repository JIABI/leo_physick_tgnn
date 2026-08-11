from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from leo_pg.train.frozen_diagnostics import EVALUATION_MODE, METRIC_CONTRACT


# ----------------------------
# Canonical frozen-diagnostic names and backward-compatible JSON aliases
# ----------------------------
CANONICAL_METRICS = [
    # key, y-label, plot title
    ("mse_mean", "Prediction MSE (mean over horizon)", "Prediction MSE vs. Horizon (mean)"),
    ("mse_last", "Prediction MSE (last step)", "Prediction MSE vs. Horizon (last)"),
    ("adjacent_aba", "Adjacent A-B-A fraction", "Adjacent A-B-A fraction vs. Horizon"),
    ("assignment_failure", "Assignment-failure fraction", "Assignment failures vs. Horizon"),
    ("load_var", "Predicted-utilization variance", "Utilization variance vs. Horizon"),
    ("load_peak", "Mean peak predicted utilization", "Mean peak utilization vs. Horizon"),
]

# Map canonical -> acceptable keys in json
ALIASES: Dict[str, List[str]] = {
    "mse_mean": ["mse_mean", "mse_mean_pred", "mse_rollout_mean"],
    "mse_last": ["mse_last", "mse_last_pred", "mse_rollout_last"],
    "adjacent_aba": ["adjacent_aba_pred", "adjacent_aba", "pingpong_pred", "pingpong_rate"],
    "assignment_failure": ["assignment_failure_pred", "assignment_failure", "ho_fail_pred", "ho_fail_rate"],
    "load_var": ["load_var", "beam_load_var", "load_var_pred", "load_variance", "beamload_var"],
    "load_peak": ["load_peak", "peak_load", "beam_load_peak", "load_peak_pred", "beamload_peak"],
}

MODEL_ORDER = ["mlp", "kan", "physick"]
MODEL_DISPLAY = {"mlp": "MLP", "kan": "KAN", "physick": "PhysiCK"}

# Black/white printable styling (don’t rely on color)
STYLE = {
    "mlp": dict(linestyle="-", marker="o"),
    "kan": dict(linestyle="--", marker="s"),
    "physick": dict(linestyle="-.", marker="^"),
}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _contract_id(blob: Dict[str, Any], *, allow_legacy: bool) -> str:
    if (
        blob.get("evaluation_mode") == EVALUATION_MODE
        and blob.get("metric_contract") == METRIC_CONTRACT
    ):
        return "canonical-v1"
    if not allow_legacy:
        raise ValueError(
            "Diagnostic JSON has a missing or incompatible evaluation_mode/metric_contract. "
            "Regenerate it with scripts/rollout.py; use --allow-legacy only for isolated inspection."
        )
    return "legacy:" + json.dumps(
        {
            "evaluation_mode": blob.get("evaluation_mode"),
            "metric_contract": blob.get("metric_contract"),
        },
        sort_keys=True,
    )


def _expand_paths(patterns: List[str]) -> List[Path]:
    """
    Accept:
      - exact file paths
      - directory paths (will search for *.json inside)
      - glob patterns (e.g., runs/**/rollout_metrics*.json)
    """
    out: List[Path] = []
    for p in patterns:
        pp = Path(p)
        if any(ch in p for ch in ["*", "?", "[", "]"]):
            out.extend([Path(x) for x in sorted(Path().glob(p))])
        elif pp.is_dir():
            out.extend(sorted(pp.glob("*.json")))
        else:
            out.append(pp)
    # de-dup
    uniq = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _extract_metric(d: Dict[str, Any], canonical: str) -> Optional[float]:
    for k in ALIASES.get(canonical, []):
        if k in d:
            try:
                return float(d[k])
            except Exception:
                return None
    return None


def _group_results_by_H(blob: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """
    Expected blob format:
      { "results": [ {"H": 30, "mse_last":..., ...}, ... ], ... }
    """
    if "results" not in blob or not isinstance(blob["results"], list):
        raise ValueError("JSON must contain a top-level list field: 'results'.")

    collected: Dict[int, Dict[str, List[float]]] = {}
    for r in blob["results"]:
        H = int(r["H"])
        collected.setdefault(H, {})
        for canonical, _, _ in CANONICAL_METRICS:
            v = _extract_metric(r, canonical)
            if v is not None and (not math.isnan(v)) and (not math.isinf(v)):
                collected[H].setdefault(canonical, []).append(v)
    return {
        horizon: {
            metric: float(np.mean(values))
            for metric, values in metrics.items()
        }
        for horizon, metrics in collected.items()
    }


def _aggregate_over_seeds(list_of_byH: List[Dict[int, Dict[str, float]]]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Return:
      agg[H][metric] = (mean, std)
    """
    all_H = sorted(set().union(*[set(m.keys()) for m in list_of_byH])) if list_of_byH else []
    agg: Dict[int, Dict[str, Tuple[float, float]]] = {H: {} for H in all_H}

    for H in all_H:
        for canonical, _, _ in CANONICAL_METRICS:
            vals = []
            for byH in list_of_byH:
                if H in byH and canonical in byH[H]:
                    vals.append(byH[H][canonical])
            if len(vals) == 0:
                continue
            v = np.asarray(vals, dtype=np.float64)
            agg[H][canonical] = (float(v.mean()), float(v.std(ddof=0)))
    return agg


def _scale_factor(values: np.ndarray) -> Tuple[float, str]:
    """Return one common axis scale factor and its original-unit suffix."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0, ""

    vmax = float(np.max(np.abs(finite)))
    if vmax == 0.0:
        return 1.0, ""

    # choose power so that scaled vmax in [1, 1000)
    power = int(math.floor(math.log10(vmax)))
    if -2 <= power <= 3:
        return 1.0, ""

    return 10.0 ** (-power), f" (×10$^{{{power}}}$)"


def _auto_scale(values: np.ndarray) -> Tuple[np.ndarray, str]:
    """Scale values with one factor selected from the full supplied collection."""
    factor, suffix = _scale_factor(values)
    return values * factor, suffix


def _clip_prob(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return max(0.0, min(1.0, x))


def _diagnose_assignment_failure(
    model_name: str,
    failure_values: List[float],
    aba_values: List[float],
) -> None:
    """Warn when the prototype assignment diagnostic is saturated."""
    finite = [v for v in failure_values if np.isfinite(v)]
    if len(finite) == 0:
        return

    all_one = all(v >= 0.999 for v in finite)
    if not all_one:
        return

    msg = [
        f"[WARN] {MODEL_DISPLAY.get(model_name, model_name)}: assignment-failure fraction is ~1.0.",
        "Check candidate visibility, SINR threshold, capacity, and load units.",
    ]
    if any(np.isfinite(v) and v > 1e-6 for v in aba_values):
        msg.append(
            "Adjacent A-B-A events are nonzero despite near-universal assignment failure; "
            "inspect event construction."
        )
    print("\n".join(msg))


def _setup_rcparams(fontsize: int = 9) -> None:
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "font.family": "serif",
        "figure.dpi": 200,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", type=str, nargs="+", required=True,
                    help="One or more paths/globs to MLP diagnostic JSON files.")
    ap.add_argument("--kan", type=str, nargs="+", required=True,
                    help="One or more paths/globs to KAN diagnostic JSON files.")
    ap.add_argument("--physick", type=str, nargs="+", required=True,
                    help="One or more paths/globs to PhysiCK diagnostic JSON files.")
    ap.add_argument("--out_dir", type=str, default="runs/plots")
    ap.add_argument("--dt", type=float, default=None,
                    help="Decision interval Δt in seconds. If set, x-axis will show both steps and seconds in label.")
    ap.add_argument("--ci", type=str, default="std", choices=["none", "std"],
                    help="Uncertainty band: 'std' (mean ± std) or 'none'.")
    ap.add_argument("--fontsize", type=int, default=9)
    ap.add_argument("--grid_alpha", type=float, default=0.15)
    ap.add_argument("--combined", action="store_true",
                    help="If set, also export a single 2x3 combined diagnostic figure.")
    ap.add_argument("--target_label", type=str, default="Prediction",
                    help="Label prefix for MSE axes, e.g. 'Beam-load prediction' or 'Association prediction'.")
    ap.add_argument("--allow-legacy", action="store_true",
                    help="Inspect one internally consistent legacy contract; never mix it with canonical files.")
    args = ap.parse_args()

    _setup_rcparams(args.fontsize)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand globs / dirs
    paths = {
        "mlp": _expand_paths(args.mlp),
        "kan": _expand_paths(args.kan),
        "physick": _expand_paths(args.physick),
    }
    for k, ps in paths.items():
        if len(ps) == 0:
            raise FileNotFoundError(f"No json files found for {k} using inputs: {getattr(args, k)}")
        for p in ps:
            if not p.exists():
                raise FileNotFoundError(f"Missing file: {p}")

    # Load and group each independent input file.
    per_model_byH: Dict[str, List[Dict[int, Dict[str, float]]]] = {m: [] for m in MODEL_ORDER}
    contract_ids = set()
    for m in MODEL_ORDER:
        for p in paths[m]:
            blob = _load_json(p)
            contract_ids.add(_contract_id(blob, allow_legacy=args.allow_legacy))
            per_model_byH[m].append(_group_results_by_H(blob))
    if len(contract_ids) != 1:
        raise ValueError("Input JSON files use different metric contracts and cannot be combined")
    contract_id = next(iter(contract_ids))
    legacy_contract = contract_id != "canonical-v1"
    if legacy_contract:
        print("[WARN] plotting an explicitly allowed legacy diagnostic contract")

    # Aggregate mean/std over seeds
    agg = {m: _aggregate_over_seeds(per_model_byH[m]) for m in MODEL_ORDER}
    Hs = sorted(set().union(*[set(agg[m].keys()) for m in MODEL_ORDER]))

    # Plot helper
    def plot_metric(ax, canonical: str, ylabel: str, title: str):
        # collect all values for scaling
        all_vals = []
        for m in MODEL_ORDER:
            ys = []
            for H in Hs:
                if H in agg[m] and canonical in agg[m][H]:
                    ys.append(agg[m][H][canonical][0])
            all_vals.extend(ys)
        all_vals_arr = np.asarray(all_vals, dtype=np.float64)

        # If prob-like metric, clip to [0,1] and do not scale
        is_prob = canonical in ["adjacent_aba", "assignment_failure"]
        if is_prob:
            scale_factor = 1.0
            scaled_suffix = ""
        else:
            scale_factor, scaled_suffix = _scale_factor(all_vals_arr)

        # Plot each model
        for m in MODEL_ORDER:
            means = []
            stds = []
            for H in Hs:
                if H in agg[m] and canonical in agg[m][H]:
                    mu, sd = agg[m][H][canonical]
                else:
                    mu, sd = float("nan"), float("nan")
                if is_prob and np.isfinite(mu):
                    mu = _clip_prob(mu)
                    sd = max(0.0, min(sd, 1.0))  # keep bounded-ish
                means.append(mu)
                stds.append(sd)

            means_arr = np.asarray(means, dtype=np.float64)
            stds_arr = np.asarray(stds, dtype=np.float64)

            means_arr = means_arr * scale_factor
            stds_arr = stds_arr * scale_factor

            ax.plot(
                Hs, means_arr,
                label=MODEL_DISPLAY[m],
                linewidth=1.8,
                markersize=5.5,
                markeredgewidth=0.8,
                **STYLE[m],
            )[0]

            if args.ci == "std":
                lo = means_arr - stds_arr
                hi = means_arr + stds_arr
                ax.fill_between(Hs, lo, hi, alpha=0.12)

        # Axes labels/titles
        xlab = "Horizon $H$ (decision steps)"
        if args.dt is not None:
            xlab = f"Horizon $H$ (steps, $\\Delta t$={args.dt:g}s)"
        ax.set_xlabel(xlab)

        if canonical in ["mse_mean", "mse_last"]:
            ylabel = ylabel.replace("Prediction", args.target_label)

        ax.set_ylabel(ylabel + scaled_suffix)
        ax.set_title(title + (" [legacy contract]" if legacy_contract else ""))

        # Grid / spines
        ax.grid(True, alpha=args.grid_alpha, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Avoid scientific offset text
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain")

    # Diagnose assignment-failure anomalies.
    for m in MODEL_ORDER:
        failure_values = [
            agg[m].get(H, {}).get("assignment_failure", (float("nan"), 0.0))[0]
            for H in Hs
        ]
        aba_values = [
            agg[m].get(H, {}).get("adjacent_aba", (float("nan"), 0.0))[0]
            for H in Hs
        ]
        _diagnose_assignment_failure(m, failure_values, aba_values)

    # Individual plots
    for canonical, ylabel, title in CANONICAL_METRICS:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        plot_metric(ax, canonical, ylabel, title)

        # Legend outside (reduces occlusion)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)

        out_path = out_dir / f"{canonical}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")

    # Combined 2x3 figure.
    if args.combined:
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0))
        axes = axes.reshape(2, 3)

        for idx, (canonical, ylabel, title) in enumerate(CANONICAL_METRICS):
            r, c = divmod(idx, 3)
            ax = axes[r, c]
            plot_metric(ax, canonical, ylabel, title)

        # One shared legend.
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = out_dir / "system_metrics_2x3.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
