from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ----------------------------
# Canonical metric names (paper-facing) and json-key aliases (code-facing)
# ----------------------------
CANONICAL_METRICS = [
    # key, y-label (paper), plot title (short)
    ("mse_mean", "Prediction MSE (mean over horizon)", "Prediction MSE vs. Horizon (mean)"),
    ("mse_last", "Prediction MSE (last step)", "Prediction MSE vs. Horizon (last)"),
    ("pingpong", "Ping-pong rate", "Ping-pong rate vs. Horizon"),
    ("ho_fail", "HO failure probability", "HO failure probability vs. Horizon"),
    ("load_var", "Beam-load variance", "Beam-load variance vs. Horizon"),
    ("load_peak", "Peak beam load", "Peak beam load vs. Horizon"),
]

# Map canonical -> acceptable keys in json
ALIASES: Dict[str, List[str]] = {
    "mse_mean": ["mse_mean", "mse_mean_pred", "mse_rollout_mean"],
    "mse_last": ["mse_last", "mse_last_pred", "mse_rollout_last"],
    "pingpong": ["pingpong", "pingpong_rate", "pingpong_prob", "pingpong_pred"],
    "ho_fail": ["ho_fail", "ho_fail_rate", "ho_fail_prob", "ho_fail_pred", "handover_fail", "handover_failure"],
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

    out: Dict[int, Dict[str, float]] = {}
    for r in blob["results"]:
        H = int(r["H"])
        out[H] = {}
        for canonical, _, _ in CANONICAL_METRICS:
            v = _extract_metric(r, canonical)
            if v is not None and (not math.isnan(v)) and (not math.isinf(v)):
                out[H][canonical] = v
    return out


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


def _auto_scale(values: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Make axis human-readable:
      - Avoid 1e-5 offset text; instead, rescale and add '×10^{k}' to ylabel suffix.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values, ""

    vmax = float(np.max(np.abs(finite)))
    if vmax == 0.0:
        return values, ""

    # choose power so that scaled vmax in [1, 1000)
    power = int(math.floor(math.log10(vmax)))
    # If already readable (10^-2..10^3), don't scale
    if -2 <= power <= 3:
        return values, ""

    # scale to bring into 10^0..10^3
    # e.g. vmax=1e-5 -> power=-5, scale by 1e5
    scale_pow = -power
    scaled = values * (10.0 ** scale_pow)
    suffix = f" (×10$^{{{power}}}$)"  # because y_scaled = y * 10^{-power}; label shows original magnitude
    # Careful: if we multiply by 10^{-power}, then original y = y_scaled × 10^{power}
    # So suffix should state '×10^{power}' for original unit. That’s what we show.
    return scaled, suffix


def _clip_prob(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return max(0.0, min(1.0, x))


def _diagnose_ho_failure(model_name: str, Hs: List[int], ho_vals: List[float], ping_vals: List[float]) -> None:
    """
    Print a focused warning when HO failure is suspicious (often equals 1.0 everywhere).
    """
    finite = [v for v in ho_vals if np.isfinite(v)]
    if len(finite) == 0:
        return

    all_one = all(v >= 0.999 for v in finite)
    if not all_one:
        return

    # If ho_fail is always 1, this usually means "always outage / no feasible link",
    # or metric definition mismatch.
    msg = [
        f"[WARN] {MODEL_DISPLAY.get(model_name, model_name)}: HO failure probability is ~1.0 for all horizons.",
        "Most common causes:",
        "  (1) Metric definition is actually 'outage ratio' (a_k(t)=∅) rather than 'handover failure'.",
        "  (2) Feasible set is empty almost always: gamma_min too high, visibility radius too small, load cap too tight, or load uses simultaneous ℓ(t) instead of observed ℓ(t-Δt).",
        "  (3) In multi-user mode, interference edges / load updates create systematic infeasibility.",
        "What to check quickly:",
        "  - In exported rollout JSON, confirm the event counted by 'ho_fail' (outage vs failure-on-handover).",
        "  - During rollout, log fraction of steps with |A_k(t)|=0 and fraction with a_k(t)=∅.",
        "  - Compare single-user vs multi-user with same gamma_min / visibility_radius.",
    ]
    if any(np.isfinite(v) and v > 1e-6 for v in ping_vals):
        msg.append("  Note: ping-pong is nonzero while ho_fail=1, which strongly suggests 'ho_fail' is not computed as intended.")
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
                    help="One or more paths/globs to MLP rollout_metrics.json (supports multiple seeds).")
    ap.add_argument("--kan", type=str, nargs="+", required=True,
                    help="One or more paths/globs to KAN rollout_metrics.json (supports multiple seeds).")
    ap.add_argument("--physick", type=str, nargs="+", required=True,
                    help="One or more paths/globs to PhysiCK rollout_metrics.json (supports multiple seeds).")
    ap.add_argument("--out_dir", type=str, default="runs/plots")
    ap.add_argument("--dt", type=float, default=None,
                    help="Decision interval Δt in seconds. If set, x-axis will show both steps and seconds in label.")
    ap.add_argument("--ci", type=str, default="std", choices=["none", "std"],
                    help="Uncertainty band: 'std' (mean ± std) or 'none'.")
    ap.add_argument("--fontsize", type=int, default=9)
    ap.add_argument("--grid_alpha", type=float, default=0.15)
    ap.add_argument("--combined", action="store_true",
                    help="If set, also export a single 2x3 combined figure (recommended for TWC).")
    ap.add_argument("--target_label", type=str, default="Prediction",
                    help="Label prefix for MSE axes, e.g. 'Beam-load prediction' or 'Association prediction'.")
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

    # Load and group each seed
    per_model_byH: Dict[str, List[Dict[int, Dict[str, float]]]] = {m: [] for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        for p in paths[m]:
            blob = _load_json(p)
            per_model_byH[m].append(_group_results_by_H(blob))

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
        is_prob = canonical in ["pingpong", "ho_fail"]
        if is_prob:
            scaled_suffix = ""
        else:
            _, scaled_suffix = _auto_scale(all_vals_arr)

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

            if (not is_prob) and np.isfinite(means_arr).any():
                means_arr, _ = _auto_scale(means_arr)
                stds_arr = stds_arr * (means_arr.max() / means_arr.max()) if False else stds_arr  # no-op

                # If we scaled means, scale std by same factor:
                # Infer scale factor from one finite point if possible.
                # (We recompute scale using the same procedure on original mean values.)
                # Use first finite mean:
                idx = np.where(np.isfinite(means_arr))[0]
                if idx.size > 0:
                    # derive scale from original means and scaled means at idx[0]
                    orig_mu = means[idx[0]]
                    scaled_mu = float(means_arr[idx[0]])
                    if orig_mu != 0:
                        scale = scaled_mu / orig_mu
                        stds_arr = stds_arr * scale

            line = ax.plot(
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
        ax.set_title(title)

        # Grid / spines
        ax.grid(True, alpha=args.grid_alpha, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Avoid scientific offset text
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain")

    # Diagnose HO failure anomalies
    for m in MODEL_ORDER:
        ho_vals = [agg[m].get(H, {}).get("ho_fail", (float("nan"), 0.0))[0] for H in Hs]
        ping_vals = [agg[m].get(H, {}).get("pingpong", (float("nan"), 0.0))[0] for H in Hs]
        _diagnose_ho_failure(m, Hs, ho_vals, ping_vals)

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

    # Combined 2x3 figure (recommended)
    if args.combined:
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0))
        axes = axes.reshape(2, 3)

        for idx, (canonical, ylabel, title) in enumerate(CANONICAL_METRICS):
            r, c = divmod(idx, 3)
            ax = axes[r, c]
            plot_metric(ax, canonical, ylabel, title)

        # One shared legend (paper style)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = out_dir / "system_metrics_2x3.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

