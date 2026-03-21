"""
benchmark.py — Comparative analysis: Baseline vs. Enhanced APF-RRT.

Runs both planners N times with the same random seeds and produces a
summary table of:
  • Success rate
  • Mean computation time
  • Mean path length (joint-space L2)
  • Mean node count (memory efficiency)
"""

from __future__ import annotations
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BENCHMARK_RUNS, TIMEOUT_SEC, Q_START, Q_GOAL
from obstacles import ObstacleSet
from rrt_base import APFRRTPlanner, PlanResult
from rrt_enhanced import EnhancedAPFRRTPlanner


def _run_single(planner, seed: int) -> PlanResult:
    np.random.seed(seed)
    return planner.plan(Q_START, Q_GOAL, timeout=TIMEOUT_SEC)


def run_benchmark(
    n_runs: int = BENCHMARK_RUNS,
    verbose: bool = True,
) -> dict:
    """
    Execute *n_runs* planning attempts for both planners.

    Returns
    -------
    results : dict
        Keys: 'baseline', 'enhanced', each mapping to a list of PlanResult.
    """
    obs = ObstacleSet()
    planner_base = APFRRTPlanner(obs, apf_weight=0.5)
    planner_enh  = EnhancedAPFRRTPlanner(obs, apf_weight=0.5)

    results = {"baseline": [], "enhanced": []}

    for i in range(n_runs):
        seed = 1000 + i

        if verbose:
            print(f"  Run {i + 1:2d}/{n_runs}  (seed={seed})", end="  →  ", flush=True)

        r_base = _run_single(planner_base, seed)
        results["baseline"].append(r_base)

        r_enh = _run_single(planner_enh, seed)
        results["enhanced"].append(r_enh)

        if verbose:
            b = "✓" if r_base.success else "✗"
            e = "✓" if r_enh.success else "✗"
            print(f"Base[{b} {r_base.time_sec:.2f}s]  Enh[{e} {r_enh.time_sec:.2f}s]")

    return results


def summarise(results: dict) -> str:
    """
    Build a formatted comparison table from benchmark results.
    """
    lines = []
    header = (
        f"{'Metric':<28s}  {'Baseline APF-RRT':>18s}  {'Enhanced APF-RRT':>18s}"
    )
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    for label, key in [("Baseline APF-RRT", "baseline"), ("Enhanced APF-RRT", "enhanced")]:
        pass  # we'll compute per-metric below

    def _stats(key, attr, only_success=False):
        vals = []
        for r in results[key]:
            if only_success and not r.success:
                continue
            vals.append(getattr(r, attr))
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    # Success rate
    sr_b = sum(r.success for r in results["baseline"]) / len(results["baseline"]) * 100
    sr_e = sum(r.success for r in results["enhanced"]) / len(results["enhanced"]) * 100
    lines.append(f"{'Success Rate (%)':<28s}  {sr_b:>17.1f}%  {sr_e:>17.1f}%")

    # Computation time (successful runs only)
    m, s = _stats("baseline", "time_sec", only_success=True)
    base_time = f"{m:.3f} ± {s:.3f}"
    m, s = _stats("enhanced", "time_sec", only_success=True)
    enh_time = f"{m:.3f} ± {s:.3f}"
    lines.append(f"{'Comp. Time (s, success)':<28s}  {base_time:>18s}  {enh_time:>18s}")

    # Path length
    m, s = _stats("baseline", "path_length", only_success=True)
    base_pl = f"{m:.3f} ± {s:.3f}"
    m, s = _stats("enhanced", "path_length", only_success=True)
    enh_pl = f"{m:.3f} ± {s:.3f}"
    lines.append(f"{'Path Length (rad, success)':<28s}  {base_pl:>18s}  {enh_pl:>18s}")

    # Node count
    m, s = _stats("baseline", "node_count", only_success=True)
    base_nc = f"{m:.0f} ± {s:.0f}"
    m, s = _stats("enhanced", "node_count", only_success=True)
    enh_nc = f"{m:.0f} ± {s:.0f}"
    lines.append(f"{'Node Count (success)':<28s}  {base_nc:>18s}  {enh_nc:>18s}")

    lines.append("=" * len(header))
    return "\n".join(lines)


def plot_benchmark_bars(results: dict, save_path: str | None = None) -> plt.Figure:
    """Bar chart comparing the four key metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    labels = ["Baseline", "Enhanced"]
    colors = ["#5DA5DA", "#60BD68"]

    # 1. Success rate
    sr = [
        sum(r.success for r in results["baseline"]) / len(results["baseline"]) * 100,
        sum(r.success for r in results["enhanced"]) / len(results["enhanced"]) * 100,
    ]
    axes[0].bar(labels, sr, color=colors, edgecolor="black")
    axes[0].set_title("Success Rate (%)", fontweight="bold")
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(sr):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")

    # 2. Comp time
    def _success_vals(key, attr):
        return [getattr(r, attr) for r in results[key] if r.success]

    t_b, t_e = _success_vals("baseline", "time_sec"), _success_vals("enhanced", "time_sec")
    means = [np.mean(t_b) if t_b else 0, np.mean(t_e) if t_e else 0]
    stds  = [np.std(t_b)  if t_b else 0, np.std(t_e)  if t_e else 0]
    axes[1].bar(labels, means, yerr=stds, color=colors, edgecolor="black", capsize=5)
    axes[1].set_title("Computation Time (s)", fontweight="bold")

    # 3. Path length
    pl_b, pl_e = _success_vals("baseline", "path_length"), _success_vals("enhanced", "path_length")
    means = [np.mean(pl_b) if pl_b else 0, np.mean(pl_e) if pl_e else 0]
    stds  = [np.std(pl_b)  if pl_b else 0, np.std(pl_e)  if pl_e else 0]
    axes[2].bar(labels, means, yerr=stds, color=colors, edgecolor="black", capsize=5)
    axes[2].set_title("Path Length (rad)", fontweight="bold")

    # 4. Node count
    nc_b, nc_e = _success_vals("baseline", "node_count"), _success_vals("enhanced", "node_count")
    means = [np.mean(nc_b) if nc_b else 0, np.mean(nc_e) if nc_e else 0]
    stds  = [np.std(nc_b)  if nc_b else 0, np.std(nc_e)  if nc_e else 0]
    axes[3].bar(labels, means, yerr=stds, color=colors, edgecolor="black", capsize=5)
    axes[3].set_title("Node Count", fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Benchmark: Baseline vs Enhanced APF-RRT (20 runs)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
