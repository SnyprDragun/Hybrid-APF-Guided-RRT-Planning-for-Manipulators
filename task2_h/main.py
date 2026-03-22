#!/usr/bin/env python3
"""
main.py — Entry point for the Hybrid APF-RRT Motion Planner.

Modes
-----
  python main.py              # quick demo: run both planners once, plot
  python main.py --benchmark  # full 20-run comparative analysis
  python main.py --animate    # generate animation frames + GIF
  python main.py --pybullet   # launch PyBullet GUI (requires pybullet)

All outputs are saved to ./output/
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Q_START, Q_GOAL, BENCHMARK_RUNS
from obstacles import ObstacleSet
from rrt_base import APFRRTPlanner
from rrt_enhanced import EnhancedAPFRRTPlanner
from visualization import (
    plot_result,
    plot_comparison,
    plot_path_animation_frames,
    create_gif,
)
from benchmark import run_benchmark, summarise, plot_benchmark_bars


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Demo: single run of each planner
# ─────────────────────────────────────────────

def demo():
    ensure_output_dir()
    obs = ObstacleSet()

    print("=" * 60)
    print("  Hybrid APF-RRT Motion Planner — Demo Run")
    print("=" * 60)

    # ── Baseline ──
    np.random.seed(42)
    planner_base = APFRRTPlanner(obs, apf_weight=0.5)
    print("\n[1/2] Running Baseline APF-RRT …")
    result_base = planner_base.plan(Q_START, Q_GOAL)
    _print_result("Baseline", result_base)

    # ── Enhanced ──
    np.random.seed(42)
    planner_enh = EnhancedAPFRRTPlanner(obs, apf_weight=0.5)
    print("\n[2/2] Running Enhanced APF-RRT …")
    result_enh = planner_enh.plan(Q_START, Q_GOAL)
    _print_result("Enhanced", result_enh)

    # ── Plots ──
    print("\nGenerating visualisations …")
    plot_result(
        result_base,
        title="Baseline APF-RRT",
        save_path=os.path.join(OUTPUT_DIR, "baseline_result.png"),
    )
    plot_result(
        result_enh,
        title="Enhanced APF-RRT (Adaptive + Smoothing)",
        save_path=os.path.join(OUTPUT_DIR, "enhanced_result.png"),
    )
    plot_comparison(
        result_base, result_enh,
        save_path=os.path.join(OUTPUT_DIR, "comparison.png"),
    )
    print(f"  Plots saved to {OUTPUT_DIR}/")
    return result_base, result_enh


# ─────────────────────────────────────────────
# Full benchmark
# ─────────────────────────────────────────────

def full_benchmark():
    ensure_output_dir()

    print("=" * 60)
    print(f"  Benchmark: {BENCHMARK_RUNS} runs per planner")
    print("=" * 60)

    results = run_benchmark(n_runs=BENCHMARK_RUNS, verbose=True)
    table = summarise(results)
    print("\n" + table)

    # Save table
    with open(os.path.join(OUTPUT_DIR, "benchmark_table.txt"), "w") as f:
        f.write(table)

    plot_benchmark_bars(
        results,
        save_path=os.path.join(OUTPUT_DIR, "benchmark_bars.png"),
    )
    print(f"\nBenchmark artifacts saved to {OUTPUT_DIR}/")
    return results


# ─────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────

def animate():
    ensure_output_dir()
    obs = ObstacleSet()

    np.random.seed(42)
    planner = EnhancedAPFRRTPlanner(obs, apf_weight=0.5)
    print("Planning path for animation …")
    result = planner.plan(Q_START, Q_GOAL)

    if not result.success:
        print("Planning failed — cannot animate.")
        return

    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    print(f"Rendering frames to {frames_dir}/ …")
    plot_path_animation_frames(result, save_dir=frames_dir, n_interp=4)

    gif_path = os.path.join(OUTPUT_DIR, "robot_motion.gif")
    create_gif(frames_dir, gif_path, duration_ms=80)


# ─────────────────────────────────────────────
# PyBullet
# ─────────────────────────────────────────────

def pybullet_demo():
    from pybullet_sim import PyBulletVisualizer, PYBULLET_AVAILABLE

    if not PYBULLET_AVAILABLE:
        print("PyBullet is not installed. Run:  pip install pybullet")
        return

    obs = ObstacleSet()
    np.random.seed(42)
    planner = EnhancedAPFRRTPlanner(obs, apf_weight=0.5)
    result = planner.plan(Q_START, Q_GOAL)

    if not result.success:
        print("Planning failed.")
        return

    viz = PyBulletVisualizer(gui=True, use_gripper=True)
    viz.load_scene()
    viz.execute_path(result.path, speed=1.0)
    input("Press Enter to exit …")
    viz.disconnect()


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _print_result(label: str, r):
    status = "SUCCESS" if r.success else "FAILED"
    print(f"  {label}: {status}")
    print(f"    Iterations : {r.iterations}")
    print(f"    Time       : {r.time_sec:.3f} s")
    print(f"    Nodes      : {r.node_count}")
    if r.success:
        print(f"    Path length: {r.path_length:.4f} rad")
        print(f"    Waypoints  : {len(r.path)}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid APF-RRT Motion Planner for 6-DOF Manipulator"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run full 20-run comparative benchmark"
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Generate animation frames and GIF"
    )
    parser.add_argument(
        "--pybullet", action="store_true",
        help="Launch PyBullet GUI visualization"
    )
    args = parser.parse_args()

    if args.pybullet:
        pybullet_demo()
    elif args.benchmark:
        demo()
        full_benchmark()
    elif args.animate:
        _, result_enh = demo()
        animate()
    else:
        demo()


if __name__ == "__main__":
    main()
