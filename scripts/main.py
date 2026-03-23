#!/usr/bin/env python3
"""
main.py — Entry point for the Hybrid APF-RRT Motion Planner (Panda 7-DOF).

Modes
-----
  python main.py                 # demo: plan + plot
  python main.py --benchmark     # 20-run comparative analysis
  python main.py --animate       # GIF animation
  python main.py --pybullet      # PyBullet with PD torque control

All outputs saved to ./output/
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Q_START, Q_GOAL, BENCHMARK_RUNS, TRAJECTORY_DURATION
from obstacles import ObstacleSet
from rrt_base import APFRRTPlanner
from rrt_enhanced import EnhancedAPFRRTPlanner
from trajectory import Trajectory
from controller import PDController
from visualization import (
    plot_result, plot_comparison, plot_trajectory_profile,
    plot_path_animation_frames, create_gif,
)
from benchmark import run_benchmark, summarise, plot_benchmark_bars

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _print_result(label, r):
    status = "SUCCESS" if r.success else "FAILED"
    print(f"  {label}: {status}")
    print(f"    Iterations : {r.iterations}")
    print(f"    Time       : {r.time_sec:.3f} s")
    print(f"    Nodes      : {r.node_count}")
    if r.success:
        print(f"    Path length: {r.path_length:.4f} rad")
        print(f"    Waypoints  : {len(r.path)}")


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

def demo():
    ensure_output()
    obs = ObstacleSet()

    print("=" * 60)
    print("  Hybrid APF-RRT Planner — Franka Panda 7-DOF")
    print("=" * 60)

    np.random.seed(42)
    planner_base = APFRRTPlanner(obs, apf_weight=0.5)
    print("\n[1/2] Baseline APF-RRT ...")
    result_base = planner_base.plan(Q_START, Q_GOAL)
    _print_result("Baseline", result_base)

    np.random.seed(42)
    planner_enh = EnhancedAPFRRTPlanner(obs, apf_weight=0.5)
    print("\n[2/2] Enhanced APF-RRT ...")
    result_enh = planner_enh.plan(Q_START, Q_GOAL)
    _print_result("Enhanced", result_enh)

    print("\nGenerating plots ...")
    plot_result(result_base, title="Baseline APF-RRT",
                save_path=os.path.join(OUTPUT_DIR, "baseline_result.png"))
    plot_result(result_enh, title="Enhanced APF-RRT (Adaptive + Smoothing)",
                save_path=os.path.join(OUTPUT_DIR, "enhanced_result.png"))
    plot_comparison(result_base, result_enh,
                    save_path=os.path.join(OUTPUT_DIR, "comparison.png"))

    # Trajectory profile for the enhanced path
    if result_enh.success:
        traj = Trajectory(result_enh.path, duration=TRAJECTORY_DURATION)
        plot_trajectory_profile(traj, save_path=os.path.join(OUTPUT_DIR, "trajectory_profile.png"))
        lim = traj.check_limits()
        print(f"\n  Trajectory limits check:")
        print(f"    Max vel per joint: {np.round(lim['max_velocity_per_joint'], 3)}")
        print(f"    Vel OK: {lim['vel_ok']}   Acc OK: {lim['acc_ok']}")

    print(f"  Plots saved to {OUTPUT_DIR}/")
    return result_base, result_enh


# ─────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────

def full_benchmark():
    ensure_output()
    print("=" * 60)
    print(f"  Benchmark: {BENCHMARK_RUNS} runs per planner")
    print("=" * 60)

    results = run_benchmark(n_runs=BENCHMARK_RUNS, verbose=True)
    table = summarise(results)
    print("\n" + table)

    with open(os.path.join(OUTPUT_DIR, "benchmark_table.txt"), "w") as f:
        f.write(table)

    plot_benchmark_bars(results, save_path=os.path.join(OUTPUT_DIR, "benchmark_bars.png"))
    print(f"\nBenchmark saved to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────

def animate():
    ensure_output()
    obs = ObstacleSet()
    np.random.seed(42)
    planner = APFRRTPlanner(obs, apf_weight=0.5)  # baseline has more waypoints
    print("Planning for animation ...")
    result = planner.plan(Q_START, Q_GOAL)
    if not result.success:
        print("Planning failed."); return

    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    print(f"Rendering {len(result.path)} waypoints ...")
    plot_path_animation_frames(result, save_dir=frames_dir, n_interp=6)
    create_gif(frames_dir, os.path.join(OUTPUT_DIR, "robot_motion.gif"), duration_ms=80)


# ─────────────────────────────────────────────
# PyBullet with torque control
# ─────────────────────────────────────────────

def pybullet_demo():
    from pybullet_sim import PyBulletVisualizer, PYBULLET_AVAILABLE
    from visualization import plot_controller_telemetry

    if not PYBULLET_AVAILABLE:
        print("PyBullet not installed. Run: pip install pybullet"); return

    # Plan
    obs = ObstacleSet()
    np.random.seed(42)
    planner = EnhancedAPFRRTPlanner(obs, apf_weight=0.5)
    print("Planning ...")
    result = planner.plan(Q_START, Q_GOAL)
    if not result.success:
        print("Planning failed."); return

    # Build trajectory
    traj = Trajectory(result.path, duration=TRAJECTORY_DURATION)
    ctrl = PDController()

    print(f"\nTrajectory: {TRAJECTORY_DURATION:.1f}s, {len(result.path)} waypoints")
    print("Launching PyBullet with PD torque control ...\n")

    viz = PyBulletVisualizer(gui=True, use_gripper=True)
    viz.load_scene()

    # Execute with controller
    log = viz.execute_with_controller(traj, ctrl, real_time=True)

    # Save telemetry
    ensure_output()
    plot_controller_telemetry(log, save_path=os.path.join(OUTPUT_DIR, "controller_telemetry.png"))
    print(f"\n  Final tracking error RMS: {log['error'][-1]:.6f} rad")
    print(f"  Telemetry saved to {OUTPUT_DIR}/controller_telemetry.png")

    input("\nPress Enter to exit ...")
    viz.disconnect()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid APF-RRT Motion Planner — Franka Panda 7-DOF"
    )
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full 20-run comparative benchmark")
    parser.add_argument("--animate", action="store_true",
                        help="Generate animation frames and GIF")
    parser.add_argument("--pybullet", action="store_true",
                        help="PyBullet GUI with PD torque control")
    args = parser.parse_args()

    if args.pybullet:
        pybullet_demo()
    elif args.benchmark:
        demo()
        full_benchmark()
    elif args.animate:
        demo()
        animate()
    else:
        demo()


if __name__ == "__main__":
    main()
