"""
visualization.py — 3D plots, trajectory analysis, and telemetry visualization.

Generates:
  - 3D scatter/line plots of RRT tree, obstacles, path, robot
  - Side-by-side baseline vs enhanced comparison
  - Trajectory profile plots (q, qd, qdd over time)
  - Controller telemetry plots (tracking error, torques)
  - Animation frames for GIF generation
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OBSTACLES, Q_START, Q_GOAL, NUM_JOINTS
from robot_model import end_effector_position, link_positions
from rrt_base import PlanResult, TreeNode


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _draw_sphere(ax, centre, radius, color="red", alpha=0.25):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def _draw_robot(ax, q, color="steelblue", linewidth=2.5, label=None):
    pts = link_positions(q)
    origins = [pts[0][0]]
    for seg in pts:
        origins.append(seg[-1])
    origins = np.array(origins)
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2],
            "-o", color=color, linewidth=linewidth, markersize=4, label=label)


def _tree_to_taskspace(tree: list[TreeNode]):
    positions = [end_effector_position(n.q) for n in tree]
    edges_from, edges_to = [], []
    for i, node in enumerate(tree):
        if node.parent is not None:
            edges_from.append(positions[node.parent])
            edges_to.append(positions[i])
    return edges_from, edges_to


def _auto_scale(ax):
    all_centres = np.array([o[0] for o in OBSTACLES])
    lo = all_centres.min(axis=0) - 0.3
    hi = all_centres.max(axis=0) + 0.3
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(max(0, lo[2] - 0.1), hi[2] + 0.1)


# ─────────────────────────────────────────────
# 3D planner results
# ─────────────────────────────────────────────

def plot_result(
    result: PlanResult, title: str = "APF-RRT Result",
    save_path: str | None = None, show_tree: bool = True,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for c, r in OBSTACLES:
        _draw_sphere(ax, c, r, color="tomato", alpha=0.20)

    if show_tree and result.tree_nodes:
        edges_from, edges_to = _tree_to_taskspace(result.tree_nodes)
        # Color edges by depth for better visibility
        n_edges = len(edges_from)
        for idx, (a, b) in enumerate(zip(edges_from, edges_to)):
            alpha = 0.15 + 0.35 * (idx / max(n_edges, 1))
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color="cornflowerblue", linewidth=0.4, alpha=alpha)

    if result.path:
        path_ee = np.array([end_effector_position(q) for q in result.path])
        ax.plot(path_ee[:, 0], path_ee[:, 1], path_ee[:, 2],
                "-", color="limegreen", linewidth=3.0, label="Path", zorder=5)

    start_ee = end_effector_position(Q_START)
    goal_ee  = end_effector_position(Q_GOAL)
    ax.scatter(*start_ee, color="blue",  s=120, marker="^", label="Start", zorder=6)
    ax.scatter(*goal_ee,  color="gold",  s=120, marker="*", label="Goal",  zorder=6)

    _draw_robot(ax, Q_START, color="royalblue",  linewidth=2, label="Robot @ start")
    _draw_robot(ax, Q_GOAL,  color="goldenrod",  linewidth=2, label="Robot @ goal")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    _auto_scale(ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(
    result_base: PlanResult, result_enh: PlanResult,
    save_path: str | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(20, 8))

    for idx, (res, label) in enumerate(
        [(result_base, "Baseline APF-RRT"), (result_enh, "Enhanced APF-RRT")]
    ):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        for c, r in OBSTACLES:
            _draw_sphere(ax, c, r, color="tomato", alpha=0.18)
        if res.tree_nodes:
            ef, et = _tree_to_taskspace(res.tree_nodes)
            n_e = len(ef)
            for i, (a, b) in enumerate(zip(ef, et)):
                alpha = 0.1 + 0.35 * (i / max(n_e, 1))
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color="cornflowerblue", linewidth=0.3, alpha=alpha)
        if res.path:
            path_ee = np.array([end_effector_position(q) for q in res.path])
            ax.plot(path_ee[:, 0], path_ee[:, 1], path_ee[:, 2],
                    "-", color="limegreen", linewidth=3.0, label="Path")

        start_ee = end_effector_position(Q_START)
        goal_ee  = end_effector_position(Q_GOAL)
        ax.scatter(*start_ee, color="blue", s=100, marker="^")
        ax.scatter(*goal_ee,  color="gold", s=100, marker="*")
        _draw_robot(ax, Q_START, color="royalblue",  linewidth=1.5)
        _draw_robot(ax, Q_GOAL,  color="goldenrod", linewidth=1.5)

        status = "OK" if res.success else "FAIL"
        ax.set_title(f"{label}  [{status}]\n"
                     f"Time: {res.time_sec:.2f}s | Nodes: {res.node_count} | "
                     f"Path: {res.path_length:.3f} rad",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        _auto_scale(ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# Trajectory profile plots
# ─────────────────────────────────────────────

def plot_trajectory_profile(
    trajectory, save_path: str | None = None,
) -> plt.Figure:
    """Plot joint positions, velocities, accelerations over time."""
    ts, qs, qds = trajectory.sample_dense(dt=0.01)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    joint_colors = plt.cm.tab10(np.linspace(0, 1, NUM_JOINTS))

    for j in range(NUM_JOINTS):
        axes[0].plot(ts, qs[:, j], color=joint_colors[j], label=f"J{j+1}")
        axes[1].plot(ts, qds[:, j], color=joint_colors[j], label=f"J{j+1}")

    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title("Joint trajectory profile", fontweight="bold")
    axes[0].legend(ncol=7, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# Controller telemetry plots
# ─────────────────────────────────────────────

def plot_controller_telemetry(
    log: dict, save_path: str | None = None,
) -> plt.Figure:
    """Plot tracking error, torques, and desired vs actual positions."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    t = log["t"]
    joint_colors = plt.cm.tab10(np.linspace(0, 1, NUM_JOINTS))

    # Position tracking
    for j in range(NUM_JOINTS):
        axes[0].plot(t, log["q_des"][:, j], "--", color=joint_colors[j], alpha=0.5)
        axes[0].plot(t, log["q_act"][:, j], "-",  color=joint_colors[j], label=f"J{j+1}")
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title("PD controller telemetry (solid=actual, dashed=desired)",
                      fontweight="bold")
    axes[0].legend(ncol=7, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Tracking error RMS
    axes[1].plot(t, log["error"], color="crimson", linewidth=1.5)
    axes[1].set_ylabel("Tracking error RMS (rad)")
    axes[1].grid(True, alpha=0.3)

    # Torques
    for j in range(NUM_JOINTS):
        axes[2].plot(t, log["tau"][:, j], color=joint_colors[j], linewidth=0.8)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# Animation frames
# ─────────────────────────────────────────────

def plot_path_animation_frames(
    result: PlanResult, save_dir: str, n_interp: int = 5,
):
    import os
    os.makedirs(save_dir, exist_ok=True)
    if not result.path:
        return

    interp_path = []
    for i in range(len(result.path) - 1):
        for t in np.linspace(0, 1, n_interp, endpoint=False):
            interp_path.append(result.path[i] + t * (result.path[i+1] - result.path[i]))
    interp_path.append(result.path[-1])

    for fi, q in enumerate(interp_path):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        for c, r in OBSTACLES:
            _draw_sphere(ax, c, r, color="tomato", alpha=0.18)
        path_ee = np.array([end_effector_position(qi) for qi in result.path])
        ax.plot(path_ee[:, 0], path_ee[:, 1], path_ee[:, 2],
                "--", color="limegreen", linewidth=1.5, alpha=0.5)
        _draw_robot(ax, q, color="steelblue", linewidth=3)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"Frame {fi+1}/{len(interp_path)}", fontsize=10)
        ax.set_box_aspect([1, 1, 1])
        _auto_scale(ax)
        fig.savefig(f"{save_dir}/frame_{fi:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def create_gif(frames_dir: str, output_path: str, duration_ms: int = 80):
    try:
        from PIL import Image
        import glob, os
        files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        if not files:
            print("No frames found."); return
        imgs = [Image.open(f) for f in files]
        imgs[0].save(output_path, save_all=True, append_images=imgs[1:],
                     duration=duration_ms, loop=0)
        print(f"GIF saved to {output_path}")
    except ImportError:
        print("Pillow not installed — skipping GIF.")
