"""
visualization.py — Rich 3D visualisation of the APF-RRT planner.

Produces publication-quality matplotlib figures showing:
  • Spherical obstacles
  • RRT tree edges (in task-space, mapped via FK)
  • Final path (bold overlay)
  • Start / goal markers
  • Robot arm in start and goal configurations

Also provides an animation helper that renders a GIF of the robot
following the planned path frame-by-frame.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # headless-safe backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from config import OBSTACLES, Q_START, Q_GOAL
from robot_model import end_effector_position, link_positions, forward_kinematics
from rrt_base import PlanResult, TreeNode


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _draw_sphere(ax, centre, radius, color="red", alpha=0.25):
    """Draw a translucent sphere on a 3D axes."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def _draw_robot(ax, q, color="steelblue", linewidth=2.5, label=None):
    """Draw the robot kinematic chain as connected line segments."""
    pts = link_positions(q)
    # Collect joint origins
    origins = [pts[0][0]]  # base
    for seg in pts:
        origins.append(seg[-1])
    origins = np.array(origins)

    ax.plot(
        origins[:, 0], origins[:, 1], origins[:, 2],
        "-o", color=color, linewidth=linewidth, markersize=4, label=label,
    )


def _tree_to_taskspace(tree: list[TreeNode]) -> tuple[list, list]:
    """Convert tree edges from joint-space to task-space (EE positions)."""
    positions = [end_effector_position(n.q) for n in tree]
    edges_from, edges_to = [], []
    for i, node in enumerate(tree):
        if node.parent is not None:
            edges_from.append(positions[node.parent])
            edges_to.append(positions[i])
    return edges_from, edges_to


# ─────────────────────────────────────────────
# Main visualisation entry points
# ─────────────────────────────────────────────

def plot_result(
    result: PlanResult,
    title: str = "APF-RRT Result",
    save_path: str | None = None,
    show_tree: bool = True,
) -> plt.Figure:
    """
    Generate a 3D plot of obstacles, the RRT tree, and the final path.

    Parameters
    ----------
    result : PlanResult
    title : str
    save_path : str, optional
        If given, save figure to this path.
    show_tree : bool
        Whether to render tree edges (can be slow for large trees).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ── Obstacles ──
    for centre, radius in OBSTACLES:
        _draw_sphere(ax, centre, radius, color="tomato", alpha=0.20)

    # ── Tree edges ──
    if show_tree and result.tree_nodes:
        edges_from, edges_to = _tree_to_taskspace(result.tree_nodes)
        for a, b in zip(edges_from, edges_to):
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="lightblue", linewidth=0.3, alpha=0.4,
            )

    # ── Path (task-space) ──
    if result.path:
        path_ee = np.array([end_effector_position(q) for q in result.path])
        ax.plot(
            path_ee[:, 0], path_ee[:, 1], path_ee[:, 2],
            "-", color="limegreen", linewidth=3.0, label="Path", zorder=5,
        )

    # ── Start / Goal ──
    start_ee = end_effector_position(Q_START)
    goal_ee  = end_effector_position(Q_GOAL)
    ax.scatter(*start_ee, color="blue",  s=120, marker="^", label="Start", zorder=6)
    ax.scatter(*goal_ee,  color="gold",  s=120, marker="*", label="Goal",  zorder=6)

    # ── Robot at start and goal ──
    _draw_robot(ax, Q_START, color="royalblue",  linewidth=2, label="Robot @ start")
    _draw_robot(ax, Q_GOAL,  color="goldenrod", linewidth=2, label="Robot @ goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    # Auto-scale
    ax.set_box_aspect([1, 1, 1])
    _auto_scale(ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(
    result_base: PlanResult,
    result_enh: PlanResult,
    save_path: str | None = None,
) -> plt.Figure:
    """Side-by-side comparison of baseline vs. enhanced planner."""
    fig = plt.figure(figsize=(20, 8))

    for idx, (res, label) in enumerate(
        [(result_base, "Baseline APF-RRT"), (result_enh, "Enhanced APF-RRT")]
    ):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")

        for centre, radius in OBSTACLES:
            _draw_sphere(ax, centre, radius, color="tomato", alpha=0.18)

        if res.tree_nodes:
            edges_from, edges_to = _tree_to_taskspace(res.tree_nodes)
            for a, b in zip(edges_from, edges_to):
                ax.plot(
                    [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color="lightblue", linewidth=0.25, alpha=0.35,
                )

        if res.path:
            path_ee = np.array([end_effector_position(q) for q in res.path])
            ax.plot(
                path_ee[:, 0], path_ee[:, 1], path_ee[:, 2],
                "-", color="limegreen", linewidth=3.0, label="Path",
            )

        start_ee = end_effector_position(Q_START)
        goal_ee  = end_effector_position(Q_GOAL)
        ax.scatter(*start_ee, color="blue", s=100, marker="^")
        ax.scatter(*goal_ee,  color="gold", s=100, marker="*")

        _draw_robot(ax, Q_START, color="royalblue",  linewidth=1.5)
        _draw_robot(ax, Q_GOAL,  color="goldenrod", linewidth=1.5)

        status = "✓" if res.success else "✗"
        subtitle = (
            f"{label}  [{status}]\n"
            f"Time: {res.time_sec:.2f}s | Nodes: {res.node_count} | "
            f"Path len: {res.path_length:.3f}"
        )
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        _auto_scale(ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_path_animation_frames(
    result: PlanResult,
    save_dir: str = "/home/claude/apf_rrt_planner/frames",
    n_interp: int = 5,
):
    """
    Render a sequence of PNG frames showing the robot following the path.
    Can be stitched into a GIF with imagemagick or pillow.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if not result.path:
        return

    # Interpolate between waypoints for smoother animation
    interp_path = []
    for i in range(len(result.path) - 1):
        for t in np.linspace(0, 1, n_interp, endpoint=False):
            interp_path.append(result.path[i] + t * (result.path[i + 1] - result.path[i]))
    interp_path.append(result.path[-1])

    for frame_idx, q in enumerate(interp_path):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for centre, radius in OBSTACLES:
            _draw_sphere(ax, centre, radius, color="tomato", alpha=0.18)

        path_ee = np.array([end_effector_position(qi) for qi in result.path])
        ax.plot(
            path_ee[:, 0], path_ee[:, 1], path_ee[:, 2],
            "--", color="limegreen", linewidth=1.5, alpha=0.5,
        )

        _draw_robot(ax, q, color="steelblue", linewidth=3)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"Frame {frame_idx + 1}/{len(interp_path)}", fontsize=10)
        ax.set_box_aspect([1, 1, 1])
        _auto_scale(ax)

        fig.savefig(f"{save_dir}/frame_{frame_idx:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def create_gif(frames_dir: str, output_path: str, duration_ms: int = 80):
    """Stitch PNGs in *frames_dir* into an animated GIF via Pillow."""
    try:
        from PIL import Image
        import glob, os

        files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        if not files:
            print("No frames found.")
            return

        imgs = [Image.open(f) for f in files]
        imgs[0].save(
            output_path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"GIF saved to {output_path}")
    except ImportError:
        print("Pillow not installed — skipping GIF creation.")


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _auto_scale(ax):
    """Set axis limits to encompass obstacles and robot workspace."""
    all_centres = np.array([o[0] for o in OBSTACLES])
    lo = all_centres.min(axis=0) - 0.3
    hi = all_centres.max(axis=0) + 0.3
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(max(0, lo[2] - 0.1), hi[2] + 0.1)
