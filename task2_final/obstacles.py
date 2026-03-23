"""
obstacles.py — Obstacle management and collision detection.

Checks both robot-vs-obstacle (swept-volume sphere model) and
robot-vs-self (pairwise link-sphere distances) collisions.
"""

from __future__ import annotations
import numpy as np
from config import OBSTACLES, COLLISION_CLEARANCE
from robot_model import link_positions, self_collision


class ObstacleSet:
    """
    Collection of spherical obstacles with collision and distance queries.

    Parameters
    ----------
    obstacles : list of (centre, radius) tuples
    clearance : float
        Extra safety margin added to every radius.
    check_self_collision : bool
        If True, config_collides also checks robot self-collision.
    """

    def __init__(
        self,
        obstacles: list[tuple[np.ndarray, float]] | None = None,
        clearance: float = COLLISION_CLEARANCE,
        check_self_collision: bool = True,
    ):
        obs = obstacles if obstacles is not None else OBSTACLES
        self.centres = np.array([o[0] for o in obs])
        self.radii   = np.array([o[1] for o in obs]) + clearance
        self.check_self = check_self_collision

    # ── Collision queries ────────────────────

    def point_collides(self, p: np.ndarray) -> bool:
        """Check if a single 3D point is inside any obstacle."""
        dists = np.linalg.norm(self.centres - p, axis=1)
        return bool(np.any(dists < self.radii))

    def config_collides(self, q: np.ndarray) -> bool:
        """
        Full collision check: robot links vs obstacles + self-collision.
        """
        # Self-collision
        if self.check_self and self_collision(q):
            return True

        # Robot-vs-obstacle (swept-volume)
        for link_pts in link_positions(q):
            for pt in link_pts:
                if self.point_collides(pt):
                    return True
        return False

    def segment_collides(
        self, q_from: np.ndarray, q_to: np.ndarray, n_checks: int = 10
    ) -> bool:
        """
        Check straight-line joint-space motion for intermediate collisions.
        """
        for t in np.linspace(0.0, 1.0, n_checks):
            q_mid = q_from + t * (q_to - q_from)
            if self.config_collides(q_mid):
                return True
        return False

    # ── Distance queries (used by APF) ───────

    def min_distance_to_obstacles(self, q: np.ndarray) -> float:
        """Smallest clearance between any link point and nearest obstacle surface."""
        d_min = np.inf
        for link_pts in link_positions(q):
            for pt in link_pts:
                dists = np.linalg.norm(self.centres - pt, axis=1) - self.radii
                d_min = min(d_min, float(np.min(dists)))
        return d_min

    def closest_obstacle_info(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """For a single 3D point, return (distance_to_surface, direction_away)."""
        diffs = point - self.centres
        dists = np.linalg.norm(diffs, axis=1)
        surface_dists = dists - self.radii
        idx = int(np.argmin(surface_dists))
        direction = diffs[idx] / (dists[idx] + 1e-12)
        return float(surface_dists[idx]), direction

    def local_obstacle_density(self, q: np.ndarray, radius: float = 0.3) -> float:
        """Count obstacle centres within *radius* of the end-effector."""
        from robot_model import end_effector_position
        ee = end_effector_position(q)
        dists = np.linalg.norm(self.centres - ee, axis=1)
        return float(np.sum(dists < radius))
