"""
obstacles.py — Obstacle management and collision detection.

Obstacles are modelled as spheres in task-space.  Collision is checked
against every sample point on every link of the robot's kinematic chain
(swept-volume approximation).
"""

from __future__ import annotations
import numpy as np
from config import OBSTACLES, COLLISION_CLEARANCE
from robot_model import link_positions


class ObstacleSet:
    """
    Collection of spherical obstacles with fast vectorised distance queries.

    Parameters
    ----------
    obstacles : list of (centre, radius) tuples.
        Each centre is np.ndarray shape (3,).
    clearance : float
        Extra safety margin added to every radius.
    """

    def __init__(
        self,
        obstacles: list[tuple[np.ndarray, float]] | None = None,
        clearance: float = COLLISION_CLEARANCE,
    ):
        obs = obstacles if obstacles is not None else OBSTACLES
        self.centres = np.array([o[0] for o in obs])        # (N, 3)
        self.radii   = np.array([o[1] for o in obs]) + clearance  # (N,)

    # ── Collision queries ────────────────────────────────────

    def point_collides(self, p: np.ndarray) -> bool:
        """Check if a single 3D point is inside any obstacle."""
        dists = np.linalg.norm(self.centres - p, axis=1)
        return bool(np.any(dists < self.radii))

    def config_collides(self, q: np.ndarray) -> bool:
        """
        Check if the full robot configuration *q* collides with any obstacle.

        Evaluates every interpolated point on every link (swept-volume).
        """
        for link_pts in link_positions(q):          # list of (S, 3)
            for pt in link_pts:
                if self.point_collides(pt):
                    return True
        return False

    def segment_collides(
        self, q_from: np.ndarray, q_to: np.ndarray, n_checks: int = 10
    ) -> bool:
        """
        Check if a straight-line motion in joint-space from *q_from* to
        *q_to* causes any intermediate collision.
        """
        for t in np.linspace(0.0, 1.0, n_checks):
            q_mid = q_from + t * (q_to - q_from)
            if self.config_collides(q_mid):
                return True
        return False

    # ── Distance queries (used by APF) ───────────────────────

    def min_distance_to_obstacles(self, q: np.ndarray) -> float:
        """
        Return the smallest Euclidean clearance between any link point
        and the nearest obstacle surface.
        """
        d_min = np.inf
        for link_pts in link_positions(q):
            for pt in link_pts:
                dists = np.linalg.norm(self.centres - pt, axis=1) - self.radii
                d_min = min(d_min, float(np.min(dists)))
        return d_min

    def closest_obstacle_info(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """
        For a single 3D point, return (distance_to_surface, direction_away).
        """
        diffs = point - self.centres                         # (N, 3)
        dists = np.linalg.norm(diffs, axis=1)                # (N,)
        surface_dists = dists - self.radii
        idx = int(np.argmin(surface_dists))

        direction = diffs[idx] / (dists[idx] + 1e-12)       # unit vector away
        return float(surface_dists[idx]), direction

    def local_obstacle_density(self, q: np.ndarray, radius: float = 0.3) -> float:
        """
        Count how many obstacle centres are within *radius* of the
        end-effector.  Used by the adaptive step-size controller.
        """
        from robot_model import end_effector_position
        ee = end_effector_position(q)
        dists = np.linalg.norm(self.centres - ee, axis=1)
        return float(np.sum(dists < radius))
