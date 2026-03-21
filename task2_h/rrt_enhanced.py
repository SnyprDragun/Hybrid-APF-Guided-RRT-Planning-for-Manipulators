"""
rrt_enhanced.py — Enhanced Hybrid APF-RRT with two improvements:

  1. **Adaptive Parameter Tuning**
     Step size and APF weights (K_att, K_rep) are dynamically adjusted
     based on local obstacle density and progress toward the goal.

     • High obstacle density  → smaller step, higher K_rep
     • Clear space / stalled  → larger step, higher K_att
     This escapes local minima where the standard APF repulsion balances
     the attraction and the tree stagnates.

  2. **Optimization-Based Path Smoothing**
     After a raw path is found, a shortcutting + gradient-descent smoother
     iteratively removes unnecessary waypoints and nudges remaining ones
     toward shorter, collision-free trajectories.
"""

from __future__ import annotations
import time
import numpy as np

from config import (
    NUM_JOINTS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
    MAX_ITERATIONS, GOAL_BIAS, GOAL_THRESHOLD,
    ADAPTIVE_STEP_MIN, ADAPTIVE_STEP_MAX,
    SMOOTHING_ITERS, Q_START, Q_GOAL,
    K_ATT, K_REP,
)
from robot_model import end_effector_position
from obstacles import ObstacleSet
from apf import joint_space_gradient
from rrt_base import TreeNode, PlanResult, APFRRTPlanner


class EnhancedAPFRRTPlanner(APFRRTPlanner):
    """
    Enhanced planner inheriting from the baseline, adding:
      • density-aware adaptive step size and APF weight tuning,
      • shortcut + gradient path smoothing.
    """

    def __init__(self, obs: ObstacleSet, apf_weight: float = 0.5):
        super().__init__(obs, apf_weight)
        self._stall_counter = 0
        self._best_dist = np.inf

    # ─── Adaptive step size ──────────────────

    def _adaptive_step(self, q: np.ndarray) -> float:
        """
        Compute step size inversely proportional to local obstacle density.
        Denser region → smaller, safer steps.
        """
        density = self.obs.local_obstacle_density(q, radius=0.3)
        # Map density [0, max_obs] → step [MAX, MIN]
        max_obs = len(self.obs.centres)
        ratio = density / max(max_obs, 1)
        step = ADAPTIVE_STEP_MAX - ratio * (ADAPTIVE_STEP_MAX - ADAPTIVE_STEP_MIN)
        return np.clip(step, ADAPTIVE_STEP_MIN, ADAPTIVE_STEP_MAX)

    # ─── Adaptive APF weights ────────────────

    def _adaptive_apf_weights(self, q: np.ndarray, q_goal: np.ndarray):
        """
        Adjust K_att and K_rep based on progress and density.

        If stalled (no progress for many iterations), temporarily boost
        K_att and suppress K_rep to break out of local minima.
        """
        dist_to_goal = np.linalg.norm(
            end_effector_position(q) - end_effector_position(q_goal)
        )

        if dist_to_goal < self._best_dist - 0.005:
            self._best_dist = dist_to_goal
            self._stall_counter = 0
        else:
            self._stall_counter += 1

        # Stall escape: after 80 stalled iterations, bias toward attraction
        if self._stall_counter > 80:
            k_att = K_ATT * 3.0
            k_rep = K_REP * 0.3
            self._stall_counter = 0      # reset
        else:
            density = self.obs.local_obstacle_density(q, radius=0.3)
            k_att = K_ATT
            k_rep = K_REP * (1.0 + 0.5 * density)

        return k_att, k_rep

    # ─── Overridden steer with adaptive params ─

    def _steer_adaptive(
        self, q_near: np.ndarray, q_sample: np.ndarray, q_goal: np.ndarray
    ) -> np.ndarray:
        """
        Steer with adaptive step size and dynamic APF weights.
        """
        step = self._adaptive_step(q_near)
        k_att, k_rep = self._adaptive_apf_weights(q_near, q_goal)

        # Random direction
        d_rand = q_sample - q_near
        norm_rand = np.linalg.norm(d_rand)
        if norm_rand < 1e-8:
            d_rand = np.random.randn(NUM_JOINTS)
            norm_rand = np.linalg.norm(d_rand)
        d_rand /= norm_rand

        # APF gradient with tuned weights
        d_apf = joint_space_gradient(q_near, self.goal_pos, self.obs, k_att, k_rep)
        if np.linalg.norm(d_apf) < 1e-8:
            d_apf = d_rand

        # Blend
        w = self.apf_weight
        d_blend = (1.0 - w) * d_rand + w * d_apf
        norm = np.linalg.norm(d_blend)
        if norm < 1e-8:
            d_blend = d_rand
            norm = 1.0
        d_blend /= norm

        q_new = q_near + step * d_blend
        return np.clip(q_new, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

    # ─── Path smoothing ─────────────────────

    def smooth_path(
        self, path: list[np.ndarray], max_iters: int = SMOOTHING_ITERS
    ) -> list[np.ndarray]:
        """
        Two-stage path smoothing:
          1. **Shortcutting** — randomly pick two non-adjacent waypoints;
             if the straight line between them is collision-free, remove
             all intermediate waypoints.
          2. **Gradient nudge** — for each remaining waypoint, nudge it
             toward the midpoint of its neighbours if that reduces length
             and stays collision-free.
        """
        path = [q.copy() for q in path]

        # Stage 1: Shortcutting
        for _ in range(max_iters):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if not self.obs.segment_collides(path[i], path[j], n_checks=10):
                path = path[: i + 1] + path[j:]

        # Stage 2: Gradient nudge
        for _ in range(max_iters // 2):
            improved = False
            for k in range(1, len(path) - 1):
                mid = 0.5 * (path[k - 1] + path[k + 1])
                nudged = 0.7 * path[k] + 0.3 * mid    # blend toward midpoint
                nudged = np.clip(nudged, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

                if self.obs.config_collides(nudged):
                    continue
                if self.obs.segment_collides(path[k - 1], nudged, 5):
                    continue
                if self.obs.segment_collides(nudged, path[k + 1], 5):
                    continue

                old_len = (np.linalg.norm(path[k] - path[k - 1])
                           + np.linalg.norm(path[k + 1] - path[k]))
                new_len = (np.linalg.norm(nudged - path[k - 1])
                           + np.linalg.norm(path[k + 1] - nudged))

                if new_len < old_len:
                    path[k] = nudged
                    improved = True

            if not improved:
                break

        return path

    # ─── Main planning loop (overridden) ─────

    def plan(
        self,
        q_start: np.ndarray = Q_START,
        q_goal: np.ndarray = Q_GOAL,
        max_iter: int = MAX_ITERATIONS,
        timeout: float = 30.0,
    ) -> PlanResult:
        """
        Enhanced planner with adaptive tuning + post-hoc smoothing.
        """
        self.goal_pos = end_effector_position(q_goal)
        self._stall_counter = 0
        self._best_dist = np.inf

        tree = [TreeNode(q=q_start.copy())]
        t0 = time.time()

        for it in range(1, max_iter + 1):
            if time.time() - t0 > timeout:
                break

            q_sample = self._sample(q_goal)
            near_idx = self._nearest(tree, q_sample)
            q_near   = tree[near_idx].q

            q_new = self._steer_adaptive(q_near, q_sample, q_goal)

            if self.obs.config_collides(q_new):
                continue
            if self.obs.segment_collides(q_near, q_new, n_checks=5):
                continue

            new_idx = len(tree)
            tree.append(TreeNode(q=q_new, parent=near_idx))

            if np.linalg.norm(q_new - q_goal) < GOAL_THRESHOLD:
                raw_path = self._extract_path(tree, new_idx)
                smooth_path = self.smooth_path(raw_path)
                elapsed = time.time() - t0
                return PlanResult(
                    success=True,
                    path=smooth_path,
                    tree_nodes=tree,
                    iterations=it,
                    time_sec=elapsed,
                    path_length=self._path_length(smooth_path),
                    node_count=len(tree),
                )

        elapsed = time.time() - t0
        return PlanResult(
            success=False,
            tree_nodes=tree,
            iterations=max_iter,
            time_sec=elapsed,
            node_count=len(tree),
        )
