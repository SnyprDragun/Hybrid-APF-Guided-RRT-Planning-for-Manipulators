"""
rrt_base.py — Baseline Hybrid APF-Guided RRT Planner for 7-DOF Panda.

Algorithm
---------
1. Initialise tree with q_start.
2. At each iteration:
   a) With probability GOAL_BIAS, sample q_goal; else uniform random.
   b) Find nearest node in tree (L2 in 7D joint-space).
   c) Compute APF joint-space gradient at the nearest node.
   d) Blend random direction with APF gradient (weighted sum).
   e) Extend tree by STEP_SIZE in the blended direction.
   f) Collision-check the new segment (obstacle + self-collision).
   g) If new node is within GOAL_THRESHOLD of q_goal, terminate.
3. Extract path by backtracking parent pointers.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field

from config import (
    NUM_JOINTS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
    MAX_ITERATIONS, STEP_SIZE, GOAL_BIAS, GOAL_THRESHOLD,
    Q_START, Q_GOAL,
)
from robot_model import end_effector_position
from obstacles import ObstacleSet
from apf import joint_space_gradient


@dataclass
class TreeNode:
    """A single node in the RRT tree."""
    q: np.ndarray
    parent: int | None = None


@dataclass
class PlanResult:
    """Container returned from a planning attempt."""
    success: bool
    path: list[np.ndarray]      = field(default_factory=list)
    tree_nodes: list[TreeNode]  = field(default_factory=list)
    iterations: int             = 0
    time_sec: float             = 0.0
    path_length: float          = 0.0
    node_count: int             = 0


class APFRRTPlanner:
    """
    Baseline Hybrid APF-RRT for the Franka Panda (7-DOF).

    Parameters
    ----------
    obs : ObstacleSet
    apf_weight : float
        Blending weight in [0, 1].  0 = pure RRT, 1 = pure APF greedy.
    """

    def __init__(self, obs: ObstacleSet, apf_weight: float = 0.5):
        self.obs = obs
        self.apf_weight = apf_weight
        self.goal_pos: np.ndarray | None = None

    def _random_config(self) -> np.ndarray:
        return np.random.uniform(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

    def _sample(self, q_goal: np.ndarray) -> np.ndarray:
        if np.random.rand() < GOAL_BIAS:
            return q_goal.copy()
        return self._random_config()

    @staticmethod
    def _nearest(tree: list[TreeNode], q: np.ndarray) -> int:
        dists = [np.linalg.norm(n.q - q) for n in tree]
        return int(np.argmin(dists))

    def _steer(
        self, q_near: np.ndarray, q_sample: np.ndarray, step: float = STEP_SIZE
    ) -> np.ndarray:
        d_rand = q_sample - q_near
        norm_rand = np.linalg.norm(d_rand)
        if norm_rand < 1e-8:
            d_rand = np.random.randn(NUM_JOINTS)
            norm_rand = np.linalg.norm(d_rand)
        d_rand /= norm_rand

        d_apf = joint_space_gradient(q_near, self.goal_pos, self.obs)
        if np.linalg.norm(d_apf) < 1e-8:
            d_apf = d_rand

        d_blend = (1.0 - self.apf_weight) * d_rand + self.apf_weight * d_apf
        norm_blend = np.linalg.norm(d_blend)
        if norm_blend < 1e-8:
            d_blend = d_rand
            norm_blend = 1.0
        d_blend /= norm_blend

        q_new = q_near + step * d_blend
        return np.clip(q_new, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

    @staticmethod
    def _extract_path(tree: list[TreeNode], goal_idx: int) -> list[np.ndarray]:
        path = []
        idx: int | None = goal_idx
        while idx is not None:
            path.append(tree[idx].q)
            idx = tree[idx].parent
        path.reverse()
        return path

    @staticmethod
    def _path_length(path: list[np.ndarray]) -> float:
        return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))

    def plan(
        self,
        q_start: np.ndarray = Q_START,
        q_goal: np.ndarray  = Q_GOAL,
        max_iter: int       = MAX_ITERATIONS,
        timeout: float      = 30.0,
    ) -> PlanResult:
        self.goal_pos = end_effector_position(q_goal)
        tree = [TreeNode(q=q_start.copy())]
        t0 = time.time()

        for it in range(1, max_iter + 1):
            if time.time() - t0 > timeout:
                break

            q_sample = self._sample(q_goal)
            near_idx = self._nearest(tree, q_sample)
            q_near   = tree[near_idx].q
            q_new    = self._steer(q_near, q_sample)

            if self.obs.config_collides(q_new):
                continue
            if self.obs.segment_collides(q_near, q_new, n_checks=5):
                continue

            new_idx = len(tree)
            tree.append(TreeNode(q=q_new, parent=near_idx))

            if np.linalg.norm(q_new - q_goal) < GOAL_THRESHOLD:
                path = self._extract_path(tree, new_idx)
                # Append the actual goal if the last-mile segment is clear
                if not self.obs.segment_collides(path[-1], q_goal, n_checks=8):
                    path.append(q_goal.copy())
                elapsed = time.time() - t0
                return PlanResult(
                    success=True, path=path, tree_nodes=tree,
                    iterations=it, time_sec=elapsed,
                    path_length=self._path_length(path),
                    node_count=len(tree),
                )

        return PlanResult(
            success=False, tree_nodes=tree,
            iterations=max_iter, time_sec=time.time() - t0,
            node_count=len(tree),
        )