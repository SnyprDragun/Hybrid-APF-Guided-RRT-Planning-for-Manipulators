"""
apf.py — Artificial Potential Field for task-space guidance of a 7-DOF arm.

Computes attractive and repulsive forces in task-space (3D workspace)
and projects them into joint-space via the Jacobian transpose.

Potential functions
-------------------
Attractive (quadratic):
    U_att(x) = 0.5 · K_att · ‖x - x_goal‖²
    F_att(x) = -K_att · (x - x_goal)

Repulsive (inverse-distance, clamped by rho_0):
    U_rep(x) = 0.5 · K_rep · (1/rho - 1/rho_0)²   if rho < rho_0
    F_rep(x) = K_rep · (1/rho - 1/rho_0) · (1/rho²) · grad_rho

Joint-space gradient (Jacobian transpose projection):
    tau = J^T · (F_att + sum(F_rep_i))

Repulsive forces are computed on EVERY link sample point, not just
the end-effector — so the planner steers away from configurations
where the elbow or wrist are near obstacles, not just the flange.
"""

import numpy as np
from config import K_ATT, K_REP, RHO_0, NUM_JOINTS
from robot_model import end_effector_position, numerical_jacobian, link_positions
from obstacles import ObstacleSet


def attractive_force(
    ee_pos: np.ndarray, goal_pos: np.ndarray, k_att: float = K_ATT
) -> np.ndarray:
    """Task-space attractive force pointing toward the goal."""
    return -k_att * (ee_pos - goal_pos)


def repulsive_force(
    point: np.ndarray, obs: ObstacleSet, k_rep: float = K_REP, rho_0: float = RHO_0
) -> np.ndarray:
    """Sum of repulsive forces on a single workspace point from all obstacles."""
    f_rep = np.zeros(3)
    for i in range(len(obs.centres)):
        diff = point - obs.centres[i]
        dist = np.linalg.norm(diff)
        rho  = max(dist - obs.radii[i], 1e-4)

        if rho < rho_0:
            grad_rho = diff / (dist + 1e-12)
            magnitude = k_rep * (1.0 / rho - 1.0 / rho_0) * (1.0 / rho ** 2)
            f_rep += magnitude * grad_rho

    return f_rep


def total_task_force(
    q: np.ndarray, goal_pos: np.ndarray, obs: ObstacleSet,
    k_att: float = K_ATT, k_rep: float = K_REP,
) -> np.ndarray:
    """Net task-space force (attractive + repulsive on all link points)."""
    ee = end_effector_position(q)
    f_att = attractive_force(ee, goal_pos, k_att)

    f_rep = np.zeros(3)
    for link_pts in link_positions(q):
        for pt in link_pts:
            f_rep += repulsive_force(pt, obs, k_rep)

    return f_att + f_rep


def joint_space_gradient(
    q: np.ndarray, goal_pos: np.ndarray, obs: ObstacleSet,
    k_att: float = K_ATT, k_rep: float = K_REP,
) -> np.ndarray:
    """
    Project the task-space potential-field force into joint-space:
        tau = J^T · F_total

    Returns a unit direction in joint-space (length NUM_JOINTS).
    """
    F   = total_task_force(q, goal_pos, obs, k_att, k_rep)
    J   = numerical_jacobian(q)
    tau = J.T @ F

    norm = np.linalg.norm(tau)
    if norm < 1e-8:
        return np.zeros(NUM_JOINTS)
    return tau / norm
