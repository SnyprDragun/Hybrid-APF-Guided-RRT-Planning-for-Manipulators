"""
apf.py — Artificial Potential Field for task-space guidance.

Computes attractive and repulsive forces in task-space (3D workspace)
and projects them back into joint-space via the Jacobian transpose.

Potential functions
-------------------
Attractive (quadratic):
    U_att(x) = 0.5 · K_att · ‖x − x_goal‖²
    F_att(x) = −K_att · (x − x_goal)

Repulsive (inverse-distance, clamped by ρ₀):
    U_rep(x) = 0.5 · K_rep · (1/ρ − 1/ρ₀)²   if ρ < ρ₀
    F_rep(x) = K_rep · (1/ρ − 1/ρ₀) · (1/ρ²) · ∇ρ

Joint-space gradient:
    τ = Jᵀ · (F_att + Σ F_rep_i)
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
    """
    Sum of repulsive forces on a single workspace point from all obstacles.
    """
    f_rep = np.zeros(3)
    for i in range(len(obs.centres)):
        diff = point - obs.centres[i]
        dist = np.linalg.norm(diff)
        rho  = dist - obs.radii[i]            # clearance to surface
        rho  = max(rho, 1e-4)                 # avoid division by zero

        if rho < rho_0:
            grad_rho = diff / (dist + 1e-12)  # ∇ρ  (unit direction away)
            magnitude = k_rep * (1.0 / rho - 1.0 / rho_0) * (1.0 / rho ** 2)
            f_rep += magnitude * grad_rho

    return f_rep


def total_task_force(
    q: np.ndarray,
    goal_pos: np.ndarray,
    obs: ObstacleSet,
    k_att: float = K_ATT,
    k_rep: float = K_REP,
) -> np.ndarray:
    """
    Net task-space force (attractive + repulsive) at the end-effector.
    """
    ee = end_effector_position(q)
    f_att = attractive_force(ee, goal_pos, k_att)

    # Sum repulsive from *all* link points, not just EE
    f_rep = np.zeros(3)
    for link_pts in link_positions(q):
        for pt in link_pts:
            f_rep += repulsive_force(pt, obs, k_rep)

    return f_att + f_rep


def joint_space_gradient(
    q: np.ndarray,
    goal_pos: np.ndarray,
    obs: ObstacleSet,
    k_att: float = K_ATT,
    k_rep: float = K_REP,
) -> np.ndarray:
    """
    Project the task-space potential-field force into joint-space:
        τ = Jᵀ · F_total

    Returns a unit direction in joint-space.
    """
    F = total_task_force(q, goal_pos, obs, k_att, k_rep)
    J = numerical_jacobian(q)
    tau = J.T @ F                             # (NUM_JOINTS,)

    norm = np.linalg.norm(tau)
    if norm < 1e-8:
        return np.zeros(NUM_JOINTS)
    return tau / norm
