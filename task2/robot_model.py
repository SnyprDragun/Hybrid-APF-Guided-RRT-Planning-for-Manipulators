"""
robot_model.py — Forward kinematics and link geometry for a 6-DOF manipulator.

Uses the Denavit-Hartenberg convention to compute:
  • End-effector pose  (4×4 homogeneous transform)
  • Intermediate link positions  (for swept-volume collision checking)
  • Analytical Jacobian        (for potential-field gradient mapping)
"""

import numpy as np
from config import DH_PARAMS, NUM_JOINTS, LINK_COLLISION_SAMPLES


def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Build the 4×4 DH transformation matrix for one joint."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0],
    ])


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """
    Compute the end-effector 4×4 pose given joint angles *q* (length 6).

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        Homogeneous transform from base to end-effector.
    """
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        a, alpha, d, offset = DH_PARAMS[i]
        T = T @ dh_matrix(a, alpha, d, q[i] + offset)
    return T


def link_positions(q: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of 3D positions along every link of the kinematic chain.

    For each of the 6 links the segment between two consecutive joint frames
    is linearly interpolated with ``LINK_COLLISION_SAMPLES`` points.  This
    gives a swept-volume approximation suitable for sphere-based collision
    checks.

    Returns
    -------
    points : list[np.ndarray]
        Each element is shape (LINK_COLLISION_SAMPLES, 3).
    """
    frames = []           # origin of each joint frame
    T = np.eye(4)
    frames.append(T[:3, 3].copy())

    for i in range(NUM_JOINTS):
        a, alpha, d, offset = DH_PARAMS[i]
        T = T @ dh_matrix(a, alpha, d, q[i] + offset)
        frames.append(T[:3, 3].copy())

    points = []
    for i in range(NUM_JOINTS):
        seg = np.linspace(frames[i], frames[i + 1], LINK_COLLISION_SAMPLES)
        points.append(seg)

    return points


def end_effector_position(q: np.ndarray) -> np.ndarray:
    """Shortcut: return only the (x, y, z) position of the end-effector."""
    return forward_kinematics(q)[:3, 3]


def numerical_jacobian(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute the 3×N positional Jacobian via central finite differences.

    Parameters
    ----------
    q : np.ndarray, shape (NUM_JOINTS,)
    eps : float
        Perturbation step.

    Returns
    -------
    J : np.ndarray, shape (3, NUM_JOINTS)
    """
    J = np.zeros((3, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        q_plus  = q.copy();  q_plus[i]  += eps
        q_minus = q.copy();  q_minus[i] -= eps
        J[:, i] = (end_effector_position(q_plus) - end_effector_position(q_minus)) / (2.0 * eps)
    return J
