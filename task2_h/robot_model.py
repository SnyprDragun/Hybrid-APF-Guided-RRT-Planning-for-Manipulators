"""
robot_model.py — Forward kinematics for the Franka Emika Panda (7-DOF).

Computes transforms directly from the URDF joint origins and axes
(stored in config.py), avoiding DH convention ambiguities.

Each joint i contributes:
    T_i = T_origin_i  ·  Rot(axis_i, q_i)

where T_origin_i is the fixed transform from the parent frame to the
child frame at q=0 (the <origin rpy="..." xyz="..."/> tag), and
Rot(axis, q) is the rotation about the joint axis.

The flange pose appends the fixed panda_joint8 offset at the end.
"""

import numpy as np
from config import (
    NUM_JOINTS, JOINT_ORIGINS, JOINT_AXES,
    FLANGE_OFFSET, LINK_COLLISION_SAMPLES,
)


# ─────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────

def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: Rot = Rz(yaw) · Ry(pitch) · Rx(roll)."""
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)

def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for rotation of *angle* about unit *axis*."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    if np.allclose(ax, [0, 0, 1]):
        return _rot_z(angle)
    elif np.allclose(ax, [0, 1, 0]):
        return _rot_y(angle)
    elif np.allclose(ax, [1, 0, 0]):
        return _rot_x(angle)
    # Rodrigues for arbitrary axis
    K = np.array([
        [0, -ax[2], ax[1]],
        [ax[2], 0, -ax[0]],
        [-ax[1], ax[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# ─────────────────────────────────────────────
# Joint origin transform (fixed part from URDF)
# ─────────────────────────────────────────────

def _joint_origin_transform(joint_idx: int) -> np.ndarray:
    """4x4 fixed transform for joint *joint_idx* from its URDF <origin> tag."""
    x, y, z, roll, pitch, yaw = JOINT_ORIGINS[joint_idx]
    R = _rpy_to_matrix(roll, pitch, yaw)
    return _make_transform(R, np.array([x, y, z]))


# ─────────────────────────────────────────────
# Forward kinematics
# ─────────────────────────────────────────────

def forward_kinematics(q: np.ndarray, include_flange: bool = True) -> np.ndarray:
    """
    Compute the end-effector (flange) 4x4 pose given joint angles *q* (len 7).

    Parameters
    ----------
    q : np.ndarray, shape (7,)
        Joint angles in radians.
    include_flange : bool
        If True, append the fixed flange offset (panda_joint8).

    Returns
    -------
    T : np.ndarray, shape (4, 4)
    """
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        T = T @ _joint_origin_transform(i)
        R_joint = _axis_angle_matrix(JOINT_AXES[i], q[i])
        T[:3, :3] = T[:3, :3] @ R_joint

    if include_flange:
        T_flange = _make_transform(np.eye(3), FLANGE_OFFSET)
        T = T @ T_flange

    return T


def joint_frames(q: np.ndarray) -> list[np.ndarray]:
    """
    Return the 4x4 pose of every joint frame (0..7) plus flange.

    Returns a list of length NUM_JOINTS + 2:
      [base, after_joint1, ..., after_joint7, flange]
    """
    frames = [np.eye(4)]      # base
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        T = T @ _joint_origin_transform(i)
        R_joint = _axis_angle_matrix(JOINT_AXES[i], q[i])
        T[:3, :3] = T[:3, :3] @ R_joint
        frames.append(T.copy())

    # Flange
    T_flange = T @ _make_transform(np.eye(3), FLANGE_OFFSET)
    frames.append(T_flange)
    return frames


def link_positions(q: np.ndarray) -> list[np.ndarray]:
    """
    Return interpolated 3D points along every link for swept-volume
    collision checking.

    Returns
    -------
    points : list of np.ndarray, each shape (LINK_COLLISION_SAMPLES, 3)
        One entry per link segment (NUM_JOINTS + 1 segments: base->j1,
        j1->j2, ..., j7->flange).
    """
    frames = joint_frames(q)
    points = []
    for i in range(len(frames) - 1):
        p0 = frames[i][:3, 3]
        p1 = frames[i + 1][:3, 3]
        seg = np.linspace(p0, p1, LINK_COLLISION_SAMPLES)
        points.append(seg)
    return points


def end_effector_position(q: np.ndarray) -> np.ndarray:
    """Return only the (x, y, z) position of the flange."""
    return forward_kinematics(q)[:3, 3]


def numerical_jacobian(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """3x7 positional Jacobian via central finite differences."""
    J = np.zeros((3, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        q_plus  = q.copy();  q_plus[i]  += eps
        q_minus = q.copy();  q_minus[i] -= eps
        J[:, i] = (end_effector_position(q_plus) - end_effector_position(q_minus)) / (2.0 * eps)
    return J
