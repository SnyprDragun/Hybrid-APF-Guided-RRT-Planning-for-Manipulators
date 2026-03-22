"""
config.py — Central configuration for the Hybrid APF-RRT Motion Planner.

All tuneable parameters live here so experiments are reproducible
and swapping environments is trivial.

Robot: Franka Emika Panda (7-DOF) — parameters extracted directly
from the provided panda.urdf / panda_with_gripper.urdf.
"""

import numpy as np
import os

# ─────────────────────────────────────────────
# Robot (Franka Panda 7-DOF)
# ─────────────────────────────────────────────
NUM_JOINTS = 7

# URDF paths (resolved relative to this file)
_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PANDA          = os.path.join(_HERE, "panda.urdf")
URDF_PANDA_GRIPPER  = os.path.join(_HERE, "panda_with_gripper.urdf")

# Joint origins (xyz) and orientations (rpy) from URDF
# Each row: [x, y, z, roll, pitch, yaw]
JOINT_ORIGINS = np.array([
    [0.0,     0.0,    0.333,  0.0,       0.0, 0.0],    # joint1
    [0.0,     0.0,    0.0,   -np.pi/2,   0.0, 0.0],    # joint2
    [0.0,    -0.316,  0.0,    np.pi/2,   0.0, 0.0],    # joint3
    [0.0825,  0.0,    0.0,    np.pi/2,   0.0, 0.0],    # joint4
    [-0.0825, 0.384,  0.0,   -np.pi/2,   0.0, 0.0],    # joint5
    [0.0,     0.0,    0.0,    np.pi/2,   0.0, 0.0],    # joint6
    [0.088,   0.0,    0.0,    np.pi/2,   0.0, 0.0],    # joint7
])

# Fixed transform from joint7 to flange (panda_joint8 / panda_hand_joint)
FLANGE_OFFSET = np.array([0.0, 0.0, 0.107])

# All joints rotate about local Z
JOINT_AXES = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])

# Actual joint limits from URDF <limit> tags (radians)
JOINT_LIMITS_LOW = np.array([
    -2.8973,   # joint1
    -1.7628,   # joint2
    -2.8973,   # joint3
    -3.0718,   # joint4
    -2.8973,   # joint5
    -0.0175,   # joint6
    -2.8973,   # joint7
])
JOINT_LIMITS_HIGH = np.array([
     2.8973,   # joint1
     1.7628,   # joint2
     2.8973,   # joint3
    -0.0698,   # joint4
     2.8973,   # joint5
     3.7520,   # joint6
     2.8973,   # joint7
])

# Number of collision-check sample points per link
LINK_COLLISION_SAMPLES = 5

# ─────────────────────────────────────────────
# Workspace / Obstacles
# ─────────────────────────────────────────────
# 8 spherical obstacles placed in the Panda's reachable workspace
# to create narrow passages between start and goal EE positions.
OBSTACLES = [
    (np.array([ 0.30,  0.15, 0.45]), 0.06),
    # (np.array([ 0.40, -0.10, 0.35]), 0.06),
    # (np.array([ 0.20,  0.30, 0.30]), 0.05),
    # (np.array([ 0.50,  0.05, 0.50]), 0.05),
    # (np.array([ 0.35,  0.25, 0.55]), 0.05),
    # (np.array([ 0.15, -0.15, 0.40]), 0.04),
    # (np.array([ 0.45,  0.20, 0.40]), 0.04),
    # (np.array([ 0.25,  0.00, 0.55]), 0.04),
]

# Safety clearance added to obstacle radii for swept-volume checks
COLLISION_CLEARANCE = 0.02  # metres

# ─────────────────────────────────────────────
# APF Parameters
# ─────────────────────────────────────────────
K_ATT  = 2.0       # attractive gain
K_REP  = 0.3       # repulsive gain
RHO_0  = 0.15      # repulsive influence radius (metres)

# ─────────────────────────────────────────────
# RRT Parameters
# ─────────────────────────────────────────────
MAX_ITERATIONS   = 8000
STEP_SIZE        = 0.25     # radians — max expansion per step
GOAL_BIAS        = 0.15     # probability of sampling goal directly
GOAL_THRESHOLD   = 0.50     # joint-space L2 distance to accept goal reached

# ─────────────────────────────────────────────
# Enhanced Planner — Adaptive + Smoothing
# ─────────────────────────────────────────────
ADAPTIVE_STEP_MIN  = 0.08
ADAPTIVE_STEP_MAX  = 0.35
SMOOTHING_ITERS    = 200     # shortcut iterations for path smoothing

# ─────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────
BENCHMARK_RUNS = 20
TIMEOUT_SEC    = 60.0       # per-run timeout

# ─────────────────────────────────────────────
# Default start / goal configurations (radians)
# Chosen within joint limits; EE positions verified collision-free.
# ─────────────────────────────────────────────
Q_START = np.array([ 0.000000, 0.610865, 0.000000, -2.007130, 0.000000, 2.617990, 0.785411])
Q_GOAL  = np.array([-0.785411, 0.610865, 0.000000, -2.007130, 0.000000, 2.617990, 0.000000])