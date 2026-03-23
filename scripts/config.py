"""
config.py — Central configuration for the Hybrid APF-RRT Motion Planner.

Robot: Franka Emika Panda (7-DOF) — parameters extracted directly
from the provided panda.urdf / panda_with_gripper.urdf.

All tuneable parameters live here so experiments are reproducible
and swapping environments is trivial.
"""

import numpy as np
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PANDA         = os.path.join(_HERE, "panda.urdf")
URDF_PANDA_GRIPPER = os.path.join(_HERE, "panda_with_gripper.urdf")
MESHES_DIR         = os.path.join(_HERE, "meshes")

# ─────────────────────────────────────────────
# Robot — Franka Panda 7-DOF
# ─────────────────────────────────────────────
NUM_JOINTS = 7

# Joint origins (xyz) and orientations (rpy) from URDF <joint> tags.
# Each row: [x, y, z, roll, pitch, yaw]
JOINT_ORIGINS = np.array([
    [0.0,      0.0,    0.333,  0.0,       0.0, 0.0],   # panda_joint1
    [0.0,      0.0,    0.0,   -np.pi/2,   0.0, 0.0],   # panda_joint2
    [0.0,     -0.316,  0.0,    np.pi/2,   0.0, 0.0],   # panda_joint3
    [0.0825,   0.0,    0.0,    np.pi/2,   0.0, 0.0],   # panda_joint4
    [-0.0825,  0.384,  0.0,   -np.pi/2,   0.0, 0.0],   # panda_joint5
    [0.0,      0.0,    0.0,    np.pi/2,   0.0, 0.0],   # panda_joint6
    [0.088,    0.0,    0.0,    np.pi/2,   0.0, 0.0],   # panda_joint7
])

# Fixed transform: joint7 → flange (panda_joint8)
FLANGE_OFFSET = np.array([0.0, 0.0, 0.107])

# All Panda joints rotate about local Z
JOINT_AXES = np.tile(np.array([0, 0, 1]), (NUM_JOINTS, 1)).astype(float)

# Joint limits from URDF <limit> tags (radians)
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

# Velocity limits from URDF (rad/s)
JOINT_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

# Approximate link radii for self-collision checking (metres).
# Indexed 0..7 for links 0 through 7 (flange).
LINK_RADII = np.array([0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.04])

# Minimum link index separation to check for self-collision
# (adjacent links can't self-collide due to joint geometry).
SELF_COLLISION_SKIP = 3

# Collision-check sample points per link segment
LINK_COLLISION_SAMPLES = 5

# ─────────────────────────────────────────────
# Workspace / Obstacles
# ─────────────────────────────────────────────
# 8 spherical obstacles creating narrow passages in the Panda's workspace
# between start and goal EE positions.
OBSTACLES = [
    # (np.array([ 0.30,  0.15, 0.45]), 0.06),
    # (np.array([ 0.40, -0.10, 0.35]), 0.06),
    # (np.array([ 0.20,  0.30, 0.30]), 0.05),
    (np.array([ 0.50,  0.05, 0.50]), 0.05),
    # (np.array([ 0.35,  0.25, 0.55]), 0.05),
    # (np.array([ 0.15, -0.15, 0.40]), 0.04),
    # (np.array([ 0.45,  0.20, 0.40]), 0.04),
    # (np.array([ 0.25,  0.10, 0.55]), 0.04),
]

# Safety clearance added to obstacle radii for swept-volume checks
COLLISION_CLEARANCE = 0.02  # metres

# ─────────────────────────────────────────────
# APF Parameters
# ─────────────────────────────────────────────
K_ATT = 2.0        # attractive gain
K_REP = 0.3        # repulsive gain
RHO_0 = 0.15       # repulsive influence radius (metres)

# ─────────────────────────────────────────────
# RRT Parameters
# ─────────────────────────────────────────────
MAX_ITERATIONS = 8000
STEP_SIZE      = 0.25       # radians — max expansion per step
GOAL_BIAS      = 0.15       # probability of sampling goal directly
GOAL_THRESHOLD = 0.50       # joint-space L2 distance to accept goal

# ─────────────────────────────────────────────
# Enhanced Planner — Adaptive + Smoothing
# ─────────────────────────────────────────────
ADAPTIVE_STEP_MIN = 0.08
ADAPTIVE_STEP_MAX = 0.35
SMOOTHING_ITERS   = 200

# ─────────────────────────────────────────────
# Trajectory Generation
# ─────────────────────────────────────────────
TRAJECTORY_DT       = 1.0 / 240.0   # simulation timestep (s)
TRAJECTORY_DURATION  = 12.0          # total motion time (s)
MAX_JOINT_VEL       = 1.5           # conservative velocity limit (rad/s)
MAX_JOINT_ACC       = 3.0           # conservative accel limit (rad/s^2)

# ─────────────────────────────────────────────
# PD Controller Gains (joint-space)
# ─────────────────────────────────────────────
# Tuned for PyBullet's Featherstone dynamics at 240Hz.
# Target: ~5 Hz bandwidth, damping ratio ~0.7-1.0 (near critical).
#
# Rule of thumb: Kp ~ omega_n^2 * I_reflected,  Kd ~ 2*zeta*omega_n*I_reflected
# For Panda in PyBullet, effective reflected inertias are ~0.5-2.0 kg·m².
# Lower gains = smoother but slower tracking; higher = stiffer but jittery.
KP_GAINS = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 30.0, 10.0])
KD_GAINS = np.array([ 8.0,  8.0,  8.0,  8.0, 4.0,  3.2,  1.6])

# ─────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────
BENCHMARK_RUNS = 20
TIMEOUT_SEC    = 60.0

# ─────────────────────────────────────────────
# Default start / goal (radians)
# ─────────────────────────────────────────────
Q_START = np.array([ 0.000000,-0.785411, 0.000000, -2.356229, 0.000000, 1.570824, 0.785411])
Q_GOAL  = np.array([-0.785411, 0.610865, 0.000000, -2.007130, 0.000000, 2.617990, 0.000000])