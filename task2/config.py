"""
config.py — Central configuration for the Hybrid APF-RRT Motion Planner.

All tuneable parameters live here so experiments are reproducible
and swapping environments is trivial.
"""

import numpy as np

# ─────────────────────────────────────────────
# Robot (6-DOF revolute chain, UR5-like DH)
# ─────────────────────────────────────────────
NUM_JOINTS = 6

# DH parameters [a, alpha, d, theta_offset] — UR5 convention
# Reference: Universal Robots UR5 datasheet
DH_PARAMS = np.array([
    [0.0,      np.pi / 2,  0.0892,  0.0],
    [-0.425,   0.0,        0.0,     0.0],
    [-0.3922,  0.0,        0.0,     0.0],
    [0.0,      np.pi / 2,  0.1093,  0.0],
    [0.0,     -np.pi / 2,  0.0947,  0.0],
    [0.0,      0.0,        0.0823,  0.0],
])

# Joint limits (radians) — ±π keeps sampling efficient
JOINT_LIMITS_LOW  = np.full(NUM_JOINTS, -np.pi)
JOINT_LIMITS_HIGH = np.full(NUM_JOINTS,  np.pi)

# Number of collision-check sample points per link
LINK_COLLISION_SAMPLES = 5

# ─────────────────────────────────────────────
# Workspace / Obstacles
# ─────────────────────────────────────────────
# Obstacle list: (centre_xyz, radius)
# Placed along the line between start-EE and goal-EE to force narrow passages.
OBSTACLES = [
    (np.array([-0.50, -0.40, 0.35]), 0.05),
    (np.array([-0.35, -0.45, 0.30]), 0.05),
    (np.array([-0.20, -0.30, 0.38]), 0.04),
    (np.array([-0.40, -0.20, 0.42]), 0.04),
    (np.array([-0.55, -0.35, 0.25]), 0.04),
    (np.array([-0.30, -0.15, 0.30]), 0.04),
    (np.array([-0.45, -0.50, 0.40]), 0.04),
    (np.array([-0.25, -0.45, 0.22]), 0.04),
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
STEP_SIZE        = 0.30     # radians — max expansion per step
GOAL_BIAS        = 0.15     # probability of sampling goal directly
GOAL_THRESHOLD   = 0.45     # joint-space L2 distance to accept goal reached

# ─────────────────────────────────────────────
# Enhanced Planner — Adaptive + Smoothing
# ─────────────────────────────────────────────
ADAPTIVE_STEP_MIN  = 0.10
ADAPTIVE_STEP_MAX  = 0.40
SMOOTHING_ITERS    = 200     # shortcut iterations for path smoothing
PSO_PARTICLES      = 20
PSO_ITERATIONS     = 50

# ─────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────
BENCHMARK_RUNS = 20
TIMEOUT_SEC    = 60.0       # per-run timeout

# ─────────────────────────────────────────────
# Default start / goal configurations (radians)
# ─────────────────────────────────────────────
Q_START = np.array([0.0, -0.6,  0.6, -1.0,  0.5, 0.0])
Q_GOAL  = np.array([1.2, -0.9,  0.9, -0.5, -0.3, 0.8])
