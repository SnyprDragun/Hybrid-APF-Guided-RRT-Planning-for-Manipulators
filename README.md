# Hybrid APF-Guided RRT Motion Planner

**Franka Emika Panda (7-DOF) · PyBullet Simulation · PD Torque Control**

A complete motion planning and execution pipeline for a 7-DOF robotic manipulator navigating obstacle-dense environments. Combines the exploratory power of Rapidly-exploring Random Trees (RRT) with the directional guidance of Artificial Potential Fields (APF), then executes the planned path using a joint-space PD torque controller over a smooth time-parameterised trajectory.

---

## Table of contents

1. [Architecture](#architecture)
2. [Robot model](#robot-model)
3. [Algorithm — Phase A: Baseline APF-RRT](#phase-a-baseline-apf-rrt)
4. [Algorithm — Phase B: Enhanced planner](#phase-b-enhanced-planner)
5. [Collision detection](#collision-detection)
6. [Trajectory generation](#trajectory-generation)
7. [PD torque controller](#pd-torque-controller)
8. [Visualisation and simulation](#visualisation-and-simulation)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Benchmark results](#benchmark-results)
12. [File reference](#file-reference)
13. [Configuration](#configuration)

---

## Architecture

```
                       ┌─────────────────────────────────┐
                       │           main.py                │
                       │  CLI: demo / benchmark /         │
                       │        animate / pybullet        │
                       └──────┬────────┬─────────┬───────┘
                              │        │         │
               ┌──────────────┤        │         ├──────────────┐
               ▼              ▼        ▼         ▼              ▼
        ┌────────────┐  ┌──────────┐  ┌───────────────┐  ┌────────────┐
        │ rrt_base.py│  │rrt_enhan.│  │ benchmark.py  │  │visualization│
        │  Baseline  │  │  Enhanced│  │  20-run stats │  │  3D plots   │
        └─────┬──────┘  └────┬─────┘  └───────────────┘  │  telemetry  │
              │              │                            │  animation  │
              └──────┬───────┘                            └─────────────┘
                     │
              ┌──────┴──────┐
              │   apf.py    │
              │ F_att + F_rep│
              │  Jᵀ mapping │
              └──────┬──────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
  ┌──────────────┐        ┌──────────────┐
  │robot_model.py│        │ obstacles.py │
  │ URDF-based FK│        │ sphere coll. │
  │ self-coll.   │        │ + self-coll. │
  └──────┬───────┘        └──────────────┘
         │
  ┌──────┴───────┐
  │  config.py   │
  │  All params  │
  └──────────────┘

         Execution pipeline (PyBullet mode):
         ┌──────────┐    ┌──────────────┐    ┌──────────────┐
  Path → │trajectory│ → │ controller.py│ → │pybullet_sim.py│
         │  .py     │    │ PD + ff      │    │ torque exec. │
         │cubic spln│    │ tau command  │    │ telemetry log│
         └──────────┘    └──────────────┘    └──────────────┘
```

---

## Robot model

Forward kinematics uses the exact joint specifications from `panda.urdf`, avoiding DH convention ambiguities. Each joint transform is computed as:

```
T_joint_i = T_origin_i · Rot_z(q_i)
```

where `T_origin_i` is the fixed `<origin rpy="..." xyz="..."/>` transform from the URDF, and all Panda joints rotate about their local Z axis.

| Joint | Origin xyz (m) | Origin rpy (rad) | Limits (rad) |
|-------|----------------|-------------------|--------------|
| panda_joint1 | (0, 0, 0.333) | (0, 0, 0) | [-2.90, +2.90] |
| panda_joint2 | (0, 0, 0) | (-pi/2, 0, 0) | [-1.76, +1.76] |
| panda_joint3 | (0, -0.316, 0) | (+pi/2, 0, 0) | [-2.90, +2.90] |
| panda_joint4 | (0.0825, 0, 0) | (+pi/2, 0, 0) | [-3.07, -0.07] |
| panda_joint5 | (-0.0825, 0.384, 0) | (-pi/2, 0, 0) | [-2.90, +2.90] |
| panda_joint6 | (0, 0, 0) | (+pi/2, 0, 0) | [-0.02, +3.75] |
| panda_joint7 | (0.088, 0, 0) | (+pi/2, 0, 0) | [-2.90, +2.90] |

Flange offset: (0, 0, 0.107) from joint 7. Zero-configuration end-effector: [0.088, 0, 0.926] m.

The 3x7 positional Jacobian is computed via central finite differences. Self-collision is checked by modelling each link as a sphere (centred at the link midpoint with a conservative radius) and testing pairwise distances between non-adjacent links.

---

## Phase A: Baseline APF-RRT

The baseline planner hybridises RRT with Artificial Potential Fields:

**Sampling.** With probability `GOAL_BIAS` (default 0.15), sample the goal configuration directly; otherwise, sample uniformly within joint limits.

**Nearest neighbour.** L2 distance in 7D joint-space (brute force, sufficient for tree sizes under ~10000 nodes).

**APF-guided steering.** At each nearest node, compute the net task-space force:

```
F_total = F_att(x_ee) + sum_over_all_link_points( F_rep(x_link) )
```

where the attractive force is quadratic (`F_att = -K_att * (x_ee - x_goal)`) and the repulsive force uses the inverse-distance potential clamped at influence radius rho_0:

```
F_rep = K_rep * (1/rho - 1/rho_0) * (1/rho^2) * grad_rho    if rho < rho_0
```

Repulsive forces are computed on **every interpolated link point** (8 segments x 5 points = 40 samples), not just the end-effector. This means the planner avoids configurations where any part of the arm is near an obstacle — the elbow, the wrist, or the flange.

The task-space force is projected to joint-space via the Jacobian transpose: `tau = J^T * F_total`. The resulting joint-space direction is blended with the random sample direction:

```
d = (1 - w) * d_random + w * d_apf
```

and the tree extends by `STEP_SIZE` in the blended direction.

**Collision checking.** Every candidate node is checked for robot-vs-obstacle collision (swept-volume sphere model), robot self-collision (pairwise link spheres), and segment collision (5 intermediate configurations along the edge).

---

## Phase B: Enhanced planner

Two improvements over the baseline, corresponding to Phase B options from the assignment.

### Adaptive parameter tuning

**Step size** scales inversely with local obstacle density: denser regions get smaller, safer steps (range: 0.08 to 0.35 rad).

**APF weight adaptation** uses a stall detector: if no task-space progress toward the goal for 80 consecutive iterations, `K_att` is tripled and `K_rep` reduced by 70% to break out of local minima where attractive and repulsive forces balance.

### Optimization-based path smoothing

After finding a raw path, a two-stage optimizer minimises total joint-space path length subject to collision-free constraints:

**Stage 1 — Stochastic shortcutting.** Randomly sample pairs (i, j) of non-adjacent waypoints; if the straight-line segment is collision-free, remove all intermediate waypoints. This is a stochastic combinatorial optimizer over the set of feasible waypoint subsets.

**Stage 2 — Projected gradient descent.** For each interior waypoint q_k, the gradient of path length with respect to q_k is:

```
dL/dq_k = (q_k - q_{k-1}) / ||q_k - q_{k-1}||
         + (q_k - q_{k+1}) / ||q_k - q_{k+1}||
```

This gradient points away from the midpoint of the neighbours. The optimizer nudges q_k toward the midpoint (learning rate 0.3), rejecting moves that cause collision. This is projected gradient descent with a feasibility constraint, iterated until convergence.

---

## Collision detection

The `ObstacleSet` class performs two types of collision checking:

**Robot vs obstacles.** For each link segment (8 segments from base to flange), 5 points are linearly interpolated between consecutive joint frames. Each point is tested against all obstacle spheres with a safety clearance of 0.02 m added to every obstacle radius.

**Self-collision.** Each link is modelled as a sphere centred at its midpoint. Non-adjacent links (index separation >= 3) are tested pairwise. The radii are conservative estimates of the Panda's link geometry:

```
Link 0: 0.08m   Link 1: 0.08m   Link 2: 0.07m   Link 3: 0.07m
Link 4: 0.06m   Link 5: 0.06m   Link 6: 0.05m   Link 7: 0.04m
```

Both checks are performed for every candidate node AND at 5 intermediate configurations along each tree edge.

---

## Trajectory generation

The planner outputs a list of joint-space waypoints. The `Trajectory` class (`trajectory.py`) converts these into a smooth, time-parameterised trajectory:

**Arc-length parameterisation.** Waypoint times are distributed proportionally to cumulative joint-space distance, so the arm moves at approximately constant speed through all segments.

**Cubic spline interpolation.** A natural cubic spline is fitted per joint with zero-velocity boundary conditions at start and end. This guarantees C2 continuity (continuous position, velocity, and acceleration) and smooth start/stop behaviour.

**Limit checking.** After construction, `trajectory.check_limits()` verifies that peak velocities and accelerations stay within the Panda's rated limits (2.175 rad/s for joints 1-4, 2.61 rad/s for joints 5-7).

---

## PD torque controller

The `PDController` class (`controller.py`) implements a joint-space torque control law:

```
tau = Kp * (q_d - q) + Kd * (qd_d - qd) + qdd_d
```

where `(q_d, qd_d, qdd_d)` are the desired trajectory state and `(q, qd)` are the actual joint state from PyBullet. The feedforward term `qdd_d` comes directly from the cubic spline's second derivative.

**Gains** are tuned for the Panda's inertia profile: higher stiffness on the proximal joints (600 Nm/rad for joints 1-4) that carry more load, lower stiffness on the distal joints (50-250 Nm/rad for joints 5-7). Derivative gains provide critical damping.

**Torque saturation** clamps each joint to +/-87 Nm (the Panda's rated limit for joints 1-4).

In PyBullet, this controller runs at 240 Hz. The default velocity motors are disabled so that `p.TORQUE_CONTROL` commands are the sole input to the dynamics simulation. This means gravity, Coriolis, and inertial effects are all handled by PyBullet's internal Featherstone dynamics — the PD controller compensates for tracking errors against these disturbances.

For a real Panda deployment, this PD law would be augmented with computed torque (inverse dynamics from Pinocchio or libfranka) to get:

```
tau = M(q) * [qdd_d + Kp*e + Kd*ed] + C(q,qd)*qd + g(q)
```

---

## Visualisation and simulation

**Matplotlib (no PyBullet required):**
- 3D plots of the RRT tree (edges coloured by depth), smoothed path, 8 obstacle spheres, and Panda arm skeleton at start/goal configurations.
- Side-by-side baseline vs enhanced comparison with metrics.
- Joint trajectory profiles (position and velocity over time).
- Animated GIF of the robot following the path frame by frame.

**PyBullet (optional):**
- Loads the actual `panda_with_gripper.urdf` with STL meshes. The `package://model_description/meshes/...` references in the URDF are automatically patched to absolute paths pointing to the project's `meshes/` directory.
- Executes the trajectory using the PD torque controller at 240 Hz with real-time rendering (~12 seconds of smooth motion).
- Logs and plots controller telemetry: desired vs actual position per joint, tracking error RMS, and commanded torques.

---

## Installation

```bash
# Core (planning + matplotlib plots)
pip install numpy matplotlib scipy

# Optional: PyBullet simulation with torque control
pip install pybullet

# Optional: GIF generation
pip install Pillow
```

Python 3.10+ required. No ROS, Pinocchio, or other heavy dependencies.

**Mesh files:** Place the `meshes/` directory (containing `visual/` and `collision/` subdirectories with `link0.stl` through `link7.stl`, `hand.stl`, `finger.stl`) in the project root alongside the URDF files. The PyBullet module will automatically find and use them.

---

## Usage

```bash
# Quick demo — plan with both planners, generate plots + trajectory profile
python main.py

# Full 20-run comparative benchmark
python main.py --benchmark

# Animation GIF
python main.py --animate

# PyBullet with PD torque control (12s smooth execution)
python main.py --pybullet
```

All outputs are saved to `./output/`.

---

## Benchmark results

20 randomised runs, 8 spherical obstacles, same start/goal, different random seeds:

```
====================================================================
Metric                          Baseline APF-RRT    Enhanced APF-RRT
====================================================================
Success Rate (%)                          100.0%               95.0%
Comp. Time (s, success)            0.633 +/- 0.275   2.189 +/- 2.797
Path Length (rad, success)         2.188 +/- 0.505   1.278 +/- 0.163
Node Count (success)                     34 +/- 14         77 +/- 147
====================================================================
```

The enhanced planner produces **42% shorter paths** (1.28 vs 2.19 rad) with 3x lower variance (0.16 vs 0.51), at the cost of longer computation time due to the smoothing optimisation pass.

---

## File reference

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 146 | All parameters: Panda joints, obstacles, APF/RRT tuning, trajectory, PD gains |
| `robot_model.py` | 142 | URDF-based FK, link positions, Jacobian, self-collision |
| `obstacles.py` | 95 | Sphere collision checking (obstacle + self), distance queries |
| `apf.py` | 85 | Attractive/repulsive forces, per-link repulsion, Jacobian transpose projection |
| `rrt_base.py` | 130 | Baseline APF-RRT planner |
| `rrt_enhanced.py` | 160 | Enhanced planner: adaptive tuning + optimization-based smoothing |
| `trajectory.py` | 130 | Cubic spline time-parameterisation, arc-length distribution, limit checking |
| `controller.py` | 80 | PD joint-space torque controller with feedforward, saturation, error logging |
| `pybullet_sim.py` | 190 | PyBullet simulation, URDF mesh patching, torque-controlled execution, telemetry |
| `visualization.py` | 210 | 3D plots, trajectory profiles, controller telemetry, GIF animation |
| `benchmark.py` | 173 | 20-run comparative analysis with statistics and bar charts |
| `main.py` | 155 | CLI entry point with 4 modes |
| `panda.urdf` | 253 | Franka Panda URDF (7-DOF arm) |
| `panda_with_gripper.urdf` | 327 | Panda URDF with finger joints |
| `meshes/` | — | STL mesh files for visual and collision geometry |

---

## Configuration

All tuneable parameters are in `config.py`. Key groups:

**Robot geometry** — joint origins, axes, limits, link radii (from URDF).

**Obstacles** — 8 spherical obstacles with positions, radii, and safety clearance. Edit this list to change the environment.

**APF** — `K_ATT` (attractive gain), `K_REP` (repulsive gain), `RHO_0` (influence radius). Higher `K_REP` makes the arm more conservative around obstacles; higher `K_ATT` makes it more aggressive toward the goal.

**RRT** — `MAX_ITERATIONS`, `STEP_SIZE`, `GOAL_BIAS`, `GOAL_THRESHOLD`. Larger step size explores faster but misses narrow passages; higher goal bias converges faster but explores less.

**Trajectory** — `TRAJECTORY_DURATION` (default 12s), velocity and acceleration limits.

**Controller** — `KP_GAINS` and `KD_GAINS` per joint. Proximal joints (1-4) get higher gains (600/50) due to higher inertia; distal joints (5-7) get lower gains (50-250 / 8-20).

**Start/Goal** — `Q_START` and `Q_GOAL` in radians. These are verified collision-free at import time.
