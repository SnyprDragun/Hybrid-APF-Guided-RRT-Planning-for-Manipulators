# Hybrid APF-Guided RRT Motion Planner — Franka Panda 7-DOF

A high-performance motion planner combining **Rapidly-exploring Random Trees (RRT)** with **Artificial Potential Fields (APF)** for navigating a Franka Emika Panda 7-DOF arm through obstacle-dense environments.

Uses the **actual Panda URDF** for forward kinematics (joint origins, axes, and limits extracted directly from `panda.urdf`).

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                      main.py                           │
│  CLI entry point — demo / benchmark / animate / sim    │
├──────────┬──────────┬──────────┬───────────────────────┤
│ rrt_base │rrt_enhan.│benchmark │   visualization.py    │
│  .py     │  .py     │  .py     │ 3D plots, animation   │
├──────────┴──────────┴──────────┴───────────────────────┤
│                      apf.py                            │
│  Attractive + Repulsive forces, Jacobian projection    │
├────────────────────────┬───────────────────────────────┤
│    robot_model.py      │       obstacles.py            │
│  Panda URDF-based FK   │ Sphere collision, distance    │
├────────────────────────┴───────────────────────────────┤
│                     config.py                          │
│  Panda joint params + all tuneable parameters          │
└────────────────────────────────────────────────────────┘
        (optional)  pybullet_sim.py  — URDF-based GUI
```

## Robot Model

Forward kinematics uses the **exact URDF joint specifications**:

| Joint | Origin xyz | Origin rpy | Limits (rad) |
|-------|-----------|-----------|-------------|
| panda_joint1 | (0, 0, 0.333) | (0, 0, 0) | [-2.90, +2.90] |
| panda_joint2 | (0, 0, 0) | (-π/2, 0, 0) | [-1.76, +1.76] |
| panda_joint3 | (0, -0.316, 0) | (+π/2, 0, 0) | [-2.90, +2.90] |
| panda_joint4 | (0.0825, 0, 0) | (+π/2, 0, 0) | [-3.07, -0.07] |
| panda_joint5 | (-0.0825, 0.384, 0) | (-π/2, 0, 0) | [-2.90, +2.90] |
| panda_joint6 | (0, 0, 0) | (+π/2, 0, 0) | [-0.02, +3.75] |
| panda_joint7 | (0.088, 0, 0) | (+π/2, 0, 0) | [-2.90, +2.90] |

Flange offset: (0, 0, 0.107) from joint7.

Each joint transform: `T_joint = T_origin · Rot_z(q_i)` — all axes are local Z.

## Algorithm

### Phase A — Baseline APF-RRT

1. Initialise tree with `q_start`.
2. Each iteration:
   - **Sample**: With probability `GOAL_BIAS` → sample `q_goal`; else uniform random in C-space (respecting actual Panda joint limits).
   - **Nearest**: Find closest tree node (L2 in 7D joint-space).
   - **Steer with APF blend**: Compute net task-space force `F = F_att + ΣF_rep` on all link points. Project to joint-space via `τ = Jᵀ · F`. Blend: `d = (1−w)·d_random + w·d_apf`.
   - **Extend**: Move by `STEP_SIZE` in blended direction.
   - **Collision check**: Swept-volume check (interpolated points per link vs spherical obstacles).
   - **Goal check**: If `‖q_new − q_goal‖ < GOAL_THRESHOLD` → extract path.

### Phase B — Enhanced (Adaptive Tuning + Optimisation-Based Smoothing)

**1. Adaptive Parameter Tuning**
- Step size inversely proportional to local obstacle density.
- Stall detection: if no progress for 80 iterations, triple `K_att` and reduce `K_rep` by 70% to escape local minima.

**2. Shortcut + Gradient Smoothing**
- **Stage 1 — Shortcutting**: Randomly pick two non-adjacent waypoints; if the straight line is collision-free, remove intermediates.
- **Stage 2 — Gradient nudge**: Nudge each waypoint toward the midpoint of its neighbours if it shortens the path without collision.

## Installation

```bash
pip install numpy matplotlib scipy        # core (no simulation needed)
pip install pybullet                       # optional: PyBullet GUI
pip install pillow                         # optional: GIF generation
```

Python 3.10+ required.

## Usage

```bash
python main.py                 # quick demo: both planners + 3D plots
python main.py --benchmark     # full 20-run comparative analysis
python main.py --animate       # render animation frames + GIF
python main.py --pybullet      # launch PyBullet with real Panda URDF
```

All outputs → `./output/`

## Visualization

**No simulation is strictly required.** The planner includes:

1. **3D matplotlib plots** — RRT tree, smoothed path, 8 obstacle spheres, Panda arm skeleton at start/goal configs.
2. **Side-by-side comparison** — baseline vs enhanced on the same figure.
3. **Benchmark bar chart** — success rate, computation time, path length, node count.
4. **(Optional) PyBullet GUI** — loads `panda.urdf` / `panda_with_gripper.urdf` directly, animates the trajectory in a physics engine with visual obstacles.

## Benchmark Results (20 runs)

```
====================================================================
Metric                          Baseline APF-RRT    Enhanced APF-RRT
====================================================================
Success Rate (%)                          100.0%              100.0%
Comp. Time (s, success)            0.475 ± 0.246       0.487 ± 0.281
Path Length (rad, success)         2.062 ± 0.325       1.356 ± 0.115
Node Count (success)                     26 ± 13             22 ± 13
====================================================================
```

The enhanced planner achieves **34% shorter paths** with lower variance, at comparable computation time.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All parameters: Panda joints, obstacles, APF/RRT tuning |
| `robot_model.py` | URDF-based FK, link positions, Jacobian |
| `apf.py` | Attractive/repulsive forces, joint-space projection |
| `obstacles.py` | Sphere collision checking, distance queries |
| `rrt_base.py` | Baseline APF-RRT planner |
| `rrt_enhanced.py` | Enhanced planner (adaptive + smoothing) |
| `visualization.py` | 3D matplotlib plots and animation |
| `benchmark.py` | 20-run comparative analysis |
| `pybullet_sim.py` | Optional PyBullet visualiser (uses real URDF) |
| `main.py` | CLI entry point |
| `panda.urdf` | Franka Panda URDF (7-DOF arm) |
| `panda_with_gripper.urdf` | Panda URDF with finger joints |
