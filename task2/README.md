# Hybrid APF-Guided RRT Motion Planner for 6-DOF Manipulator

A high-performance motion planner that combines the exploratory power of **Rapidly-exploring Random Trees (RRT)** with the directional guidance of **Artificial Potential Fields (APF)** for navigating a 6-DOF robotic arm (UR5-like kinematics) through obstacle-dense environments.

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
│  DH FK, link geometry  │ Sphere collision, distance    │
├────────────────────────┴───────────────────────────────┤
│                     config.py                          │
│  All tuneable parameters in one place                  │
└────────────────────────────────────────────────────────┘
         (optional)  pybullet_sim.py  — GUI physics viz
```

## Algorithm

### Phase A — Baseline APF-RRT

1. Initialise tree with `q_start`.
2. Each iteration:
   - **Sample**: With probability `GOAL_BIAS`, sample `q_goal`; else uniform random in C-space.
   - **Nearest**: Find closest tree node (L2 in joint-space).
   - **Steer with APF blend**: Compute the APF joint-space gradient at the nearest node via `τ = Jᵀ · (F_att + ΣF_rep)`. Blend with the random direction: `d = (1−w)·d_rand + w·d_apf`.
   - **Extend**: Move by `STEP_SIZE` in the blended direction.
   - **Collision check**: Validate via swept-volume sampling against spherical obstacles.
   - **Goal check**: If within `GOAL_THRESHOLD`, extract path via backtracking.

### Phase B — Enhanced Planner (Adaptive + Smoothing)

Two improvements over the baseline:

**1. Adaptive Parameter Tuning**
- **Step size** scales inversely with local obstacle density (more obstacles → smaller, safer steps).
- **APF weights** (`K_att`, `K_rep`) auto-adjust: if the tree stalls (no progress for 80 iterations), `K_att` is tripled and `K_rep` reduced to break out of local minima.

**2. Optimization-Based Path Smoothing**
- **Stage 1 — Shortcutting**: Randomly pick two non-adjacent waypoints; if the straight line is collision-free, remove all intermediate nodes.
- **Stage 2 — Gradient nudge**: For each remaining waypoint, nudge it toward the midpoint of its neighbours if that reduces total path length and remains collision-free.

## Installation

```bash
# Core (no simulation)
pip install numpy matplotlib scipy

# Optional: physics-based PyBullet visualization
pip install pybullet

# Optional: GIF generation
pip install pillow
```

**Python 3.10+** required (uses `X | Y` type union syntax).

## Usage

```bash
# Quick demo — one run of each planner, saves plots
python main.py

# Full 20-run comparative benchmark
python main.py --benchmark

# Generate animation frames + GIF
python main.py --animate

# PyBullet GUI (requires pybullet)
python main.py --pybullet
```

All outputs go to `./output/`.

## Configuration

Edit `config.py` to change:
- Robot DH parameters and joint limits
- Obstacle positions and sizes
- APF gains (`K_ATT`, `K_REP`, `RHO_0`)
- RRT parameters (`MAX_ITERATIONS`, `STEP_SIZE`, `GOAL_BIAS`)
- Start/goal configurations
- Benchmark run count

## Visualization

The planner produces three types of visual output:

1. **3D matplotlib plots** — tree edges, final path, obstacles, robot arm at start/goal.
2. **Side-by-side comparison** — baseline vs. enhanced on the same axes.
3. **Animated GIF** — robot arm following the smoothed path frame-by-frame.
4. **(Optional) PyBullet GUI** — real-time physics simulation with UR5/Panda URDF.

No simulation is strictly required — the forward kinematics model (`robot_model.py`) provides full 3D geometry for both planning and visualization. PyBullet adds physics fidelity but is optional.

## Deliverables

| Deliverable | File |
|---|---|
| Codebase | All `.py` files in this repository |
| Comparative analysis table | `output/benchmark_table.txt` |
| Bar chart comparison | `output/benchmark_bars.png` |
| 3D tree + path plots | `output/baseline_result.png`, `output/enhanced_result.png` |
| Side-by-side comparison | `output/comparison.png` |
| Animation | `output/robot_motion.gif` |

## License

MIT
