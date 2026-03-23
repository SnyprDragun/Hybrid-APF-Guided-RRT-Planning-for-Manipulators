"""
pybullet_sim.py — PyBullet simulation with PD torque-controlled execution.

Loads the Panda URDF, spawns obstacle spheres, and executes planned
trajectories using the joint-space PD controller (controller.py) with
time-parameterised trajectories (trajectory.py).

The robot moves smoothly over the configured duration (default 12s)
with proper dynamics — not instantaneous teleportation.

URDF mesh handling
------------------
The URDF references meshes via ``package://model_description/meshes/...``.
This module patches the URDF at load time to use absolute paths pointing
to the meshes/ folder in the project directory.
"""

from __future__ import annotations
import os
import re
import time
import tempfile
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from config import (
    OBSTACLES, NUM_JOINTS, URDF_PANDA, URDF_PANDA_GRIPPER,
    MESHES_DIR, TRAJECTORY_DURATION,
)


PANDA_ARM_JOINTS = list(range(7))


def _patch_urdf_mesh_paths(urdf_path: str, meshes_dir: str) -> str:
    """
    Read the URDF and replace ALL mesh file references with absolute
    paths pointing to *meshes_dir*.  Handles three formats:

    1. package://model_description/meshes/visual/link0.stl
    2. /home/.../model_description/meshes/visual/link0.stl  (hardcoded abs)
    3. meshes/visual/link0.stl                              (relative)

    Returns the path to a patched temp URDF.
    """
    with open(urdf_path, "r") as f:
        content = f.read()

    abs_meshes = os.path.abspath(meshes_dir)

    # Strategy: find every mesh filename="..." and rewrite just the path
    # portion so that it points to abs_meshes/{visual|collision}/file.stl.
    def _rewrite_mesh_path(match):
        original = match.group(1)
        # Extract the tail: visual/linkN.stl or collision/hand.stl etc.
        # Look for the meshes/ directory boundary
        for marker in ["/meshes/", "meshes/"]:
            idx = original.find(marker)
            if idx >= 0:
                tail = original[idx + len(marker):]
                return f'filename="{abs_meshes}/{tail}"'
        # Fallback: if the file is just a bare name, prepend meshes dir
        basename = os.path.basename(original)
        return f'filename="{abs_meshes}/{basename}"'

    content = re.sub(
        r'filename="([^"]*\.stl)"',
        _rewrite_mesh_path,
        content,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(content)
    tmp.close()
    return tmp.name


class PyBulletVisualizer:
    """
    PyBullet simulation with PD torque control.

    Parameters
    ----------
    gui : bool
        If True, open GUI window.
    use_gripper : bool
        If True, load panda_with_gripper.urdf.
    """

    def __init__(self, gui: bool = True, use_gripper: bool = False):
        if not PYBULLET_AVAILABLE:
            raise RuntimeError("PyBullet not installed. Run: pip install pybullet")

        mode = p.GUI if gui else p.DIRECT
        self.cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)

        self.use_gripper = use_gripper
        self.robot_id: int | None = None
        self.obstacle_ids: list[int] = []
        self._patched_urdf: str | None = None

    def load_scene(self):
        """Load ground, Panda URDF (with patched mesh paths), and obstacles."""
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        urdf_src = URDF_PANDA_GRIPPER if self.use_gripper else URDF_PANDA
        if not os.path.exists(urdf_src):
            raise FileNotFoundError(f"URDF not found: {urdf_src}")

        # Patch mesh paths if meshes directory exists
        if os.path.isdir(MESHES_DIR):
            self._patched_urdf = _patch_urdf_mesh_paths(urdf_src, MESHES_DIR)
            urdf_to_load = self._patched_urdf
            print(f"  Meshes found at {MESHES_DIR} — loading with visuals.")
        else:
            urdf_to_load = urdf_src
            print(f"  Meshes dir not found ({MESHES_DIR}) — loading without visuals.")

        try:
            self.robot_id = p.loadURDF(
                urdf_to_load,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.cid,
            )
        except Exception as e:
            print(f"  URDF load with meshes failed ({e})")
            print(f"  Retrying without visual/collision meshes (kinematics still work)...")
            # As a last resort, load with mesh ignoring flags
            try:
                self.robot_id = p.loadURDF(
                    urdf_to_load, basePosition=[0, 0, 0],
                    useFixedBase=True, physicsClientId=self.cid,
                    flags=p.URDF_IGNORE_VISUAL_SHAPES | p.URDF_IGNORE_COLLISION_SHAPES,
                )
            except Exception:
                # If even the patched file fails, try the original with ignore flags
                self.robot_id = p.loadURDF(
                    urdf_src, basePosition=[0, 0, 0],
                    useFixedBase=True, physicsClientId=self.cid,
                    flags=p.URDF_IGNORE_VISUAL_SHAPES | p.URDF_IGNORE_COLLISION_SHAPES,
                )

        # Print joint info
        n_joints = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        print(f"  Loaded Panda with {n_joints} joints.")
        for i in range(min(n_joints, 12)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.cid)
            name = info[1].decode("utf-8")
            jtype = {0: "revolute", 1: "prismatic", 4: "fixed"}.get(info[2], "other")
            print(f"    [{i}] {name:25s}  {jtype}  [{info[8]:.3f}, {info[9]:.3f}]")

        # Disable default velocity motors (required for torque control)
        for ji in PANDA_ARM_JOINTS:
            p.setJointMotorControl2(
                self.robot_id, ji,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.cid,
            )

        # Obstacles
        for centre, radius in OBSTACLES:
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.cid)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                         rgbaColor=[0.9, 0.2, 0.2, 0.4],
                                         physicsClientId=self.cid)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                                     baseVisualShapeIndex=vis_id,
                                     basePosition=centre.tolist(),
                                     physicsClientId=self.cid)
            self.obstacle_ids.append(body)

        # Camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=[0.3, 0.1, 0.3],
            physicsClientId=self.cid,
        )

    # ── Joint state access ───────────────────

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) for the 7 arm joints."""
        states = p.getJointStates(self.robot_id, PANDA_ARM_JOINTS, physicsClientId=self.cid)
        q  = np.array([s[0] for s in states])
        qd = np.array([s[1] for s in states])
        return q, qd

    def set_joint_positions(self, q: np.ndarray):
        """Snap joints (no dynamics) — used for initial configuration."""
        for i, ji in enumerate(PANDA_ARM_JOINTS):
            if i < len(q):
                p.resetJointState(self.robot_id, ji, q[i], physicsClientId=self.cid)

    # ── Torque-controlled execution ──────────

    def execute_with_controller(
        self,
        trajectory,
        controller,
        real_time: bool = True,
    ) -> dict:
        """
        Execute a trajectory using the PD torque controller.

        Parameters
        ----------
        trajectory : trajectory.Trajectory
            Time-parameterised trajectory.
        controller : controller.PDController
            Joint-space PD controller.
        real_time : bool
            If True, sleep to match real-time playback.

        Returns
        -------
        log : dict
            Telemetry: times, q_desired, q_actual, torques, tracking_error.
        """
        dt = 1.0 / 240.0
        log = {"t": [], "q_des": [], "q_act": [], "qd_act": [],
               "tau": [], "error": []}

        # Set initial configuration
        q_d0, _, _ = trajectory.sample(0.0)
        self.set_joint_positions(q_d0)
        # Let physics settle with position hold
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.cid)

        t = 0.0
        step_count = 0
        while t <= trajectory.t_end + 0.5:
            # Sample desired trajectory
            q_d, qd_d, qdd_d = trajectory.sample(min(t, trajectory.t_end))

            # Read actual state
            q_act, qd_act = self.get_joint_states()

            # Compute PD torques
            tau = controller.compute(q_d, qd_d, qdd_d, q_act, qd_act)

            # Note: gravity compensation via calculateInverseDynamics is
            # not used because PyBullet fails on URDFs with fixed joints
            # in the chain. The PD controller with the current gains is
            # stiff enough to hold against gravity with minimal steady-state
            # error. For a real robot, use Pinocchio RNEA or libfranka.

            # Apply torques
            p.setJointMotorControlArray(
                self.robot_id, PANDA_ARM_JOINTS,
                controlMode=p.TORQUE_CONTROL,
                forces=tau.tolist(),
                physicsClientId=self.cid,
            )

            p.stepSimulation(physicsClientId=self.cid)

            # Log every 10th step
            if step_count % 10 == 0:
                log["t"].append(t)
                log["q_des"].append(q_d.copy())
                log["q_act"].append(q_act.copy())
                log["qd_act"].append(qd_act.copy())
                log["tau"].append(tau.copy())
                log["error"].append(controller.tracking_error_rms())

            if real_time:
                time.sleep(dt)

            t += dt
            step_count += 1

        # Convert to arrays
        for key in ["t", "q_des", "q_act", "qd_act", "tau", "error"]:
            log[key] = np.array(log[key])

        return log

    # ── Simple kinematic replay (legacy) ─────

    def execute_path_kinematic(
        self, path: list[np.ndarray], duration: float = 12.0
    ):
        """Replay path with linear interpolation — no dynamics."""
        n_total = int(duration * 240)
        interp_path = []
        for i in range(len(path) - 1):
            n_seg = max(1, n_total // (len(path) - 1))
            for t in np.linspace(0, 1, n_seg, endpoint=(i == len(path)-2)):
                interp_path.append(path[i] + t * (path[i+1] - path[i]))

        for q in interp_path:
            self.set_joint_positions(q)
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(1.0 / 240.0)

    # ── Cleanup ──────────────────────────────

    def disconnect(self):
        if self._patched_urdf and os.path.exists(self._patched_urdf):
            os.unlink(self._patched_urdf)
        p.disconnect(self.cid)