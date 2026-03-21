"""
pybullet_sim.py — PyBullet simulation using the real Panda URDF.

Loads the user-provided panda.urdf or panda_with_gripper.urdf and
executes planned trajectories in a physics environment with visual
obstacle spheres.

Usage
-----
    from pybullet_sim import PyBulletVisualizer
    viz = PyBulletVisualizer(gui=True, use_gripper=True)
    viz.load_scene()
    viz.execute_path(result.path, speed=1.0)
    viz.disconnect()

Install
-------
    pip install pybullet
"""

from __future__ import annotations
import time
import os
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from config import (
    OBSTACLES, NUM_JOINTS, URDF_PANDA, URDF_PANDA_GRIPPER,
)


# Panda revolute joint indices in PyBullet (0-indexed)
# Joints 0..6 are the 7 revolute arm joints in the URDF.
PANDA_ARM_JOINT_INDICES = list(range(7))


class PyBulletVisualizer:
    """
    Wrapper around PyBullet for visualising the Franka Panda executing
    a planned trajectory in a cluttered environment.

    Parameters
    ----------
    gui : bool
        If True, open a GUI window; otherwise run headless (DIRECT).
    use_gripper : bool
        If True, load panda_with_gripper.urdf; else panda.urdf.
    """

    def __init__(self, gui: bool = True, use_gripper: bool = False):
        if not PYBULLET_AVAILABLE:
            raise RuntimeError(
                "PyBullet is not installed.  Run:  pip install pybullet"
            )
        mode = p.GUI if gui else p.DIRECT
        self.cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        self.use_gripper = use_gripper
        self.robot_id: int | None = None
        self.obstacle_ids: list[int] = []

    # ── Scene setup ──────────────────────────

    def load_scene(self):
        """Load ground plane, Panda URDF, and obstacle spheres."""

        # Ground
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        # Select URDF
        urdf = URDF_PANDA_GRIPPER if self.use_gripper else URDF_PANDA
        if not os.path.exists(urdf):
            raise FileNotFoundError(
                f"URDF not found at {urdf}. "
                "Make sure panda.urdf / panda_with_gripper.urdf are in the project root."
            )

        # The URDF references meshes via package://model_description/meshes/...
        # PyBullet resolves package:// paths by searching its additional search paths.
        # Strategy: add both the URDF directory AND its parent (in case the
        # model_description package folder sits alongside the project).
        urdf_dir = os.path.dirname(os.path.abspath(urdf))
        p.setAdditionalSearchPath(urdf_dir)
        parent_dir = os.path.dirname(urdf_dir)
        p.setAdditionalSearchPath(parent_dir)

        # Also check if a model_description folder exists nearby
        pkg_candidates = [
            os.path.join(urdf_dir, "model_description"),
            os.path.join(parent_dir, "model_description"),
        ]
        for pkg_path in pkg_candidates:
            if os.path.isdir(pkg_path):
                p.setAdditionalSearchPath(os.path.dirname(pkg_path))
                break

        try:
            self.robot_id = p.loadURDF(
                urdf,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.cid,
                flags=p.URDF_USE_SELF_COLLISION,
            )
        except p.error as e:
            # If mesh loading fails, retry without meshes (kinematics still work)
            print(f"  Warning: URDF mesh loading failed ({e})")
            print("  Retrying without visual meshes — kinematics still functional.")
            print("  To fix: place the model_description/meshes/ folder next to the URDF.")
            self.robot_id = p.loadURDF(
                urdf,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.cid,
                flags=p.URDF_IGNORE_VISUAL_SHAPES | p.URDF_IGNORE_COLLISION_SHAPES,
            )

        # Print joint info for debugging
        n_joints = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        print(f"  Loaded Panda with {n_joints} joints from: {urdf}")
        for i in range(min(n_joints, 12)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.cid)
            name = info[1].decode("utf-8")
            jtype = ["revolute", "prismatic", "spherical", "planar", "fixed"][info[2]]
            print(f"    [{i}] {name:30s}  type={jtype}  limits=[{info[8]:.3f}, {info[9]:.3f}]")

        # Obstacle spheres (visual only, no dynamics)
        for centre, radius in OBSTACLES:
            col_id = p.createCollisionShape(
                p.GEOM_SPHERE, radius=radius, physicsClientId=self.cid,
            )
            vis_id = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius,
                rgbaColor=[0.9, 0.2, 0.2, 0.4],
                physicsClientId=self.cid,
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=centre.tolist(),
                physicsClientId=self.cid,
            )
            self.obstacle_ids.append(body_id)

        # Camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.3, 0.1, 0.3],
            physicsClientId=self.cid,
        )

    # ── Joint control ────────────────────────

    def set_joint_positions(self, q: np.ndarray):
        """Snap the 7 arm joints to configuration *q* (no dynamics)."""
        if self.robot_id is None:
            return
        for i, ji in enumerate(PANDA_ARM_JOINT_INDICES):
            if i < len(q):
                p.resetJointState(
                    self.robot_id, ji, q[i], physicsClientId=self.cid,
                )

    def execute_path(
        self,
        path: list[np.ndarray],
        speed: float = 1.0,
        n_interp: int = 10,
    ):
        """
        Animate the robot along the planned path.

        Parameters
        ----------
        path : list of joint configurations (each length 7)
        speed : float
            Playback multiplier (1.0 = real-time).
        n_interp : int
            Interpolated sub-steps between waypoints for smoothness.
        """
        for i in range(len(path) - 1):
            for t in np.linspace(0, 1, n_interp):
                q = path[i] + t * (path[i + 1] - path[i])
                self.set_joint_positions(q)
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(1.0 / (240.0 * speed))

    # ── Cleanup ──────────────────────────────

    def disconnect(self):
        p.disconnect(self.cid)
