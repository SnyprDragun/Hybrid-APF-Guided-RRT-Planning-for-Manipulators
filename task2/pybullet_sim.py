"""
pybullet_sim.py — Optional PyBullet simulation for physics-based rendering.

This module is **not required** to run the planner or produce results.
When PyBullet is installed, it provides:
  • A GUI window showing the UR5 executing the planned path
  • Spherical obstacle bodies in the physics scene
  • Real-time or recorded video output

Usage
-----
    from pybullet_sim import PyBulletVisualizer
    viz = PyBulletVisualizer(gui=True)
    viz.load_scene()
    viz.execute_path(result.path, speed=0.5)
    viz.disconnect()

Install
-------
    pip install pybullet
"""

from __future__ import annotations
import time
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from config import OBSTACLES, NUM_JOINTS


class PyBulletVisualizer:
    """
    Thin wrapper around PyBullet for visualising the 6-DOF manipulator
    executing a planned trajectory in a cluttered environment.

    Parameters
    ----------
    gui : bool
        If True, open a GUI window; otherwise run headless (DIRECT).
    """

    def __init__(self, gui: bool = True):
        if not PYBULLET_AVAILABLE:
            raise RuntimeError(
                "PyBullet is not installed.  Run: pip install pybullet"
            )
        mode = p.GUI if gui else p.DIRECT
        self.cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        self.robot_id = None
        self.obstacle_ids: list[int] = []

    # ── Scene setup ──────────────────────────

    def load_scene(self, urdf_path: str | None = None):
        """
        Load the ground plane, robot, and obstacles.

        Parameters
        ----------
        urdf_path : str, optional
            Path to a UR5 URDF.  If None, uses the default Franka Panda
            from pybullet_data (most universally available).
        """
        # Ground
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        # Robot
        if urdf_path is None:
            # Try UR5 from pybullet_data; fallback to Franka
            try:
                self.robot_id = p.loadURDF(
                    "franka_panda/panda.urdf",
                    basePosition=[0, 0, 0],
                    useFixedBase=True,
                    physicsClientId=self.cid,
                )
            except Exception:
                print("Could not load Franka URDF; skipping robot.")
                self.robot_id = None
        else:
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.cid,
            )

        # Obstacles (visual-only spheres)
        for centre, radius in OBSTACLES:
            col_id = p.createCollisionShape(
                p.GEOM_SPHERE, radius=radius, physicsClientId=self.cid
            )
            vis_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=radius,
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
            cameraTargetPosition=[0.1, 0.1, 0.3],
            physicsClientId=self.cid,
        )

    # ── Trajectory execution ─────────────────

    def set_joint_positions(self, q: np.ndarray):
        """Snap robot joints to configuration *q* (no dynamics)."""
        if self.robot_id is None:
            return
        n = min(NUM_JOINTS, p.getNumJoints(self.robot_id, physicsClientId=self.cid))
        for i in range(n):
            p.resetJointState(
                self.robot_id, i, q[i], physicsClientId=self.cid
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
        path : list of joint configurations
        speed : float
            Playback multiplier (1.0 = real-time).
        n_interp : int
            Interpolated sub-steps between waypoints.
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
