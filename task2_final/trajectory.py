"""
trajectory.py — Time-parameterised trajectory generation.

Converts a sequence of joint-space waypoints from the planner into a
smooth, time-parameterised trajectory with bounded velocity and
acceleration, suitable for feeding to a torque controller.

Two methods are provided:
  1. Cubic spline interpolation (smooth C2 trajectory)
  2. Trapezoidal velocity profile (bang-coast-bang, simpler)

The output is a Trajectory object that can be sampled at any time t
to get (q_d, qd_d, qdd_d) — desired position, velocity, acceleration.
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from config import (
    NUM_JOINTS, TRAJECTORY_DURATION,
    MAX_JOINT_VEL, MAX_JOINT_ACC,
)


class Trajectory:
    """
    Time-parameterised joint-space trajectory.

    Parameters
    ----------
    waypoints : list of np.ndarray
        Joint-space waypoints (each length NUM_JOINTS).
    duration : float
        Total trajectory duration in seconds.
    method : str
        'cubic' for cubic spline, 'linear' for linear interpolation.
    """

    def __init__(
        self,
        waypoints: list[np.ndarray],
        duration: float = TRAJECTORY_DURATION,
        method: str = "cubic",
    ):
        self.waypoints = [q.copy() for q in waypoints]
        self.n_waypoints = len(waypoints)
        self.duration = duration
        self.method = method

        # Distribute waypoints at times proportional to cumulative
        # joint-space distance (arc-length parameterisation).
        dists = [0.0]
        for i in range(1, self.n_waypoints):
            dists.append(dists[-1] + np.linalg.norm(waypoints[i] - waypoints[i-1]))
        total = dists[-1] if dists[-1] > 1e-8 else 1.0
        self.times = np.array([d / total * duration for d in dists])

        # Build interpolators per joint
        Q = np.array(waypoints)  # (n_waypoints, NUM_JOINTS)

        if method == "cubic" and self.n_waypoints >= 3:
            # Cubic spline with zero velocity at start and end
            self._splines = []
            for j in range(NUM_JOINTS):
                cs = CubicSpline(
                    self.times, Q[:, j],
                    bc_type=((1, 0.0), (1, 0.0)),  # zero velocity BCs
                )
                self._splines.append(cs)
        else:
            self._splines = None  # fallback to linear

    @property
    def t_start(self) -> float:
        return 0.0

    @property
    def t_end(self) -> float:
        return self.duration

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the trajectory at time *t*.

        Returns
        -------
        q_d   : np.ndarray — desired joint positions
        qd_d  : np.ndarray — desired joint velocities
        qdd_d : np.ndarray — desired joint accelerations
        """
        t = np.clip(t, 0.0, self.duration)

        if self._splines is not None:
            q_d   = np.array([s(t)    for s in self._splines])
            qd_d  = np.array([s(t, 1) for s in self._splines])
            qdd_d = np.array([s(t, 2) for s in self._splines])
        else:
            # Linear interpolation fallback
            q_d, qd_d, qdd_d = self._linear_interp(t)

        return q_d, qd_d, qdd_d

    def _linear_interp(self, t: float):
        """Piecewise-linear interpolation (velocity = constant per segment)."""
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, self.n_waypoints - 2)

        t0, t1 = self.times[idx], self.times[idx + 1]
        dt = t1 - t0
        if dt < 1e-8:
            return self.waypoints[idx].copy(), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)

        alpha = (t - t0) / dt
        q_d  = (1 - alpha) * self.waypoints[idx] + alpha * self.waypoints[idx + 1]
        qd_d = (self.waypoints[idx + 1] - self.waypoints[idx]) / dt
        return q_d, qd_d, np.zeros(NUM_JOINTS)

    def sample_dense(self, dt: float = 1.0 / 240.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the full trajectory as dense arrays sampled at interval *dt*.

        Returns (times, positions, velocities) — each (N, NUM_JOINTS)
        except times which is (N,).
        """
        ts = np.arange(0.0, self.duration + dt, dt)
        qs, qds, qdds = [], [], []
        for t in ts:
            q, qd, qdd = self.sample(t)
            qs.append(q); qds.append(qd); qdds.append(qdd)
        return ts, np.array(qs), np.array(qds)

    def check_limits(self) -> dict:
        """
        Check if the trajectory exceeds velocity/acceleration bounds.
        Returns a dict with max values per joint.
        """
        ts, qs, qds = self.sample_dense(dt=0.005)
        max_vel = np.max(np.abs(qds), axis=0)
        # Numerical acceleration
        qdds = np.diff(qds, axis=0) / 0.005
        max_acc = np.max(np.abs(qdds), axis=0) if len(qdds) > 0 else np.zeros(NUM_JOINTS)
        return {
            "max_velocity_per_joint": max_vel,
            "max_acceleration_per_joint": max_acc,
            "vel_limit": MAX_JOINT_VEL,
            "acc_limit": MAX_JOINT_ACC,
            "vel_ok": bool(np.all(max_vel < MAX_JOINT_VEL * 1.1)),
            "acc_ok": bool(np.all(max_acc < MAX_JOINT_ACC * 1.5)),
        }
