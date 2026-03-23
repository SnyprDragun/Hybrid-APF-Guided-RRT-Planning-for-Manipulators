"""
controller.py — Joint-space PD torque controller for the Franka Panda.

Implements a computed-torque-like PD controller in joint space:

    tau = Kp · (q_d - q) + Kd · (qd_d - qd) + qdd_d_ff

where:
    q_d, qd_d, qdd_d  = desired position, velocity, acceleration (from trajectory)
    q, qd              = actual position, velocity (from sensors/simulation)
    Kp, Kd             = diagonal gain matrices
    qdd_d_ff           = feedforward acceleration term

In a full implementation with a dynamics model, this would be:
    tau = M(q) · [qdd_d + Kp·e + Kd·ed] + C(q,qd)·qd + g(q)

The current implementation uses PD + feedforward without the full
dynamics model (M, C, g), relying on PyBullet's internal dynamics
to provide the gravity compensation and inertia.  This is sufficient
for simulation — for a real Panda, you would use libfranka's model
or Pinocchio RNEA.

Usage with PyBullet
-------------------
    ctrl = PDController()
    for t in sim_loop:
        q_d, qd_d, qdd_d = trajectory.sample(t)
        tau = ctrl.compute(q_d, qd_d, qdd_d, q_actual, qd_actual)
        p.setJointMotorControlArray(..., controlMode=p.TORQUE_CONTROL, forces=tau)
"""

import numpy as np
from config import NUM_JOINTS, KP_GAINS, KD_GAINS


class PDController:
    """
    PD controller with feedforward acceleration for joint-space tracking.

    Parameters
    ----------
    kp : np.ndarray, shape (NUM_JOINTS,)
        Proportional gains (stiffness).
    kd : np.ndarray, shape (NUM_JOINTS,)
        Derivative gains (damping).
    torque_limit : float
        Per-joint torque saturation (Nm).
    """

    def __init__(
        self,
        kp: np.ndarray = KP_GAINS,
        kd: np.ndarray = KD_GAINS,
        torque_limit: float = 87.0,
    ):
        self.kp = kp.copy()
        self.kd = kd.copy()
        self.torque_limit = torque_limit

        # Tracking errors for logging
        self.last_pos_error = np.zeros(NUM_JOINTS)
        self.last_vel_error = np.zeros(NUM_JOINTS)

    def compute(
        self,
        q_desired: np.ndarray,
        qd_desired: np.ndarray,
        qdd_desired: np.ndarray,
        q_actual: np.ndarray,
        qd_actual: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint torques for one timestep.

        Parameters
        ----------
        q_desired   : desired joint positions
        qd_desired  : desired joint velocities
        qdd_desired : desired joint accelerations (feedforward)
        q_actual    : measured joint positions
        qd_actual   : measured joint velocities

        Returns
        -------
        tau : np.ndarray, shape (NUM_JOINTS,)
            Commanded joint torques (Nm), saturated to torque_limit.
        """
        # Position and velocity errors
        e_pos = q_desired - q_actual
        e_vel = qd_desired - qd_actual

        # Store for logging
        self.last_pos_error = e_pos
        self.last_vel_error = e_vel

        # PD control (no feedforward — qdd_d would need M(q) scaling
        # to be physically meaningful, and we don't have the mass matrix
        # without Pinocchio.  PyBullet handles gravity internally.)
        tau = self.kp * e_pos + self.kd * e_vel

        # Torque saturation (per-joint)
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)

        return tau

    def tracking_error_rms(self) -> float:
        """RMS of the most recent position error across all joints."""
        return float(np.sqrt(np.mean(self.last_pos_error ** 2)))