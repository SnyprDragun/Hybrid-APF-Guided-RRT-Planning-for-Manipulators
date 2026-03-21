#!/usr/bin/env python3
"""
Directly inspect what's in your URDF - no fluff
"""

import pinocchio as pin
import numpy as np
from pathlib import Path

urdf_path = Path("/home/focaslab/ros2_ws/src/Hybrid-APF-Guided-RRT-Planning-for-Manipulators/model_description/panda_with_gripper.urdf")
package_dir = str(urdf_path.parent.parent)

robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dir)
model = robot.model

print("JOINT LIST:")
print("-" * 80)
for i, joint in enumerate(model.joints):
    print(f"Joint {i}: {model.names[i]:<30} type={joint.shortname():<8} nq={joint.nq}")

print("\n\nTESTING DIFFERENT JOINT CONFIGURATIONS:")
print("-" * 80)

ee_frame_id = model.nframes - 2

# Test: vary ONLY joint 7
print("\nVarying ONLY joint 7 (index 7):")
for val in [0, 0.5, 1.0, 1.5, 2.0]:
    q = np.array([0, 0, 0, 0, 0, 0, 1, val, -np.pi/4])
    pin.forwardKinematics(model, robot.data, q)
    pin.updateFramePlacement(model, robot.data, ee_frame_id)
    ee = robot.data.oMf[ee_frame_id].translation
    print(f"  j7={val:+.1f}: EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

# Test: vary ONLY joint 8
print("\nVarying ONLY joint 8 (index 8):")
for val in [-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi]:
    q = np.array([0, 0, 0, 0, 0, 0, 1, 0, val])
    pin.forwardKinematics(model, robot.data, q)
    pin.updateFramePlacement(model, robot.data, ee_frame_id)
    ee = robot.data.oMf[ee_frame_id].translation
    print(f"  j8={val:+.2f}: EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

# Test: vary both
print("\nVarying BOTH joints:")
for j7 in [-1, 0, 1]:
    for j8 in [-np.pi, -np.pi/2, 0]:
        q = np.array([0, 0, 0, 0, 0, 0, 1, j7, j8])
        pin.forwardKinematics(model, robot.data, q)
        pin.updateFramePlacement(model, robot.data, ee_frame_id)
        ee = robot.data.oMf[ee_frame_id].translation
        print(f"  j7={j7:+.1f}, j8={j8:+.2f}: EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")