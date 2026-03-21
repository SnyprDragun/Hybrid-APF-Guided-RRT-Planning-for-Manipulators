#!/usr/bin/env python3
"""
URDF Diagnostic Tool
Helps identify model structure and required configuration vector size
"""

import pinocchio as pin
import numpy as np
from pathlib import Path


def diagnose_robot(urdf_path: str, package_dir: str = None):
    """Analyze URDF and print robot structure"""
    
    print("=" * 80)
    print(f"ROBOT URDF DIAGNOSTIC: {urdf_path}")
    print("=" * 80)
    
    try:
        if package_dir:
            robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dir)
        else:
            robot = pin.RobotWrapper.BuildFromURDF(urdf_path, 
                                                    pin.StdVec_StdString())
    except Exception as e:
        print(f"❌ Error loading URDF: {e}")
        return None
    
    model = robot.model
    
    print(f"\n✓ Successfully loaded robot model")
    print(f"\n{'='*80}")
    print(f"MODEL STRUCTURE")
    print(f"{'='*80}")
    
    print(f"\nConfiguration Space Dimension (nq): {model.nq}")
    print(f"Velocity Space Dimension (nv):     {model.nv}")
    print(f"Number of Joints:                   {model.njoints}")
    print(f"Number of Frames:                   {model.nframes}")
    
    print(f"\n{'='*80}")
    print(f"JOINT DETAILS")
    print(f"{'='*80}\n")
    
    for i, joint in enumerate(model.joints):
        print(f"Joint {i}: {model.names[i]}")
        print(f"  Type: {joint.shortname()}")
        print(f"  Config dim: {joint.nq}, Velocity dim: {joint.nv}")
        if hasattr(joint, 'placement'):
            print(f"  Placement:\n{joint.placement}")
        print()
    
    print(f"{'='*80}")
    print(f"FRAME DETAILS (Useful for EE tracking)")
    print(f"{'='*80}\n")
    
    for frame_id, frame in enumerate(model.frames):
        print(f"Frame {frame_id}: {frame.name}")
        print(f"  Parent Joint: {model.names[frame.parentJoint]}")
        if "hand" in frame.name.lower() or "ee" in frame.name.lower() or frame_id == model.nframes - 1:
            print(f"  ⭐ Likely End-Effector")
        print()
    
    print(f"{'='*80}")
    print(f"RECOMMENDED ACTIONS")
    print(f"{'='*80}\n")
    
    if model.nq == 7:
        print("✓ Found 7-DOF model (Franka Panda with floating base or gripper)")
        print("  Option 1: Use full 7-DOF")
        print("  Option 2: Lock last DOF (gripper) and use 6-DOF")
        q_example = np.zeros(model.nq)
        print(f"\n  Example config vector (all zeros): shape {q_example.shape}")
        
    elif model.nq == 9:
        print("❌ Found 9-DOF model")
        print("  This likely includes:")
        print("    - Floating base (SE3: 7 DOF) OR")
        print("    - 6-DOF arm + 3-DOF hand")
        print("\n  SOLUTION: Check URDF for floating base")
        print("    If floating base exists, either:")
        print("      A) Remove floating base from URDF")
        print("      B) Use 9-DOF planning (expert mode)")
        
    elif model.nq == 6:
        print("✓ Perfect! Standard 6-DOF manipulator")
        q_example = np.zeros(model.nq)
        print(f"  Config vector shape: {q_example.shape}")
        
    else:
        print(f"⚠ Unusual DOF count: {model.nq}")
        print("  Check URDF structure carefully")
    
    # Test forward kinematics
    print(f"\n{'='*80}")
    print(f"FORWARD KINEMATICS TEST")
    print(f"{'='*80}\n")
    
    try:
        q_test = np.zeros(model.nq)
        pin.forwardKinematics(model, robot.data, q_test)
        
        # Find likely end-effector frame
        ee_frame_candidates = [name for name in model.names 
                              if any(x in name.lower() for x in 
                                    ['hand', 'ee', 'tool', 'tcp', 'panda_hand'])]
        
        if ee_frame_candidates:
            ee_name = ee_frame_candidates[0]
            ee_frame_id = model.getFrameId(ee_name)
            pin.updateFramePlacement(model, robot.data, ee_frame_id)
            ee_pos = robot.data.oMf[ee_frame_id].translation
            print(f"✓ FK test successful!")
            print(f"  End-effector candidate: {ee_name}")
            print(f"  Position at q=0: {ee_pos}")
        else:
            print(f"⚠ Could not identify end-effector frame")
            print(f"  Available frames: {[model.names[i] for i in range(1, min(10, model.nframes))]}")
        
    except Exception as e:
        print(f"❌ FK test failed: {e}")
    
    print(f"\n{'='*80}\n")
    
    return robot, model


def generate_corrected_config(model, use_only_arm: bool = True):
    """Generate corrected configuration vector"""
    
    print(f"{'='*80}")
    print(f"GENERATING CORRECTED CONFIGURATION")
    print(f"{'='*80}\n")
    
    nq = model.nq
    
    if nq == 9:
        print("Detected 9-DOF model (floating base + 6-arm likely)")
        
        if use_only_arm:
            print("\n✓ OPTION A: Use only arm DOF (last 6 joints)")
            print("  Set first 3 DOF = position [0, 0, 0]")
            print("  Set next 3 DOF = quaternion [0, 0, 0, 1]")
            print("  This effectively fixes the base in space\n")
            
            # Example config for planning
            q_example = np.array([
                0, 0, 0,      # Base position (x, y, z)
                0, 0, 0, 1,   # Base orientation (qx, qy, qz, qw)
                0, 0           # Arm joints (placeholder - adjust for your arm)
            ])
            print(f"  Example: {q_example}")
            
        else:
            print("\n✓ OPTION B: Use full 9-DOF (advanced)")
            q_example = np.zeros(9)
            print(f"  Example: {q_example}")
    
    elif nq == 7:
        print("Detected 7-DOF model (likely Panda + gripper)")
        print("\n✓ Use all 7 DOF in planning")
        q_example = np.zeros(7)
        print(f"  Example: {q_example}")
        
        print("\nOr, to use 6-DOF (lock gripper):")
        print("  - Remove gripper joint from URDF, OR")
        print("  - Plan only first 6 DOF, fix gripper = 0")
    
    else:
        print(f"DOF count: {nq}")
        q_example = np.zeros(nq)
        print(f"  Config example: {q_example}")
    
    return q_example


if __name__ == "__main__":
    """
    Usage Examples:
    
    # For standard Franka Panda
    robot, model = diagnose_robot("franka_panda.urdf")
    
    # If URDF requires package directory
    robot, model = diagnose_robot(
        "/path/to/franka.urdf",
        package_dir="/path/to/franka_description"
    )
    
    # Generate corrected config
    q_correct = generate_corrected_config(model, use_only_arm=True)
    """
    
    # Example: uncomment and update path to your URDF
    robot, model = diagnose_robot("model_description/panda_with_gripper.urdf")
    q_corrected = generate_corrected_config(model)
    
    print("Update the __main__ section with your URDF path and run this script.")