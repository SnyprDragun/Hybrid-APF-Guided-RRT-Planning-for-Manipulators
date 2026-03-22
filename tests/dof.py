#!/usr/bin/env python3
"""
Panda Home Pose Finder
Determines the correct home configuration and tests collision detection
"""

import pinocchio as pin
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_panda_home_pose():
    """Test different configurations to find proper home pose"""
    
    urdf_path = Path("/home/focaslab/ros2_ws/src/Hybrid-APF-Guided-RRT-Planning-for-Manipulators/model_description/panda_with_gripper.urdf")
    package_dir = str(urdf_path.parent.parent)
    
    print("=" * 80)
    print("PANDA HOME POSE FINDER")
    print("=" * 80 + "\n")
    
    # Load robot
    robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dir)
    model = robot.model
    
    print(f"Robot DOF: {model.nq}\n")
    
    # Test several configurations
    configs_to_test = {
        "All zeros (q=0)": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
        
        "Standard Panda Home (deg)": np.array([
            0, 0, 0, 0, 0, 0, 1,  # base (fixed)
            0,                      # arm joint 1: 0 deg
            0                       # arm joint 2: 0 deg
        ]),
        
        "Neutral Ready (radians)": np.array([
            0, 0, 0, 0, 0, 0, 1,  # base (fixed)
            0,                      # arm joint 1: 0 rad
            -np.pi/4                # arm joint 2: -45 deg
        ]),
        
        "Safe Folded Position": np.array([
            0, 0, 0, 0, 0, 0, 1,  # base (fixed)
            0,                      # arm joint 1
            -np.pi/2                # arm joint 2: -90 deg
        ]),
        
        "Reaching Forward": np.array([
            0, 0, 0, 0, 0, 0, 1,  # base (fixed)
            np.pi/4,                # arm joint 1: 45 deg
            -np.pi/3                # arm joint 2: -60 deg
        ]),
    }
    
    ee_frame_id = model.nframes - 2
    
    for config_name, q in configs_to_test.items():
        print(f"\nTesting: {config_name}")
        print(f"  Config: {q}")
        
        try:
            pin.forwardKinematics(model, robot.data, q)
            pin.updateFramePlacement(model, robot.data, ee_frame_id)
            
            ee_pos = robot.data.oMf[ee_frame_id].translation
            print(f"  ✓ EE Position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Detailed analysis of zero configuration
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: ALL-ZERO CONFIGURATION")
    print("=" * 80 + "\n")
    
    q_zero = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
    pin.forwardKinematics(model, robot.data, q_zero)
    
    print("Joint Positions for q=zeros(9):")
    for i in range(min(10, model.nframes)):
        pin.updateFramePlacement(model, robot.data, i)
        pos = robot.data.oMf[i].translation
        frame_name = model.frames[i].name
        print(f"  Frame {i:2d} ({frame_name:20s}): ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")
    
    # =========================================================================
    # ANALYSIS: WHY COLLISION DETECTION ISN'T WORKING
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("COLLISION DETECTION ANALYSIS")
    print("=" * 80 + "\n")
    
    print("ISSUE IDENTIFIED:")
    print("  ❌ Your code only checks EE (end-effector) vs obstacles")
    print("  ❌ With small workspace, EE might never hit obstacles")
    print("  ✅ Real robots need full robot link collision checking\n")
    
    # Test current collision detection
    print("Testing collision detection with sample obstacles:")
    
    # Obstacle very close to base
    obstacles = [
        (np.array([0.1, 0.0, 0.3]), 0.1),  # Close to base
        (np.array([0.5, 0.0, 0.3]), 0.1),  # In middle workspace
        (np.array([1.0, 0.0, 0.5]), 0.1),  # Far away
    ]
    
    pin.forwardKinematics(model, robot.data, q_zero)
    pin.updateFramePlacement(model, robot.data, ee_frame_id)
    ee_pos = robot.data.oMf[ee_frame_id].translation
    
    print(f"\nEE Position at q=0: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    print(f"EE to Obstacle distances:")
    
    for obs_center, obs_radius in obstacles:
        dist = np.linalg.norm(ee_pos - obs_center) - obs_radius
        collision = "❌ COLLISION" if dist < 0.05 else "✓ Clear"
        print(f"  Obs at {obs_center}: distance={dist:.3f}m {collision}")
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80 + "\n")
    
    print("1. HOME POSE SELECTION:")
    print("   - ❌ All zeros [0,0,0,0,0,0,1,0,0] may be singularity")
    print("   - ✅ Use Panda standard home: joint2 = -π/4 rad (-45°)")
    print("   - Suggested q_start:")
    
    q_home = np.array([0, 0, 0, 0, 0, 0, 1, 0, -np.pi/4])
    print(f"     {q_home}")
    
    pin.forwardKinematics(model, robot.data, q_home)
    pin.updateFramePlacement(model, robot.data, ee_frame_id)
    ee_pos_home = robot.data.oMf[ee_frame_id].translation
    print(f"   EE position at home: ({ee_pos_home[0]:.3f}, {ee_pos_home[1]:.3f}, {ee_pos_home[2]:.3f})")
    
    print(f"\n2. GOAL POSE SELECTION:")
    q_goal = np.array([0, 0, 0, 0, 0, 0, 1, 0.5, -1.0])
    print(f"   q_goal: {q_goal}")
    
    pin.forwardKinematics(model, robot.data, q_goal)
    pin.updateFramePlacement(model, robot.data, ee_frame_id)
    ee_pos_goal = robot.data.oMf[ee_frame_id].translation
    print(f"   EE position at goal: ({ee_pos_goal[0]:.3f}, {ee_pos_goal[1]:.3f}, {ee_pos_goal[2]:.3f})")
    
    print(f"\n3. COLLISION DETECTION FIX:")
    print(f"   ❌ Current code only checks EE")
    print(f"   ✅ Should check: base + all arm links")
    print(f"   → Use multiple frame points or convex hull")
    
    print(f"\n4. OBSTACLE PLACEMENT:")
    print(f"   ✅ Place obstacles BETWEEN start and goal EE positions")
    print(f"   Current EE range: [{ee_pos_home[0]:.2f}:{ee_pos_goal[0]:.2f}]")
    print(f"                     [{ee_pos_home[1]:.2f}:{ee_pos_goal[1]:.2f}]")
    print(f"                     [{ee_pos_home[2]:.2f}:{ee_pos_goal[2]:.2f}]")
    
    print(f"\n5. VISUALIZATION:")
    visualize_workspace(robot, model, q_home, q_goal, obstacles)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80 + "\n")
    
    print("Update test5_final_working.py with:")
    print("""
# In main() function:

q_start = np.array([0, 0, 0, 0, 0, 0, 1, 0, -np.pi/4])  # Home pose
q_goal = np.array([0, 0, 0, 0, 0, 0, 1, 0.5, -1.0])     # Goal pose

# Place obstacles BETWEEN start and goal
obstacles = [
    (np.array([0.3, 0.0, 0.4]), 0.08),   # In path
    (np.array([0.5, 0.2, 0.3]), 0.1),    # In path
]
""")


def visualize_workspace(robot, model, q_start, q_goal, obstacles):
    """Visualize start, goal, and obstacle positions"""
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ee_frame_id = model.nframes - 2
    
    # Start position
    pin.forwardKinematics(model, robot.data, q_start)
    pin.updateFramePlacement(model, robot.data, ee_frame_id)
    start_pos = robot.data.oMf[ee_frame_id].translation
    ax.scatter(*start_pos, s=200, c='green', marker='o', label='Start', zorder=5)
    
    # Goal position
    pin.forwardKinematics(model, robot.data, q_goal)
    pin.updateFramePlacement(model, robot.data, ee_frame_id)
    goal_pos = robot.data.oMf[ee_frame_id].translation
    ax.scatter(*goal_pos, s=200, c='red', marker='s', label='Goal', zorder=5)
    
    # Draw straight line from start to goal
    ax.plot([start_pos[0], goal_pos[0]], 
            [start_pos[1], goal_pos[1]], 
            [start_pos[2], goal_pos[2]], 
            'k--', linewidth=2, label='Direct path (no obstacles)', zorder=1)
    
    # Obstacles
    for obs_center, obs_radius in obstacles:
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_center[0]
        y = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_center[1]
        z = obs_radius * np.outer(np.ones(len(u)), np.cos(v)) + obs_center[2]
        ax.plot_surface(x, y, z, alpha=0.4, color='red', zorder=2)
    
    # Base location
    ax.scatter(0, 0, 0, s=100, c='blue', marker='^', label='Robot Base', zorder=5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Panda Workspace: Start → Goal with Obstacles')
    ax.legend()
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('workspace_analysis.png', dpi=150)
    print("✓ Saved workspace visualization to workspace_analysis.png")
    plt.show()


if __name__ == "__main__":
    find_panda_home_pose()