#!/usr/bin/env python3
"""
APF-RRT* Planner - CORRECTED for Actual Panda Structure
Real arm joints are 1-7, not 7-8!
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


URDF_PATH = Path("/home/focaslab/ros2_ws/src/Hybrid-APF-Guided-RRT-Planning-for-Manipulators/model_description/panda_with_gripper.urdf")


@dataclass
class PlannerConfig:
    max_iterations: int = 5000
    goal_sample_rate: float = 0.15
    base_step: float = 0.3
    k_att: float = 1.2
    k_rep: float = 2.5
    rep_range: float = 0.5
    rewire_radius: float = 1.0
    collision_samples: int = 20


class PandaRobotWrapper:
    
    def __init__(self, urdf_path: Path):
        
        print(f"Loading URDF: {urdf_path}")
        
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        package_dir = str(urdf_path.parent.parent)
        self.robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dir)
        
        self.model = self.robot.model
        self.data = self.robot.data
        self.nq = self.model.nq
        
        print(f"✓ Loaded robot with {self.nq} DOF")
        print(f"  Joint structure:")
        for i in range(self.model.njoints):
            print(f"    [{i}] {self.model.names[i]}")
        
        # Find EE frame
        self.ee_frame_id = self._find_ee_frame()
        print(f"✓ End-effector: {self.model.frames[self.ee_frame_id].name}\n")
        
        # Collision check frames
        self.collision_check_frames = self._find_collision_frames()
        
        # Joint limits
        self.joint_limits = self._get_joint_limits()
    
    def _find_ee_frame(self) -> int:
        ee_patterns = ['hand', 'ee', 'tool', 'tcp', 'gripper', 'panda_hand']
        for pattern in ee_patterns:
            for frame_id, frame in enumerate(self.model.frames):
                if pattern.lower() in frame.name.lower():
                    return frame_id
        return self.model.nframes - 2
    
    def _find_collision_frames(self) -> List[int]:
        frames = []
        important_keywords = ['link', 'panda', 'arm']
        for frame_id, frame in enumerate(self.model.frames):
            name_lower = frame.name.lower()
            if any(kw in name_lower for kw in important_keywords):
                if 'world' not in name_lower and 'finger' not in name_lower:
                    frames.append(frame_id)
        frames.append(self.ee_frame_id)
        return list(set(frames))[:10]
    
    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        limits = []
        for i in range(1, self.model.njoints):
            try:
                if hasattr(self.model, 'lowerPositionLimit'):
                    low = float(self.model.lowerPositionLimit[i-1])
                    high = float(self.model.upperPositionLimit[i-1])
                else:
                    low, high = -np.pi, np.pi
            except:
                low, high = -np.pi, np.pi
            limits.append((low, high))
        return limits
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        if len(q) != self.nq:
            raise ValueError(f"Expected {self.nq} DOF, got {len(q)}")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return self.data.oMf[self.ee_frame_id].translation.copy()
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J
    
    def enforce_limits(self, q: np.ndarray) -> np.ndarray:
        q_safe = q.copy()
        for i, (low, high) in enumerate(self.joint_limits[:len(q)]):
            q_safe[i] = np.clip(q_safe[i], low, high)
        return q_safe
    
    def collision_check(self, q: np.ndarray,
                       obstacles: List[Tuple[np.ndarray, float]],
                       clearance: float = 0.08) -> bool:
        try:
            pin.forwardKinematics(self.model, self.data, q)
            for frame_id in self.collision_check_frames:
                pin.updateFramePlacement(self.model, self.data, frame_id)
                frame_pos = self.data.oMf[frame_id].translation
                for obs_center, obs_radius in obstacles:
                    dist = np.linalg.norm(frame_pos - obs_center)
                    if dist < obs_radius + clearance:
                        return False
            return True
        except:
            return False
    
    def swept_volume_collision_free(self, q1: np.ndarray, q2: np.ndarray,
                                   obstacles: List[Tuple[np.ndarray, float]],
                                   num_samples: int = 20) -> bool:
        for t in np.linspace(0, 1, num_samples):
            q_interp = (1 - t) * q1 + t * q2
            if not self.collision_check(q_interp, obstacles):
                return False
        return True


class AdaptiveAPF:
    
    def __init__(self, robot: PandaRobotWrapper, config: PlannerConfig):
        self.robot = robot
        self.config = config
    
    def workspace_distance(self, q: np.ndarray,
                          obstacles: List[Tuple[np.ndarray, float]]) -> Tuple[float, np.ndarray]:
        ee_pos = self.robot.forward_kinematics(q)
        min_dist = np.inf
        closest_grad = np.zeros(3)
        
        for obs_center, obs_radius in obstacles:
            vec = ee_pos - obs_center
            dist = np.linalg.norm(vec) - obs_radius
            if dist < min_dist:
                min_dist = dist
                if dist > 1e-6:
                    closest_grad = vec / np.linalg.norm(vec)
        
        return min_dist, closest_grad
    
    def compute_force(self, q: np.ndarray, q_goal: np.ndarray,
                     obstacles: List[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
        f_att = self.config.k_att * (q_goal - q)
        dist_obs, grad_obs = self.workspace_distance(q, obstacles)
        f_rep = np.zeros(self.robot.nq)
        
        if dist_obs < self.config.rep_range and dist_obs > 1e-6:
            mag = self.config.k_rep * (1.0/dist_obs - 1.0/self.config.rep_range)
            mag /= (dist_obs ** 2)
            try:
                J = self.robot.jacobian(q)
                J_pos = J[:3, :]
                J_pinv = np.linalg.pinv(J_pos)
                f_rep = J_pinv.T @ (mag * grad_obs)
            except:
                pass
        
        return f_att + f_rep, dist_obs


@dataclass
class Node:
    q: np.ndarray
    cost: float = 0.0
    parent: Optional['Node'] = None


class APFRRTStar:
    
    def __init__(self, robot: PandaRobotWrapper, config: PlannerConfig):
        self.robot = robot
        self.config = config
        self.apf = AdaptiveAPF(robot, config)
    
    def steer(self, q_from: np.ndarray, q_to: np.ndarray,
             obstacles: List[Tuple[np.ndarray, float]]) -> Tuple[bool, np.ndarray]:
        q_new = q_from.copy()
        for step in range(15):
            force, dist = self.apf.compute_force(q_new, q_to, obstacles)
            force_norm = np.linalg.norm(force) + 1e-8
            step_size = self.config.base_step * (1.0 + np.clip(dist, 0, 2))
            q_new = q_new + (step_size / force_norm) * force
            q_new = self.robot.enforce_limits(q_new)
            if not self.robot.collision_check(q_new, obstacles):
                return False, q_new
        return True, q_new
    
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray,
            obstacles: List[Tuple[np.ndarray, float]]) -> Tuple[Optional[List[np.ndarray]], List[Node], Dict]:
        
        if len(q_start) != self.robot.nq or len(q_goal) != self.robot.nq:
            raise ValueError(f"Expected {self.robot.nq} DOF")
        
        print(f"{'='*70}")
        print(f"APF-RRT* Planning")
        print(f"{'='*70}\n")
        
        tree = [Node(q=q_start.copy(), cost=0.0)]
        best_cost = np.inf
        best_path = None
        metrics = {'iterations': 0, 'nodes_added': 0, 'collisions': 0, 'first_solution_iter': None}
        
        for iteration in range(self.config.max_iterations):
            if np.random.random() < self.config.goal_sample_rate:
                q_rand = q_goal.copy()
            else:
                q_rand = np.zeros(self.robot.nq)
                q_rand[0] = 0  # universe joint
                for i in range(1, self.robot.nq):
                    low, high = self.robot.joint_limits[i-1]
                    q_rand[i] = np.random.uniform(low, high)
            
            dists = [np.linalg.norm(n.q - q_rand) for n in tree]
            nearest_idx = np.argmin(dists)
            
            success, q_new = self.steer(tree[nearest_idx].q, q_rand, obstacles)
            if not success:
                metrics['collisions'] += 1
                continue
            
            new_node = Node(q=q_new.copy())
            radius = self.config.rewire_radius / np.sqrt(len(tree) + 1)
            neighbors = [n for n in tree
                        if np.linalg.norm(n.q - q_new) < radius
                        and self.robot.swept_volume_collision_free(n.q, q_new, obstacles)]
            
            best_parent = tree[nearest_idx]
            best_cost_temp = best_parent.cost + np.linalg.norm(best_parent.q - q_new)
            
            for neighbor in neighbors:
                cost = neighbor.cost + np.linalg.norm(neighbor.q - q_new)
                if cost < best_cost_temp:
                    best_cost_temp = cost
                    best_parent = neighbor
            
            new_node.parent = best_parent
            new_node.cost = best_cost_temp
            tree.append(new_node)
            metrics['nodes_added'] += 1
            
            for neighbor in neighbors:
                cost = new_node.cost + np.linalg.norm(new_node.q - neighbor.q)
                if cost < neighbor.cost:
                    neighbor.parent = new_node
                    neighbor.cost = cost
            
            if np.linalg.norm(q_new - q_goal) < 0.3:
                goal_node = Node(q=q_goal.copy())
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + np.linalg.norm(q_new - q_goal)
                tree.append(goal_node)
                
                path = self._extract_path(goal_node)
                if goal_node.cost < best_cost:
                    best_cost = goal_node.cost
                    best_path = path
                    if metrics['first_solution_iter'] is None:
                        metrics['first_solution_iter'] = iteration
                    print(f"[{iteration}] ✓ Found path! Cost: {best_cost:.3f}")
            
            if (iteration + 1) % 500 == 0:
                print(f"[{iteration+1}] Nodes: {len(tree)}, Collisions: {metrics['collisions']}")
            
            metrics['iterations'] = iteration
        
        print(f"\n{'='*70}")
        if best_path:
            print(f"✓ SUCCESS: Found path with {len(best_path)} waypoints")
            print(f"  Cost: {best_cost:.3f}")
        else:
            print(f"❌ NO PATH FOUND")
            print(f"  Collisions: {metrics['collisions']}/{metrics['nodes_added']}")
        print(f"{'='*70}\n")
        
        return best_path, tree, metrics
    
    def _extract_path(self, node: Node) -> List[np.ndarray]:
        path = []
        while node:
            path.append(node.q.copy())
            node = node.parent
        return path[::-1]


def visualize(robot: PandaRobotWrapper, path: Optional[List[np.ndarray]],
             obstacles: List[Tuple[np.ndarray, float]]):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if path is None:
        print("No path to visualize")
        return
    
    try:
        path_array = np.array([robot.forward_kinematics(q) for q in path])
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                'b-', linewidth=2, label='Path')
        ax.scatter(*path_array[0], s=100, c='green', marker='o', label='Start')
        ax.scatter(*path_array[-1], s=100, c='red', marker='s', label='Goal')
    except Exception as e:
        print(f"Visualization error: {e}")
        return
    
    for obs_center, obs_radius in obstacles:
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_center[0]
        y = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_center[1]
        z = obs_radius * np.outer(np.ones(len(u)), np.cos(v)) + obs_center[2]
        ax.plot_surface(x, y, z, alpha=0.3, color='red')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('APF-RRT* Path')
    ax.legend()
    plt.tight_layout()
    plt.savefig('path_visualization.png', dpi=150)
    print("✓ Saved to path_visualization.png")
    plt.show()


def main():
    
    print("\n" + "="*70)
    print("APF-RRT* PLANNER - PANDA ARM")
    print("="*70 + "\n")
    
    try:
        robot = PandaRobotWrapper(URDF_PATH)
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    config = PlannerConfig()
    
    # ========================================================================
    # REAL ARM JOINTS ARE 1-7, NOT 7-8!
    # Based on your output:
    # - Joint 0: universe (rotation, nq=1)
    # - Joints 1-7: arm joints (the real Panda 7-DOF arm)
    # - Joints 8-9: gripper (doesn't affect EE position)
    # ========================================================================
    
    # Home pose (9 DOF: universe is index 0, then panda_joint1-7, then gripper)
    q_start = np.array([
        0.0,          # universe joint (nq=1)
        0.0,          # panda_joint1
        0.0,          # panda_joint2
        0.0,          # panda_joint3
        0.0,          # panda_joint4
        0.0,          # panda_joint5
        0.0,          # panda_joint6
        0.0,          # panda_joint7
        0.0           # panda_finger_joint1 (gripper - don't care)
    ])
    
    # Goal: move the arm to a different configuration
    # From your output, joint 2 and joint 4 have the most effect on X,Y movement
    q_goal = np.array([
        0.0,          # universe
        1.0,          # panda_joint1 (rotate base)
        -0.5,         # panda_joint2 (big effect!)
        0.5,          # panda_joint3
        -1.5,         # panda_joint4 (big effect!)
        0.3,          # panda_joint5
        1.0,          # panda_joint6
        0.0,          # panda_joint7
        0.0           # panda_finger_joint1 (gripper)
    ])
    
    # Compute EE positions
    ee_start = robot.forward_kinematics(q_start)
    ee_goal = robot.forward_kinematics(q_goal)
    
    print(f"Start config: joints 1-7 = {q_start[1:8]}")
    print(f"  EE: ({ee_start[0]:.3f}, {ee_start[1]:.3f}, {ee_start[2]:.3f})")
    print(f"\nGoal config: joints 1-7 = {q_goal[1:8]}")
    print(f"  EE: ({ee_goal[0]:.3f}, {ee_goal[1]:.3f}, {ee_goal[2]:.3f})")
    print(f"\nEE displacement: ({ee_goal[0]-ee_start[0]:+.3f}, {ee_goal[1]-ee_start[1]:+.3f}, {ee_goal[2]-ee_start[2]:+.3f})\n")
    
    # Obstacles between start and goal
    obstacles = [
        (np.array([0.2, 0.5, 0.8]), 0.1),
        (np.array([0.0, 1.0, 0.8]), 0.1),
    ]
    
    # Plan
    planner = APFRRTStar(robot, config)
    path, tree, metrics = planner.plan(q_start, q_goal, obstacles)
    
    if path:
        visualize(robot, path, obstacles)
    else:
        print(f"Start is collision-free: {robot.collision_check(q_start, obstacles)}")
        print(f"Goal is collision-free: {robot.collision_check(q_goal, obstacles)}")


if __name__ == "__main__":
    main()