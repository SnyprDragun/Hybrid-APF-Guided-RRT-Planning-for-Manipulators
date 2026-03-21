#!/usr/bin/env python3
"""
PyBullet Simulation for APF-RRT* Panda Planner
Visualizes planning and execution in real-time
"""

import pybullet as p
import pybullet_data
import numpy as np
import pinocchio as pin
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


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
    """Pinocchio-based robot for planning"""
    
    def __init__(self, urdf_path: Path):
        package_dir = str(urdf_path.parent.parent)
        self.robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dir)
        self.model = self.robot.model
        self.data = self.robot.data
        self.nq = self.model.nq
        
        self.ee_frame_id = self._find_ee_frame()
        self.collision_check_frames = self._find_collision_frames()
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
        for frame_id, frame in enumerate(self.model.frames):
            name_lower = frame.name.lower()
            if any(kw in name_lower for kw in ['link', 'panda', 'arm']):
                if 'world' not in name_lower and 'finger' not in name_lower:
                    frames.append(frame_id)
        frames.append(self.ee_frame_id)
        return list(set(frames))[:10]
    
    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        limits = []
        for i in range(1, self.model.njoints):
            try:
                low = float(self.model.lowerPositionLimit[i-1])
                high = float(self.model.upperPositionLimit[i-1])
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
        return pin.getFrameJacobian(self.model, self.data, self.ee_frame_id,
                                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
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
        
        tree = [Node(q=q_start.copy(), cost=0.0)]
        best_cost = np.inf
        best_path = None
        metrics = {'iterations': 0, 'nodes_added': 0, 'collisions': 0, 'first_solution_iter': None}
        
        for iteration in range(self.config.max_iterations):
            if np.random.random() < self.config.goal_sample_rate:
                q_rand = q_goal.copy()
            else:
                # Biased sampling: prefer configs closer to goal
                q_rand = np.zeros(self.robot.nq)
                for i in range(self.robot.nq):
                    low, high = self.robot.joint_limits[i]
                    # 70% towards goal, 30% random
                    if np.random.random() < 0.7:
                        q_rand[i] = q_goal[i] + np.random.normal(0, 0.3)
                    else:
                        q_rand[i] = np.random.uniform(low, high)
                    q_rand[i] = np.clip(q_rand[i], low, high)
            
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
            
            if np.linalg.norm(q_new - q_goal) < 0.5:
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
        
        # If no path found, try simple linear interpolation as fallback
        if best_path is None:
            print("\nNo RRT path found. Trying linear interpolation fallback...")
            if self.robot.swept_volume_collision_free(q_start, q_goal, obstacles, num_samples=50):
                best_path = [q_start + (q_goal - q_start) * t for t in np.linspace(0, 1, 20)]
                print("✓ Linear path is collision-free!")
            else:
                print("✗ Even linear path has collisions")
        
        return best_path, tree, metrics
    
    def _extract_path(self, node: Node) -> List[np.ndarray]:
        path = []
        while node:
            path.append(node.q.copy())
            node = node.parent
        return path[::-1]


class PyBulletSimulator:
    """PyBullet simulation environment"""
    
    def __init__(self, gui: bool = True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load Panda
        self.panda_id = p.loadURDF(str(URDF_PATH), useFixedBase=True)
        
        # Get joint indices (skip fixed/universe joints)
        self.joint_ids = []
        for i in range(p.getNumJoints(self.panda_id)):
            info = p.getJointInfo(self.panda_id, i)
            if info[2] != p.JOINT_FIXED:  # Skip fixed joints
                self.joint_ids.append(i)
        
        print(f"PyBullet: Loaded Panda with {len(self.joint_ids)} active joints")
    
    def set_joint_angles(self, q: np.ndarray):
        """Set robot joint angles"""
        for i, joint_id in enumerate(self.joint_ids[:len(q)]):
            p.resetJointState(self.panda_id, joint_id, q[i])
    
    def add_obstacle(self, position: np.ndarray, radius: float, color=(1, 0, 0, 0.5)):
        """Add sphere obstacle"""
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                   baseVisualShapeIndex=visual,
                                   basePosition=position)
        return obj_id
    
    def visualize_path(self, path: List[np.ndarray], delay: float = 0.1):
        """Replay planned path"""
        print("\nReplaying path in simulation...")
        for i, q in enumerate(path):
            self.set_joint_angles(q)
            p.stepSimulation()
            time.sleep(delay)
            if (i + 1) % 5 == 0:
                print(f"  Waypoint {i+1}/{len(path)}")
        print("Path execution complete!")
    
    def close(self):
        p.disconnect()


def main():
    print("\n" + "="*70)
    print("APF-RRT* PANDA PLANNER WITH PYBULLET SIMULATION")
    print("="*70 + "\n")
    
    # Initialize planner
    robot = PandaRobotWrapper(URDF_PATH)
    config = PlannerConfig(max_iterations=3000)  # Reduced for faster planning
    planner = APFRRTStar(robot, config)
    
    # Define problem
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_goal = np.array([0.0, 1.0, -0.5, 0.5, -1.5, 0.3, 1.0, 0.0, 0.0])
    
    # No obstacles initially - let's find a valid path first
    obstacles = []
    
    print("Planning without obstacles...")
    ee_start = robot.forward_kinematics(q_start)
    ee_goal = robot.forward_kinematics(q_goal)
    
    print(f"Start EE: ({ee_start[0]:.3f}, {ee_start[1]:.3f}, {ee_start[2]:.3f})")
    print(f"Goal EE:  ({ee_goal[0]:.3f}, {ee_goal[1]:.3f}, {ee_goal[2]:.3f})\n")
    
    path, tree, metrics = planner.plan(q_start, q_goal, obstacles)
    
    if not path:
        print("❌ Planning failed even without obstacles!")
        return
    
    print(f"✓ Path found with {len(path)} waypoints\n")
    
    # Initialize PyBullet simulator
    print("Starting PyBullet simulation...")
    sim = PyBulletSimulator(gui=True)
    
    # Add some obstacles if you want
    # sim.add_obstacle(np.array([0.3, 0.0, 0.5]), 0.1)
    
    # Visualize path
    sim.visualize_path(path, delay=0.05)
    
    # Keep window open
    print("\nSimulation running. Press Ctrl+C to exit...")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("\nClosing simulation...")
    finally:
        sim.close()


if __name__ == "__main__":
    main()