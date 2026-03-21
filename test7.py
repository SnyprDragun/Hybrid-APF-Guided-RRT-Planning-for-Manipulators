#!/usr/bin/env python3
"""
APF-RRT* with Kinetostatic Danger Field for Arm Link Obstacle Avoidance
Implements the Danger Field approach from Lacevic et al. (IEEE TRO 2013)
for real-time collision avoidance of the entire robot arm
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
class DangerFieldParams:
    """Danger field parameters from paper"""
    k1: float = 1.0      # Distance-based repulsion coefficient
    k2: float = 0.25     # Velocity-based repulsion coefficient
    gamma: float = 1.0   # Velocity direction weight
    rep_range: float = 0.5  # Repulsion influence radius


@dataclass
class PlannerConfig:
    max_iterations: int = 3000
    goal_sample_rate: float = 0.15
    base_step: float = 0.3
    k_att: float = 1.2
    k_rep: float = 2.5
    rep_range: float = 0.5
    rewire_radius: float = 1.0
    collision_samples: int = 20


class DangerField:
    """
    Implements Kinetostatic Danger Field (KSDF) from Lacevic et al.
    KSDF captures both position and velocity of robot links
    """
    
    def __init__(self, robot_model, params: DangerFieldParams):
        self.model = robot_model.model
        self.data = robot_model.data
        self.params = params
        self.robot = robot_model
    
    def elementary_ksdf(self, r: np.ndarray, r_t: np.ndarray, 
                        v_t: np.ndarray) -> float:
        """
        Equation (11) from paper: Elementary KSDF
        DF(r, r_t, v_t) = k1 / ||r - r_t|| + 
                          k2 * ||v_t|| * [γ + cos(∠(r - r_t, v_t))] / ||r - r_t||²
        
        r: point in space where field is evaluated
        r_t: position of moving element (link point)
        v_t: velocity of moving element
        """
        
        rho = r - r_t  # Vector from moving element to point of interest
        rho_norm = np.linalg.norm(rho) + 1e-6
        
        if rho_norm < 1e-6:
            return 1e6  # Avoid singularity
        
        # Distance-based term (always repulsive)
        distance_term = self.params.k1 / rho_norm
        
        # Velocity-based term
        v_norm = np.linalg.norm(v_t)
        velocity_term = 0.0
        
        if v_norm > 1e-6:
            # Angle between (r - r_t) and v_t
            cos_angle = np.dot(rho, v_t) / (rho_norm * v_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
            
            # Equation (11): velocity component
            velocity_term = (self.params.k2 * v_norm * 
                           (self.params.gamma + cos_angle) / (rho_norm ** 2))
        
        return distance_term + velocity_term
    
    def cumulative_ksdf_link(self, r: np.ndarray, link_pos: np.ndarray,
                            link_vel: np.ndarray, next_pos: np.ndarray,
                            next_vel: np.ndarray) -> float:
        """
        Cumulative KSDF for a single link (line segment)
        Integrates danger field along the link from current to next joint
        Numerical approximation of equation (2) and (12)
        """
        
        cumulative_df = 0.0
        num_samples = 5  # Number of points along the link
        
        for s in np.linspace(0, 1, num_samples):
            # Interpolate position and velocity along link
            r_s = link_pos + s * (next_pos - link_pos)
            v_s = link_vel + s * (next_vel - link_vel)
            
            # Add elementary KSDF contribution
            cumulative_df += self.elementary_ksdf(r, r_s, v_s)
        
        return cumulative_df / num_samples
    
    def total_danger_field(self, q: np.ndarray, q_dot: np.ndarray,
                          r: np.ndarray) -> float:
        """
        Total cumulative KSDF for entire robot arm
        Equation (9): CDF(r) = Σ_i CDF_i(r)
        
        Evaluates danger at point r due to entire moving kinematic chain
        """
        
        # Forward kinematics and velocities
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        
        total_danger = 0.0
        
        # Iterate through each link
        for i in range(1, self.model.njoints - 2):  # Skip universe, gripper joints
            # Get link position and velocity
            pin.updateFramePlacement(self.model, self.data, i)
            link_pos = self.data.oMi[i].translation.copy()
            
            # Next joint position
            if i + 1 < self.model.njoints:
                pin.updateFramePlacement(self.model, self.data, i + 1)
                next_pos = self.data.oMi[i + 1].translation.copy()
            else:
                next_pos = link_pos.copy()
            
            # Compute link velocities using Jacobian
            J = pin.getFrameJacobian(self.model, self.data, i,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            link_vel = J[:3, :] @ q_dot
            
            if i + 1 < self.model.njoints:
                J_next = pin.getFrameJacobian(self.model, self.data, i + 1,
                                              pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                next_vel = J_next[:3, :] @ q_dot
            else:
                next_vel = link_vel.copy()
            
            # Cumulative danger from this link
            link_danger = self.cumulative_ksdf_link(r, link_pos, link_vel,
                                                   next_pos, next_vel)
            total_danger += link_danger
        
        return total_danger
    
    def danger_field_gradient(self, q: np.ndarray, q_dot: np.ndarray,
                             r: np.ndarray, delta: float = 1e-5) -> np.ndarray:
        """
        Numerical gradient of danger field in workspace
        Used for computing repulsive force
        """
        
        grad = np.zeros(3)
        
        for i in range(3):
            r_plus = r.copy()
            r_plus[i] += delta
            
            r_minus = r.copy()
            r_minus[i] -= delta
            
            df_plus = self.total_danger_field(q, q_dot, r_plus)
            df_minus = self.total_danger_field(q, q_dot, r_minus)
            
            grad[i] = (df_plus - df_minus) / (2 * delta)
        
        return grad


class PandaRobotWrapper:
    
    def __init__(self, urdf_path: Path):
        package_dir = str(urdf_path.parent.parent)
        self.robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dir)
        self.model = self.robot.model
        self.data = self.robot.data
        self.nq = self.model.nq
        self.ee_frame_id = self._find_ee_frame()
        self.joint_limits = self._get_joint_limits()
    
    def _find_ee_frame(self) -> int:
        ee_patterns = ['hand', 'ee', 'tool', 'tcp', 'gripper', 'panda_hand']
        for pattern in ee_patterns:
            for frame_id, frame in enumerate(self.model.frames):
                if pattern.lower() in frame.name.lower():
                    return frame_id
        return self.model.nframes - 2
    
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
    
    def collision_check_danger_field(self, q: np.ndarray, q_dot: np.ndarray,
                                    danger_field: 'DangerField',
                                    threshold: float = 5.0) -> bool:
        """
        Check collision using danger field threshold
        Returns False if danger exceeds threshold (collision)
        """
        
        ee_pos = self.forward_kinematics(q)
        danger = danger_field.total_danger_field(q, q_dot, ee_pos)
        return danger < threshold
    
    def swept_volume_collision_free(self, q1: np.ndarray, q2: np.ndarray,
                                   danger_field: 'DangerField',
                                   threshold: float = 5.0,
                                   num_samples: int = 20) -> bool:
        """Check collision along trajectory using danger field"""
        
        for t in np.linspace(0, 1, num_samples):
            q_interp = (1 - t) * q1 + t * q2
            # Approximate velocity
            q_dot = (q2 - q1) / max(num_samples * 0.01, 0.01)
            
            if not self.collision_check_danger_field(q_interp, q_dot, 
                                                    danger_field, threshold):
                return False
        return True


class AdaptiveAPFDangerField:
    """APF with Danger Field integration"""
    
    def __init__(self, robot: PandaRobotWrapper, config: PlannerConfig,
                danger_field: DangerField):
        self.robot = robot
        self.config = config
        self.danger_field = danger_field
    
    def compute_force(self, q: np.ndarray, q_dot: np.ndarray,
                     q_goal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute APF force with danger field repulsion
        Combines:
        1. Attractive force to goal
        2. Repulsive force from danger field
        """
        
        # Attractive force
        f_att = self.config.k_att * (q_goal - q)
        
        # Danger field repulsion in workspace
        ee_pos = self.robot.forward_kinematics(q)
        danger = self.danger_field.total_danger_field(q, q_dot, ee_pos)
        
        f_rep = np.zeros(self.robot.nq)
        
        if danger > 0:
            # Get danger field gradient
            grad_danger = self.danger_field.danger_field_gradient(q, q_dot, ee_pos)
            
            # Map workspace gradient to joint-space via Jacobian
            try:
                J = self.robot.jacobian(q)
                J_pos = J[:3, :]
                J_pinv = np.linalg.pinv(J_pos)
                
                # Repulsive force magnitude increases with danger level
                repulsion_magnitude = self.config.k_rep * danger
                f_rep = repulsion_magnitude * J_pinv.T @ grad_danger
            except:
                pass
        
        return f_att + f_rep, danger


@dataclass
class Node:
    q: np.ndarray
    cost: float = 0.0
    parent: Optional['Node'] = None


class APFRRTStarDangerField:
    """RRT* with Danger Field collision avoidance"""
    
    def __init__(self, robot: PandaRobotWrapper, config: PlannerConfig,
                df_params: DangerFieldParams):
        self.robot = robot
        self.config = config
        self.danger_field = DangerField(robot, df_params)
        self.apf = AdaptiveAPFDangerField(robot, config, self.danger_field)
        self.danger_threshold = 5.0
    
    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Steer with danger field guidance"""
        
        q_new = q_from.copy()
        q_dot = np.zeros(self.robot.nq)
        
        for step in range(15):
            force, danger = self.apf.compute_force(q_new, q_dot, q_to)
            
            # Stop if danger is too high
            if danger > self.danger_threshold * 1.5:
                return False, q_new
            
            force_norm = np.linalg.norm(force) + 1e-8
            step_size = self.config.base_step * (1.0 + np.clip(danger / 10, 0, 2))
            
            q_new = q_new + (step_size / force_norm) * force
            q_new = self.robot.enforce_limits(q_new)
            q_dot = (q_new - q_from) / max(step * 0.1, 0.01)
            
            # Check if collision according to danger field
            if not self.robot.collision_check_danger_field(q_new, q_dot, 
                                                          self.danger_field,
                                                          self.danger_threshold):
                return False, q_new
        
        return True, q_new
    
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Tuple[Optional[List[np.ndarray]], List[Node], Dict]:
        """Main planning loop"""
        
        print(f"\n{'='*70}")
        print(f"APF-RRT* with Danger Field Collision Avoidance")
        print(f"{'='*70}\n")
        
        tree = [Node(q=q_start.copy(), cost=0.0)]
        best_cost = np.inf
        best_path = None
        metrics = {'iterations': 0, 'nodes_added': 0, 'danger_violations': 0, 
                  'first_solution_iter': None}
        
        for iteration in range(self.config.max_iterations):
            # Sample
            if np.random.random() < self.config.goal_sample_rate:
                q_rand = q_goal.copy()
            else:
                q_rand = np.zeros(self.robot.nq)
                for i in range(self.robot.nq):
                    low, high = self.robot.joint_limits[i]
                    q_rand[i] = np.random.uniform(low, high)
            
            # Nearest
            dists = [np.linalg.norm(n.q - q_rand) for n in tree]
            nearest_idx = np.argmin(dists)
            
            # Steer
            success, q_new = self.steer(tree[nearest_idx].q, q_rand)
            if not success:
                metrics['danger_violations'] += 1
                continue
            
            # Add node
            new_node = Node(q=q_new.copy())
            
            # Rewire
            radius = self.config.rewire_radius / np.sqrt(len(tree) + 1)
            neighbors = [n for n in tree
                        if np.linalg.norm(n.q - q_new) < radius
                        and self.robot.swept_volume_collision_free(
                            n.q, q_new, self.danger_field, self.danger_threshold)]
            
            # Best parent
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
            
            # Rewiring
            for neighbor in neighbors:
                cost = new_node.cost + np.linalg.norm(new_node.q - neighbor.q)
                if cost < neighbor.cost:
                    neighbor.parent = new_node
                    neighbor.cost = cost
            
            # Goal check
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
                print(f"[{iteration+1}] Nodes: {len(tree)}, Danger violations: {metrics['danger_violations']}")
            
            metrics['iterations'] = iteration
        
        print(f"\n{'='*70}")
        if best_path:
            print(f"✓ SUCCESS: Found path with {len(best_path)} waypoints")
            print(f"  Cost: {best_cost:.3f}")
        else:
            print(f"❌ NO PATH FOUND")
            print(f"  Danger violations: {metrics['danger_violations']}/{metrics['nodes_added']}")
        print(f"{'='*70}\n")
        
        return best_path, tree, metrics
    
    def _extract_path(self, node: Node) -> List[np.ndarray]:
        path = []
        while node:
            path.append(node.q.copy())
            node = node.parent
        return path[::-1]


def main():
    """Main execution"""
    
    # Load robot
    robot = PandaRobotWrapper(URDF_PATH)
    
    # Configuration
    config = PlannerConfig(max_iterations=2000)
    df_params = DangerFieldParams(k1=1.0, k2=0.25, gamma=1.0, rep_range=0.5)
    
    # Define problem
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_goal = np.array([0.0, 1.0, -0.5, 0.5, -1.5, 0.3, 1.0, 0.0, 0.0])
    
    # No obstacles - danger field handles arm self-awareness
    
    # Plan
    planner = APFRRTStarDangerField(robot, config, df_params)
    path, tree, metrics = planner.plan(q_start, q_goal)
    
    if path:
        print(f"✓ Path found with {len(path)} waypoints")
        print(f"  Using Danger Field for entire arm link avoidance")
    else:
        print(f"❌ Planning failed")


if __name__ == "__main__":
    main()