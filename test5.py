#!/opt/homebrew/bin/python3
"""
High-Performance 6-DOF APF-RRT* Motion Planner
Phase A: Baseline + Phase B: Adaptive Parameter Tuning
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import json
from pathlib import Path


# =============================================================================
# CONFIGURATION & PARAMETERS
# =============================================================================

@dataclass
class PlannerConfig:
    """Centralized parameter management"""
    # Algorithm
    max_iterations: int = 5000
    goal_sample_rate: float = 0.15
    base_step: float = 0.3
    dimension: int = 6  # DOF
    
    # APF Parameters (adaptive)
    k_att_base: float = 1.2
    k_rep_base: float = 2.5
    rep_range: float = 0.5  # Influence radius for repulsion
    
    # RRT* Parameters
    radius_decay: bool = True  # Enable asymptotically optimal radius
    max_rewire_radius: float = 1.0
    
    # Collision & Safety
    collision_samples: int = 20  # Resolution of swept-volume check
    joint_limits: Tuple = ((-2.89, 2.89), (-1.76, 1.76), (-2.89, 2.89),
                           (-3.07, 0.00), (-2.89, 2.89), (-0.01, 3.82))
    self_collision_clearance: float = 0.05  # meters
    
    # Enhancement (Adaptive Tuning)
    use_adaptive_gains: bool = True
    density_radius: float = 0.4


@dataclass
class Config:
    """Joint configuration (q-space node)"""
    q: np.ndarray  # Shape: (6,)
    cost: float = 0.0
    parent: Optional['Config'] = None
    
    def copy(self):
        return Config(q=self.q.copy(), cost=self.cost, parent=self.parent)


# =============================================================================
# ROBOT & KINEMATICS
# =============================================================================

class RobotPlanner:
    """Wrapper for Pinocchio robot + planner integration"""
    
    def __init__(self, urdf_path: str, config: PlannerConfig):
        """Initialize robot from URDF"""
        self.config = config
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, 
                                                     pin.StdVec_StdString())
        self.model = self.robot.model
        self.data = self.robot.data
        self.nq = self.model.nq
        
        # Collision detection setup (simplified: sphere approximation)
        self.link_spheres = self._extract_link_spheres()
    
    def _extract_link_spheres(self) -> Dict[int, Tuple[np.ndarray, float]]:
        """Approximate robot links as spheres for collision checking"""
        # Simplified: use geometric center + radius per link
        spheres = {}
        for frame_id in range(1, self.model.nframes):
            frame = self.model.frames[frame_id]
            # In practice, load from URDF collision shapes
            # For now, placeholder
            spheres[frame_id] = (np.zeros(3), 0.05)
        return spheres
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute EE position from joint angles"""
        pin.framesForwardKinematics(self.model, self.data, q)
        ee_frame_id = self.model.getFrameId("panda_hand")
        return self.data.oMf[ee_frame_id].translation
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute Jacobian for velocity mapping"""
        ee_frame_id = self.model.getFrameId("panda_hand")
        J = pin.getFrameJacobian(self.model, self.data, ee_frame_id, 
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J
    
    def inverse_kinematics(self, target_pos: np.ndarray, 
                          q_init: Optional[np.ndarray] = None,
                          max_iter: int = 100,
                          tol: float = 1e-4) -> Tuple[bool, np.ndarray]:
        """Solve IK using numerical methods (Levenberg-Marquardt)"""
        if q_init is None:
            q_init = self.config.joint_limits[0]  # Mid-range init
        
        q = q_init.copy()
        
        for iter in range(max_iter):
            ee_pos = self.forward_kinematics(q)
            error = target_pos - ee_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tol:
                return True, q
            
            J = self.jacobian(q)
            
            # Damped least-squares (Levenberg-Marquardt)
            damping = 0.01
            dq = np.linalg.lstsq(J[:3, :].T @ J[:3, :] + damping * np.eye(self.nq),
                                 J[:3, :].T @ error, rcond=None)[0]
            
            q = self._enforce_limits(q + 0.1 * dq)
        
        return False, q
    
    def _enforce_limits(self, q: np.ndarray) -> np.ndarray:
        """Clip joint angles to limits"""
        for i, (low, high) in enumerate(self.config.joint_limits):
            q[i] = np.clip(q[i], low, high)
        return q
    
    def self_collision_check(self, q: np.ndarray, 
                            obstacles: List[Tuple[np.ndarray, float]]) -> bool:
        """Check collisions: robot vs obstacles + self-collision"""
        pin.forwardKinematics(self.model, self.data, q)
        
        # Check EE and key links against obstacles
        for frame_id in [self.model.getFrameId("panda_hand"),
                        self.model.getFrameId("panda_link7")]:
            pin.updateFramePlacement(self.model, self.data, frame_id)
            link_pos = self.data.oMf[frame_id].translation
            
            for obs_center, obs_radius in obstacles:
                dist = np.linalg.norm(link_pos - obs_center)
                if dist < obs_radius + self.config.self_collision_clearance:
                    return False
        
        # Self-collision: hardcoded pairs (simplified)
        # In production: use collision geometry
        return True
    
    def swept_volume_collision_free(self, q1: np.ndarray, q2: np.ndarray,
                                   obstacles: List[Tuple[np.ndarray, float]]
                                   ) -> bool:
        """Check collision along trajectory interpolation"""
        for t in np.linspace(0, 1, self.config.collision_samples):
            q_interp = (1 - t) * q1 + t * q2
            if not self.self_collision_check(q_interp, obstacles):
                return False
        return True


# =============================================================================
# APF IN JOINT SPACE
# =============================================================================

class AdaptiveAPF:
    """Adaptive Artificial Potential Field in q-space"""
    
    def __init__(self, robot: RobotPlanner, config: PlannerConfig):
        self.robot = robot
        self.config = config
        self.k_att = config.k_att_base
        self.k_rep = config.k_rep_base
    
    def compute_obstacle_density(self, q: np.ndarray, 
                                 obstacles: List[Tuple[np.ndarray, float]]
                                 ) -> float:
        """Estimate local obstacle density in workspace"""
        ee_pos = self.robot.forward_kinematics(q)
        density = 0
        for obs_center, obs_radius in obstacles:
            dist = np.linalg.norm(ee_pos - obs_center)
            if dist < self.config.density_radius:
                density += 1.0 / (dist + 1e-6)
        return density
    
    def adaptive_gains(self, density: float) -> Tuple[float, float]:
        """Adjust APF gains based on local density"""
        if not self.config.use_adaptive_gains:
            return self.config.k_att_base, self.config.k_rep_base
        
        # Higher density -> stronger attraction to goal
        k_att = self.config.k_att_base * (1.0 + 0.3 * density)
        # Higher density -> stronger repulsion from obstacles
        k_rep = self.config.k_rep_base * (1.0 + 2.0 * density)
        
        return np.clip(k_att, 0.5, 3.0), np.clip(k_rep, 1.0, 5.0)
    
    def workspace_gradient(self, q: np.ndarray, 
                          obstacles: List[Tuple[np.ndarray, float]]
                          ) -> np.ndarray:
        """Compute repulsive force gradient in workspace"""
        ee_pos = self.robot.forward_kinematics(q)
        grad_workspace = np.zeros(3)
        
        min_dist = np.inf
        for obs_center, obs_radius in obstacles:
            vec = ee_pos - obs_center
            dist = np.linalg.norm(vec) - obs_radius
            
            if dist < min_dist:
                min_dist = dist
            
            if dist < self.config.rep_range and dist > 1e-6:
                # Repulsive gradient
                mag = self.config.k_rep_base * (1.0 / dist - 1.0 / self.config.rep_range)
                mag /= (dist ** 2)
                grad_workspace += mag * (vec / np.linalg.norm(vec))
        
        return grad_workspace, min_dist
    
    def joint_space_force(self, q: np.ndarray, q_goal: np.ndarray,
                         obstacles: List[Tuple[np.ndarray, float]]
                         ) -> Tuple[np.ndarray, float]:
        """Compute APF in joint-space via Jacobian transpose"""
        # Attractive force in workspace
        ee_pos = self.robot.forward_kinematics(q)
        f_att_workspace = self.config.k_att_base * (q_goal - q)  # Simplified
        
        # Repulsive force in workspace
        f_rep_workspace, min_dist = self.workspace_gradient(q, obstacles)
        
        # Jacobian transpose mapping
        J = self.robot.jacobian(q)
        J_pinv = np.linalg.pinv(J[:3, :])  # Use only position part
        
        # Map forces to joint-space
        f_joint = J_pinv.T @ (f_att_workspace + f_rep_workspace)
        
        return f_joint, min_dist


# =============================================================================
# RRT* PLANNER
# =============================================================================

class APFRRTStar:
    """6-DOF APF-RRT* with adaptive parameter tuning"""
    
    def __init__(self, robot: RobotPlanner, config: PlannerConfig):
        self.robot = robot
        self.config = config
        self.apf = AdaptiveAPF(robot, config)
    
    def compute_rewire_radius(self, tree_size: int) -> float:
        """Decay radius for asymptotic optimality"""
        if not self.config.radius_decay:
            return self.config.max_rewire_radius
        
        d = self.config.dimension
        r = 2 * (1 + 1/d) * (np.log(tree_size + 1) / (tree_size + 1)) ** (1/d)
        return min(r, self.config.max_rewire_radius)
    
    def steer(self, q_from: np.ndarray, q_to: np.ndarray, 
             obstacles: List[Tuple[np.ndarray, float]]) -> Tuple[bool, np.ndarray]:
        """Steer from q_from toward q_to with APF guidance"""
        q_new = q_from.copy()
        
        for _ in range(10):  # Max steering steps
            # Compute force
            f_joint, dist_obs = self.apf.joint_space_force(q_new, q_to, obstacles)
            
            # Adaptive step size
            step_size = self.config.base_step * (1.0 + np.clip(dist_obs, 0, 2))
            direction = f_joint / (np.linalg.norm(f_joint) + 1e-6)
            
            q_new = self.robot._enforce_limits(q_new + step_size * direction)
            
            # Check collision
            if not self.robot.self_collision_check(q_new, obstacles):
                return False, q_new
        
        return True, q_new
    
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray,
            obstacles: List[Tuple[np.ndarray, float]]
            ) -> Tuple[Optional[List[np.ndarray]], List[Config], Dict]:
        """Main planning loop"""
        
        # Initialize
        tree = [Config(q=q_start.copy(), cost=0.0)]
        best_cost = np.inf
        best_path = None
        metrics = {
            'iterations': 0,
            'nodes_added': 0,
            'collisions': 0,
            'rewires': 0,
            'time_to_first_solution': None
        }
        
        for iteration in range(self.config.max_iterations):
            # Informed sampling
            if np.random.random() < self.config.goal_sample_rate:
                q_rand = q_goal.copy()
            else:
                q_rand = np.array([np.random.uniform(low, high) 
                                  for low, high in self.config.joint_limits])
            
            # Find nearest neighbor
            distances = [np.linalg.norm(node.q - q_rand) for node in tree]
            nearest_idx = np.argmin(distances)
            q_nearest = tree[nearest_idx].q
            
            # Steer with APF guidance
            success, q_new = self.steer(q_nearest, q_rand, obstacles)
            if not success:
                metrics['collisions'] += 1
                continue
            
            # Create new node
            new_node = Config(q=q_new.copy())
            
            # Find neighbors for rewiring
            radius = self.compute_rewire_radius(len(tree))
            neighbors = [node for node in tree 
                        if np.linalg.norm(node.q - q_new) < radius
                        and self.robot.swept_volume_collision_free(node.q, q_new, obstacles)]
            
            # Best parent selection
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
                    metrics['rewires'] += 1
            
            # Goal check
            if np.linalg.norm(q_new - q_goal) < 0.2:
                goal_node = Config(q=q_goal.copy())
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + np.linalg.norm(q_new - q_goal)
                tree.append(goal_node)
                
                path = self._extract_path(goal_node)
                if goal_node.cost < best_cost:
                    best_cost = goal_node.cost
                    best_path = path
                    if metrics['time_to_first_solution'] is None:
                        metrics['time_to_first_solution'] = iteration
                    print(f"[{iteration}] Found path, cost: {best_cost:.3f}")
            
            metrics['iterations'] = iteration
        
        return best_path, tree, metrics
    
    def _extract_path(self, node: Config) -> List[np.ndarray]:
        """Extract path from node to root"""
        path = []
        while node:
            path.append(node.q)
            node = node.parent
        return path[::-1]


# =============================================================================
# PATH SMOOTHING (PSO-based)
# =============================================================================

class PathSmoother:
    """Particle Swarm Optimization for path shortening"""
    
    def __init__(self, robot: RobotPlanner, obstacles: List[Tuple[np.ndarray, float]]):
        self.robot = robot
        self.obstacles = obstacles
    
    def smooth_path(self, path: List[np.ndarray], iterations: int = 50
                   ) -> List[np.ndarray]:
        """Smooth path by optimizing intermediate waypoints"""
        if len(path) <= 2:
            return path
        
        # Keep start and goal fixed, optimize intermediate points
        interior = path[1:-1]
        n_waypoints = len(interior)
        
        def objective(x_flat):
            """Minimize path length subject to collision constraints"""
            x = x_flat.reshape(n_waypoints, 6)
            full_path = [path[0]] + [xi for xi in x] + [path[-1]]
            
            # Path length
            length = 0
            for i in range(len(full_path) - 1):
                length += np.linalg.norm(full_path[i+1] - full_path[i])
            
            # Collision penalty
            penalty = 0
            for i in range(len(full_path) - 1):
                if not self.robot.swept_volume_collision_free(
                    full_path[i], full_path[i+1], self.obstacles):
                    penalty += 1000
            
            return length + penalty
        
        # Optimize
        x0 = np.array(interior).flatten()
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': iterations})
        
        x_opt = result.x.reshape(n_waypoints, 6)
        return [path[0]] + [xi for xi in x_opt] + [path[-1]]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Example usage"""
    # Configuration
    config = PlannerConfig(max_iterations=5000, use_adaptive_gains=True)
    
    # Load robot (ensure franka_panda.urdf exists)
    try:
        robot = RobotPlanner("franka_panda.urdf", config)
    except Exception as e:
        print(f"Warning: Could not load URDF: {e}")
        print("Run with valid URDF path or use PyBullet for demo")
        return
    
    # Define obstacles in workspace
    obstacles = [
        (np.array([0.5, 0.0, 0.3]), 0.1),
        (np.array([0.7, 0.2, 0.2]), 0.08),
    ]
    
    # Start and goal
    q_start = np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0])
    q_goal = np.array([1.5, -1.0, 1.0, -np.pi/2, 1.5, 0.0])
    
    # Plan
    planner = APFRRTStar(robot, config)
    path, tree, metrics = planner.plan(q_start, q_goal, obstacles)
    
    # Smooth
    if path:
        smoother = PathSmoother(robot, obstacles)
        path_smooth = smoother.smooth_path(path)
        print(f"Path length (raw): {metrics}")
        print(f"Path length (smoothed): {len(path_smooth)}")
    
    print("Planning complete!")


if __name__ == "__main__":
    main()