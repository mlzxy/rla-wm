import torch
import numpy as np
from mani_skill.utils.structs.pose import Pose
from datalib.play.geometry import sample_interaction_points, AntipodalSampler
import scipy.spatial.transform as transform

def is_actor_static(actor, lin_thresh=2e-3, ang_thresh=2e-2):
    """Checks if an actor is practically static. Slightly increased thresholds to ignore minor simulation jitter."""
    # ManiSkill 3 actors have linear_velocity and angular_velocity as tensors (N, 3)
    lin_vel = torch.linalg.norm(actor.linear_velocity, dim=-1)
    ang_vel = torch.linalg.norm(actor.angular_velocity, dim=-1)
    return (lin_vel < lin_thresh) & (ang_vel < ang_thresh)

def poke(env, actor, point, normal, env_idx=0, pre_dist=0.05, approach_dist=0.08, max_steps=50):
    """
    Executes a poke interaction.
    Args:
        env: The ManiSkill environment
        actor: The target actor
        point: Point on surface (world frame)
        normal: Normal at point (pointing OUT of surface)
        env_idx: Environment index
        pre_dist: Distance from surface for pre-interaction pose
        approach_dist: Total distance to move from pre-pose (should go slightly through surface)
    """
    device = env.device
    
    # 1. Target Pose calculation
    # We want the gripper to approach along the negative normal
    # For Panda, the EEF Z-axis is the approach direction usually.
    # We need to construct a pose where EEF Z points towards -normal.
    
    # Pre-interaction point: point + normal * pre_dist
    pre_point = point + normal * pre_dist
    
    # Target approach point: point - normal * 0.02 (poke 2cm inside)
    target_point = point - normal * 0.02
    
    # For now, let's use EEF delta control for simplicity if we can.
    # But ManiSkill envs usually expect actions in every step.
    
    # This primitive will be a generator or a loop that yields actions.
    # In a real environment, we'd have a 'controller' that manages this.
    
    # Let's implement it as a function that runs the loop for this specific env_idx.
    # Note: This blocks other envs if not handled carefully, but for test/verification it's fine.
    
    # Move to pre_point
    # We'll use a simple IK or PDEE controller if available.
    
    # In ManiSkill 3, we can set the EEF target if using PDEEPoseController.
    # env.step(action) where action is the target pose.
    
    # Let's assume the env uses pd_ee_delta_pose for now as per SPEC.md.
    
    return pre_point, target_point

class InteractionPrimitives:
    def __init__(self, env, step_callback=None):
        self.env = env
        self.device = env.device
        self.step_callback = step_callback
        self.grasp_sampler = AntipodalSampler(gripper_max_width=0.08)

    def step(self, action):
        """Wrapper for env.step that handles callback."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.step_callback:
            self.step_callback()
        return obs, reward, terminated, truncated, info

    def get_action_to_pose(self, target_pos, target_quat=None, gripper_val=0.0, env_idx=0, pos_gain=2.0, rot_gain=1.0, max_lin_vel=0.1, max_ang_vel=0.2):
        """Computes 6DoF delta eef action to reach a pose with velocity limiting."""
        curr_pose = self.env.agent.tcp.pose
        p_curr = curr_pose.p[env_idx].cpu().numpy()
        q_curr = curr_pose.q[env_idx].cpu().numpy() # [w, x, y, z]

        # Position delta
        delta_p = target_pos - p_curr
        
        # Action size depends on robot controller
        action_dim = self.env.action_space.shape[-1]
        action = np.zeros(action_dim)
        action[:3] = delta_p * pos_gain

        # Rotation delta
        if target_quat is not None:
            # ManiSkill [w, x, y, z] to Scipy [x, y, z, w]
            q_curr_scipy = np.array([q_curr[1], q_curr[2], q_curr[3], q_curr[0]])
            q_target_scipy = np.array([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
            
            r_curr = transform.Rotation.from_quat(q_curr_scipy)
            r_target = transform.Rotation.from_quat(q_target_scipy)
            
            # Compute relative rotation in EEF frame or world frame?
            # pd_ee_delta_pose usually expects deltas in world-aligned or body-aligned frame.
            # In ManiSkill 3, it's usually world-aligned if frame="root_translation:root_aligned_body_rotation"
            rel_rot = r_target * r_curr.inv()
            rot_vec = rel_rot.as_rotvec() # [ax, ay, az]
            action[3:6] = rot_vec * rot_gain

        # Velocity limiting for smoothness
        action[:3] = np.clip(action[:3], -max_lin_vel, max_lin_vel)
        action[3:6] = np.clip(action[3:6], -max_ang_vel, max_ang_vel)

        # Fill gripper values for all remaining dimensions
        if action_dim > 6:
            action[6:] = gripper_val
        return action

    def execute_path(self, waypoints, gripper_val=0.0, env_idx=0, logger=None, speed_scale=1.0, timeout=100, duration=None):
        """
        Executes a sequence of waypoints.
        Each waypoint is (pos, quat) or just pos.
        'duration': if set, holds the final waypoint for this many steps.
        """
        for i, wp in enumerate(waypoints):
            target_p = wp[0] if isinstance(wp, (list, tuple, np.ndarray)) and len(wp) == 2 else wp
            target_q = wp[1] if isinstance(wp, (list, tuple, np.ndarray)) and len(wp) == 2 else None
            
            # If it's the last waypoint and duration is set, we might want to hold it.
            actual_timeout = timeout
            if i == len(waypoints) - 1 and duration is not None:
                actual_timeout = duration

            for step_count in range(actual_timeout):
                action = self.get_action_to_pose(target_p, target_q, gripper_val=gripper_val, env_idx=env_idx)
                # Apply speed scaling
                action[:6] *= speed_scale
                
                full_action = np.zeros((self.env.num_envs, self.env.action_space.shape[-1]))
                full_action[env_idx] = action
                action_tensor = torch.from_numpy(full_action).to(self.device).float()
                
                if logger:
                    logger.log_action(action_tensor)
                    logger.record_state(self.env)
                    
                self.step(action_tensor)
                
                # If not in "hold duration" mode, check for convergence
                if duration is None or i < len(waypoints) - 1:
                    curr_p = self.env.agent.tcp.pose.p[env_idx].cpu().numpy()
                    dist = np.linalg.norm(curr_p - target_p)
                    if dist < 0.002: # Tightened from 0.005
                        break
        return True

    def execute_guarded_move(self, target_p, actor, env_idx=0, max_steps=50, logger=None, gripper_val=0.0, target_q=None, speed_scale=0.5):
        """Moves towards target_p and stops on contact or movement of actor."""
        for _ in range(max_steps):
            action = self.get_action_to_pose(target_p, target_q, gripper_val=gripper_val, env_idx=env_idx)
            action[:6] *= speed_scale
            
            full_action = np.zeros((self.env.num_envs, self.env.action_space.shape[-1]))
            full_action[env_idx] = action
            action_tensor = torch.from_numpy(full_action).to(self.device).float()
            
            if logger:
                logger.log_action(action_tensor)
                logger.record_state(self.env)
                
            obs, reward, terminated, truncated, info = self.step(action_tensor)
            
            # Check for contact or actor movement
            if not is_actor_static(actor)[env_idx]:
                return True # Success
                
            # Check distance to target
            curr_p = self.env.agent.tcp.pose.p[env_idx]
            if torch.linalg.norm(curr_p - torch.from_numpy(target_p).to(self.device)) < 0.01:
                break
        return False

    def push(self, actor, point, normal, env_idx=0, logger=None):
        """Poke and then push with orientation alignment."""
        # Align EEF Z with -normal
        # Simple alignment: look from normal direction
        from scipy.spatial.transform import Rotation
        z = -normal / np.linalg.norm(normal)
        # Find an orthogonal x
        x = np.array([1, 0, 0]) if np.abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = np.cross(x, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rot_mat = np.stack([x, y, z], axis=1)
        quat = Rotation.from_matrix(rot_mat).as_quat()
        quat_ms = np.array([quat[3], quat[0], quat[1], quat[2]])

        pre_p = point + normal * 0.08
        self.execute_path([(pre_p, quat_ms)], env_idx=env_idx, logger=logger, speed_scale=2.0)
        
        target_p = point - normal * 0.08 # Increased overshoot from 0.05
        return self.execute_guarded_move(target_p, actor, env_idx, max_steps=60, logger=logger, target_q=quat_ms, speed_scale=0.5)

    def rotate(self, actor, env_idx=0, logger=None):
        """Pokes an off-center point to induce spin."""
        points, normals = sample_interaction_points(actor, num_samples=50, env_idx=env_idx)
        actor_pos = actor.pose.p[env_idx].cpu().numpy()
        dists_from_center = np.linalg.norm(points[:, :2] - actor_pos[:2], axis=1)
        idx = np.argmax(dists_from_center)
        return self.push(actor, points[idx], normals[idx], env_idx=env_idx, logger=logger)

    def poke(self, actor, point, normal, env_idx=0, logger=None):
        """Light contact from normal direction with alignment."""
        from scipy.spatial.transform import Rotation
        z = -normal / np.linalg.norm(normal)
        x = np.array([1, 0, 0]) if np.abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = np.cross(x, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rot_mat = np.stack([x, y, z], axis=1)
        quat = Rotation.from_matrix(rot_mat).as_quat()
        quat_ms = np.array([quat[3], quat[0], quat[1], quat[2]])

        pre_p = point + normal * 0.05
        self.execute_path([(pre_p, quat_ms)], env_idx=env_idx, logger=logger, speed_scale=2.0)
        
        target_p = point - normal * 0.03 # Increased overshoot from 0.01
        return self.execute_guarded_move(target_p, actor, env_idx, max_steps=40, logger=logger, target_q=quat_ms, speed_scale=0.3)

    def slide(self, actor, point, normal, env_idx=0, logger=None):
        """Touch and then move along a tangent surface."""
        # Just use poke alignment for the touch
        from scipy.spatial.transform import Rotation
        z = -normal / np.linalg.norm(normal)
        x = np.array([1, 0, 0]) if np.abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = np.cross(x, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rot_mat = np.stack([x, y, z], axis=1)
        quat = Rotation.from_matrix(rot_mat).as_quat()
        quat_ms = np.array([quat[3], quat[0], quat[1], quat[2]])

        self.poke(actor, point, normal, env_idx, logger)
        # Random tangent
        tangent = np.cross(normal, [0, 0, 1])
        if np.linalg.norm(tangent) < 0.1: tangent = np.cross(normal, [0, 1, 0])
        tangent = tangent / np.linalg.norm(tangent)
        target_p = point + tangent * 0.15
        return self.execute_path([(target_p, quat_ms)], env_idx=env_idx, logger=logger, speed_scale=0.5)

    def pick(self, actor, env_idx=0, logger=None):
        """Robust grasp and lift using AntipodalSampler."""
        grasps = self.grasp_sampler.sample_grasps(actor, env_idx=env_idx, max_grasps=1)
        if not grasps:
            return False
        
        grasp = grasps[0]
        center, quat, width = grasp["center"], grasp["quat"], grasp["width"]
        
        # Open gripper
        curr_p = self.env.agent.tcp.pose.p[env_idx].cpu().numpy()
        self.execute_path([curr_p], duration=10, gripper_val=1.0, env_idx=env_idx, logger=logger)
        
        # Approach: Move to 10cm above grasp center
        pre_p = center + np.array([0, 0, 0.1])
        self.execute_path([(pre_p, quat)], gripper_val=1.0, env_idx=env_idx, logger=logger, speed_scale=2.0)
        
        # Move to grasp pose
        self.execute_path([(center, quat)], gripper_val=1.0, env_idx=env_idx, logger=logger, speed_scale=1.0)
        
        # Close gripper
        # Action -1.0 is closed for Panda in this config
        self.execute_path([(center, quat)], duration=20, gripper_val=-1.0, env_idx=env_idx, logger=logger)
        
        # Lift
        lift_p = center + np.array([0, 0, 0.2])
        self.execute_path([(lift_p, quat)], gripper_val=-1.0, env_idx=env_idx, logger=logger, speed_scale=1.5)
        
        # Verify success
        if actor.pose.p[env_idx, 2] > 0.08:
            return True
        return False

    def place(self, target_xy, env_idx=0, logger=None):
        """Lower and release with retraction."""
        curr_p = self.env.agent.tcp.pose.p[env_idx].cpu().numpy()
        curr_q = self.env.agent.tcp.pose.q[env_idx].cpu().numpy()
        
        # Hover over XY
        nav_p = np.array([target_xy[0], target_xy[1], curr_p[2]])
        self.execute_path([(nav_p, curr_q)], gripper_val=-1.0, env_idx=env_idx, logger=logger, speed_scale=2.0)
        
        # Lower
        place_p = np.array([target_xy[0], target_xy[1], 0.05])
        self.execute_path([(place_p, curr_q)], gripper_val=-1.0, env_idx=env_idx, logger=logger, speed_scale=0.8)
        
        # Release
        self.execute_path([(place_p, curr_q)], duration=10, gripper_val=1.0, env_idx=env_idx, logger=logger)
        
        # Retreat
        retreat_p = nav_p + np.array([0, 0, 0.1])
        self.execute_path([(retreat_p, curr_q)], gripper_val=1.0, env_idx=env_idx, logger=logger, speed_scale=2.0)
        return True

    def tool_push(self, tool_actor, target_actor, env_idx=0, logger=None):
        """Pick up tool_actor and use it to push target_actor."""
        if self.pick(tool_actor, env_idx=env_idx, logger=logger):
            # Target position
            target_p = target_actor.pose.p[env_idx].cpu().numpy()
            tool_p = tool_actor.pose.p[env_idx].cpu().numpy()
            
            # Approach target_actor from behind relative to tool?
            # For simplicity, move tool to side of target and push
            push_point = target_p + np.array([0.1, 0, 0])
            self.execute_path([(push_point + np.array([0, 0, 0.05]), self.env.agent.tcp.pose.q[env_idx].cpu().numpy())], 
                              gripper_val=-1.0, env_idx=env_idx, logger=logger, speed_scale=1.5)
            
            # Perform push through center
            target_p_through = target_p - np.array([0.1, 0, 0])
            return self.execute_guarded_move(target_p_through, target_actor, env_idx=env_idx, logger=logger, gripper_val=-1.0)
        return False

    def flip(self, actor, env_idx=0, logger=None):
        """Push a low edge to flip with alignment."""
        points, normals = sample_interaction_points(actor, num_samples=100, env_idx=env_idx)
        actor_p = actor.pose.p[env_idx].cpu().numpy()
        mask = points[:, 2] < actor_p[2]
        if not np.any(mask): return False
        
        low_points = points[mask]
        low_normals = normals[mask]
        idx = np.random.randint(len(low_points))
        p, n = low_points[idx], low_normals[idx]
        
        # Force normal to be mostly horizontal for flip
        n[2] = 0
        n /= np.linalg.norm(n)
        
        return self.push(actor, p, n, env_idx=env_idx, logger=logger)

    def stack(self, actor_a, actor_b, env_idx=0, logger=None):
        """Pick A and place on B."""
        if self.pick(actor_a, env_idx=env_idx, logger=logger):
            target_p = actor_b.pose.p[env_idx].cpu().numpy()
            # Elevate slightly for stacking
            stack_p = target_p + np.array([0, 0, 0.08])
            return self.place(stack_p[:2], env_idx=env_idx, logger=logger)
        return False
