import numpy as np
import sapien
import trimesh
from mani_skill.utils.structs.pose import Pose
import torch

def sample_interaction_points(actor, num_samples=20, env_idx=0):
    """
    Samples reachable surface points and normals from a SAPIEN actor.
    
    Args:
        actor: ManiSkill Actor or SAPIEN Actor
        num_samples: Number of points to sample
        env_idx: Environment index (if actor is batched)
        
    Returns:
        points: (num_samples, 3) numpy array of points in world frame
        normals: (num_samples, 3) numpy array of normals in world frame
    """
    # Get collision mesh (local frame)
    mesh = actor.get_first_collision_mesh(to_world_frame=False)
    if mesh is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    # Use trimesh to sample uniform points on the surface
    points, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
    normals = mesh.face_normals[face_indices]
    
    # Get actor pose
    # In ManiSkill 3, actor.pose is a Pose object with .p (N, 3) and .q (N, 4)
    if hasattr(actor, "pose") and isinstance(actor.pose, Pose):
        p = actor.pose.p[env_idx].detach().cpu().numpy()
        q = actor.pose.q[env_idx].detach().cpu().numpy()
    else:
        # Fallback for raw SAPIEN actor
        p = actor.pose.p
        q = actor.pose.q
    
    # Convert quaternion to rotation matrix
    # SAPIEN/ManiSkill use [w, x, y, z] or [x, y, z, w]?
    # ManiSkill Pose uses [w, x, y, z] by default? No, usually [x, y, z, w] in many contexts, but ManiSkill/SAPIEN uses [w, x, y, z].
    # Let's use trimesh/scipy or sapien for conversion.
    
    # Sapien pose to 4x4
    from mani_skill.utils.structs.pose import Pose as MSPose
    # We can use MSPose to transform
    
    local_points = torch.from_numpy(points).float()
    local_normals = torch.from_numpy(normals).float()
    
    # Create a Pose for the sampled points
    # Points are (N, 3), we want to transform them.
    # Pose.transform_points expects (..., 3)
    
    if hasattr(actor, "pose") and isinstance(actor.pose, Pose):
        # ManiSkill Pose (batched)
        mat = actor.pose.to_transformation_matrix()[env_idx].detach().cpu().numpy()
        rot = mat[:3, :3]
        trans = mat[:3, 3]
        
        world_points = (rot @ points.T).T + trans
        world_normals = (rot @ normals.T).T
        
        return world_points, world_normals
    else:
         # Raw SAPIEN actor fallback
         mat = actor.pose.to_transformation_matrix()
         rot = mat[:3, :3]
         trans = mat[:3, 3]
         
         world_points = (rot @ points.T).T + trans
         world_normals = (rot @ normals.T).T
         return world_points, world_normals

def filter_reachable_points(points, normals, min_z=0.01):
    """
    Filter points likely to be reachable by the robot.
    Basic heuristic: points above table level and facing upwards or sideways.
    """
    mask = points[:, 2] > min_z
    # We could also filter by normal direction (e.g. facing towards robot at x < 0)
    return points[mask], normals[mask]

class AntipodalSampler:
    def __init__(self, gripper_max_width=0.08, num_samples=500):
        self.gripper_max_width = gripper_max_width
        self.num_samples = num_samples

    def sample_grasps(self, actor, env_idx=0, max_grasps=10):
        """
        Samples antipodal grasps from the actor's collision mesh.
        Returns:
            grasps: List of dicts containing {center, orientation, width, score}
        """
        mesh = actor.get_first_collision_mesh(to_world_frame=False)
        if mesh is None:
            return []

        # Sample points and normals in local frame
        points, face_indices = trimesh.sample.sample_surface_even(mesh, self.num_samples)
        normals = mesh.face_normals[face_indices]

        # Get transformation matrix
        if hasattr(actor, "pose") and isinstance(actor.pose, Pose):
            mat = actor.pose.to_transformation_matrix()[env_idx].detach().cpu().numpy()
        else:
            mat = actor.pose.to_transformation_matrix()
        
        grasps = []
        
        # O(N^2) search - can be optimized but N=500 is small enough for occasional use
        # We look for pairs (p1, n1), (p2, n2)
        for i in range(len(points)):
            p1, n1 = points[i], normals[i]
            for j in range(i + 1, len(points)):
                p2, n2 = points[j], normals[j]
                
                # Check distance
                vec = p2 - p1
                dist = np.linalg.norm(vec)
                if dist > self.gripper_max_width or dist < 0.01:
                    continue
                
                # Check if normals are opposing
                dot_nn = np.dot(n1, n2)
                if dot_nn > -0.7: # Not opposing enough
                    continue
                
                # Check if vec is aligned with n1
                vec_u = vec / (dist + 1e-6)
                dot_vn = np.abs(np.dot(vec_u, n1))
                if dot_vn < 0.8: # Not aligned with surface normal
                    continue
                
                # We found a potential grasp
                center_local = (p1 + p2) / 2.0
                
                # Transform to world
                center_world = (mat[:3, :3] @ center_local) + mat[:3, 3]
                
                # Orientation: EEF Z should point towards center (approach)
                # EEF X should be the closing direction (along vec_u)
                # EEF Y = Z x X
                
                grasp_x = (mat[:3, :3] @ vec_u)
                # Random approach direction for now, ideally perpendicular to x
                # Let's pick a random Z that is perpendicular to X
                z_candidates = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
                best_z = None
                for z_can in z_candidates:
                    if np.abs(np.dot(z_can, grasp_x)) < 0.5:
                        # Project z_can onto plane perpendicular to grasp_x
                        z = z_can - np.dot(z_can, grasp_x) * grasp_x
                        best_z = z / np.linalg.norm(z)
                        break
                
                if best_z is None: continue
                
                grasp_y = np.cross(best_z, grasp_x)
                grasp_rot = np.stack([grasp_x, grasp_y, best_z], axis=1)
                
                # Convert to quat
                from scipy.spatial.transform import Rotation
                quat = Rotation.from_matrix(grasp_rot).as_quat() # [x, y, z, w]
                # ManiSkill uses [w, x, y, z]
                quat_ms = np.array([quat[3], quat[0], quat[1], quat[2]])
                
                grasps.append({
                    "center": center_world,
                    "quat": quat_ms,
                    "width": dist,
                    "score": dot_vn * (-dot_nn) # Higher is better
                })
                
                if len(grasps) >= max_grasps * 5: break
            if len(grasps) >= max_grasps * 5: break
            
        # Sort by score and return top samples
        grasps.sort(key=lambda x: x["score"], reverse=True)
        return grasps[:max_grasps]
