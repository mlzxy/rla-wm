import torch
import trimesh
import pytorch_kinematics as pk
import third_party.pytorch_volumetric as pv
import os.path as osp
import numpy as np
from typing import Tuple, Optional, Union
from third_party.pytorch_volumetric.visualization import get_transformed_meshes

def to_mesh(sdf: pv.RobotSDF, save_path: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the robot geometry from a RobotSDF instance as differentiable tensors.
    
    Args:
        sdf: A RobotSDF instance with its state set (via set_joint_configuration).
        save_path: Optional path to save the mesh file (e.g., "robot.obj").
                   Saving is done with a detached, non-differentiable copy.

    Returns:
        vertices: (N, 3) float tensor, differentiable w.r.t joint angles and root pose.
        faces: (T, 3) long tensor, containing concatenated face indices.
    """
    # Get transforms from link frames to object/robot frame
    # obj_frame_to_link_frame is (robot) H (link), so inverse gives (link) H (robot)
    # We want to transform vertices from link frame to robot frame
    link_to_obj_transforms = sdf.sdf.obj_frame_to_link_frame.inverse()
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Process each link's mesh
    for i in range(len(sdf.sdf.sdfs)):
        # Get the mesh from the SDF's object factory
        mesh = sdf.sdf.sdfs[i].obj_factory._mesh
        
        # Get vertices and faces from the mesh
        # Open3D TriangleMesh uses 'triangles' not 'faces'
        # Convert to numpy arrays first, then to tensors on the same device as the SDF
        mesh_vertices = torch.tensor(
            np.asarray(mesh.vertices),
            dtype=sdf.dtype,
            device=sdf.device
        )  # (N_i, 3)
        
        mesh_faces = torch.tensor(
            np.asarray(mesh.triangles),
            dtype=torch.long,
            device=sdf.device
        )  # (T_i, 3)
        
        # Get the transform matrix for this specific link
        transform_slice = sdf.sdf.ith_transform_slice(i)
        tf_matrix = link_to_obj_transforms.get_matrix()
        
        # Extract the transform matrix for this link
        # Handle different matrix shapes (batched vs non-batched)
        if len(tf_matrix.shape) == 3:
            # Shape is (num_links, 4, 4) - non-batched
            link_tf_matrix = tf_matrix[transform_slice]
        elif len(tf_matrix.shape) == 4:
            # Shape is (batch, num_links, 4, 4) - batched, use first batch
            link_tf_matrix = tf_matrix[0, transform_slice]
        else:
            # Try to index directly
            link_tf_matrix = tf_matrix[transform_slice]
        
        # Ensure link_tf_matrix is (4, 4)
        if len(link_tf_matrix.shape) > 2:
            link_tf_matrix = link_tf_matrix.squeeze()
        
        # Transform vertices using homogeneous coordinates
        ones = torch.ones(mesh_vertices.shape[0], 1, dtype=mesh_vertices.dtype, device=mesh_vertices.device)
        vertices_hom = torch.cat([mesh_vertices, ones], dim=1)  # (N_i, 4)
        transformed_hom = vertices_hom @ link_tf_matrix.T  # (N_i, 4)
        transformed_vertices = transformed_hom[:, :3]  # (N_i, 3)
        
        # Adjust face indices to account for vertex offset
        adjusted_faces = mesh_faces + vertex_offset
        
        all_vertices.append(transformed_vertices)
        all_faces.append(adjusted_faces)
        
        vertex_offset += mesh_vertices.shape[0]
    
    # Concatenate all vertices and faces
    vertices = torch.cat(all_vertices, dim=0)  # (N, 3)
    faces = torch.cat(all_faces, dim=0)  # (T, 3)
    
    # Save to file if requested (using detached copy)
    if save_path is not None:
        # Create a trimesh object from the detached tensors for saving
        vertices_np = vertices.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        combined_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        combined_mesh.export(save_path)
    
    return vertices, faces
  