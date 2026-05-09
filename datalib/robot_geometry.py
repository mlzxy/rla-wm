import torch
import numpy as np
import third_party.pytorch_kinematics as pk
import third_party.pytorch_volumetric as pv
import open3d as o3d


def to_o3d_mesh(s: pv.RobotSDF) -> o3d.geometry.TriangleMesh:
    meshes = pv.get_transformed_meshes(s)
    combined_mesh = o3d.geometry.TriangleMesh()
    # Merge all links into the combined mesh (this now includes primitives like cylinders)
    for mesh in meshes:
        combined_mesh += mesh
    combined_mesh.compute_vertex_normals()
    return combined_mesh
    
def save_o3d_mesh(mesh: o3d.geometry.TriangleMesh, filename: str) -> bool:
    """
    Save an Open3D TriangleMesh to a file.

    The mesh format is inferred from the file extension (e.g., .ply, .obj, .stl).

    Args:
        mesh: Open3D TriangleMesh to save.
        filename: Path to the output file.

    Returns:
        True if the write succeeds, False otherwise (see Open3D docs).
    """
    return o3d.io.write_triangle_mesh(filename, mesh)




def mesh_to_arrays(mesh: o3d.geometry.TriangleMesh):
    """
    Convert Open3D mesh to vertices, faces, and vertex colors arrays.
    
    Args:
        mesh: Open3D TriangleMesh
        
    Returns:
        vertices: (N, 3) numpy array of vertex positions
        faces: (T, 3) numpy array of triangle indices
        vcolors: (N, 3) numpy array of vertex colors (defaults to white if not present)
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    if mesh.has_vertex_colors():
        vcolors = np.asarray(mesh.vertex_colors)
    else:
        # Default color: white
        vcolors = np.ones((vertices.shape[0], 3))
    
    return vertices, faces, vcolors



class DifferentiableRobotGeometry(torch.nn.Module):
    def __init__(self, urdf_path: str, base_dir: str, joint_names: list[str] | None = None):
        super().__init__()
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()
            self.chain = pk.build_chain_from_urdf(urdf_content)
        self.sdf = pv.RobotSDF(self.chain, path_prefix=base_dir)
        self.joint_names = joint_names

    def set_pose(self, q: torch.Tensor, root_pose: torch.Tensor | None = None):
        """
        Args:
            q: (B, num_joints)
            root_pose: (B, 7) [x, y, z, qw, qx, qy, qz]
        """
        # 1. Create a new RobotSDF instance with the same chain and path_prefix.
        # The meshes/SDFs are likely cached or shared, so this should be relatively efficient.
        
        # If joint names are provided, map the tensor to named joints so pk orders them correctly.
        if self.joint_names is not None:
            # q shape is e.g. (B, num_joints)
            # Create a dictionary mapping: name -> (B, 1) tensor
            q_in = {name: q[..., i] for i, name in enumerate(self.joint_names)}
        else:
            q_in = q
            
        self.sdf.set_joint_configuration(q_in)

        if root_pose is not None:
            # Convert [x, y, z, qw, qx, qy, qz] to pk.Transform3d
            # Note: root_pose is the pose of the robot base in world frame
            # We need world_to_obj transform, which is the inverse of obj_to_world
            # obj_to_world is the transform that takes points from object frame to world frame
            obj_to_world = pk.Transform3d(
                pos=root_pose[:, :3],
                rot=root_pose[:, 3:], # pk expects [w, x, y, z]
                device=q.device
            )
            # world_to_obj is the inverse
            world_to_obj = obj_to_world.inverse()
            
            # Apply transform to the SDF
            self.sdf.set_base_transform(world_to_obj)

    