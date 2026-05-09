import torch
import kaolin
import torch.nn.functional as F


def compute_sdf(pointclouds, verts, faces, use_face_centers=True):
    """
    Computes SDF, closest points (approximated by face centers), and gradient.

    Args:
        pointclouds (torch.Tensor): Shape (B, P, 3)
        verts (torch.Tensor): Shape (B, V, 3) - Mesh vertices
        faces (torch.Tensor): Shape (F, 3) - Mesh face indices
        use_face_centers (bool):
            If True, computes Euclidean distance to the center of the nearest face.
            If False, uses the exact distance calculated by Kaolin's projection logic.

    Returns:
        closest_points (torch.Tensor): Shape (B, P, 3) - Center of the nearest face
        sdf (torch.Tensor): Shape (B, P) - Signed distance values
        gradient (torch.Tensor): Shape (B, P, 3) - Normalized vector pointing to surface
    """

    # 1. Prepare Face Vertices
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(verts, faces)

    # 2. Find Nearest Faces & Kaolin Distances
    # min_dist_sq: (B, P) - Exact squared distance to the mesh surface
    # face_idx: (B, P) - Indices of the closest faces
    min_dist_sq, face_idx, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        pointclouds, face_vertices
    )

    # 3. Compute "Closest Points" (Emulated by Face Centers)
    batch_size, num_points, _ = pointclouds.shape
    batch_indices = (
        torch.arange(batch_size, device=verts.device)
        .unsqueeze(1)
        .expand(-1, num_points)
    )

    # Gather closest face vertices: (B, P, 3, 3)
    closest_face_verts = face_vertices[batch_indices, face_idx]

    # Center of the triangle: (B, P, 3)
    closest_points = closest_face_verts.mean(dim=2)

    # 4. Compute Distance
    if use_face_centers:
        # Distance to the face center
        diff_vector = closest_points - pointclouds
        dist = torch.norm(diff_vector, p=2, dim=-1)
    else:
        # Exact distance to the mesh surface (from Kaolin)
        dist = torch.sqrt(min_dist_sq)

    # 5. Compute Gradient
    # Note: Gradient is always computed pointing toward the Face Center to remain
    # consistent with the "closest_points" return value requested.
    diff_vector = closest_points - pointclouds
    gradient = F.normalize(diff_vector, p=2, dim=-1, eps=1e-8)

    # 6. Determine Sign (Inside vs Outside)
    # True = Inside, False = Outside
    is_inside = kaolin.ops.mesh.check_sign(verts, faces, pointclouds)

    # Convert to sign float: Inside (-1.0), Outside (1.0)
    sign = torch.where(is_inside, -1.0, 1.0)

    # 7. Final SDF
    sdf = dist * sign

    return closest_points, sdf, gradient


# Example Usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simple Tetrahedron
    verts = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    faces = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.long, device=device
    )
    points = torch.tensor([[[2.0, 2.0, 2.0]]], device=device)

    # Option 1: Distance to Center
    _, sdf_center, _ = compute_sdf(points, verts, faces, use_face_centers=True)

    # Option 2: Exact Projection Distance
    _, sdf_exact, _ = compute_sdf(points, verts, faces, use_face_centers=False)

    print(f"SDF (Face Center): {sdf_center}")
    print(f"SDF (Exact):       {sdf_exact}")
