import copy
import logging

import torch

from . import voxel
from . import sdf
from . import model_to_sdf
from matplotlib import pyplot as plt
import matplotlib.colors

logger = logging.getLogger(__name__)


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    if x == 0:
        return "surface"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


def draw_sdf_slice(
    s: sdf.ObjectFrameSDF,
    query_range,
    resolution=0.01,
    interior_padding=0.2,
    cmap="Greys_r",
    device="cpu",
    plot_grad=False,
    do_plot=True,
):
    """

    :param s: SDF to query on
    :param query_range: (min, max) for each dimension x,y,z. One dimension must have min=max to be sliced along, with
    the other dimensions shown. Note that this should be given in the SDF's frame.
    :param resolution:
    :param interior_padding:
    :param cmap: matplotlib compatible colormap
    :param device: pytorch device
    :param plot_grad: whether to plot the gradient field
    :return:
    """
    coords, pts = voxel.get_coordinates_and_points_in_grid(
        resolution, query_range, device=device
    )
    # add a small amount of noise to avoid querying regular grid
    pts += torch.randn_like(pts) * 1e-6
    dim_labels = ["x", "y", "z"]
    slice_dim = None
    for i in range(len(dim_labels)):
        if len(coords[i]) == 1:
            slice_dim = i
            break

    # to properly draw a slice, the coords for that dimension must have only 1 element
    if slice_dim is None:
        raise RuntimeError(
            f"Sliced SDF requires a single query value for the sliced, but all query dimensions > 1"
        )

    shown_dims = [i for i in range(3) if i != slice_dim]

    sdf_val, sdf_grad = s(pts)
    norm = matplotlib.colors.Normalize(
        vmin=sdf_val.min().cpu() - interior_padding, vmax=sdf_val.max().cpu()
    )

    x = coords[shown_dims[0]].cpu()
    z = coords[shown_dims[1]].cpu()
    v = sdf_val.reshape(len(x), len(z)).transpose(0, 1).cpu()
    ax = None
    cset1 = None
    cset2 = None
    if do_plot:
        ax = plt.gca()
        ax.set_xlabel(dim_labels[shown_dims[0]])
        ax.set_ylabel(dim_labels[shown_dims[1]])
        cset1 = ax.contourf(x, z, v, norm=norm, cmap=cmap)
        cset2 = ax.contour(x, z, v, colors="k", levels=[0], linestyles="dashed")
        if plot_grad:
            sdf_grad_uv = sdf_grad.reshape(len(x), len(z), 3).permute(1, 0, 2).cpu()
            # subsample arrows
            subsample_n = 5
            ax.quiver(
                x[::subsample_n],
                z[::subsample_n],
                sdf_grad_uv[::subsample_n, ::subsample_n, shown_dims[0]],
                sdf_grad_uv[::subsample_n, ::subsample_n, shown_dims[1]],
                color="g",
            )
        ax.clabel(cset2, cset2.levels, inline=True, fontsize=13, fmt=fmt)
        plt.colorbar(cset1)
        # fig = plt.gcf()
        # fig.canvas.draw()
        plt.draw()
        plt.pause(0.005)
    return sdf_val, sdf_grad, pts, ax, cset1, cset2, v


def get_transformed_meshes(robot_sdf: model_to_sdf.RobotSDF, obj_to_world_tsf=None):
    """Get the meshes of each link of the robot, transformed to the world frame.
    Each link is assumed to be a MeshSDF (which now includes primitives like cylinders).
    You can use this like:

    import open3d as o3d
    meshes = get_transformed_meshes(robot_sdf)
    o3d.visualization.draw_geometries(meshes)
    """

    meshes = []
    # link to obj in the form of (massively composed object) H (link)
    # Since RobotSDF bakes base_transform into obj_frame_to_link_frame,
    # if base_transform is not None, obj_frame_to_link_frame is already World -> Link.
    # Therefore, inverse() gives Link -> World (actually World -> Link inverse, i.e., T_link_in_world?)
    # Wait: pk.Transform3d.inverse() gives the inverse transform.
    # obj_frame_to_link_frame (T_L_O) takes points in Object frame and puts them in Link frame.
    # So it is the coordinate transform from Object to Link.
    # Its inverse (T_O_L) transform points from Link frame to Object frame.
    # If Object frame is actually World frame (due to baking), then it transforms Link -> World.
    # And this is exactly what we need for open3d mesh.transform() (which expects T_world_link).

    tsfs = robot_sdf.sdf.obj_frame_to_link_frame.inverse()

    # If base_transform is NOT set, then 'Object Frame' is just the robot root frame.
    # In that case, we might want to apply an external obj_to_world_tsf if provided.
    if robot_sdf.base_transform is None and obj_to_world_tsf is not None:
        # obj_to_world_tsf is World -> Object (Root).
        # tsfs is Object (Root) -> Link? No, tsfs is Link -> Object (Root).
        # We want Link -> World = (World -> Object) * (Object -> Link) ?
        # Wait, if tsfs is Link -> Root. And obj_to_world_tsf is Root -> World (frame transform? or point transform?)
        # pk.Transform3d is usually point transform T_A_B (points in B to points in A).
        # So T_world_link = T_world_root * T_root_link.
        # robot_sdf.base_transform is usually T_world_root (points in root to points in world??)
        # Doc says "Transform from world to object frame" (T_object_world).
        # If so, point in World -> point in Object.
        # Then its inverse is T_world_object (point in Object -> point in World).

        # Let's assume obj_to_world_tsf passed here is meant to be T_world_object (Root -> World).
        tsfs = obj_to_world_tsf.compose(tsfs)

    tsfs = tsfs.get_matrix()
    for i in range(len(robot_sdf.sdf_to_link_name)):
        # All SDFs are now MeshSDF (primitives are wrapped in MeshSDF)
        if hasattr(robot_sdf.sdf.sdfs[i], "obj_factory") and hasattr(
            robot_sdf.sdf.sdfs[i].obj_factory, "_mesh"
        ):
            mesh = copy.deepcopy(robot_sdf.sdf.sdfs[i].obj_factory._mesh)
            mesh = mesh.transform(tsfs[i].cpu().numpy())
            meshes.append(mesh)
        else:
            logger.warning(
                f"Link {robot_sdf.sdf_to_link_name[i]} does not have a mesh to extract"
            )
    return meshes


def to_vertices_and_faces(s):
    import open3d as o3d, numpy as np

    o = sum(get_transformed_meshes(s), o3d.geometry.TriangleMesh())
    return np.asarray(o.vertices), np.asarray(o.triangles)
