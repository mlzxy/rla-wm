import typing

import numpy as np
import torch
import third_party.pytorch_kinematics as pk
from . import sdf
import logging

logger = logging.getLogger(__file__)


class RobotSDF(sdf.ObjectFrameSDF):
    """Create an SDF for a robot model described by a third_party.pytorch_kinematics Chain.
    The SDF is conditioned on a joint configuration which must be set."""

    def __init__(
        self,
        chain: pk.Chain,
        default_joint_config=None,
        path_prefix="",
        use_collision_mesh: bool = False,
        link_sdf_cls: typing.Callable[
            [sdf.ObjectFactory], sdf.ObjectFrameSDF
        ] = sdf.MeshSDF,
    ):
        """

        :param chain: Robot description; each link should be a mesh type - non-mesh geometries are ignored
        :param default_joint_config: values for each joint of the robot by default; None results in all zeros
        :param path_prefix: path to search for referenced meshes inside the robot description (e.g. URDF) which may use
        relative paths. This given path is prefixed onto those relative paths in order to find the meshes.
        :param use_collision_mesh: if True, use collision geometry instead of visual geometry when available
        :param link_sdf_cls: Factory of each link's SDFs; **kwargs are forwarded to this factory
        :param kwargs: Keyword arguments fed to link_sdf_cls
        """
        if isinstance(chain, list):
            self.chains = chain
        else:
            self.chains = [chain]

        # Assume all chains share the same device and dtype from the first chain
        self.dtype = self.chains[0].dtype
        self.device = self.chains[0].device

        self.joint_names = []
        self.frame_names = []
        for c in self.chains:
            self.joint_names.extend(c.get_joint_parameter_names())
            # self.frame_names.extend(c.get_frame_names(exclude_fixed=False))

        # We need to handle frame names carefully if there are duplicates across chains
        # For now, we linearize standard frame names
        self.frame_names = []
        for i, c in enumerate(self.chains):
            fnames = c.get_frame_names(exclude_fixed=False)
            # If we have multiple chains, we might want to qualify names to avoid collision if needed
            # but usually downstream logic assumes specific link names.
            # We will just append all.
            self.frame_names.extend(fnames)

        self.q = None
        self.q = None
        self.object_to_link_frames: typing.Optional[pk.Transform3d] = None
        self.link_transforms_root_frame: typing.Optional[pk.Transform3d] = None
        # self.joint_names = self.chain.get_joint_parameter_names()
        # self.frame_names = self.chain.get_frame_names(exclude_fixed=False)
        self.sdf: typing.Optional[sdf.ComposedSDF] = None
        self.sdf_to_link_name = []
        self.configuration_batch = None
        self.base_transform: typing.Optional[pk.Transform3d] = (
            None  # Transform from world to object frame
        )

        sdfs = []
        offsets = []
        primitive_factories = []  # Store primitive factories for mesh extraction
        self.use_collision_mesh = use_collision_mesh

        # get the link meshes from the frames and create meshes
        # Iterate over all chains
        for chain_idx, chain in enumerate(self.chains):
            # We iterate over this chain's frames
            chain_frame_names = chain.get_frame_names(exclude_fixed=False)

            for frame_name in chain_frame_names:
                frame = chain.find_frame(frame_name)
                # Select which set of geometries to use: collision (preferred when requested) or visual
                link_geoms = getattr(frame.link, "visuals", [])
                if self.use_collision_mesh:
                    collision_geoms = getattr(frame.link, "collisions", [])
                    if collision_geoms:
                        logger.info(
                            f"Using collision geometries for link {frame.link.name}"
                        )
                        link_geoms = collision_geoms
                    else:
                        logger.info(
                            f"No collision geometries for link {frame.link.name}; falling back to visuals"
                        )

                for link_geom in link_geoms:
                    if link_geom.geom_type == "mesh":
                        logger.info(f"{frame.link.name} offset {link_geom.offset}")
                        # Check if path_prefix is a list or single string
                        if isinstance(path_prefix, list):
                            # Assume it corresponds to chains
                            pp = (
                                path_prefix[chain_idx]
                                if chain_idx < len(path_prefix)
                                else ""
                            )
                        else:
                            pp = path_prefix

                        link_obj = sdf.MeshObjectFactory(
                            link_geom.geom_param[0],
                            scale=link_geom.geom_param[1],
                            path_prefix=pp,
                        )
                        link_sdf = link_sdf_cls(link_obj)
                        self.sdf_to_link_name.append(frame.link.name)
                        sdfs.append(link_sdf)
                        offsets.append(link_geom.offset)
                        primitive_factories.append(None)  # Not a primitive
                    elif link_geom.geom_type == "cylinder":
                        # Handle cylinder primitive
                        radius = link_geom.geom_param[0]
                        length = link_geom.geom_param[1]
                        logger.info(
                            f"{frame.link.name} cylinder primitive: radius={radius}, length={length}, offset={link_geom.offset}"
                        )
                        link_obj = sdf.CylinderObjectFactory(
                            radius=radius, length=length
                        )
                        # Always use MeshSDF wrapper for primitives to ensure consistent interface
                        link_sdf = sdf.MeshSDF(link_obj)
                        self.sdf_to_link_name.append(frame.link.name)
                        sdfs.append(link_sdf)
                        offsets.append(link_geom.offset)
                        primitive_factories.append(link_obj)
                    elif link_geom.geom_type == "sphere":
                        radius = link_geom.geom_param[0]
                        logger.info(
                            f"{frame.link.name} sphere primitive: radius={radius}, offset={link_geom.offset}"
                        )
                        link_obj = sdf.SphereObjectFactory(radius=radius)
                        link_sdf = sdf.MeshSDF(link_obj)
                        self.sdf_to_link_name.append(frame.link.name)
                        sdfs.append(link_sdf)
                        offsets.append(link_geom.offset)
                        primitive_factories.append(link_obj)
                    elif link_geom.geom_type == "box":
                        size = link_geom.geom_param[0]
                        if isinstance(size, (int, float)):
                            size = link_geom.geom_param
                        logger.info(
                            f"{frame.link.name} box primitive: size={size}, offset={link_geom.offset}"
                        )
                        link_obj = sdf.BoxObjectFactory(size=size)
                        link_sdf = sdf.MeshSDF(link_obj)
                        self.sdf_to_link_name.append(frame.link.name)
                        sdfs.append(link_sdf)
                        offsets.append(link_geom.offset)
                        primitive_factories.append(link_obj)
                    else:
                        logger.warning(
                            f"Cannot handle non-mesh link geometry type {link_geom} for {frame.link.name}"
                        )

        self.primitive_factories = primitive_factories
        self.link_to_chain_idx = []
        # Re-iterate to store chain index for each link
        # We need to replicate the exact order of self.sdf_to_link_name
        # The above loops appended to self.sdf_to_link_name in order of chain, then frame
        # So we can just rebuild it or store it during the loop.
        # But I don't want to re-write the big loop above if I can avoid it.
        # Actually, let's just make a list of length equal to sdfs.
        # We know the order: chain 0 frames, chain 1 frames, etc.

        current_sdf_idx = 0
        for chain_idx, chain in enumerate(self.chains):
            chain_frame_names = chain.get_frame_names(exclude_fixed=False)
            for frame_name in chain_frame_names:
                frame = chain.find_frame(frame_name)
                # Re-check logic to count how many SDFs this frame generated
                link_geoms = getattr(frame.link, "visuals", [])
                if self.use_collision_mesh:
                    collision_geoms = getattr(frame.link, "collisions", [])
                    if collision_geoms:
                        link_geoms = collision_geoms

                for link_geom in link_geoms:
                    if link_geom.geom_type in ["mesh", "cylinder", "sphere", "box"]:
                        self.link_to_chain_idx.append(chain_idx)
                        current_sdf_idx += 1

        self.offset_transforms = (
            offsets[0].stack(*offsets[1:]).to(device=self.device, dtype=self.dtype)
        )
        self.sdf = sdf.ComposedSDF(sdfs, self.object_to_link_frames)
        self.set_joint_configuration(default_joint_config)

    def surface_bounding_box(self, **kwargs):
        return self.sdf.surface_bounding_box(**kwargs)

    def link_bounding_boxes(self):
        """
        Get the bounding box of each link in the robot's frame under the current configuration.
        Note that the bounding box is not necessarily axis-aligned, so the returned bounding box is not just
        the min and max of the points.
        :return: [A x] [B x] 8 x 3 points of the bounding box for each link in the robot's frame
        """
        tfs = self.sdf.obj_frame_to_link_frame.inverse()
        bbs = []
        for i in range(len(self.sdf.sdfs)):
            sdf = self.sdf.sdfs[i]
            bb = aabb_to_ordered_end_points(sdf.surface_bounding_box(padding=0))
            bb = tfs.transform_points(
                torch.tensor(bb, device=tfs.device, dtype=tfs.dtype)
            )[self.sdf.ith_transform_slice(i)]
            bbs.append(bb)
        return torch.stack(bbs).squeeze()

    def set_joint_configuration(self, joint_config=None):
        """
        Set the joint configuration of the robot
        :param joint_config: [A x] M optionally arbitrarily batched joint configurations. There are M joints; A can be
        any number of batched dimensions.
        Can also be a list of tensors, one for each chain.
        :return:
        """
        # Overall M is sum of all chains' DOFs
        M = len(self.joint_names)

        # If no config provided, zero out all
        if joint_config is None:
            joint_config = torch.zeros(M, device=self.device, dtype=self.dtype)

        if isinstance(joint_config, dict):
            first_val = next(iter(joint_config.values()))
            elem_shape = first_val.shape if hasattr(first_val, 'shape') else ()
            th = torch.zeros([*elem_shape, M], device=self.device, dtype=self.dtype)
            for joint_name, joint_position in joint_config.items():
                if joint_name in self.joint_names:
                    jnt_idx = self.joint_names.index(joint_name)
                    th[..., jnt_idx] = joint_position
            joint_config = th

        # Handle list input
        if isinstance(joint_config, list):
            # Concatenate list of tensors if it matches our total DOF
            # Or handle per-chain logic.
            # Simplest is to concat into one big tensor, then split for FK.
            # Assuming batch dims match.
            joint_config = torch.cat(joint_config, dim=-1)

        # Transform3D only works with 1 batch dimension, so we need to manually flatten any additional ones
        # save the batch dimensions for when retrieving points
        if len(joint_config.shape) > 1:
            self.configuration_batch = joint_config.shape[:-1]
            joint_config = joint_config.reshape(-1, M)
        else:
            self.configuration_batch = None

        # Now split joint_config for each chain
        # We need to know how many DOFs each chain has
        current_idx = 0
        all_link_transforms = {}

        for chain in self.chains:
            n_dof = len(chain.get_joint_parameter_names())
            chain_config = joint_config[..., current_idx : current_idx + n_dof]
            current_idx += n_dof

            if hasattr(chain, "_serial_frames"):
                tf = chain.forward_kinematics(chain_config, end_only=False)
            else:
                tf = chain.forward_kinematics(chain_config)

            # Merge dictionary
            for k, v in tf.items():
                all_link_transforms[k] = v

        tsfs = []
        for link_name in self.sdf_to_link_name:
            tsfs.append(all_link_transforms[link_name].get_matrix())

        # make offset transforms have compatible batch dimensions
        offset_tsf = self.offset_transforms.inverse()
        if self.configuration_batch is not None:
            # must be of shape (num_links, *self.configuration_batch, 4, 4) before flattening
            expand_dims = (None,) * len(self.configuration_batch)
            offset_tsf_mat = offset_tsf.get_matrix()[(slice(None),) + expand_dims]
            offset_tsf_mat = offset_tsf_mat.repeat(1, *self.configuration_batch, 1, 1)
            offset_tsf = pk.Transform3d(matrix=offset_tsf_mat.reshape(-1, 4, 4))

        tsfs = torch.cat(tsfs)

        self.link_transforms_root_frame = offset_tsf.compose(
            pk.Transform3d(matrix=tsfs).inverse()
        )
        self.object_to_link_frames = self.link_transforms_root_frame

        # Apply base transforms if necessary (bake into SDF transforms)
        self._update_sdf_transforms()

        if self.sdf is not None:
            self.sdf.set_transforms(
                self.object_to_link_frames, batch_dim=self.configuration_batch
            )

    def set_base_transform(
        self,
        world_to_obj_transform: typing.Union[
            pk.Transform3d, typing.List[pk.Transform3d]
        ],
    ):
        """
        Set the base transform of the robot. This transform maps from world frame to object (robot base) frame.
        After setting this, SDF queries should be made with points in the world frame.

        :param world_to_obj_transform: Transform3d from world frame to object frame.
                                       Can be a list of Transform3d, one per chain.
        """
        if isinstance(world_to_obj_transform, list):
            # Ensure we have one transform per chain
            if len(world_to_obj_transform) != len(self.chains):
                raise ValueError(
                    f"Expected {len(self.chains)} base transforms, got {len(world_to_obj_transform)}"
                )

            # Store as list, but move each to device
            self.base_transform = [
                t.to(device=self.device, dtype=self.dtype)
                for t in world_to_obj_transform
            ]
        else:
            # Single transform for all chains
            self.base_transform = world_to_obj_transform.to(
                device=self.device, dtype=self.dtype
            )

        # Re-update transforms to potentially bake in the base transform
        if (
            self.link_transforms_root_frame is not None
        ):  # Check if joint config has been set
            self._update_sdf_transforms()
            if self.sdf is not None:
                self.sdf.set_transforms(
                    self.object_to_link_frames, batch_dim=self.configuration_batch
                )

    def _update_sdf_transforms(self):
        # Helper to combine joint config transforms and base transforms
        # This needs to run whenever joint config or base transform changes
        # We start from the clean Root->Mesh transforms (link_transforms_root_frame)
        # And compose with World->Root (base_transform)

        if self.link_transforms_root_frame is None:
            return

        if self.base_transform is None:
            self.object_to_link_frames = self.link_transforms_root_frame
            return

        if isinstance(self.base_transform, list):
            # We have per-chain base transforms (World -> Root_i)
            base_matrices_per_link = []
            for chain_idx in self.link_to_chain_idx:
                tsf = self.base_transform[chain_idx]
                base_matrices_per_link.append(tsf.get_matrix())

            stacked_base_mats = torch.stack(base_matrices_per_link, dim=0)

            if self.configuration_batch is not None:
                B_total = 1
                for d in self.configuration_batch:
                    B_total *= d

                base_mats_flat = stacked_base_mats.reshape(
                    len(self.link_to_chain_idx), B_total, 4, 4
                )
                base_mats_for_compose = base_mats_flat.reshape(-1, 4, 4)
            else:
                base_mats_for_compose = stacked_base_mats.reshape(-1, 4, 4)

            base_tsf_links = pk.Transform3d(matrix=base_mats_for_compose)
            self.object_to_link_frames = self.link_transforms_root_frame.compose(
                base_tsf_links
            )

        else:
            # Single base transform for all chains
            base_mat = self.base_transform.get_matrix()  # (..., 4, 4)

            if self.configuration_batch is not None:
                B_total = 1
                for d in self.configuration_batch:
                    B_total *= d

                if base_mat.shape[:-2] != self.configuration_batch:
                    # Try to broadcast or raise error
                    # Assuming base_mat matches batch
                    pass

                base_mats_for_compose = base_mat.unsqueeze(0).expand(
                    len(self.sdf_to_link_name), *base_mat.shape
                )
                base_mats_for_compose = base_mats_for_compose.reshape(-1, 4, 4)
            else:
                base_mats_for_compose = base_mat.unsqueeze(0).expand(
                    len(self.sdf_to_link_name), *base_mat.shape
                )
                base_mats_for_compose = base_mats_for_compose.reshape(-1, 4, 4)

            base_tsf_links = pk.Transform3d(matrix=base_mats_for_compose)
            self.object_to_link_frames = self.link_transforms_root_frame.compose(
                base_tsf_links
            )

    def __call__(self, points_in_object_frame):
        """
        Query for SDF value and SDF gradients for points in the robot's frame.
        Note: If multiple chains with different base transforms are used, 'points_in_object_frame'
        should actually be 'points_in_world_frame'.

        :param points_in_object_frame: [B x] N x 3 optionally arbitrarily batched points. If base_transform is set,
        these points are interpreted as being in the world frame; otherwise, they are in the robot's object frame.
        :return: [A x] [B x] N SDF value, and [A x] [B x] N x 3 SDF gradient. A are the configurations' arbitrary
        number of batch dimensions.
        """
        # Since we bake the base_transform into the SDF transforms (in _update_sdf_transforms),
        # self.sdf handles everything. We pass the points directly.
        # If base_transform is set, the points are expected to be in World frame.
        # If base_transform is None, the points are expected to be in Object (Root) frame.

        sdf_val, sdf_grad = self.sdf(points_in_object_frame)
        return sdf_val, sdf_grad


def cache_link_sdf_factory(resolution=0.01, padding=0.1, **kwargs):
    def create_sdf(obj_factory: sdf.ObjectFactory):
        gt_sdf = sdf.MeshSDF(obj_factory)
        return sdf.CachedSDF(
            obj_factory.name,
            resolution,
            obj_factory.bounding_box(padding=padding),
            gt_sdf,
            **kwargs,
        )

    return create_sdf


def aabb_to_ordered_end_points(aabb, arrange_in_sequential_order=False):
    aabbMin = aabb[:, 0]
    aabbMax = aabb[:, 1]
    if arrange_in_sequential_order:
        arr = [
            [aabbMin[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMin[1], aabbMin[2]],
            [aabbMin[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMax[1], aabbMax[2]],
            [aabbMax[0], aabbMax[1], aabbMin[2]],
            [aabbMax[0], aabbMax[1], aabbMax[2]],
            [aabbMin[0], aabbMax[1], aabbMax[2]],
            [aabbMin[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMax[1], aabbMax[2]],
            [aabbMin[0], aabbMin[1], aabbMax[2]],
        ]
    else:
        arr = [
            [aabbMin[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMin[1], aabbMin[2]],
            [aabbMin[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMin[1], aabbMax[2]],
            [aabbMin[0], aabbMax[1], aabbMax[2]],
            [aabbMax[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMax[1], aabbMin[2]],
            [aabbMax[0], aabbMax[1], aabbMax[2]],
        ]
    if torch.is_tensor(aabb):
        return torch.tensor(arr, device=aabb.device, dtype=aabb.dtype)
    return np.array(arr)
