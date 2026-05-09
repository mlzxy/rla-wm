"""
Utility functions for extracting and manipulating controller target qpos.

These functions provide reverse operations for CombinedController.set_action:
- extract_target_qpos_from_controller: Extracts _target_qpos from each sub-controller
  (if sets_target_qpos is True) and merges them into a single vector based on action_mapping
- split_target_qpos_to_controller_dict: Splits a merged vector back into a dict
  of vectors (one for each sub-controller) based on action_mapping
"""

import numpy as np
import torch
from typing import Dict, Optional, Union
from mani_skill.agents.controllers.base_controller import CombinedController


def extract_target_qpos_from_controller(
    controller: CombinedController, return_numpy: bool = True
) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Extract all _target_qpos from each sub-controller (if sets_target_qpos is True)
    and merge them into a single vector based on action_mapping.

    This is the reverse operation of CombinedController.set_action.

    Args:
        controller: A CombinedController instance
        return_numpy: If True, return numpy array; if False, return torch tensor

    Returns:
        Merged target qpos vector of shape (num_envs, total_action_dim),
        or None if no controller has sets_target_qpos=True
    """
    num_envs = controller.scene.num_envs

    # Determine total action dimension
    if controller.scene.num_envs > 1:
        total_action_dim = controller.action_space.shape[1]
    else:
        total_action_dim = controller.action_space.shape[0]

    # Initialize output tensor
    device = controller.device
    merged_target_qpos = torch.full(
        (num_envs, total_action_dim),
        fill_value=-1000.0,
        device=device,
        dtype=torch.float32,
    )

    # Extract _target_qpos from each sub-controller and place in merged vector
    for uid, sub_controller in controller.controllers.items():
        if not sub_controller.sets_target_qpos:
            continue

        # Get the action slice indices for this controller
        start, end = controller.action_mapping[uid]
        action_dim = end - start

        # Extract _target_qpos from this controller
        if (
            hasattr(sub_controller, "_target_qpos")
            and sub_controller._target_qpos is not None
        ):
            target_qpos = sub_controller._target_qpos

            # target_qpos has shape (num_envs, num_active_joints)
            # We need to extract the dimensions that correspond to the action space

            # Check if this is a mimic controller (has control_joint_indices)
            if hasattr(sub_controller, "control_joint_indices"):
                # For mimic controllers, extract only the control joints
                control_indices = sub_controller.control_joint_indices
                # Extract target_qpos for control joints only
                target_qpos_control = target_qpos[:, control_indices]

                # The action dimension should match the number of control joints
                if target_qpos_control.shape[1] == action_dim:
                    target_qpos_slice = target_qpos_control
                elif target_qpos_control.shape[1] > action_dim:
                    # Take first action_dim dimensions
                    target_qpos_slice = target_qpos_control[:, :action_dim]
                else:
                    # Pad if needed (shouldn't happen normally)
                    padded = torch.zeros(
                        (num_envs, action_dim), device=device, dtype=target_qpos.dtype
                    )
                    padded[:, : target_qpos_control.shape[1]] = target_qpos_control
                    target_qpos_slice = padded
            else:
                # For regular controllers, check if action includes velocity
                # (e.g., PDJointPosVelController has action_dim = 2 * num_joints)
                # In that case, _target_qpos corresponds to the first half (position part)
                num_joints = target_qpos.shape[1]

                if action_dim == num_joints:
                    # Position-only controller: use all of target_qpos
                    target_qpos_slice = target_qpos
                elif action_dim == 2 * num_joints:
                    # Position+velocity controller: extract only position part
                    # The first num_joints dimensions of action correspond to position
                    # Note: We're only extracting position, but the merged vector needs
                    # to accommodate the full action_dim, so we pad with zeros for velocity part
                    padded = torch.zeros(
                        (num_envs, action_dim), device=device, dtype=target_qpos.dtype
                    )
                    padded[:, :num_joints] = target_qpos
                    target_qpos_slice = padded
                elif action_dim < num_joints:
                    # Action controls subset of joints (shouldn't happen for standard controllers)
                    target_qpos_slice = target_qpos[:, :action_dim]
                else:
                    # Action has more dimensions than joints (unexpected case)
                    # Use target_qpos and pad the rest with zeros
                    padded = torch.zeros(
                        (num_envs, action_dim), device=device, dtype=target_qpos.dtype
                    )
                    padded[:, :num_joints] = target_qpos
                    target_qpos_slice = padded

            # Place in the merged vector at the correct position
            merged_target_qpos[:, start:end] = target_qpos_slice

    if return_numpy:
        return merged_target_qpos.cpu().numpy()
    else:
        return merged_target_qpos


def split_target_qpos_to_controller_dict(
    controller: CombinedController, merged_target_qpos: Union[np.ndarray, torch.Tensor]
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Split a merged target qpos vector back into a dict of vectors
    (one for each sub-controller) based on action_mapping.

    This is the reverse operation of extract_target_qpos_from_controller.

    Args:
        controller: A CombinedController instance
        merged_target_qpos: Merged target qpos vector of shape (num_envs, total_action_dim)

    Returns:
        Dictionary mapping controller UIDs to their target qpos vectors
        Only includes controllers where sets_target_qpos is True
    """
    # Convert to tensor if needed
    if isinstance(merged_target_qpos, np.ndarray):
        merged_target_qpos = torch.from_numpy(merged_target_qpos).to(controller.device)
    else:
        merged_target_qpos = merged_target_qpos.to(controller.device)

    num_envs = controller.scene.num_envs

    # Sanity check
    if controller.scene.num_envs > 1:
        expected_dim = controller.action_space.shape[1]
    else:
        expected_dim = controller.action_space.shape[0]

    assert merged_target_qpos.shape == (num_envs, expected_dim), (
        f"Expected shape ({num_envs}, {expected_dim}), got {merged_target_qpos.shape}"
    )

    # Split the merged vector into sub-controller vectors
    controller_dict = {}

    for uid, sub_controller in controller.controllers.items():
        if not sub_controller.sets_target_qpos:
            continue

        # Get the action slice indices for this controller
        start, end = controller.action_mapping[uid]

        # Extract the slice for this controller
        target_qpos_slice = merged_target_qpos[:, start:end]

        controller_dict[uid] = target_qpos_slice

    return controller_dict
