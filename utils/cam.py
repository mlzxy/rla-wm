import numpy as np

def convert_extrinsics_3x4_to_4x4(extrinsics_data: np.ndarray) -> np.ndarray:
    """Convert extrinsics matrices from 3x4 to 4x4 format.
    
    Args:
        extrinsics_data: Extrinsics data with shape (..., 3, 4) where the last two
            dimensions are (3, 4). Can handle shapes like (3, 4), (T, 3, 4), 
            (1, 3, 4), (Cams, 1, 3, 4), etc.
    
    Returns:
        Extrinsics data with shape (..., 4, 4) where the 3x4 matrix is converted
        to 4x4 by adding [0, 0, 0, 1] as the last row.
    """
    if not isinstance(extrinsics_data, np.ndarray):
        return extrinsics_data
    
    orig_shape = extrinsics_data.shape
    # Handle different shapes: (3, 4), (T, 3, 4), (1, 3, 4), (Cams, 1, 3, 4), etc.
    if len(orig_shape) >= 2 and orig_shape[-2:] == (3, 4):
        # Reshape to (..., 3, 4) for easier processing
        flat_shape = (-1, 3, 4)
        reshaped = extrinsics_data.reshape(flat_shape)
        # Convert each 3x4 matrix to 4x4
        extrinsics_4x4 = np.zeros((reshaped.shape[0], 4, 4), dtype=extrinsics_data.dtype)
        extrinsics_4x4[:, :3, :] = reshaped
        extrinsics_4x4[:, 3, 3] = 1.0
        # Reshape back to original shape but with 4x4 instead of 3x4
        new_shape = orig_shape[:-2] + (4, 4)
        return extrinsics_4x4.reshape(new_shape)
    else:
        # Already 4x4 or unexpected shape, return as-is
        return extrinsics_data