import h5py
import os.path as osp
import pickle
import subprocess
import shutil
import numpy as np
import cv2
import json
import os
import glob
import re
from typing import Dict, Optional, List, Any, Union, Tuple, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from rich import print
import third_party.pytorch_kinematics as pk
from utils.vis import to_pil
try:
    import decord

    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("[yellow]Decord not installed, using fallback opencv[/yellow]")

if os.environ.get("DISABLE_DECORD", "0") != "0":
    HAS_DECORD = False
    print("[yellow]DISABLE_DECORD is set, using fallback opencv[/yellow]")


try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False




def resize_image(
    img: np.ndarray, target_size: int, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize an image to target_size x target_size.

    Args:
        img: Input image of shape [H, W] or [H, W, C]
        target_size: Target size (both height and width will be target_size)
        interpolation: OpenCV interpolation method

    Returns:
        Resized image of shape [target_size, target_size] or [target_size, target_size, C]
    """
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    return cv2.resize(img, (target_size, target_size), interpolation=interpolation)


def resize_intrinsics(
    intrinsics: np.ndarray, orig_h: int, orig_w: int, new_h: int, new_w: int
) -> np.ndarray:
    """
    Adjust camera intrinsics matrix for image resize.

    Args:
        intrinsics: Camera intrinsics matrix of shape [..., 3, 3]
        orig_h, orig_w: Original image dimensions
        new_h, new_w: New image dimensions

    Returns:
        Adjusted intrinsics matrix with same shape
    """
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # Create a copy to avoid modifying original
    adjusted = intrinsics.copy()

    # Scale focal lengths and principal point
    adjusted[..., 0, 0] *= scale_x  # fx
    adjusted[..., 0, 2] *= scale_x  # cx
    adjusted[..., 1, 1] *= scale_y  # fy
    adjusted[..., 1, 2] *= scale_y  # cy

    return adjusted


class KeyType(Enum):
    """Type of key for video encoding/decoding."""

    RGB = "rgb"  # Standard RGB video
    DEPTH = "depth"  # Depth encoded as uint16
    MASK = "mask"  # Binary mask
    GRAYSCALE = "grayscale"  # Grayscale video


class VideoEncoding(Enum):
    """Video encoding format for depth, mask, and grayscale streams. Stored in metadata for consistent readback."""

    FFV1 = "ffv1"  # Lossless, OpenCV FFV1 → .avi
    H264_LOSSLESS = "h264_lossless"  # Lossless H.264 (qp=0, yuv444p) → .mp4
    H264_CRF23 = "h264_crf23"  # High-quality lossy H.264 (crf=23) → .mp4
    H264_CRF28 = "h264_crf28"  # Higher compression H.264 (crf=28) → .mp4


def encode_depth_int16_to_uint8(
    depth_int16: np.ndarray, foreground_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Encode int16 depth (in millimeters) to uint8 using min/max normalization.
    Min/max are computed only from foreground pixels if mask is provided.

    Args:
        depth_int16: Depth array in int16 (millimeters), shape (H, W)
        foreground_mask: Optional boolean mask for foreground pixels, shape (H, W)
                        If provided, min/max are computed only from foreground pixels

    Returns:
        Tuple of:
        - uint8 depth image with shape (H, W), normalized to 0-255
        - min_depth: Minimum depth value (from foreground if mask provided)
        - max_depth: Maximum depth value (from foreground if mask provided)
    """
    # Compute min/max from foreground pixels only if mask is provided
    if foreground_mask is not None:
        # Ensure mask is boolean
        if foreground_mask.dtype != bool:
            foreground_mask = foreground_mask.astype(bool)

        # Get foreground pixels
        foreground_pixels = depth_int16[foreground_mask]

        if len(foreground_pixels) > 0:
            min_depth = float(np.min(foreground_pixels))
            max_depth = float(np.max(foreground_pixels))
        else:
            # No foreground pixels, use full image
            min_depth = float(np.min(depth_int16))
            max_depth = float(np.max(depth_int16))
    else:
        # No mask, use full image
        min_depth = float(np.min(depth_int16))
        max_depth = float(np.max(depth_int16))

    # Initialize output array (will set background to 255 later if mask provided)
    depth_uint8 = np.zeros_like(depth_int16, dtype=np.uint8)

    # Handle edge case where min == max (constant depth)
    if max_depth == min_depth:
        # For constant depth, set foreground to a middle value (127)
        if foreground_mask is not None:
            if foreground_mask.dtype != bool:
                foreground_mask = foreground_mask.astype(bool)
            depth_uint8[foreground_mask] = 127
        else:
            depth_uint8.fill(127)
    else:
        # Normalize foreground pixels to 0-254 range (leaving 255 for background)
        if foreground_mask is not None:
            # Ensure mask is boolean
            if foreground_mask.dtype != bool:
                foreground_mask = foreground_mask.astype(bool)

            # Normalize only foreground pixels
            foreground_pixels = depth_int16[foreground_mask].astype(np.float32)
            depth_normalized = (foreground_pixels - min_depth) / (max_depth - min_depth)
            depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
            # Map to 0-254 range (leaving 255 for background)
            depth_uint8[foreground_mask] = np.round(depth_normalized * 254.0).astype(np.uint8)
            # Set background pixels to 255
            depth_uint8[~foreground_mask] = 255
        else:
            # No mask: normalize all pixels to 0-254 (no background)
            depth_normalized = (depth_int16.astype(np.float32) - min_depth) / (
                max_depth - min_depth
            )
            depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
            depth_uint8 = np.round(depth_normalized * 254.0).astype(np.uint8)

    return depth_uint8, min_depth, max_depth


def decode_depth_uint8_to_int16(
    depth_uint8: np.ndarray,
    min_depth: float,
    max_depth: float,
    background_value: int = 255,
) -> np.ndarray:
    """
    Decode uint8 depth image back to int16 depth (in millimeters) using min/max denormalization.
    Background pixels (value 255 by default) are set to 0.

    Args:
        depth_uint8: uint8 depth image with shape (H, W), normalized to 0-255
        min_depth: Minimum depth value used for normalization
        max_depth: Maximum depth value used for normalization
        background_value: Value used to mark background pixels (default 255)

    Returns:
        Depth array in int16 (millimeters), shape (H, W)
        Background pixels are set to 0
    """
    # Create output array
    depth_int16 = np.zeros_like(depth_uint8, dtype=np.int16)

    # Identify foreground pixels (not background)
    foreground_mask = depth_uint8 != background_value

    # Handle edge case where min == max
    if max_depth == min_depth:
        # Set foreground pixels to constant value
        depth_int16[foreground_mask] = int(min_depth)
    else:
        # Denormalize from 0-255 back to original range for foreground pixels only
        # Note: We use 254 as max to avoid conflict with background value 255
        # This means foreground pixels are normalized to 0-254 range
        depth_normalized = depth_uint8[foreground_mask].astype(np.float32) / 254.0
        depth_int16[foreground_mask] = np.round(
            depth_normalized * (max_depth - min_depth) + min_depth
        ).astype(np.int16)

    # Background pixels remain 0 (already set by zeros_like)
    return depth_int16


@dataclass
class KeyConfig:
    """Configuration for how a key should be treated."""

    key_type: KeyType = KeyType.RGB
    depth_scaler: Optional[float] = None  # Only used for DEPTH type
    depth_encoding: Union[VideoEncoding, str] = VideoEncoding.FFV1  # DEPTH codec/level
    mask_encoding: Union[VideoEncoding, str] = VideoEncoding.H264_LOSSLESS  # MASK codec/level
    threshold: float = 127.0  # Only used for MASK type


@dataclass
class TrajectoryData:
    """Structured trajectory data with explicit separation of video streams and metadata."""

    success: bool
    video_streams: Dict[str, np.ndarray] = field(
        default_factory=dict
    )  # Video data: {key: frames_array}
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Non-video data: {key: value}


@dataclass
class RobotInfo:
    uid: str
    urdf_path: str
    urdf_config: dict
    joint_names: list[str]
    action_space: tuple[list[float], list[float], list[int]]
    action_mapping: dict[str, tuple[int, int]]


def _resolve_video_encoding(encoding: Union[VideoEncoding, str]) -> VideoEncoding:
    """Normalize video encoding to enum."""
    if isinstance(encoding, VideoEncoding):
        return encoding
    return VideoEncoding(str(encoding).lower())


def _video_encoding_extension(encoding: VideoEncoding) -> str:
    """File extension for the video (e.g. .avi for FFV1, .mp4 for H264)."""
    if encoding == VideoEncoding.FFV1:
        return ".avi"
    return ".mp4"


def _get_ffmpeg_exe() -> Optional[str]:
    """Return path to ffmpeg (imageio-ffmpeg or system)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _start_ffmpeg_grayscale_h264(
    path: str, width: int, height: int, fps: int, encoding: VideoEncoding
) -> Optional[subprocess.Popen]:
    """
    Start ffmpeg to encode raw grayscale (H, W) frames to H264. Returns process or None.
    Piping (H, W) avoids 3x memory from duplicating channels.
    """
    exe = _get_ffmpeg_exe()
    if not exe:
        return None
    # Input: raw grayscale from stdin
    cmd = [
        exe,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        "{}x{}".format(width, height),
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
    ]
    if encoding == VideoEncoding.H264_LOSSLESS:
        cmd.extend(["-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv444p"])
    elif encoding == VideoEncoding.H264_CRF23:
        cmd.extend(["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p"])
    elif encoding == VideoEncoding.H264_CRF28:
        cmd.extend(["-crf", "28", "-preset", "fast", "-pix_fmt", "yuv420p"])
    else:
        return None
    cmd.append(path)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except Exception:
        return None


class VideoRecorder:
    def __init__(
        self, path: str, fps: int, res: tuple, key_config: Optional[KeyConfig] = None
    ):
        self.path = path
        self.fps = fps
        self.width, self.height = res
        self.key_config = key_config or KeyConfig()
        self.writer = None
        self._imageio_writer = None  # Fallback when ffmpeg pipe not used
        self._ffmpeg_process = None  # H264 depth via raw grayscale pipe (no channel dup)
        # Track min/max depth values per frame for depth encoding
        self.depth_min_max = []  # List of (min, max) tuples per frame

    def init_writer(self):
        # Determine codec and extension based on key type
        if self.key_config.key_type in (KeyType.DEPTH, KeyType.MASK, KeyType.GRAYSCALE):
            if self.key_config.key_type == KeyType.DEPTH:
                encoding = _resolve_video_encoding(self.key_config.depth_encoding)
            elif self.key_config.key_type == KeyType.MASK:
                encoding = _resolve_video_encoding(self.key_config.mask_encoding)
            else: # GRAYSCALE
                encoding = VideoEncoding.H264_LOSSLESS # Default for grayscale

            ext = _video_encoding_extension(encoding)
            if not self.path.endswith(ext):
                self.path = os.path.splitext(self.path)[0] + ext

            if encoding == VideoEncoding.FFV1:
                fourcc = cv2.VideoWriter_fourcc(*"FFV1")
                self.writer = cv2.VideoWriter(
                    self.path, fourcc, self.fps, (self.width, self.height), False
                )
                self._encoding_used = encoding
                return

            # H264: prefer ffmpeg pipe with raw (H, W) grayscale to avoid 3x memory
            proc = _start_ffmpeg_grayscale_h264(
                self.path, self.width, self.height, self.fps, encoding
            )
            if proc is not None:
                self._ffmpeg_process = proc
                self._encoding_used = encoding
                return

            # Fallback: imageio with (H, W, 3) by stacking channels
            if not HAS_IMAGEIO:
                raise RuntimeError(
                    "Encoding {} requires ffmpeg (on PATH or imageio-ffmpeg) or imageio. "
                    "Install: pip install imageio imageio-ffmpeg".format(encoding.value)
                )
            
            if encoding == VideoEncoding.H264_LOSSLESS:
                self._imageio_writer = imageio.get_writer(
                    self.path,
                    fps=self.fps,
                    codec="libx264",
                    pixelformat="yuv444p",
                    ffmpeg_params=["-qp", "0", "-preset", "ultrafast"],
                    macro_block_size=None,
                )
            elif encoding == VideoEncoding.H264_CRF23:
                self._imageio_writer = imageio.get_writer(
                    self.path,
                    fps=self.fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    ffmpeg_params=["-crf", "23", "-preset", "fast"],
                    macro_block_size=None,
                )
            elif encoding == VideoEncoding.H264_CRF28:
                self._imageio_writer = imageio.get_writer(
                    self.path,
                    fps=self.fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    ffmpeg_params=["-crf", "28", "-preset", "fast"],
                    macro_block_size=None,
                )
            else:
                raise ValueError("Unsupported encoding: {}".format(encoding))
            self._encoding_used = encoding
            return

        # RGB or default
        ext = ".mp4"
        if not self.path.endswith(ext):
            self.path = os.path.splitext(self.path)[0] + ext
        
        if HAS_IMAGEIO:
            self._imageio_writer = imageio.get_writer(
                self.path,
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
                ffmpeg_params=["-crf", "23", "-preset", "fast"],
                macro_block_size=None,
            )
            self._encoding_used = "h264_crf23"
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.path, fourcc, self.fps, (self.width, self.height), True
            )
            self._encoding_used = "mp4v"

    def write(self, frame: np.ndarray, foreground_mask: Optional[np.ndarray] = None):
        if (
            self.writer is None
            and self._imageio_writer is None
            and self._ffmpeg_process is None
        ):
            self.init_writer()

        if self.key_config.key_type == KeyType.DEPTH:
            # Convert depth to int16 if needed (ManiSkill depth is already int16 in mm)
            if frame.dtype != np.int16:
                frame = frame.astype(np.int16)
            # Encode int16 depth to uint8 using min/max normalization
            depth_uint8, min_depth, max_depth = encode_depth_int16_to_uint8(
                frame, foreground_mask
            )
            self.depth_min_max.append((min_depth, max_depth))
            if self._ffmpeg_process is not None:
                # Pipe raw (H, W) grayscale to ffmpeg — no extra memory for channels
                self._ffmpeg_process.stdin.write(
                    np.ascontiguousarray(depth_uint8).tobytes()
                )
            elif self._imageio_writer is not None:
                # Fallback: imageio expects RGB (H, W, 3)
                frame_rgb = np.stack([depth_uint8] * 3, axis=-1)
                self._imageio_writer.append_data(frame_rgb)
            else:
                self.writer.write(depth_uint8)
        elif self.key_config.key_type == KeyType.MASK:
            # Convert mask to 1-channel or 3-channel depending on writer
            if frame.dtype == bool:
                mask_vis = frame.astype(np.uint8) * 255
            else:
                mask_vis = (frame > self.key_config.threshold).astype(np.uint8) * 255
            
            if self._ffmpeg_process is not None:
                # Pipe raw (H, W) grayscale to ffmpeg
                self._ffmpeg_process.stdin.write(
                    np.ascontiguousarray(mask_vis).tobytes()
                )
            elif self._imageio_writer is not None:
                # imageio expects RGB (H, W, 3) for the current h264 setup
                mask_rgb = np.stack([mask_vis] * 3, axis=-1)
                self._imageio_writer.append_data(mask_rgb)
            else:
                # Legacy OpenCV writer
                mask_3ch = np.stack([mask_vis] * 3, axis=-1)
                self.writer.write(mask_3ch)
        elif self.key_config.key_type == KeyType.GRAYSCALE:
            # Convert grayscale to 1-channel or 3-channel
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            if self._ffmpeg_process is not None:
                self._ffmpeg_process.stdin.write(
                    np.ascontiguousarray(gray).tobytes()
                )
            elif self._imageio_writer is not None:
                gray_rgb = np.stack([gray] * 3, axis=-1)
                self._imageio_writer.append_data(gray_rgb)
            else:
                gray_3ch = np.stack([gray] * 3, axis=-1)
                self.writer.write(gray_3ch)
        else:  # RGB or default
            # Expecting RGB input
            if self._imageio_writer is not None:
                # imageio h264 writer expects RGB
                self._imageio_writer.append_data(frame)
            else:
                # OpenCV writer expects BGR
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.writer.write(frame)

    def close(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        if self._ffmpeg_process is not None:
            self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None
        if self._imageio_writer is not None:
            self._imageio_writer.close()
            self._imageio_writer = None

        # Save depth min/max and encoding format to JSON so readers can decode any format
        if self.key_config.key_type == KeyType.DEPTH and len(self.depth_min_max) > 0:
            json_path = os.path.splitext(self.path)[0] + "_depth_minmax.json"
            encoding_used = getattr(
                self, "_encoding_used", VideoEncoding.FFV1
            )
            min_max_data = {
                "encoding": encoding_used.value if hasattr(encoding_used, "value") else str(encoding_used),
                "frames": [
                    {"min": float(min_val), "max": float(max_val)}
                    for min_val, max_val in self.depth_min_max
                ],
            }
            with open(json_path, "w") as f:
                json.dump(min_max_data, f, indent=2)
        
        # Save encoding info for mask/grayscale even if no min/max
        elif self.key_config.key_type in (KeyType.MASK, KeyType.GRAYSCALE):
            json_path = os.path.splitext(self.path)[0] + "_encoding.json"
            encoding_used = getattr(self, "_encoding_used", VideoEncoding.H264_LOSSLESS)
            enc_data = {
                "encoding": encoding_used.value if hasattr(encoding_used, "value") else str(encoding_used)
            }
            with open(json_path, "w") as f:
                json.dump(enc_data, f, indent=2)


class ManiSkillTrajectoryDataset:
    _FRAME_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    _DEFAULT_FORCED_FRAME_ALIGNED_METADATA_KEYS = frozenset(
        {"eef_pose", "object_poses"}
    )

    def __init__(
        self,
        root_dir: str,
        key_configs: Optional[Dict[str, KeyConfig]] = None,
        depth_scaler: float = 1000.0,
        depth_encoding: Union[VideoEncoding, str] = VideoEncoding.FFV1,
        mask_encoding: Union[VideoEncoding, str] = VideoEncoding.H264_LOSSLESS,
        start_traj_id: Optional[str] = None,
        end_traj_id: Optional[str] = None,
        force_reindex: bool = False,
        rgb_variant_mode: str = "base",
        forced_frame_aligned_metadata_keys: Optional[List[str]] = None,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory for the dataset
            key_configs: Dictionary mapping key names to KeyConfig objects
                         Determines how each key is encoded/decoded as video
            depth_scaler: Default scaler for depth keys (meters to mm)
            depth_encoding: Default depth video encoding (FFV1, h264_lossless, h264_crf23, h264_crf28).
                           Used for keys inferred as DEPTH from suffix; overridable per key via key_configs.
            mask_encoding: Default mask video encoding (h264_lossless, h264_crf23, h264_crf28).
                           Used for keys inferred as MASK from suffix; overridable per key via key_configs.
            start_traj_id: Optional inclusive start trajectory ID filter.
                           If provided, only trajectories >= this ID (natural order) are visible.
            end_traj_id: Optional inclusive end trajectory ID filter.
                         If provided, only trajectories <= this ID (natural order) are visible.
            rgb_variant_mode: RGB reading mode when video_keys is None in read_trajectory:
                - "base": read only `*_rgb` streams
                - "wo_robot": read only `*_rgb.wo_robot` streams
                - "both": read both when available
            forced_frame_aligned_metadata_keys: Metadata keys to force-treat as
                frame-aligned when reading, even if `_frame_aligned` is not set
                in stored data. Defaults to ["eef_pose", "object_poses"].
        """
        self.root = root_dir
        os.makedirs(self.root, exist_ok=True)
        self.metadata_path = os.path.join(self.root, "metadata.json")
        self.index_path = os.path.join(self.root, "index.json")
        self.depth_scaler = depth_scaler
        self.depth_encoding = depth_encoding
        self.mask_encoding = mask_encoding
        self.start_traj_id = start_traj_id
        self.end_traj_id = end_traj_id
        if rgb_variant_mode not in {"base", "wo_robot", "both"}:
            raise ValueError(
                f"Invalid rgb_variant_mode={rgb_variant_mode}. "
                "Expected one of: base, wo_robot, both"
            )
        self.rgb_variant_mode = rgb_variant_mode
        if forced_frame_aligned_metadata_keys is None:
            self.forced_frame_aligned_metadata_keys = set(
                self._DEFAULT_FORCED_FRAME_ALIGNED_METADATA_KEYS
            )
        else:
            self.forced_frame_aligned_metadata_keys = set(
                forced_frame_aligned_metadata_keys
            )

        # Key configurations for video encoding/decoding
        self.key_configs = key_configs or {}

        if (
            self.start_traj_id is not None
            and self.end_traj_id is not None
            and self._traj_id_sort_key(self.start_traj_id)
            > self._traj_id_sort_key(self.end_traj_id)
        ):
            raise ValueError(
                f"start_traj_id ({self.start_traj_id}) must be <= end_traj_id ({self.end_traj_id})"
            )

        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w") as f:
                json.dump({"robot_infos": []}, f, indent=2)

        # Load index if it exists, otherwise build it
        self._index = None
        self._load_index()
        if self._index is None or force_reindex:
            self.build_index(force=True, verbose=False)

    def _traj_id_sort_key(self, traj_id: str):
        """
        Create a natural sort key for trajectory IDs.
        Example: traj_2 < traj_10 and 001_front < 010_front.
        """
        parts = re.split(r"(\d+)", str(traj_id))
        return tuple(int(p) if p.isdigit() else p.lower() for p in parts)

    def _traj_id_in_range(self, traj_id: str) -> bool:
        """Check whether trajectory ID is inside configured inclusive range."""
        key = self._traj_id_sort_key(traj_id)
        if self.start_traj_id is not None:
            if key < self._traj_id_sort_key(self.start_traj_id):
                return False
        if self.end_traj_id is not None:
            if key > self._traj_id_sort_key(self.end_traj_id):
                return False
        return True

    def _filter_trajectory_ids(self, traj_ids: List[str]) -> List[str]:
        """Filter trajectory IDs according to configured start/end range."""
        if self.start_traj_id is None and self.end_traj_id is None:
            return traj_ids
        return [traj_id for traj_id in traj_ids if self._traj_id_in_range(traj_id)]

    def save_robot_infos(self, robot_infos: Union[List[RobotInfo], RobotInfo], replace: bool=True):
        """
        Save robot information. Merges with existing metadata instead of overwriting.

        Args:
            robot_infos: RobotInfo object or list of RobotInfo objects
        """
        # Convert single RobotInfo to list
        if isinstance(robot_infos, RobotInfo):
            robot_infos = [robot_infos]

        # Convert RobotInfo objects to dictionaries for JSON serialization
        robot_info_dicts = [asdict(ri) for ri in robot_infos]

        # Load existing metadata if it exists
        existing_data = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                existing_data = json.load(f)

        # Initialize robot_infos list if it doesn't exist
        if "robot_infos" not in existing_data:
            existing_data["robot_infos"] = []

        # Extend with new robot infos
        if replace:
            existing_data["robot_infos"] = robot_info_dicts
        else:
            existing_data["robot_infos"].extend(robot_info_dicts)

        with open(self.metadata_path, "w") as f:
            json.dump(existing_data, f, indent=2)

    def get_robot_infos(self, dedup=True) -> Optional[List[RobotInfo]]:
        """
        Get the robot information, returns None if it doesn't exist.

        Args:
            dedup: Whether to deduplicate robot infos

        Returns:
            List of RobotInfo objects, or None if not found
        """
        if not os.path.exists(self.metadata_path):
            return None
        uids = set()
        with open(self.metadata_path, "r") as f:
            data = json.load(f)
            robot_infos_data = data.get("robot_infos")
            if robot_infos_data is None:
                return None
            robot_infos = []
            for item in robot_infos_data:
                if dedup and item["uid"] in uids:
                    continue
                stl_urdf_path = item["urdf_path"].replace(".urdf", ".stl.urdf")
                if osp.exists(stl_urdf_path):
                    item["urdf_path"] = stl_urdf_path
                robot_infos.append(RobotInfo(**item))
                uids.add(item["uid"])
            return robot_infos

    def save_robot_urdfs(self, urdf_paths: list[str]):
        """
        Save robot URDF paths (legacy method for backward compatibility).
        This method is deprecated - use save_robot_infos instead.

        Args:
            urdf_paths: List of URDF file paths (strings)
        """
        # Convert URDF paths to minimal RobotInfo objects
        robot_infos = [
            RobotInfo(
                uid="unknown",
                urdf_path=path,
                urdf_config={},
                joint_names=[],
                action_space=(0.0, 0.0, []),
                action_mapping={},
            )
            for path in urdf_paths
        ]
        self.save_robot_infos(robot_infos)

    def get_robot_urdfs(self) -> Optional[list[str]]:
        """
        Get the robot URDF paths (legacy method for backward compatibility).
        This method is deprecated - use get_robot_infos instead.

        Returns:
            List of URDF file paths (strings), or None if not found
        """
        robot_infos = self.get_robot_infos()
        if robot_infos is None:
            return None
        return [ri.urdf_path for ri in robot_infos]

    def _load_index(self):
        """Load index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = None
        else:
            self._index = None

    def build_index(self, force: bool = False, verbose: bool = True):
        """
        Build an index of all trajectories and their keys for fast lookup.

        Args:
            force: If True, rebuild index even if it exists
            verbose: If True, print progress
        """
        if self._index is not None and not force:
            if verbose:
                print(
                    f"Index already exists at {self.index_path}. Use force=True to rebuild."
                )
            return

        if verbose:
            print("Building index...")

        index = {"trajectories": {}, "trajectory_list": []}

        # Scan all trajectory directories
        traj_dirs = sorted(glob.glob(os.path.join(self.root, "traj_*")))
        total = len(traj_dirs)

        for i, traj_dir in enumerate(traj_dirs):
            traj_id = os.path.basename(traj_dir).replace("traj_", "")
            index["trajectory_list"].append(traj_id)

            # Get keys for this trajectory
            keys = self._list_keys_scan(traj_id)
            index["trajectories"][traj_id] = {"keys": keys, "path": traj_dir}

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total} trajectories...")

        # Save index
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

        self._index = index

        if verbose:
            print(
                f"Index built successfully: {len(index['trajectory_list'])} trajectories"
            )

    def _list_keys_scan(self, traj_id: str) -> List[str]:
        """
        Scan filesystem to list keys for a trajectory (used during indexing).
        Supports both video-file streams and frame-directory streams.
        """
        traj_dir = os.path.join(self.root, f"traj_{traj_id}")
        if not os.path.exists(traj_dir):
            return []

        keys = set()

        # Keys from H5 file
        metadata_path = os.path.join(traj_dir, "metadata.h5")
        if os.path.exists(metadata_path):
            try:
                with h5py.File(metadata_path, "r") as f:
                    keys.update(f.keys())
            except (OSError, IOError):
                pass

        # Keys from video files
        pattern = os.path.join(traj_dir, "*.*")
        try:
            files = glob.glob(pattern)
            for f in files:
                basename = os.path.basename(f)
                if basename.startswith("traj_"):
                    continue
                if "_scaler" in basename:
                    key = basename.split("_scaler")[0]
                else:
                    key = os.path.splitext(basename)[0]
                if key and key != "metadata":
                    keys.add(key)
        except (OSError, IOError):
            pass

        # Keys from frame directories (e.g. traj_xxx/cam0_rgb/000001.jpg)
        try:
            for entry in os.listdir(traj_dir):
                entry_path = os.path.join(traj_dir, entry)
                if not os.path.isdir(entry_path):
                    continue
                if entry.startswith("traj_"):
                    continue
                if len(self._list_frame_images(entry_path)) > 0:
                    keys.add(entry)
        except (OSError, IOError):
            pass

        return sorted(keys)

    @staticmethod
    def _frame_sort_key(name: str):
        parts = re.split(r"(\d+)", str(name))
        return tuple(int(p) if p.isdigit() else p.lower() for p in parts)

    def _list_frame_images(
        self, frames_dir: str, allowed_exts: Optional[set[str]] = None
    ) -> List[str]:
        if not os.path.isdir(frames_dir):
            return []
        valid_exts = (
            set(e.lower() for e in allowed_exts)
            if allowed_exts is not None
            else set(self._FRAME_IMAGE_EXTENSIONS)
        )
        frame_paths = []
        for entry in os.listdir(frames_dir):
            path = os.path.join(frames_dir, entry)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(entry)[1].lower() in valid_exts:
                frame_paths.append(path)
        frame_paths.sort(key=lambda p: self._frame_sort_key(os.path.basename(p)))
        return frame_paths

    def _resolve_stream_source_for_key(
        self, traj_dir: str, key: str
    ) -> Tuple[Literal["video_file", "frames_dir"], str]:
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        def resolve_exact(stem: str) -> Optional[Tuple[Literal["video_file", "frames_dir"], str]]:
            frames_dir = os.path.join(traj_dir, stem)
            if os.path.isdir(frames_dir) and len(self._list_frame_images(frames_dir)) > 0:
                return ("frames_dir", frames_dir)

            for ext in video_extensions:
                candidate = os.path.join(traj_dir, f"{stem}{ext}")
                if os.path.exists(candidate):
                    return ("video_file", candidate)

            pattern = os.path.join(traj_dir, f"{stem}.*")
            for path in sorted(glob.glob(pattern)):
                file_stem = os.path.splitext(os.path.basename(path))[0]
                file_ext = os.path.splitext(path)[1].lower()
                if file_stem == stem and file_ext in set(video_extensions):
                    return ("video_file", path)
            return None

        if self._is_wo_robot_rgb_key(key):
            resolved = resolve_exact(key)
            if resolved is not None:
                return resolved
            raise FileNotFoundError(f"No stream found for key {key} in {traj_dir}")

        if self._is_base_rgb_key(key):
            resolved = resolve_exact(key)
            if resolved is not None:
                return resolved
            raise FileNotFoundError(
                f"No base RGB stream found for key {key} in {traj_dir}"
            )

        resolved = resolve_exact(key)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"No stream found for key {key} in {traj_dir}")

    def list_trajectories(self) -> List[str]:
        """
        Returns a summary list of all trajectories.
        Uses index if available, otherwise scans filesystem.
        """
        if self._index is not None:
            return self._filter_trajectory_ids(self._index["trajectory_list"].copy())

        # Fallback to filesystem scan
        traj_dirs = sorted(glob.glob(os.path.join(self.root, "traj_*")))
        trajs = []
        for d in traj_dirs:
            traj_id = os.path.basename(d).replace("traj_", "")
            trajs.append(traj_id)
        return self._filter_trajectory_ids(trajs)

    def list_keys(self, traj_id: str) -> List[str]:
        """
        List available keys for a trajectory.
        Uses index if available, otherwise scans filesystem.

        Args:
            traj_id: Trajectory ID

        Returns:
            List of available keys
        """
        if not self._traj_id_in_range(traj_id):
            return []

        # Try index first
        if self._index is not None:
            if traj_id in self._index["trajectories"]:
                return self._index["trajectories"][traj_id]["keys"].copy()

        # Fallback to filesystem scan
        return self._list_keys_scan(traj_id)

    def _get_video_dims(self, value: np.ndarray) -> tuple:
        """Get (T, H, W) from a video array."""
        if len(value.shape) == 3:  # (T, H, W)
            return value.shape[0], value.shape[1], value.shape[2]
        elif len(value.shape) == 4:  # (T, H, W, C)
            return value.shape[0], value.shape[1], value.shape[2]
        else:
            raise ValueError(
                f"Cannot determine video dimensions from shape {value.shape}"
            )

    def _get_key_config(self, key: str) -> KeyConfig:
        """
        Get KeyConfig for a key, with automatic detection based on suffix.

        Rules:
        1. Check explicit configs in self.key_configs
        2. Check if key ends with _rgb, _depth, _mask, _grayscale (or exact match)
        3. Default to RGB
        """
        # First check explicit configs
        if key in self.key_configs:
            return self.key_configs[key]

        # Check suffix-based rules (case insensitive)
        key_lower = key.lower()

        if key_lower.endswith("_rgb") or key_lower == "rgb":
            return KeyConfig(KeyType.RGB)
        elif key_lower.endswith("_rgb.wo_robot"):
            return KeyConfig(KeyType.RGB)
        elif key_lower.endswith("_depth") or key_lower == "depth":
            return KeyConfig(
                KeyType.DEPTH,
                depth_scaler=self.depth_scaler,
                depth_encoding=self.depth_encoding,
                mask_encoding=self.mask_encoding,
            )
        elif key_lower.endswith("_mask") or key_lower == "mask":
            return KeyConfig(KeyType.MASK, threshold=127.0, mask_encoding=self.mask_encoding)
        elif (
            key_lower.endswith("_grayscale")
            or key_lower == "grayscale"
            or key_lower.endswith("_gray")
        ):
            return KeyConfig(KeyType.GRAYSCALE, mask_encoding=self.mask_encoding)

        # Default to RGB
        return KeyConfig(KeyType.RGB)

    @staticmethod
    def _is_wo_robot_rgb_key(key: str) -> bool:
        return key.lower().endswith("_rgb.wo_robot")

    @staticmethod
    def _is_base_rgb_key(key: str) -> bool:
        key_lower = key.lower()
        return key_lower.endswith("_rgb") and not key_lower.endswith("_rgb.wo_robot")

    def _filter_video_keys_by_rgb_mode(self, keys: List[str]) -> List[str]:
        if self.rgb_variant_mode == "both":
            return keys
        filtered = []
        for key in keys:
            is_base = self._is_base_rgb_key(key)
            is_wo = self._is_wo_robot_rgb_key(key)
            if self.rgb_variant_mode == "base":
                if is_wo:
                    continue
            elif self.rgb_variant_mode == "wo_robot":
                if is_base:
                    continue
            filtered.append(key)
        return filtered

    def _resolve_video_path_for_key(self, traj_dir: str, key: str) -> str:
        source_type, source_path = self._resolve_stream_source_for_key(traj_dir, key)
        if source_type != "video_file":
            raise FileNotFoundError(
                f"Key {key} in {traj_dir} is stored as frame directory ({source_path})"
            )
        return source_path

    def write_trajectory(self, traj_id: str, data: TrajectoryData):
        """
        Write trajectory data with explicit separation of video streams and metadata.

        Args:
            traj_id: Trajectory identifier
            data: TrajectoryData with success, video_streams, and metadata fields
        """
        traj_dir = os.path.join(self.root, f"traj_{traj_id}")
        os.makedirs(traj_dir, exist_ok=True)

        # Determine number of frames for frame-aligned metadata detection
        num_frames = None
        if data.video_streams:
            first_key = next(iter(data.video_streams))
            first_frames = data.video_streams[first_key]
            num_frames = self._get_video_dims(first_frames)[0]

        # Write H5 metadata
        with h5py.File(os.path.join(traj_dir, "metadata.h5"), "w") as f:
            f.attrs["success"] = data.success

            for key, value in data.metadata.items():
                # Determine if this metadata is frame-aligned
                value_len = None
                if isinstance(value, (list, np.ndarray)):
                    value_len = len(value)
                is_frame_aligned = num_frames is not None and value_len == num_frames

                if isinstance(value, np.ndarray):
                    # Check if array has object dtype (h5py can't store object arrays)
                    if value.dtype == object:
                        if is_frame_aligned:
                            # Store per-element for frame-aligned metadata
                            grp = f.create_group(key)
                            grp.attrs["_frame_aligned"] = True
                            grp.attrs["_length"] = len(value)
                            for i, elem in enumerate(value):
                                pickled = pickle.dumps(elem)
                                grp.create_dataset(
                                    f"elem_{i}",
                                    data=np.frombuffer(pickled, dtype=np.uint8),
                                )
                        else:
                            # Serialize entire array using pickle
                            pickled = pickle.dumps(value)
                            f.create_dataset(
                                key, data=np.frombuffer(pickled, dtype=np.uint8)
                            )
                            f[key].attrs["_pickled"] = (
                                True  # Mark as pickled for reading
                            )
                    else:
                        if is_frame_aligned:
                            # Store with frame alignment marker for slicing
                            f.create_dataset(key, data=value)
                            f[key].attrs["_frame_aligned"] = True
                        else:
                            f.create_dataset(key, data=value)
                elif isinstance(value, (int, float, bool)):
                    f.attrs[key] = value
                elif isinstance(value, str):
                    f.attrs[key] = value
                elif isinstance(value, list):
                    # Convert list to numpy array if possible
                    try:
                        arr = np.array(value)
                        # Check if array has object dtype (contains objects that h5py can't store)
                        if arr.dtype == object:
                            if is_frame_aligned:
                                # Store per-element for frame-aligned metadata
                                grp = f.create_group(key)
                                grp.attrs["_frame_aligned"] = True
                                grp.attrs["_length"] = len(value)
                                for i, elem in enumerate(value):
                                    pickled = pickle.dumps(elem)
                                    grp.create_dataset(
                                        f"elem_{i}",
                                        data=np.frombuffer(pickled, dtype=np.uint8),
                                    )
                            else:
                                # Serialize entire list using pickle
                                pickled = pickle.dumps(value)
                                f.create_dataset(
                                    key, data=np.frombuffer(pickled, dtype=np.uint8)
                                )
                                f[key].attrs["_pickled"] = (
                                    True  # Mark as pickled for reading
                                )
                        else:
                            if is_frame_aligned:
                                # Store with frame alignment marker
                                f.create_dataset(key, data=arr)
                                f[key].attrs["_frame_aligned"] = True
                            else:
                                f.create_dataset(key, data=arr)
                    except (ValueError, TypeError):
                        # Store as JSON string if can't convert to array at all
                        if is_frame_aligned:
                            # For frame-aligned JSON-serializable lists, store per-element
                            grp = f.create_group(key)
                            grp.attrs["_frame_aligned"] = True
                            grp.attrs["_length"] = len(value)
                            for i, elem in enumerate(value):
                                grp.attrs[f"elem_{i}"] = json.dumps(elem)
                        else:
                            f.attrs[key] = json.dumps(value)
                    except Exception:
                        # If h5py.create_dataset fails (e.g., for object arrays not caught above),
                        # fall back to pickling
                        try:
                            if is_frame_aligned:
                                # Store per-element
                                grp = f.create_group(key)
                                grp.attrs["_frame_aligned"] = True
                                grp.attrs["_length"] = len(value)
                                for i, elem in enumerate(value):
                                    pickled = pickle.dumps(elem)
                                    grp.create_dataset(
                                        f"elem_{i}",
                                        data=np.frombuffer(pickled, dtype=np.uint8),
                                    )
                            else:
                                pickled = pickle.dumps(value)
                                f.create_dataset(
                                    key, data=np.frombuffer(pickled, dtype=np.uint8)
                                )
                                f[key].attrs["_pickled"] = True
                        except Exception:
                            # Last resort: store as JSON string
                            if is_frame_aligned:
                                grp = f.create_group(key)
                                grp.attrs["_frame_aligned"] = True
                                grp.attrs["_length"] = len(value)
                                for i, elem in enumerate(value):
                                    try:
                                        grp.attrs[f"elem_{i}"] = json.dumps(elem)
                                    except (TypeError, ValueError):
                                        # If even JSON fails, pickle individual element
                                        pickled = pickle.dumps(elem)
                                        grp.create_dataset(
                                            f"elem_{i}",
                                            data=np.frombuffer(pickled, dtype=np.uint8),
                                        )
                                        grp[f"elem_{i}"].attrs["_pickled"] = True
                            else:
                                f.attrs[key] = json.dumps(value)

        # Write videos
        if data.video_streams:
            # Determine video dimensions from first video stream
            first_key = next(iter(data.video_streams))
            first_frames = data.video_streams[first_key]
            T, H, W = self._get_video_dims(first_frames)

            # Validate that all video streams have the same number of frames
            for key, frames in data.video_streams.items():
                stream_T, stream_H, stream_W = self._get_video_dims(frames)
                if stream_T != T:
                    raise ValueError(
                        f"All video streams must have the same number of frames. "
                        f"Stream '{first_key}' has {T} frames, but stream '{key}' has {stream_T} frames."
                    )
                if stream_H != H or stream_W != W:
                    raise ValueError(
                        f"All video streams must have the same resolution. "
                        f"Stream '{first_key}' has resolution {H}x{W}, but stream '{key}' has resolution {stream_H}x{stream_W}."
                    )

            for key, frames in data.video_streams.items():
                # Get key config (with automatic suffix detection)
                key_config = self._get_key_config(key)

                # For depth keys, find corresponding foreground mask for min/max computation
                mask_frames = None
                if key_config.key_type == KeyType.DEPTH:
                    # Try to find corresponding foreground mask
                    # Pattern: {cam_name}_depth -> {cam_name}_foreground_mask
                    if key.endswith("_depth"):
                        mask_key = (
                            key[:-6] + "_foreground_mask"
                        )  # Remove "_depth" and add "_foreground_mask"
                    else:
                        # Try exact match pattern or other common patterns
                        mask_key = key.replace("_depth", "_foreground_mask")

                    if mask_key in data.video_streams:
                        mask_frames = data.video_streams[mask_key]
                        # Ensure mask and depth have same number of frames
                        if len(mask_frames) != len(frames):
                            mask_frames = None  # Mismatch, don't use mask

                # Create video recorder
                rec = VideoRecorder(
                    os.path.join(traj_dir, key), 30, (W, H), key_config=key_config
                )

                # Write frames with foreground mask if available
                for i, f in enumerate(frames):
                    foreground_mask = None
                    if mask_frames is not None and i < len(mask_frames):
                        mask_frame = mask_frames[i]
                        # Convert mask to boolean (foreground = True, background = False)
                        if mask_frame.dtype == bool:
                            foreground_mask = mask_frame
                        elif mask_frame.dtype == np.uint8:
                            # Threshold at 127 (default threshold for masks)
                            foreground_mask = mask_frame > 127
                        else:
                            # Try to convert to boolean
                            foreground_mask = mask_frame.astype(bool)

                        # Ensure shapes match
                        if f.shape[:2] != foreground_mask.shape[:2]:
                            foreground_mask = None  # Shape mismatch, don't use mask

                    rec.write(f, foreground_mask=foreground_mask)
                rec.close()

    def _get_video_frame_count(self, traj_id: str, key: str) -> Optional[int]:
        """
        Get total frame count for a video key without reading frames.
        Uses decord if available, otherwise falls back to OpenCV.
        """
        if not self._traj_id_in_range(traj_id):
            return None

        traj_dir = os.path.join(self.root, f"traj_{traj_id}")
        try:
            source_type, source_path = self._resolve_stream_source_for_key(traj_dir, key)
        except FileNotFoundError:
            return None

        if source_type == "frames_dir":
            return len(self._list_frame_images(source_path))

        if HAS_DECORD:
            vr = decord.VideoReader(source_path, ctx=decord.cpu(0))
            frame_count = len(vr)
            del vr
        else:
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                return None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return frame_count

    def read_frames(
        self,
        traj_id: str,
        key: str,
        start: int = 0,
        end: Optional[int] = None,
        max_frames: Optional[int] = None,
        frame_indices: Optional[Union[List[int], np.ndarray]] = None,
        img_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Read frames for a specific key (video).
        Uses decord for fast random frame access, falls back to OpenCV if missing.

        Args:
            traj_id: Trajectory ID
            key: Key name
            start: Start frame index (ignored if frame_indices is provided)
            end: End frame index (exclusive, ignored if frame_indices is provided)
            max_frames: Maximum number of frames to read (downsampling, ignored if frame_indices is provided)
            frame_indices: Optional list/array of specific frame indices to read. If provided,
                         takes precedence over start/end/max_frames.
            img_size: If specified, resize all frames to img_size x img_size.
                     RGB uses bilinear interpolation, depth uses bilinear on float values,
                     masks use nearest neighbor interpolation.

        Returns:
            Tuple of:
            - Array of frames (resized if img_size is specified)
            - Original dimensions (orig_H, orig_W) if resizing was performed, None otherwise
        """
        if not self._traj_id_in_range(traj_id):
            raise ValueError(
                f"Trajectory {traj_id} is outside configured range "
                f"[{self.start_traj_id}, {self.end_traj_id}]"
            )

        traj_dir = os.path.join(self.root, f"traj_{traj_id}")

        source_type, source_path = self._resolve_stream_source_for_key(traj_dir, key)

        # Get key config (with automatic suffix detection)
        key_config = self._get_key_config(key)

        # Load depth min/max and encoding from metadata (same interface for FFV1/H264)
        depth_min_max = None
        if key_config.key_type == KeyType.DEPTH:
            json_path = os.path.join(traj_dir, f"{key}_depth_minmax.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        min_max_data = json.load(f)
                        depth_min_max = [
                            (item["min"], item["max"])
                            for item in min_max_data.get("frames", [])
                        ]
                        # encoding is optional (legacy files default to ffv1)
                except (json.JSONDecodeError, IOError, KeyError):
                    depth_min_max = None
        elif key_config.key_type in (KeyType.MASK, KeyType.GRAYSCALE):
            # For masks/grayscale, we might have an encoding JSON, but no min/max
            json_path = os.path.join(traj_dir, f"{key}_encoding.json")
            # We don't strictly need to read it here unless we need the codec info for decord,
            # but decord/opencv usually handle that automatically from the file header.
            # However, it's good for consistency.
            pass

        if source_type == "frames_dir":
            frame_paths = (
                self._list_frame_images(source_path, allowed_exts={".png"})
                if key_config.key_type == KeyType.MASK
                else self._list_frame_images(source_path)
            )
            total_frames = len(frame_paths)
            if total_frames == 0:
                if key_config.key_type == KeyType.MASK:
                    raise IOError(
                        f"No PNG mask frames found in directory {source_path}. "
                        "Mask streams in frame-folder mode must use PNG files."
                    )
                raise IOError(f"No frame images found in directory {source_path}")
        else:
            # Use decord for faster random frame access
            if HAS_DECORD:
                vr = decord.VideoReader(source_path, ctx=decord.cpu(0))
                total_frames = len(vr)
            else:
                cap = cv2.VideoCapture(source_path)
                if not cap.isOpened():
                    raise IOError(f"Failed to open video file {source_path}")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If frame_indices is provided, use it directly
        if frame_indices is not None:
            # Convert to numpy array and ensure it's 1D
            indices = np.asarray(frame_indices).flatten()
            # Ensure all indices are within bounds
            indices = np.clip(indices, 0, total_frames - 1).astype(int)
        else:
            # Use start/end/max_frames logic
            if end is None:
                end = total_frames

            # Clamp end to valid range
            end = min(end, total_frames)
            start = max(0, min(start, total_frames - 1))

            # Calculate indices
            indices = np.arange(start, end)
            if max_frames and len(indices) > max_frames:
                indices = np.linspace(start, end - 1, max_frames, dtype=int)

            # Ensure all indices are within bounds
            indices = np.clip(indices, 0, total_frames - 1).astype(int)

        if source_type == "frames_dir":
            decord_frames = []
            fallback_shape = None
            for idx in indices:
                img_path = frame_paths[int(idx)]
                frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if frame is None:
                    if fallback_shape is None:
                        raise IOError(f"Failed to read frame image: {img_path}")
                    frame = np.zeros(fallback_shape, dtype=np.uint8)
                else:
                    if len(frame.shape) == 3:
                        if frame.shape[2] == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        elif frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                    fallback_shape = frame.shape
                decord_frames.append(frame)
            decord_frames = np.array(decord_frames)
        else:
            # Read frames using decord (much faster for random access)
            if HAS_DECORD:
                decord_frames = vr.get_batch(indices.tolist()).asnumpy()
            else:
                # Fallback to OpenCV
                decord_frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # OpenCV reads in BGR, convert to RGB to match decord
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        decord_frames.append(frame_rgb)
                    else:
                        # If read fails, append zero frame
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        decord_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                decord_frames = np.array(decord_frames)
                cap.release()

        # Process frames based on key type
        frames = []
        for i, idx in enumerate(indices):
            frame = decord_frames[i]

            if key_config.key_type == KeyType.RGB:
                # decord returns RGB by default
                frames.append(frame)
            elif key_config.key_type == KeyType.MASK:
                # Convert RGB to grayscale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                # Convert to boolean using threshold
                frames.append(gray > key_config.threshold)
            elif key_config.key_type == KeyType.DEPTH:
                # Read depth as single-channel grayscale
                if len(frame.shape) == 2:
                    # Already single-channel
                    depth_uint8 = frame
                elif len(frame.shape) == 3:
                    # Multi-channel, extract grayscale
                    depth_uint8 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    depth_uint8 = frame

                # Get min/max for this frame
                if depth_min_max is not None and idx < len(depth_min_max):
                    min_depth, max_depth = depth_min_max[idx]
                    # Decode uint8 depth back to int16 using min/max
                    depth_int16 = decode_depth_uint8_to_int16(
                        depth_uint8, min_depth, max_depth
                    )
                else:
                    # Fallback: if min/max not available, use rough approximation
                    depth_int16 = (
                        depth_uint8.astype(np.int16) * 256
                    )  # Rough approximation (not accurate)

                frames.append(depth_int16.astype(np.float32))
            elif key_config.key_type == KeyType.GRAYSCALE:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                frames.append(gray)
            else:
                # Default: treat as RGB (decord already returns RGB)
                frames.append(frame)

        if source_type == "video_file" and HAS_DECORD:
            del vr
        frames_array = np.array(frames)

        # Apply resizing if img_size is specified
        orig_dims = None
        if img_size is not None and img_size > 0:
            orig_H, orig_W = frames_array.shape[1], frames_array.shape[2]
            if orig_H != img_size or orig_W != img_size:
                orig_dims = (orig_H, orig_W)
                # Choose interpolation based on key type
                if key_config.key_type in (KeyType.MASK, KeyType.DEPTH):
                    interp = cv2.INTER_NEAREST
                else:
                    interp = cv2.INTER_LINEAR

                # Resize each frame
                resized_frames = []
                for frame in frames_array:
                    if key_config.key_type == KeyType.DEPTH:
                        # Convert to float32 for depth before resizing
                        frame_float = frame.astype(np.float32)
                        resized = resize_image(frame_float, img_size, interp)
                        resized_frames.append(resized)
                    else:
                        resized = resize_image(frame, img_size, interp)
                        resized_frames.append(resized)

                frames_array = np.array(resized_frames)

        return frames_array, orig_dims

    def read_trajectory(
        self,
        traj_id: str,
        video_keys: Optional[List[str]] = None,
        metadata_keys: Optional[List[str]] = None,
        start: int = 0,
        end: Optional[int] = None,
        max_frames: Optional[int] = None,
        frame_indices: Optional[Union[List[int], np.ndarray]] = None,
        metadata_frame_indices: Optional[Union[List[int], np.ndarray]] = None,
        img_size: Optional[int] = None,
        frame_aligned_metadata_keys: Optional[List[str]] = None,
    ) -> TrajectoryData:
        """
        Read trajectory data with explicit separation of video streams and metadata.

        Args:
            traj_id: Trajectory ID
            video_keys: List of video stream keys to read (None = all video streams)
            metadata_keys: List of metadata keys to read (None = all metadata)
            start: Start frame index (for video streams, ignored if frame_indices is provided)
            end: End frame index (exclusive, for video streams, ignored if frame_indices is provided)
            max_frames: Maximum number of frames to read (for video streams, ignored if frame_indices is provided)
            frame_indices: Optional list/array of specific frame indices to read for video streams.
                         If provided, takes precedence over start/end/max_frames.
            metadata_frame_indices: Optional list/array of specific frame indices to read for frame-aligned metadata.
                                   If None, uses frame_indices. If frame_indices is also None, uses start/end/max_frames.
            img_size: If specified, resize all video frames to img_size x img_size and adjust intrinsics accordingly.
            frame_aligned_metadata_keys: Extra metadata keys to force-treat as
                frame-aligned for this read call. Combined with
                self.forced_frame_aligned_metadata_keys.

        Returns:
            TrajectoryData with success, video_streams, and metadata
        """
        if not self._traj_id_in_range(traj_id):
            raise ValueError(
                f"Trajectory {traj_id} is outside configured range "
                f"[{self.start_traj_id}, {self.end_traj_id}]"
            )

        traj_dir = os.path.join(self.root, f"traj_{traj_id}")
        if not os.path.exists(traj_dir):
            raise FileNotFoundError(f"Trajectory {traj_id} not found")

        # Read metadata from H5
        metadata = {}
        success = False
        metadata_path = os.path.join(traj_dir, "metadata.h5")
        if os.path.exists(metadata_path):
            with h5py.File(metadata_path, "r") as f:
                forced_frame_aligned_keys = set(self.forced_frame_aligned_metadata_keys)
                if frame_aligned_metadata_keys is not None:
                    forced_frame_aligned_keys.update(frame_aligned_metadata_keys)

                def _is_forced_frame_aligned_key(name: str) -> bool:
                    # h5py visititems can provide nested names (e.g., "a/b/c")
                    # while callers usually specify leaf keys.
                    leaf_name = name.split("/")[-1]
                    return (
                        name in forced_frame_aligned_keys
                        or leaf_name in forced_frame_aligned_keys
                    )

                def _maybe_slice_forced_frame_aligned_value(
                    name: str, value: Any
                ) -> Any:
                    if (
                        not _is_forced_frame_aligned_key(name)
                        or metadata_frame_indices_to_use is None
                        or actual_frame_count is None
                    ):
                        return value

                    if isinstance(value, np.ndarray) and len(value) == actual_frame_count:
                        return value[metadata_frame_indices_to_use]

                    if isinstance(value, list) and len(value) == actual_frame_count:
                        return [
                            value[idx]
                            for idx in metadata_frame_indices_to_use
                            if idx < len(value)
                        ]

                    return value

                # Read success
                success = bool(f.attrs["success"])

                # Determine frame range for slicing frame-aligned metadata
                # Get actual frame count from video if available
                actual_frame_count = None
                if video_keys is not None and len(video_keys) > 0:
                    # Try to get frame count from first video
                    actual_frame_count = self._get_video_frame_count(
                        traj_id, video_keys[0]
                    )
                else:
                    # Try to get frame count from any available video key
                    available_keys_temp = self.list_keys(traj_id)
                    for test_key in available_keys_temp:
                        # Skip metadata keys (they won't have video files)
                        if test_key in f.keys() or test_key in f.attrs:
                            continue
                        actual_frame_count = self._get_video_frame_count(
                            traj_id, test_key
                        )
                        if actual_frame_count is not None:
                            break

                # Calculate which indices to read for frame-aligned metadata
                # Use metadata_frame_indices if provided, otherwise use frame_indices, otherwise compute from start/end/max_frames
                metadata_frame_indices_to_use = None
                if metadata_frame_indices is not None:
                    metadata_frame_indices_to_use = list(metadata_frame_indices)
                elif frame_indices is not None:
                    metadata_frame_indices_to_use = list(frame_indices)
                elif actual_frame_count is not None:
                    if end is None:
                        end_idx = actual_frame_count
                    else:
                        end_idx = min(end, actual_frame_count)
                    metadata_frame_indices_to_use = list(range(start, end_idx))
                    if max_frames and len(metadata_frame_indices_to_use) > max_frames:
                        # Match the downsampling used for video frames
                        metadata_frame_indices_to_use = list(
                            np.linspace(start, end_idx - 1, max_frames, dtype=int)
                        )

                # Read all groups (for per-element pickled data)
                def read_group(name, obj):
                    if isinstance(obj, h5py.Group):
                        if metadata_keys is None or name in metadata_keys:
                            # Check if this is a frame-aligned group
                            is_frame_aligned_group = (
                                ("_frame_aligned" in obj.attrs and obj.attrs["_frame_aligned"])
                                or _is_forced_frame_aligned_key(name)
                            )
                            if is_frame_aligned_group:
                                length = obj.attrs.get("_length", len(obj.keys()))

                                # Determine which elements to read
                                if (
                                    metadata_frame_indices_to_use is not None
                                    and length == actual_frame_count
                                ):
                                    # Frame-aligned: read only requested indices
                                    elems = []
                                    for idx in metadata_frame_indices_to_use:
                                        if idx < length:
                                            elem_key = f"elem_{idx}"
                                            if elem_key in obj:
                                                elem_obj = obj[elem_key]
                                                if isinstance(elem_obj, h5py.Dataset):
                                                    # Check if pickled
                                                    if (
                                                        "_pickled" in elem_obj.attrs
                                                        and elem_obj.attrs["_pickled"]
                                                    ):
                                                        pickled_data = bytes(
                                                            elem_obj[:]
                                                        )
                                                        elems.append(
                                                            pickle.loads(pickled_data)
                                                        )
                                                    else:
                                                        elems.append(elem_obj[:])
                                                elif elem_key in obj.attrs:
                                                    # Stored as attribute (JSON)
                                                    val = obj.attrs[elem_key]
                                                    if isinstance(val, str):
                                                        try:
                                                            elems.append(
                                                                json.loads(val)
                                                            )
                                                        except (
                                                            json.JSONDecodeError,
                                                            ValueError,
                                                        ):
                                                            elems.append(val)
                                                    else:
                                                        elems.append(val)
                                    metadata[name] = elems
                                else:
                                    # Not frame-aligned or full read: read all elements
                                    elems = []
                                    for i in range(length):
                                        elem_key = f"elem_{i}"
                                        if elem_key in obj:
                                            elem_obj = obj[elem_key]
                                            if isinstance(elem_obj, h5py.Dataset):
                                                if (
                                                    "_pickled" in elem_obj.attrs
                                                    and elem_obj.attrs["_pickled"]
                                                ):
                                                    pickled_data = bytes(elem_obj[:])
                                                    elems.append(
                                                        pickle.loads(pickled_data)
                                                    )
                                                else:
                                                    elems.append(elem_obj[:])
                                            elif elem_key in obj.attrs:
                                                val = obj.attrs[elem_key]
                                                if isinstance(val, str):
                                                    try:
                                                        elems.append(json.loads(val))
                                                    except (
                                                        json.JSONDecodeError,
                                                        ValueError,
                                                    ):
                                                        elems.append(val)
                                                else:
                                                    elems.append(val)
                                    metadata[name] = elems

                # Read all datasets
                def read_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if metadata_keys is None or name in metadata_keys:
                            # Check if this dataset contains pickled data
                            if "_pickled" in obj.attrs and obj.attrs["_pickled"]:
                                # Unpickle the data
                                pickled_data = bytes(obj[:])
                                value = pickle.loads(pickled_data)
                                metadata[name] = _maybe_slice_forced_frame_aligned_value(
                                    name, value
                                )
                            else:
                                # Check if frame-aligned and should be sliced
                                is_frame_aligned_dataset = (
                                    ("_frame_aligned" in obj.attrs and obj.attrs["_frame_aligned"])
                                    or _is_forced_frame_aligned_key(name)
                                )
                                if is_frame_aligned_dataset:
                                    if (
                                        metadata_frame_indices_to_use is not None
                                        and len(obj) == actual_frame_count
                                    ):
                                        # Slice to match frame range
                                        metadata[name] = obj[
                                            metadata_frame_indices_to_use
                                        ]
                                    else:
                                        # Read full dataset
                                        metadata[name] = obj[:]
                                else:
                                    metadata[name] = obj[:]

                f.visititems(read_group)
                f.visititems(read_dataset)

                # Read all attributes (except success)
                for key, value in f.attrs.items():
                    if key != "success" and (
                        metadata_keys is None or key in metadata_keys
                    ):
                        # Try to parse JSON if it's a string
                        if isinstance(value, str) and value.startswith("["):
                            try:
                                parsed_value = json.loads(value)
                                metadata[key] = _maybe_slice_forced_frame_aligned_value(
                                    key, parsed_value
                                )
                            except (json.JSONDecodeError, ValueError):
                                metadata[key] = value
                        else:
                            metadata[key] = _maybe_slice_forced_frame_aligned_value(
                                key, value
                            )

        # Read video streams
        video_streams = {}
        available_keys = self.list_keys(traj_id)

        # Determine which video keys to read
        if video_keys is None:
            # Try to read all keys as videos (those that have video files)
            keys_to_try = self._filter_video_keys_by_rgb_mode(available_keys)
        else:
            keys_to_try = video_keys

        # Track original dimensions for intrinsics adjustment
        orig_dims_for_intrinsics = None

        for key in keys_to_try:
            # Skip if it's in metadata (already read)
            if key in metadata:
                continue

            # Try to read as video
            try:
                frames, orig_dims = self.read_frames(
                    traj_id,
                    key,
                    start,
                    end,
                    max_frames,
                    frame_indices=frame_indices,
                    img_size=img_size,
                )
                # Track original dimensions from first resized video
                if orig_dims is not None and orig_dims_for_intrinsics is None:
                    orig_dims_for_intrinsics = orig_dims
                video_streams[key] = frames
            except FileNotFoundError:
                # Not a video key, skip
                continue

        # Validate that all video streams have the same number of frames
        if video_streams:
            frame_counts = {
                key: frames.shape[0] for key, frames in video_streams.items()
            }
            unique_frame_counts = set(frame_counts.values())
            if len(unique_frame_counts) > 1:
                raise ValueError(
                    f"All video streams must have the same number of frames. "
                    f"Frame counts: {frame_counts}"
                )

            # Expose per-key video encoding (from depth/mask metadata) so same interface for any format
            metadata["video_encoding"] = {}
            for key in video_streams:
                config = self._get_key_config(key)
                if config.key_type == KeyType.DEPTH:
                    json_path = os.path.join(traj_dir, f"{key}_depth_minmax.json")
                elif config.key_type in (KeyType.MASK, KeyType.GRAYSCALE):
                    json_path = os.path.join(traj_dir, f"{key}_encoding.json")
                else:
                    continue

                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                            metadata["video_encoding"][key] = data.get(
                                "encoding", "ffv1" if config.key_type == KeyType.DEPTH else "h264_lossless"
                            )
                    except (json.JSONDecodeError, IOError):
                        metadata["video_encoding"][key] = "ffv1" if config.key_type == KeyType.DEPTH else "h264_lossless"

        # Adjust intrinsics if images were resized
        if orig_dims_for_intrinsics is not None and img_size is not None:
            orig_H, orig_W = orig_dims_for_intrinsics
            # Find and adjust all intrinsics metadata (keys ending with _intrinsics)
            for key in list(metadata.keys()):
                if key.endswith("_intrinsics"):
                    intrinsics_data = metadata[key]
                    if isinstance(intrinsics_data, np.ndarray):
                        metadata[key] = resize_intrinsics(
                            intrinsics_data, orig_H, orig_W, img_size, img_size
                        )

        return TrajectoryData(
            success=success, video_streams=video_streams, metadata=metadata
        )


if __name__ == "__main__":
    import argparse
    import tempfile
    import shutil
    import sys

    parser = argparse.ArgumentParser(description="ManiSkillTrajectoryDataset CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Run tests")

    # Build-index subcommand
    build_index_parser = subparsers.add_parser(
        "build-index", help="Build index for fast trajectory/key lookup"
    )
    build_index_parser.add_argument(
        "root_dir", type=str, help="Root directory of the dataset"
    )
    build_index_parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if index exists"
    )
    build_index_parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.command == "test":
        print("=" * 60)
        print("Testing ManiSkillTrajectoryDataset API (Generic)")
        print("=" * 60)

        # Create a temporary directory for testing
        test_dir = tempfile.mkdtemp(prefix="test_dataset_")
        print(f"\n[1] Created test directory: {test_dir}")

        try:
            # Test 1: Initialize dataset with custom key configs
            print("\n[2] Testing dataset initialization...")
            custom_configs = {
                "my_depth": KeyConfig(KeyType.DEPTH, depth_scaler=2000.0),
                "my_mask": KeyConfig(KeyType.MASK, threshold=128.0),
            }
            dataset = ManiSkillTrajectoryDataset(test_dir, key_configs=custom_configs)
            print("✓ Dataset initialized successfully")

            # Test 2: Save robot infos
            print("\n[3] Testing save_robot_infos and get_robot_infos...")
            # Create test RobotInfo objects
            robot_info1 = RobotInfo(
                uid="robot1",
                urdf_path="/path/to/robot1.urdf",
                urdf_config={"param1": "value1"},
                joint_names=["joint1", "joint2", "joint3"],
                action_space=(-1.0, 1.0, [0, 1, 2]),
                action_mapping={"arm": (0, 2), "gripper": (3, 3)},
            )
            robot_info2 = RobotInfo(
                uid="robot2",
                urdf_path="/path/to/robot2.urdf",
                urdf_config={"param2": "value2"},
                joint_names=["joint1", "joint2"],
                action_space=(-0.5, 0.5, [0, 1]),
                action_mapping={"arm": (0, 1)},
            )

            # Test saving robot infos
            dataset.save_robot_infos([robot_info1, robot_info2])
            saved_infos = dataset.get_robot_infos()
            assert saved_infos is not None
            assert len(saved_infos) == 2
            assert saved_infos[0].uid == "robot1"
            assert saved_infos[0].urdf_path == "/path/to/robot1.urdf"
            assert saved_infos[0].joint_names == ["joint1", "joint2", "joint3"]
            assert saved_infos[1].uid == "robot2"
            assert saved_infos[1].urdf_path == "/path/to/robot2.urdf"
            print("✓ Robot infos saved and retrieved")

            # Test that it doesn't override existing metadata
            # Manually add some other metadata to the file
            with open(dataset.metadata_path, "r") as f:
                existing_data = json.load(f)
            existing_data["other_metadata"] = "test_value"
            with open(dataset.metadata_path, "w") as f:
                json.dump(existing_data, f, indent=2)

            # Save robot infos again - should preserve other_metadata
            dataset.save_robot_infos([robot_info1, robot_info2])
            with open(dataset.metadata_path, "r") as f:
                merged_data = json.load(f)
            assert "robot_infos" in merged_data
            assert "other_metadata" in merged_data
            assert merged_data["other_metadata"] == "test_value"
            assert len(merged_data["robot_infos"]) == 4  # 2 original + 2 new
            print("✓ Robot infos save preserves existing metadata")

            # Test backward compatibility with legacy methods
            print(
                "\n[3b] Testing backward compatibility with save_robot_urdfs/get_robot_urdfs..."
            )
            urdf_paths = ["/path/to/robot3.urdf", "/path/to/robot4.urdf"]
            dataset.save_robot_urdfs(urdf_paths)
            saved_paths = dataset.get_robot_urdfs()
            assert saved_paths is not None
            assert len(saved_paths) == 6  # 4 previous + 2 new
            assert "/path/to/robot3.urdf" in saved_paths
            assert "/path/to/robot4.urdf" in saved_paths
            print("✓ Legacy methods work correctly")

            # Test 3: Create sample trajectory data with explicit video_streams and metadata
            print(
                "\n[4] Creating structured trajectory data with video_streams and metadata..."
            )
            T, H, W = 10, 64, 64

            # Trajectory 1: Front camera
            # Note: keys ending with _rgb, _depth, _mask will be auto-detected for encoding
            traj_data_front = TrajectoryData(
                success=True,
                video_streams={
                    "front_rgb": np.random.randint(
                        0, 255, (T, H, W, 3), dtype=np.uint8
                    ),  # Auto-detected as RGB
                    "front_depth": np.random.rand(T, H, W).astype(np.float32)
                    * 2.0,  # Auto-detected as DEPTH
                    "front_mask": np.random.randint(
                        0, 255, (T, H, W), dtype=np.uint8
                    ),  # Auto-detected as MASK
                    "custom_image": np.random.randint(
                        0, 255, (T, H, W, 3), dtype=np.uint8
                    ),  # Defaults to RGB
                    "my_depth": np.random.rand(T, H, W).astype(np.float32)
                    * 2.0,  # Explicit config overrides
                    "my_mask": np.random.randint(
                        0, 255, (T, H, W), dtype=np.uint8
                    ),  # Explicit config overrides
                },
                metadata={
                    "qpos": np.random.randn(T, 7).astype(np.float32),
                    "actions": np.random.randn(T, 6).astype(np.float32),
                    "extrinsics": np.eye(4, dtype=np.float32),
                    "intrinsics": np.array(
                        [[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float32
                    ),
                    "some_metadata": 42,
                    "some_list": [1, 2, 3, 4, 5],
                },
            )

            # Trajectory 2: Side camera (separate trajectory)
            traj_data_side = TrajectoryData(
                success=True,
                video_streams={
                    "side_rgb": np.random.randint(
                        0, 255, (T, H, W, 3), dtype=np.uint8
                    ),  # Auto-detected as RGB
                    "side_depth": np.random.rand(T, H, W).astype(np.float32)
                    * 2.0,  # Auto-detected as DEPTH
                },
                metadata={
                    "qpos": np.random.randn(T, 7).astype(
                        np.float32
                    ),  # Same robot state
                    "actions": np.random.randn(T, 6).astype(np.float32),  # Same actions
                    "extrinsics": np.eye(4, dtype=np.float32),
                    "intrinsics": np.array(
                        [[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float32
                    ),
                },
            )

            print(f"✓ Created 2 trajectories with {T} frames, {H}x{W} resolution")

            # Test 4: Write trajectories
            print("\n[5] Testing write_trajectory (generic)...")
            traj_id_front = "001_front"
            traj_id_side = "001_side"
            dataset.write_trajectory(traj_id_front, traj_data_front)
            dataset.write_trajectory(traj_id_side, traj_data_side)

            # Verify files
            traj_dir_front = os.path.join(test_dir, f"traj_{traj_id_front}")
            assert os.path.exists(os.path.join(traj_dir_front, "metadata.h5"))
            assert os.path.exists(
                os.path.join(traj_dir_front, "front_rgb.mp4")
            )  # Auto-detected RGB
            assert os.path.exists(
                os.path.join(traj_dir_front, "front_depth.avi")
            )  # Auto-detected DEPTH
            assert os.path.exists(
                os.path.join(traj_dir_front, "front_mask.mp4")
            )  # Auto-detected MASK
            assert os.path.exists(
                os.path.join(traj_dir_front, "custom_image.mp4")
            )  # Defaults to RGB
            assert os.path.exists(
                os.path.join(traj_dir_front, "my_depth.avi")
            )  # Explicit config
            assert os.path.exists(
                os.path.join(traj_dir_front, "my_mask.mp4")
            )  # Explicit config
            print("✓ Trajectories written successfully")

            # Test 5: List trajectories
            print("\n[6] Testing list_trajectories...")
            trajs = dataset.list_trajectories()
            assert traj_id_front in trajs and traj_id_side in trajs
            print(f"✓ Found {len(trajs)} trajectories: {trajs}")

            # Test 5b: Range-filtered dataset view
            print("\n[6b] Testing trajectory ID range filtering...")
            ranged_dataset = ManiSkillTrajectoryDataset(
                test_dir, start_traj_id=traj_id_front, end_traj_id=traj_id_front
            )
            ranged_trajs = ranged_dataset.list_trajectories()
            assert ranged_trajs == [traj_id_front]
            try:
                ranged_dataset.read_trajectory(traj_id_side)
                raise AssertionError("Expected ValueError for out-of-range trajectory")
            except ValueError:
                pass
            print("✓ Range filtering works for list/read operations")

            # Test 6: List keys
            print("\n[7] Testing list_keys...")
            keys_front = dataset.list_keys(traj_id_front)
            assert "front_rgb" in keys_front
            assert "front_depth" in keys_front
            assert "front_mask" in keys_front
            assert "custom_image" in keys_front
            assert "my_depth" in keys_front
            assert "my_mask" in keys_front
            assert "qpos" in keys_front
            assert "actions" in keys_front
            print(f"✓ Found keys for front: {keys_front}")

            # Test 7: Read trajectory (all keys) and verify structured format
            print("\n[8] Testing read_trajectory (all keys) with structured format...")
            result = dataset.read_trajectory(traj_id_front)
            assert isinstance(result, TrajectoryData)
            assert result.success
            assert "qpos" in result.metadata
            assert "actions" in result.metadata
            assert "front_rgb" in result.video_streams
            assert "front_depth" in result.video_streams
            assert "front_mask" in result.video_streams
            assert "custom_image" in result.video_streams
            assert "my_depth" in result.video_streams
            assert "my_mask" in result.video_streams
            assert "extrinsics" in result.metadata
            assert "intrinsics" in result.metadata
            assert "some_metadata" in result.metadata
            assert result.metadata["some_metadata"] == 42
            assert result.video_streams["front_rgb"].shape == (T, H, W, 3)
            assert result.video_streams["front_depth"].shape == (T, H, W)
            assert result.video_streams["front_mask"].shape == (T, H, W)
            assert (
                result.video_streams["front_mask"].dtype == bool
            )  # Auto-detected MASK should be decoded as bool
            assert result.video_streams["custom_image"].shape == (
                T,
                H,
                W,
                3,
            )  # Defaults to RGB
            assert result.video_streams["my_depth"].shape == (T, H, W)
            assert result.video_streams["my_mask"].shape == (T, H, W)
            assert (
                result.video_streams["my_mask"].dtype == bool
            )  # Explicit config MASK should be decoded as bool
            print(
                f"✓ Read trajectory: {len(result.video_streams)} video streams, {len(result.metadata)} metadata keys"
            )

            # Test 7b: Verify auto-detection worked correctly
            print("\n[8b] Verifying suffix-based auto-detection...")
            # front_depth should be decoded as float (depth type)
            assert result.video_streams["front_depth"].dtype == np.float32, (
                "front_depth should be float32"
            )
            # front_mask should be decoded as bool (mask type)
            assert result.video_streams["front_mask"].dtype == bool, (
                "front_mask should be bool"
            )
            # front_rgb should be decoded as uint8 (RGB type)
            assert result.video_streams["front_rgb"].dtype == np.uint8, (
                "front_rgb should be uint8"
            )
            print("✓ Auto-detection verified: suffixes correctly identified key types")

            # Test 8: Read trajectory (selected keys)
            print(
                "\n[9] Testing read_trajectory (selected video_keys and metadata_keys)..."
            )
            result_subset = dataset.read_trajectory(
                traj_id_front, video_keys=["front_rgb"], metadata_keys=["qpos"]
            )
            assert "front_rgb" in result_subset.video_streams
            assert "qpos" in result_subset.metadata
            assert "actions" not in result_subset.metadata
            assert "front_depth" not in result_subset.video_streams
            print("✓ Read subset of keys")

            # Test 9: Read with frame range
            print("\n[10] Testing read_trajectory with frame range...")
            result_range = dataset.read_trajectory(
                traj_id_front, video_keys=["front_rgb"], start=2, end=7
            )
            assert result_range.video_streams["front_rgb"].shape[0] == 5
            print("✓ Read with frame range")

            # Test 10: Verify data integrity
            print("\n[11] Testing data integrity...")
            result = dataset.read_trajectory(traj_id_front)
            assert np.allclose(
                result.metadata["qpos"], traj_data_front.metadata["qpos"]
            )
            assert np.allclose(
                result.metadata["actions"], traj_data_front.metadata["actions"]
            )
            assert np.allclose(
                result.metadata["extrinsics"], traj_data_front.metadata["extrinsics"]
            )
            assert np.allclose(
                result.metadata["intrinsics"], traj_data_front.metadata["intrinsics"]
            )
            print("✓ Data integrity verified")

            # Test 11: Test different trajectory (side camera)
            print("\n[12] Testing different trajectory (side camera)...")
            result_side = dataset.read_trajectory(traj_id_side)
            assert result_side.success
            assert "side_rgb" in result_side.video_streams
            assert "side_depth" in result_side.video_streams
            assert "front_rgb" not in result_side.video_streams  # Different trajectory
            # Verify auto-detection for side camera
            assert result_side.video_streams["side_rgb"].dtype == np.uint8, (
                "side_rgb should be uint8"
            )
            assert result_side.video_streams["side_depth"].dtype == np.float32, (
                "side_depth should be float32"
            )
            print("✓ Side camera trajectory read successfully with auto-detection")

            # Test 12: Test index building
            print("\n[13] Testing build_index...")
            dataset.build_index(force=True, verbose=False)
            assert os.path.exists(dataset.index_path), "Index file should be created"
            # Verify index works
            trajs_from_index = dataset.list_trajectories()
            assert (
                traj_id_front in trajs_from_index and traj_id_side in trajs_from_index
            )
            keys_from_index = dataset.list_keys(traj_id_front)
            assert "front_rgb" in keys_from_index
            assert "qpos" in keys_from_index
            print("✓ Index built and verified")

            print("\n" + "=" * 60)
            print("All tests passed! ✓")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        finally:
            # Cleanup
            print(f"\n[Cleanup] Removing test directory: {test_dir}")
            shutil.rmtree(test_dir, ignore_errors=True)
            print("✓ Cleanup complete")

    elif args.command == "build-index":
        if not os.path.exists(args.root_dir):
            print(f"Error: Directory {args.root_dir} does not exist")
            sys.exit(1)

        dataset = ManiSkillTrajectoryDataset(args.root_dir)
        dataset.build_index(force=args.force, verbose=not args.quiet)
        print(f"Index saved to: {dataset.index_path}")

    else:
        parser.print_help()
        sys.exit(1)
