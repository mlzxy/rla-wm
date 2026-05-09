"""Headless mesh rendering of articulated robots for action visualization.

Uses pyrender + EGL (with an OSMesa fallback). The camera follows a cinematic
orbit centred on the robot base. The shared :class:`PyrenderSceneRenderer`
class is reused by ``utils.render_iws`` to render IWS skeletons and EE
markers in the same scene style.
"""

from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass

import numpy as np
import torch

# Force a headless GL backend before importing pyrender / OpenGL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender  # noqa: E402
import trimesh  # noqa: E402

from datalib.robot_geometry import (  # noqa: E402
    DifferentiableRobotGeometry,
    mesh_to_arrays,
    to_o3d_mesh,
)


_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


# ---------------------------------------------------------------------------
# Shared pyrender scene helpers
# ---------------------------------------------------------------------------


def _try_offscreen_renderer(width: int, height: int) -> pyrender.OffscreenRenderer:
    """Create an OffscreenRenderer, falling back from EGL to OSMesa."""
    try:
        return pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    except Exception as e_egl:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        try:
            return pyrender.OffscreenRenderer(
                viewport_width=width, viewport_height=height
            )
        except Exception as e_os:
            raise RuntimeError(
                f"Failed to initialize pyrender OffscreenRenderer with EGL ({e_egl}) "
                f"and OSMesa ({e_os})."
            ) from e_os


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed look-at producing a camera pose (camera-to-world) for pyrender."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _orbit_camera_pose(
    center: np.ndarray,
    radius: float,
    height: float,
    yaw: float,
) -> np.ndarray:
    eye = center + np.array(
        [radius * np.cos(yaw), radius * np.sin(yaw), height], dtype=np.float64
    )
    return _look_at(eye, center, np.array([0.0, 0.0, 1.0]))


def _build_checker_floor(
    center: np.ndarray, half_size: float, cell: float = 0.25
) -> list[trimesh.Trimesh]:
    """Build two trimesh planes (pure white + pure black) tiled in a checker pattern."""
    n = max(2, int(np.ceil(2 * half_size / cell)))
    cx, cy = float(center[0]), float(center[1])
    x0 = cx - half_size
    y0 = cy - half_size

    verts_light: list[list[float]] = []
    verts_dark: list[list[float]] = []
    faces_light: list[list[int]] = []
    faces_dark: list[list[int]] = []

    def _add_quad(target_v, target_f, x, y):
        base = len(target_v)
        target_v.extend(
            [
                [x, y, 0.0],
                [x + cell, y, 0.0],
                [x + cell, y + cell, 0.0],
                [x, y + cell, 0.0],
            ]
        )
        target_f.extend(
            [
                [base, base + 1, base + 2],
                [base, base + 2, base + 3],
            ]
        )

    for i in range(n):
        for j in range(n):
            x = x0 + i * cell
            y = y0 + j * cell
            if (i + j) % 2 == 0:
                _add_quad(verts_light, faces_light, x, y)
            else:
                _add_quad(verts_dark, faces_dark, x, y)

    out: list[trimesh.Trimesh] = []
    for verts, faces, color in (
        (verts_light, faces_light, [255, 255, 255, 255]),
        (verts_dark, faces_dark, [0, 0, 0, 255]),
    ):
        if not verts:
            continue
        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        tm.visual.face_colors = np.tile(
            np.array(color, dtype=np.uint8), (len(f), 1)
        )
        out.append(tm)
    return out


@dataclass
class SceneItem:
    """A trimesh + (optional) override pyrender material to add to the scene."""

    mesh: trimesh.Trimesh
    material: pyrender.Material | None = None
    smooth: bool = True


class PyrenderSceneRenderer:
    """Reusable pyrender scene with floor, lights, cinematic orbit camera."""

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        orbit_period_frames: int = 90,
        orbit_radius: float | None = None,
        orbit_height: float = 0.6,
        bg_color: tuple[float, float, float, float] = (0.78, 0.82, 0.88, 1.0),
        ambient: tuple[float, float, float] = (0.18, 0.18, 0.20),
        floor_cell: float = 0.25,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.orbit_period_frames = max(1, int(orbit_period_frames))
        self.orbit_radius = orbit_radius
        self.orbit_height = float(orbit_height)
        self.bg_color = bg_color
        self.ambient = np.asarray(ambient, dtype=np.float64)
        self.floor_cell = float(floor_cell)
        self._renderer = _try_offscreen_renderer(self.width, self.height)
        self._yfov = np.pi / 3.0

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.delete()
            except Exception:
                pass
            self._renderer = None

    def __del__(self) -> None:
        self.close()

    def _add_lights(self, scene: pyrender.Scene, center: np.ndarray) -> None:
        key_pose = _look_at(
            center + np.array([1.2, 0.8, 2.4]), center, np.array([0.0, 0.0, 1.0])
        )
        scene.add(
            pyrender.DirectionalLight(color=np.array([1.0, 0.97, 0.92]), intensity=2.4),
            pose=key_pose,
        )
        fill_pose = _look_at(
            center + np.array([-1.4, -1.0, 1.4]), center, np.array([0.0, 0.0, 1.0])
        )
        scene.add(
            pyrender.DirectionalLight(color=np.array([0.85, 0.9, 1.0]), intensity=1.0),
            pose=fill_pose,
        )
        rim_pose = _look_at(
            center + np.array([0.0, -2.0, 0.8]), center, np.array([0.0, 0.0, 1.0])
        )
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=0.8),
            pose=rim_pose,
        )

    def _add_floor(
        self, scene: pyrender.Scene, center: np.ndarray, radius: float
    ) -> None:
        ground_size = max(4.0, 3.0 * radius)
        for gm in _build_checker_floor(
            center=center, half_size=ground_size / 2.0, cell=self.floor_cell
        ):
            scene.add(pyrender.Mesh.from_trimesh(gm, smooth=False))

    def _add_camera(
        self, scene: pyrender.Scene, center: np.ndarray, radius: float, yaw: float
    ) -> None:
        cam_pose = _orbit_camera_pose(center, radius, self.orbit_height, yaw)
        scene.add(
            pyrender.PerspectiveCamera(
                yfov=self._yfov, aspectRatio=self.width / self.height
            ),
            pose=cam_pose,
        )

    def render_frame(
        self,
        items: list[SceneItem],
        center: np.ndarray,
        radius: float,
        yaw: float,
    ) -> np.ndarray:
        """Render a single frame given content meshes + camera params."""
        scene = pyrender.Scene(bg_color=self.bg_color, ambient_light=self.ambient)
        for item in items:
            mesh = pyrender.Mesh.from_trimesh(
                item.mesh, smooth=item.smooth, material=item.material
            )
            scene.add(mesh)
        self._add_floor(scene, center, radius)
        self._add_camera(scene, center, radius, yaw)
        self._add_lights(scene, center)
        color, _ = self._renderer.render(scene)
        return np.ascontiguousarray(color[..., :3])


# ---------------------------------------------------------------------------
# Robot mesh renderer
# ---------------------------------------------------------------------------


@dataclass
class RobotSpec:
    """One robot's geometry + per-frame qpos/root_pose."""

    uid: str
    geom: DifferentiableRobotGeometry
    joint_names: list[str]


def _resolve_urdf_path(urdf_path_rel: str) -> str:
    """Resolve a stored urdf_path against the repo root, with safe fallback."""
    if osp.isabs(urdf_path_rel) and osp.isfile(urdf_path_rel):
        return urdf_path_rel
    candidate = osp.join(_REPO_ROOT, urdf_path_rel)
    if osp.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"Could not resolve URDF path: {urdf_path_rel} (tried {candidate})"
    )


def build_robot_specs(robot_infos: list[dict]) -> list[RobotSpec]:
    """Instantiate `DifferentiableRobotGeometry` for each robot in robot_infos."""
    specs: list[RobotSpec] = []
    for info in robot_infos:
        urdf_path = _resolve_urdf_path(info["urdf_path"])
        urdf_dir = osp.dirname(urdf_path)
        joint_names = list(info.get("joint_names") or [])
        if not joint_names:
            raise ValueError(
                f"robot_info for uid={info.get('uid')} has empty joint_names; "
                "cannot build DifferentiableRobotGeometry."
            )
        geom = DifferentiableRobotGeometry(
            urdf_path=urdf_path,
            base_dir=urdf_dir,
            joint_names=joint_names,
        )
        specs.append(
            RobotSpec(uid=info.get("uid") or "robot", geom=geom, joint_names=joint_names)
        )
    return specs


def _split_qpos_per_robot(
    qpos_t: np.ndarray, specs: list[RobotSpec]
) -> list[np.ndarray]:
    """Slice a single-frame qpos vector into per-robot pieces."""
    if qpos_t.ndim == 2 and qpos_t.shape[0] == len(specs):
        return [qpos_t[i] for i in range(len(specs))]
    if qpos_t.ndim == 1:
        if len(specs) == 1:
            return [qpos_t]
        out = []
        offset = 0
        for spec in specs:
            n = len(spec.joint_names)
            out.append(qpos_t[offset : offset + n])
            offset += n
        return out
    raise ValueError(
        f"Unsupported qpos shape {qpos_t.shape} for {len(specs)} robot(s)."
    )


def _expand_to_joint_dim(action_q: np.ndarray, n_joints: int) -> np.ndarray:
    """Pad/truncate an action vector to match a robot's joint count."""
    a = np.asarray(action_q).reshape(-1)
    if a.shape[0] == n_joints:
        return a
    if a.shape[0] < n_joints:
        pad = np.full(n_joints - a.shape[0], a[-1], dtype=a.dtype)
        return np.concatenate([a, pad], axis=0)
    return a[:n_joints]


def _split_root_poses_per_robot(
    root_t: np.ndarray, n_robots: int
) -> list[np.ndarray]:
    if root_t.ndim == 2 and root_t.shape[0] == n_robots and root_t.shape[1] == 7:
        return [root_t[i] for i in range(n_robots)]
    if root_t.ndim == 1:
        if root_t.shape[0] == 7 and n_robots == 1:
            return [root_t]
        if root_t.shape[0] == n_robots * 7:
            return [root_t[i * 7 : (i + 1) * 7] for i in range(n_robots)]
    raise ValueError(
        f"Unsupported root_poses shape {root_t.shape} for {n_robots} robot(s)."
    )


_ROBOT_MATERIAL = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=(0.62, 0.66, 0.74, 1.0),
    metallicFactor=0.25,
    roughnessFactor=0.55,
)


def _build_meshes_for_frame(
    specs: list[RobotSpec],
    qpos_t: np.ndarray,
    root_t: np.ndarray,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    qpos_per = _split_qpos_per_robot(qpos_t, specs)
    root_per = _split_root_poses_per_robot(root_t, len(specs))

    meshes: list[trimesh.Trimesh] = []
    centers: list[np.ndarray] = []
    for spec, q, rp in zip(specs, qpos_per, root_per):
        q_exp = _expand_to_joint_dim(q, len(spec.joint_names))
        q_t = torch.as_tensor(q_exp, dtype=torch.float32).unsqueeze(0)
        rp_t = torch.as_tensor(rp, dtype=torch.float32).unsqueeze(0)
        spec.geom.set_pose(q_t, rp_t)
        verts, faces, _ = mesh_to_arrays(to_o3d_mesh(spec.geom.sdf))
        if len(verts) == 0 or len(faces) == 0:
            continue
        face_colors = np.tile(
            np.array([170, 175, 185, 255], dtype=np.uint8), (len(faces), 1)
        )
        tm = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            face_colors=face_colors,
            process=False,
        )
        meshes.append(tm)
        centers.append(np.asarray(rp[:3], dtype=np.float64))

    if meshes:
        all_verts = np.concatenate([m.vertices for m in meshes], axis=0)
        center = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2.0
    elif centers:
        center = np.mean(np.stack(centers, axis=0), axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)
    return meshes, center


class MeshRobotRenderer:
    """Render a robot trajectory to a sequence of RGB frames."""

    def __init__(
        self,
        robot_infos: list[dict],
        width: int = 512,
        height: int = 512,
        orbit_period_frames: int = 90,
        orbit_radius: float | None = None,
        orbit_height: float = 0.6,
    ) -> None:
        self.specs = build_robot_specs(robot_infos)
        self.scene = PyrenderSceneRenderer(
            width=width,
            height=height,
            orbit_period_frames=orbit_period_frames,
            orbit_radius=orbit_radius,
            orbit_height=orbit_height,
        )

    def close(self) -> None:
        self.scene.close()

    def __del__(self) -> None:
        self.close()

    def _estimate_radius(self, meshes: list[trimesh.Trimesh]) -> float:
        if self.scene.orbit_radius is not None:
            return float(self.scene.orbit_radius)
        if not meshes:
            return 1.5
        all_verts = np.concatenate([m.vertices for m in meshes], axis=0)
        extent = float(np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0)))
        return max(1.5, 1.4 * extent)

    def render_trajectory(
        self,
        target_qpos: np.ndarray,
        root_poses: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        T = target_qpos.shape[0]
        if root_poses.shape[0] != T:
            raise ValueError(
                f"target_qpos T={T} != root_poses T={root_poses.shape[0]}"
            )
        if valid_mask is None:
            valid_mask = np.ones(T, dtype=bool)

        radius = None
        frames: list[np.ndarray] = []
        rendered_idx = 0
        for t in range(T):
            if not bool(valid_mask[t]):
                continue
            meshes, center = _build_meshes_for_frame(
                self.specs, target_qpos[t], root_poses[t]
            )
            if radius is None:
                radius = self._estimate_radius(meshes)
            yaw = (
                2.0
                * np.pi
                * (rendered_idx % self.scene.orbit_period_frames)
                / self.scene.orbit_period_frames
            )
            items = [
                SceneItem(mesh=m, material=_ROBOT_MATERIAL, smooth=True) for m in meshes
            ]
            frames.append(self.scene.render_frame(items, center, radius, yaw))
            rendered_idx += 1
        if not frames:
            return np.zeros(
                (0, self.scene.height, self.scene.width, 3), dtype=np.uint8
            )
        return np.stack(frames, axis=0)
