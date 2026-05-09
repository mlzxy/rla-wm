"""
Waypoint Visualizer for rendering trajectory waypoints in SAPIEN.

Uses actor pooling to efficiently render waypoints without constantly
spawning/deleting actors. Color codes waypoints by action type.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rich import print
import time
import sapien
import sapien.internal_renderer as R
from sapien.utils.viewer.plugin import Plugin
from .trace import Trace, Waypoint
from .primitives import PrimitiveResult


# Color mapping for waypoint types (RGBA)
WAYPOINT_COLORS = {
    "start": (0.0, 0.9, 0.0, 0.8),      # Green - start position
    "end": (0.9, 0.0, 0.0, 0.8),         # Red - end position
    "pick": (0.0, 0.8, 0.8, 0.8),        # Cyan - pick action
    "place": (0.8, 0.0, 0.8, 0.8),       # Magenta - place action
    "push": (0.9, 0.9, 0.0, 0.8),        # Yellow - push action
    "tool_push": (1.0, 0.5, 0.0, 0.8),   # Orange - tool push
    "home": (0.5, 0.5, 0.5, 0.8),        # Gray - home position
    "move": (0.3, 0.5, 0.9, 0.8),        # Blue - generic move
    "retract": (1.0, 0.6, 0.0, 0.8),     # Dark Orange - vertical retract
    "transit_to_center": (1.0, 0.8, 0.0, 0.8), # Gold - horizontal transit
    "transit_to_approach": (1.0, 1.0, 0.0, 0.8), # Yellow - final transit
    "lift": (0.0, 0.8, 0.0, 0.8),        # Green - vertical lift
    "gripper_closed": (1.0, 0.1, 0.0, 1.0), # Vivid Red-Orange - gripper closed
    "default": (0.5, 0.5, 0.5, 0.8),     # Gray fallback
}

# Size settings
WAYPOINT_RADIUS = 0.012  # 1.2cm spheres
WAYPOINT_END_RADIUS = 0.018  # Larger for start/end

# Hidden position (far away to "hide" actors)
HIDDEN_POS = [100.0, 100.0, 100.0]


class WaypointVisualizer(Plugin):
    """
    Visualizes trajectory waypoints as coordinate frames in SAPIEN.
    
    Uses viewer.add_coordinate_frame to draw RGB axes at each waypoint.
    Supports alpha dimming for transit vs interaction waypoints.
    Fallbacks to sphere actors if viewer is not available (e.g. headless).
    """
    
    def __init__(
        self,
        env,
        pool_size: int = 200,
        waypoint_radius: float = WAYPOINT_RADIUS
    ):
        """
        Initialize the waypoint visualizer.
        
        Args:
            env: ManiSkill environment with SAPIEN scene
            pool_size: Number of actors to pre-allocate in pool
            waypoint_radius: Radius of waypoint spheres (ignored for axes)
        """
        super().__init__()
        self.env = env
        self.scene = env.unwrapped.scene
        self.pool_size = pool_size
        
        # UI overlays
        self.last_result: Optional[PrimitiveResult] = None
        self.ui_window = None
        
        # Pool of visualization objects: list of [obj, in_use_flag]
        self._pool: List[Tuple[Any, bool]] = []
        self._active_count = 0
        self._use_gizmos = False
        
        # Track custom frames created with alpha (need separate cleanup)
        self._custom_frames: List[Any] = []
        
        # Lazily create pool on first use
        self._pool_initialized = False
        
        # Persistent TCP frame
        self._tcp_node = None
    
    
    def set_result(self, result: PrimitiveResult):
        """Set the latest execution result for display in the HUD."""
        self.last_result = result
        
    def get_ui_windows(self):
        """Create the HUD overlay for execution results."""
        if self.ui_window is None:
            # Positioned at top-center/right
            self.ui_window = R.UIWindow().Label("Primitive Result HUD").Pos(400, 20).Size(400, 150)
            
        self.ui_window.remove_children()
        
        if self.last_result is not None:
            status_text = "SUCCESS" if self.last_result.success else "FAILURE"
            # Format action name: convert underscores to hyphens and uppercase
            action_display = self.last_result.action_name.replace("_", "-").upper()
            # We don't have direct color control in R.UIDisplayText but we can use labels
            self.ui_window.append(R.UIDisplayText().Text(f"Action: {action_display}"))
            self.ui_window.append(R.UIDisplayText().Text(f"Status: {status_text}"))
            self.ui_window.append(R.UIDisplayText().Text(f"Message: {self.last_result.message}"))
            if self.last_result.steps_taken > 0:
                self.ui_window.append(R.UIDisplayText().Text(f"Steps: {self.last_result.steps_taken}"))
                
        return [self.ui_window]


    def _ensure_pool(self):
        """Initialize object pool with sphere actors for "tube" path."""
        if self._pool_initialized:
            return
            
        # Always use sphere actors for the pool to create a "tube" effect
        # and support dynamic color updates.
        self._use_gizmos = False
        for i in range(self.pool_size):
            actor = self._create_sphere_actor(f"waypoint_marker_{i}")
            actor.set_pose(sapien.Pose(p=HIDDEN_POS))
            self._pool.append([actor, False])
        
        self._pool_initialized = True
    
    def _create_sphere_actor(self, name: str):
        """Create a visual-only sphere actor (fallback)."""
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=0.015,
            material=sapien.render.RenderMaterial(base_color=(0.5, 0.5, 0.5, 0.8))
        )
        actor = builder.build_static(name=name)
        return actor
    
    def _get_render_shapes(self, obj: Any) -> List[Any]:
        """Get all render shapes from an object (Link, Actor, or ManiSkill Actor)."""
        shapes = []
        if hasattr(obj, "get_visual_bodies"):
            for visual_body in obj.get_visual_bodies():
                for render_shape in visual_body.get_render_shapes():
                    shapes.append(render_shape)
        elif hasattr(obj, "_objs"):
            # ManiSkill Actor wrapper: underlying entities have RenderBodyComponent
            for entity in obj._objs:
                for comp in entity.components:
                    if isinstance(comp, sapien.render.RenderBodyComponent):
                        for render_shape in comp.render_shapes:
                            shapes.append(render_shape)
        elif hasattr(obj, "components"):
            # Raw SAPIEN entity with components
            for comp in obj.components:
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    for render_shape in comp.render_shapes:
                        shapes.append(render_shape)
        return shapes

    def _get_object(self) -> Optional[Any]:
        """Get an available object from the pool."""
        self._ensure_pool()
        
        for entry in self._pool:
            if not entry[1]:  # Not in use
                entry[1] = True
                self._active_count += 1
                return entry[0]
        
        return None
    
    def _create_coordinate_frame_node(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        alpha: float = 1.0,
        length: float = 0.06,
        radius: float = 0.06,
        colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None
    ) -> Optional[Any]:
        """
        Create a coordinate frame node with custom transparency and colors.
        
        Args:
            position: World position [3]
            quaternion: Quaternion [4] in xyzw format
            alpha: Transparency (0.0 = invisible, 1.0 = opaque)
            length: Length of each axis
            radius: Radius of axis cylinders
            colors: Custom colors for X, Y, Z axes. If None, uses RGB.
            
        Returns:
            Node object, or None if viewer unavailable
        """
        viewer = self.env.unwrapped.viewer
        if viewer is None:
            return None
        
        renderer_context = viewer.renderer_context
        render_scene = viewer.render_scene
        
        # Default colors (RGB)
        if colors is None:
            colors = {
                "x": [1, 0, 0, alpha],
                "y": [0, 1, 0, alpha],
                "z": [0, 0, 1, alpha]
            }
        else:
            # Apply alpha to custom colors if not already provided
            for key in colors:
                if len(colors[key]) == 3:
                    colors[key] = list(colors[key]) + [alpha]
                elif len(colors[key]) == 4:
                    colors[key] = list(colors[key][:3]) + [alpha]

        # Create materials
        mat_x = renderer_context.create_material(colors["x"], [0, 0, 0, 1], 0, 1, 0)
        mat_y = renderer_context.create_material(colors["y"], [0, 0, 0, 1], 0, 1, 0)
        mat_z = renderer_context.create_material(colors["z"], [0, 0, 0, 1], 0, 1, 0)
        
        # Create meshes
        cone = renderer_context.create_cone_mesh(16)
        capsule = renderer_context.create_capsule_mesh(radius=radius, half_length=0.5, segments=16, half_rings=4)
        
        # Create models
        model_x_cone = renderer_context.create_model([cone], [mat_x])
        model_y_cone = renderer_context.create_model([cone], [mat_y])
        model_z_cone = renderer_context.create_model([cone], [mat_z])
        model_x_capsule = renderer_context.create_model([capsule], [mat_x])
        model_y_capsule = renderer_context.create_model([capsule], [mat_y])
        model_z_capsule = renderer_context.create_model([capsule], [mat_z])
        
        cone_scale = [0.08, 0.06, 0.06]
        capsule_scale = [1.0, 1.0, 1.0]
        
        # Build node hierarchy
        node = render_scene.add_node()
        
        # X-axis
        obj = render_scene.add_object(model_x_cone, node)
        obj.set_scale(cone_scale)
        obj.set_position([length, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False
        
        obj = render_scene.add_object(model_x_capsule, node)
        obj.set_scale(capsule_scale)
        obj.set_position([length * 0.52, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False
        
        # Y-axis
        obj = render_scene.add_object(model_y_cone, node)
        obj.set_scale(cone_scale)
        obj.set_position([0, length, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False
        
        obj = render_scene.add_object(model_y_capsule, node)
        obj.set_scale(capsule_scale)
        obj.set_position([0, length * 0.51, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False
        
        # Z-axis
        obj = render_scene.add_object(model_z_cone, node)
        obj.set_scale(cone_scale)
        obj.set_position([0, 0, length])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False
        
        obj = render_scene.add_object(model_z_capsule, node)
        obj.set_scale(capsule_scale)
        obj.set_position([0, 0, length * 0.5])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False
        
        # Set pose
        # node.set_scale([length, length, length]) # REMOVED: Redundant scaling makes it tiny
        node.set_position(position)
        
        # Convert xyzw to wxyz for SAPIEN
        q_wxyz = quaternion[[3, 0, 1, 2]]
        node.set_rotation(q_wxyz)
        
        return node

    def update_tcp_pose(self, position: np.ndarray, quaternion: np.ndarray):
        """
        Update the persistent TCP coordinate frame visualization.
        
        Args:
            position: TCP position [3]
            quaternion: TCP quaternion [4] in xyzw format
        """
        viewer = self.env.unwrapped.viewer
        if viewer is None:
            return
            
        if self._tcp_node is None:
            # Create a semi-transparent Cyan (X), Pink (Y), Yellow (Z) frame for TCP
            # to distinguish from waypoint RGB frame
            colors = {
                "x": [0, 1, 1, 1],  # Cyan
                "y": [1, 0.2, 0.6, 1],  # Pink/Magenta
                "z": [1, 1, 0, 1]   # Yellow
            }
            # self._tcp_node = self._create_coordinate_frame_node(
            #     position, quaternion, alpha=0.9, length=0.1, radius=0.008, colors=colors
            # )
        else:
            # self._tcp_node.set_position(position)
            # Convert xyzw to wxyz for SAPIEN
            q_wxyz = quaternion[[3, 0, 1, 2]]
            # self._tcp_node.set_rotation(q_wxyz)
    
    def _add_transparent_coordinate_frame(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        alpha: float = 1.0,
        length: float = 0.06,
        radius: float = 0.06
    ) -> Optional[Any]:
        """
        Legacy wrapper for _create_coordinate_frame_node.
        """
        return self._create_coordinate_frame_node(
            position, quaternion, alpha=alpha, length=length, radius=radius
        )
    
    def visualize_trace(self, trace: Trace, use_alpha_dimming: bool = True) -> int:
        """
        Visualize waypoints from a trace.
        
        Args:
            trace: Trace object with waypoints
            use_alpha_dimming: If True, dim non-interaction waypoints
            
        Returns:
            Number of waypoints rendered
        """
        # Clear previous visualization
        self.clear()
        
        rendered = 0
        viewer = self.env.unwrapped.viewer
        
        for i, waypoint in enumerate(trace):
            pos = waypoint.position
            quat = waypoint.orientation if waypoint.orientation is not None else np.array([1, 0, 0, 0])
            
            # Metadata: is this a critical pose that needs RGB axes?
            is_special = waypoint.metadata.get("is_special", False)
            phase = waypoint.metadata.get("phase", "move")
            is_interaction = waypoint.metadata.get("is_interaction", False)
            
            if is_special and viewer is not None:
                # 1. SPECIAL POSE: Render as full RGB Coordinate Frame
                # Use viewer's native coordinate frame which is more robust
                q_wxyz = quat[[3, 0, 1, 2]]
                node = viewer.add_coordinate_frame(sapien.Pose(p=pos, q=q_wxyz), length=0.06, radius=0.05)
                if node is not None:
                    self._custom_frames.append(node)
                    rendered += 1
            else:
                # 2. TUBE PATH: Render as pooled sphere marker
                obj = self._get_object()
                if obj is None:
                    break
                
                # Determine color based on phase
                if waypoint.gripper_state == "closed":
                    color = WAYPOINT_COLORS["gripper_closed"]
                else:
                    color = WAYPOINT_COLORS.get(phase, WAYPOINT_COLORS["default"])
                
                # Dim transit spheres slightly if alpha dimming is on
                if use_alpha_dimming and not is_interaction:
                    color = list(color[:3]) + [0.4]
                
                # Set pose (convert to wxyz)
                q_wxyz = quat[[3, 0, 1, 2]]
                obj.set_pose(sapien.Pose(p=pos, q=q_wxyz))
                
                # Try to set color (dynamic visual update in SAPIEN)
                try:
                    for render_shape in self._get_render_shapes(obj):
                        mat = render_shape.material
                        if hasattr(mat, "set_base_color"):
                            mat.set_base_color(color)
                        elif hasattr(mat, "base_color"):
                            mat.base_color = color
                except Exception as e:
                    print(f"[red]Failed to set color: {e}[/red]")
                    import traceback
                    traceback.print_exc()
                
                rendered += 1
        
        return rendered
    
    def clear(self):
        """Hide all objects and remove custom frames."""
        # Clear pooled objects
        if self._pool_initialized:
            for entry in self._pool:
                if entry[1]:  # In use
                    obj = entry[0]
                    if self._use_gizmos:
                        obj.set_position(HIDDEN_POS)
                    else:
                        obj.set_pose(sapien.Pose(p=HIDDEN_POS))
                    entry[1] = False
            self._active_count = 0
        
        # Remove custom transparent frames
        if self._custom_frames:
            viewer = self.env.unwrapped.viewer
            if viewer is not None:
                for node in self._custom_frames:
                    try:
                        viewer.render_scene.remove_node(node)
                    except Exception:
                        pass
            self._custom_frames.clear()
        
        # NOTE: We persistent the TCP node across clears to keep the same frame
        # If we need to hide it, we would add logic here.
    
    def destroy(self):
        """Remove all objects from the scene (cleanup)."""
        self.clear()
        
        # Destroy TCP frame
        if self._tcp_node:
            viewer = self.env.unwrapped.viewer
            if viewer is not None:
                try:
                    viewer.render_scene.remove_node(self._tcp_node)
                except Exception:
                    pass
            self._tcp_node = None
            
        if not self._pool_initialized:
            return
            
        for entry in self._pool:
            obj = entry[0]
            try:
                if self._use_gizmos:
                    obj.set_position(HIDDEN_POS)
                else:
                    self.scene.remove_actor(obj)
            except Exception:
                pass
        self._pool.clear()
        self._pool_initialized = False
        self._active_count = 0


class ArrowMarker:
    """
    Optional arrow marker for showing direction at waypoints.
    
    Not used by default, but available for enhanced visualization.
    """
    
    @staticmethod
    def create_arrow(
        scene,
        position: np.ndarray,
        direction: np.ndarray,
        color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
        length: float = 0.05,
        name: str = "arrow"
    ):
        """
        Create an arrow actor pointing in a direction.
        
        Uses a cylinder + cone combination for arrow shape.
        """
        builder = scene.create_actor_builder()
        
        # Cylinder for shaft
        builder.add_cylinder_visual(
            radius=0.003,
            half_length=length * 0.7 / 2,
            material=sapien.render.RenderMaterial(base_color=color)
        )
        
        # TODO: Add cone for arrowhead (requires transform)
        
        arrow = builder.build_static(name=name)
        arrow.set_pose(sapien.Pose(p=position))
        return arrow
