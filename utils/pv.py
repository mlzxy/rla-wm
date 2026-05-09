import asyncio
from loguru import logger
import pickle

import torch
import numpy as np
import traceback
from torch import Tensor
import os.path as osp
import os


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

try:
    import pyvista as pv
    from pyvista.trame.ui.vuetify3 import (
        button,
        divider,
        slider,
        text_field,
    )
    from trame.widgets import vuetify3 as vuetify
    from trame.widgets import html

    pv.set_jupyter_backend("trame")
    pv.set_plot_theme("paraview")
except ImportError:
    logger.warning("pyvista not installed, some features will be disabled")

COLORS = "#4169E1,#40E0D0,#6B8E23,#E6E6FA,#FFD700,#FFDAB9,#8A2BE2,#00CED1,#00FA9A,#6495ED,#ADD8E6,#BDB76B,#00BFFF,#EEE8AA,#6B8E23,#8B4513,#48D1CC,#FFD700,#D8BFD8,#7FFFD4,#F0E68C,#6A5ACD,#D2B48C,#FFEBCD,#00FF00,#FF69B4,#F5DEB3,#32CD32,#9932CC,#483D8B,#228B22,#FFDAB9,#40E0D0,#20B2AA,#FFEFD5,#FFDEAD,#F5FFFA,#3CB371,#FFFACD,#556B2F,#F4A460,#B0C4DE,#B8860B,#87CEFA,#BA55D3,#FF4500,#006400,#BA55D3,#FAEBD7,#CD853F,#87CEEB,#FFF0F5,#FAFAD2,#00CED1,#DDA0DD,#8A2BE2,#FFE4B5,#8FBC8F,#66CDAA,#FDF5E6,#FFB6C1,#4682B4,#AFEEEE,#E0FFFF,#8B008B,#FFEFD5,#FF1493,#F5F5DC,#FF7F50,#9400D3,#008000,#FFE4B5,#008B8B,#5F9EA0,#FFE4C4,#7CFC00,#DAA520,#DA70D6,#6A5ACD,#7B68EE,#7B68EE,#FFFFE0,#FF6347,#00FFFF,#EEE8AA,#483D8B,#FFF5EE,#FFC0CB,#DB7093,#DEB887,#FFFFFF,#FFFF00,#F5F5F5,#4B0082,#DA70D6,#FFE4E1,#800080,#FAFAD2,#BC8F8F,#48D1CC,#F0E68C,#BDB76B,#EE82EE,#9ACD32,#90EE90,#808000,#A0522D,#556B2F,#008080,#C71585,#008080,#98FB98,#9370DB,#D2691E,#2E8B57,#1E90FF,#20B2AA,#FFA500,#00FFFF,#FF8C00,#9370DB".split(
    ","
)


def checkbox(model, icons, tooltip, **kwargs):  # numpydoc ignore=PR01
    """Create a vuetify checkbox."""
    with vuetify.VTooltip(location="bottom"):
        with vuetify.Template(v_slot_activator=("{ props }",)):
            with html.Div(v_bind=("props",)):
                vuetify.VCheckbox(
                    v_model=model,
                    true_icon=icons[0],
                    false_icon=icons[1],
                    density="compact",
                    hide_details=True,
                    classes="ma-1 py-1",
                    **kwargs,
                )
        html.Span(tooltip)


DEFAULT_COLORS = (
    "red",
    "blue",
    "yellow",
    "magenta",
    "green",
    "indigo",
    "darkorange",
    "cyan",
    "pink",
    "yellowgreen",
)


def vis_link(
    plotter,
    joints,
    connections,
    prefix="",
    radius=0.1,
    render=True,
    joints_color=None,
    links_color=None,
):
    actors = []
    for i, joint in enumerate(joints):
        if joints_color is not None:
            color = joints_color[i]
        else:
            color = COLORS[i % len(COLORS)]
        a = plotter.add_mesh(
            pv.Sphere(center=joint, radius=radius),
            color=color,
            name=f"{prefix}joint-{i}",
            render=render,
        )
        actors.append(a)

    for i, connection in enumerate(connections):
        joint1 = joints[connection[0]]
        joint2 = joints[connection[1]]
        if links_color is not None:
            color = links_color[i]
        else:
            color = "red"
        a = plotter.add_mesh(
            pv.Tube(joint1, joint2, radius=radius / 2),
            color=color,
            name=f"{prefix}tube-{i}",
            render=render,
        )
        actors.append(a)
    return actors


def to_np(x):
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


def to255(rgbs):
    if rgbs.max() <= 1:
        return (rgbs * 255).astype(np.uint8)
    else:
        return rgbs


def mask2color(mask: np.ndarray):
    mask = to_np(mask)
    mask = mask.astype(int)
    mask = mask - mask.min()
    num_classes = mask.max() + 1
    if num_classes > len(DEFAULT_COLORS):
        colors = COLORS
    else:
        colors = DEFAULT_COLORS
    colors = np.array([mcolors.to_rgb(c) for c in colors])
    oshape = mask.shape
    colored_mask = to255(colors[mask.flatten() % len(colors)])
    colored_mask = colored_mask.reshape(oshape + (3,))
    return colored_mask


def o3d_mesh_to_pv(o3d_mesh):
    """
    Convert an Open3D TriangleMesh to a PyVista PolyData mesh.

    Parameters
    ----------
    o3d_mesh : open3d.geometry.TriangleMesh
        The Open3D mesh to convert.

    Returns
    -------
    pyvista.PolyData
        A PyVista mesh with vertices, faces, and optionally normals and colors.

    Examples
    --------
    >>> import open3d as o3d
    >>> from utils import pv
    >>> o3d_mesh = o3d.geometry.TriangleMesh()
    >>> pv_mesh = pv.o3d_mesh_to_pv(o3d_mesh)
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required for o3d_mesh_to_pv")

    # Extract vertices and faces from Open3D mesh
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # PyVista requires faces in a specific format:
    # [n, v0, v1, v2, n, v0, v1, v2, ...] where n is the number of vertices per face
    # For triangles, n=3, so we prepend 3 to each face
    faces_pv = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()

    # Create PyVista mesh
    pv_mesh = pv.PolyData(vertices, faces_pv)

    # Optionally copy vertex normals if they exist
    if o3d_mesh.has_vertex_normals():
        normals = np.asarray(o3d_mesh.vertex_normals)
        pv_mesh.point_data["Normals"] = normals

    # Optionally copy vertex colors if they exist
    if o3d_mesh.has_vertex_colors():
        colors = np.asarray(o3d_mesh.vertex_colors)
        # Open3D colors are in [0, 1], PyVista expects [0, 255] for RGB
        pv_mesh.point_data["Colors"] = (colors * 255).astype(np.uint8)

    return pv_mesh


class Plotter:
    def render(
        self, render=True, offline=False, reset_before_render=True, pyrender_fix=False
    ):
        if self.backend.startswith("remote:"):
            backend = self.backend.split("remote:")[-1]
            self.dump(backend)
            if self.logging:
                logger.info(f"pyvista rendering to {backend}")
            return
        elif self.backend.startswith("from:"):
            backend = self.backend.split("from:")[-1]
            if not osp.exists(backend):
                if self.logging:
                    logger.info(f"file {backend} does not exist yet")
                return
            remote_mtime = osp.getmtime(backend)
            if self.gui_state["pv"].get("remote_mtime", -1) != remote_mtime:
                if self.logging:
                    logger.info(f"rendering from {backend}")
                self.load(backend)
            self.gui_state["pv"]["remote_mtime"] = remote_mtime

        if self._viewer is None:
            if self.logging:
                logger.info("Viewer not initialized, skip rendering")
            return

        fid = self.gui_state["ui"]["frame_id"]
        # view_mode = self.gui_state["ui"]["view"]
        node_radius = self.gui_state["ui"]["node_radius"]

        def remove_points(name_prefix="splats"):
            if name_prefix == "splats":
                # compatibility with old single-group code
                points = self.actors.pop("points", None)
                if points is not None:
                    self.plotter.remove_actor(points)

            # Remove any actor starting with this prefix
            # This is a bit safer if we have multiple groups
            to_remove = []
            for k, v in self.actors.items():
                if k.startswith(name_prefix):
                    self.plotter.remove_actor(v)
                    to_remove.append(k)
            for k in to_remove:
                self.actors.pop(k)

        def remove_graph():
            for a in self.actors.pop("graph", []):
                self.plotter.remove_actor(a)

        def render_points_group(
            name, means_data, colors_data=None, opacity=1.0, color=None
        ):
            # means_data: [T, N, 3] or [N, 3] check inside
            # colors_data: [T, N, 3] or [N, 3] or None

            actor_name = f"splats_{name}"
            arrow_actor_name = f"arrows_{name}"

            # If data is sequence
            current_means = means_data
            if (
                len(means_data) > fid
                and hasattr(means_data, "shape")
                and len(means_data.shape) == 3
            ):  # heuristics for [T, N, 3]
                current_means = means_data[fid]
            elif isinstance(means_data, list):
                current_means = means_data[fid]

            current_colors = colors_data
            if colors_data is not None:
                if (
                    len(colors_data) > fid
                    and hasattr(colors_data, "shape")
                    and len(colors_data.shape) == 3
                ):  # heuristics
                    current_colors = colors_data[fid]
                elif isinstance(colors_data, list):
                    current_colors = colors_data[fid]

            # Handle flow visualization for (*, 6) data
            current_means_np = to_np(current_means)
            if hasattr(current_means_np, "shape") and current_means_np.shape[-1] == 6:
                points_A = current_means_np[:, :3]
                points_B = current_means_np[:, 3:]
                vectors = points_B - points_A

                cloud = pv.PolyData(points_A)
                cloud["vectors"] = vectors

                # Create a custom thin arrow
                # "very thin, thinner than the point itself"
                arrow_source = pv.Arrow(
                    tip_length=0.2,
                    tip_radius=0.03,
                    shaft_radius=0.005,
                    start=(0, 0, 0),
                    direction=(1, 0, 0),
                )
                arrows = cloud.glyph(
                    geom=arrow_source, orient="vectors", scale="vectors", factor=1.0
                )

                # Use yellow for arrows, unless points are yellow, then red? or just fixed yellow.
                arrow_color = "yellow"

                try:
                    self.actors[arrow_actor_name] = self.plotter.add_mesh(
                        arrows,
                        color=arrow_color,
                        opacity=0.3,  # "dim semi transparent"
                        name=arrow_actor_name,
                        render=False,
                    )
                except Exception:
                    remove_points(arrow_actor_name)
                    if self.logging:
                        logger.info(str(traceback.format_exc()))

                current_means = points_A  # Use start points for the dots
            else:
                # Clean up arrows if they exist but data is not flow anymore
                remove_points(arrow_actor_name)

            try:
                self.actors[actor_name] = self.plotter.add_points(
                    to_np(current_means),
                    opacity=opacity,
                    render_points_as_spheres=True,
                    emissive=True,
                    rgb=current_colors is not None,
                    name=actor_name,
                    scalars=to255(to_np(current_colors))
                    if current_colors is not None
                    else None,
                    color=color,
                    render=False,
                )
            except Exception as e:
                remove_points(actor_name)
                if self.logging:
                    logger.info(str(traceback.format_exc()))

        def render_points():
            means = self.params.get("means")
            if means is None:
                return

            if isinstance(means, dict):
                # Multiple groups
                colors = self.params.get("colors", {})
                if not isinstance(colors, dict):
                    # Fallback if colors is single array but means is dict?
                    # Probably safer to assume if means is dict, colors should be dict or None
                    colors = {}

                # Create widgets and render if new group
                # We iterate to find index for color generation
                # Note: dict keys order is preserved in Python 3.7+
                for i, name in enumerate(means.keys()):
                    # Determine color for this group
                    grp_colors_input = colors.get(name, None)

                    widget_color = "green"  # Default
                    points_color_override = (
                        None  # If not None, forces uniform color on points
                    )

                    if grp_colors_input is None:
                        # Case: No color data provided. Auto-generate.
                        auto_color = COLORS[i % len(COLORS)]
                        widget_color = auto_color
                        points_color_override = auto_color
                    elif isinstance(grp_colors_input, str):
                        # Case: Single color string
                        widget_color = grp_colors_input
                        points_color_override = grp_colors_input
                    else:
                        # Case: Data array provided.
                        # Assign a consistent ID color for widget to distinguish it
                        widget_color = COLORS[i % len(COLORS)]
                        points_color_override = None

                    if name not in self._point_widgets:

                        def _create_callback(n, c):
                            def _callback(state):
                                self.gui_state["ui"][f"show_points_{n}"] = state
                                if f"splats_{n}" in self.actors:
                                    self.actors[f"splats_{n}"].SetVisibility(state)
                                if f"arrows_{n}" in self.actors:
                                    self.actors[f"arrows_{n}"].SetVisibility(state)
                                if f"label_{n}" in self._point_labels:
                                    # Dim label instead of hiding
                                    try:
                                        actor = self._point_labels[f"label_{n}"]
                                        target_c = c if state else "grey"
                                        actor.GetTextProperty().SetColor(
                                            mcolors.to_rgb(target_c)
                                        )
                                    except Exception:
                                        pass

                            return _callback

                        self._point_widgets[name] = (
                            self.plotter.add_checkbox_button_widget(
                                _create_callback(name, widget_color),
                                value=self.gui_state["ui"].get(
                                    f"show_points_{name}", True
                                ),
                                position=(10.0, self._widget_y_pos),
                                size=20,
                                border_size=1,
                                color_on=widget_color,
                                color_off="grey",
                                background_color="white",
                            )
                        )

                        self._point_labels[f"label_{name}"] = self.plotter.add_text(
                            name,
                            position=(35.0, self._widget_y_pos),
                            font_size=10,
                            color=widget_color,
                            shadow=True,
                            name=f"label_{name}",
                        )

                        self._widget_y_pos += 25

                    # Check visibility
                    should_show = self.gui_state["ui"].get(f"show_points_{name}", True)

                    if should_show:
                        render_points_group(
                            name,
                            means[name],
                            grp_colors_input,
                            self.params.get("opacities", 1.0),
                            color=points_color_override,
                        )
                    else:
                        remove_points(f"splats_{name}")
                        remove_points(f"arrows_{name}")

            else:
                # Original single group behavior
                if self.gui_state["ui"].get("show_points", True):
                    render_points_group(
                        "",
                        means,
                        self.params.get("colors"),
                        self.params.get("opacities", 1.0),
                    )
                else:
                    remove_points("splats_")
                    remove_points("arrows_")

        def render_graph():
            if (
                "graph" in self.params
                and "joints" in self.params["graph"]
                and self.params["graph"]["joints"] is not None
                and "links" in self.params["graph"]
                and self.params["graph"]["links"] is not None
            ):
                try:
                    if pyrender_fix:
                        for a in self.actors.get("graph", []):
                            a.SetVisibility(False)

                    self.actors["graph"] = vis_link(
                        self.plotter,
                        to_np(self.params["graph"]["joints"][fid]),
                        self.params["graph"]["links"],
                        prefix="splats-",
                        radius=node_radius,
                        render=False,
                        joints_color=self.params["graph"].get("joints_color", None),
                        links_color=self.params["graph"].get("links_color", None),
                    )

                    if pyrender_fix:
                        for a in self.actors.get("graph", []):
                            a.SetVisibility(True)
                except Exception as e:
                    remove_graph()
                    if self.logging:
                        logger.info(str(traceback.format_exc()))

        def render_meshes():
            if len(self.params.get("meshes", [])) > 0:
                try:
                    actors = []
                    for i, m_dict in enumerate(self.params["meshes"]):
                        if isinstance(m_dict, list):
                            m_dict = m_dict[int(fid)]
                        actors.append(
                            self.plotter.add_mesh(
                                m_dict["mesh"],
                                **m_dict.get("kwargs", {}),
                                render=False,
                                name=m_dict.get("name", str(i)),
                            )
                        )
                    self.actors["meshes"] = actors
                except Exception as e:
                    remove_meshes()
                    if self.logging:
                        logger.info(str(traceback.format_exc()))

        def remove_meshes():
            for a in self.actors.pop("meshes", []):
                self.plotter.remove_actor(a)

        if reset_before_render:
            # Clear all
            # remove_points() # This needs to be smarter now
            self.plotter.clear_actors()
            self.actors = {}
            # remove_graph()
            # remove_meshes()

            # Reset widgets state
            self._point_widgets = {}
            self._point_labels = {}
            self._widget_y_pos = 10.0

        # Render based on flags
        render_points()

        if self.gui_state["ui"].get("show_graph", True):
            render_graph()
        else:
            remove_graph()

        if self.gui_state["ui"].get("show_meshes", True):
            render_meshes()
        else:
            remove_meshes()

        if render:
            if offline:
                self.plotter.write_frame()
            else:
                self.plotter.render()

    def __init__(
        self,
        node_radius=0.1,
        num_frames=500,
        backend="/tmp/pvlib.state",
        notebook=True,
        off_screen=False,
        frame_selection=True,
        logging=True,
        graph_enabled=False,
    ):
        self.plotter = None
        self._viewer = None
        self._point_widgets = {}
        self._point_labels = {}
        self._widget_y_pos = 10.0
        self.frame_selection = frame_selection
        self.params = {}
        self.actors = {}
        self.backend = backend
        self.gui_state = {
            "pv": {},
            "ui": {
                "node_radius": node_radius,
                "cursor_step_size": 0.1,
                "frame_id": 0,
                # "view": "all", # Replaced by individual flags
                "show_points": True,  # Default for single group
                "show_graph": True,
                "show_meshes": True,
                "play_step": 10,
                "play_sleep": 0.5,
            },
            "app": {
                "num_frames": num_frames,
            },
        }
        self.notebook = notebook
        self.off_screen = off_screen
        self.logging = logging
        self.graph_enabled = graph_enabled

    def dump(self, path="/tmp/pvlib.state"):
        state = {"params": self.params, "gui_state": {"app": self.gui_state["app"]}}
        with open(path, "wb") as f:
            f.write(pickle.dumps(state))

    def load(self, path="/tmp/pvlib.state"):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.params = state["params"]
        self.gui_state["app"] = state["gui_state"]["app"]

    def reset(self):
        if self.plotter is not None:
            self.plotter.close()
        self._viewer = None
        self._point_widgets = {}  # Reset widgets
        self._point_labels = {}
        self._widget_y_pos = 10.0
        self.plotter = pv.Plotter(notebook=self.notebook, off_screen=self.off_screen)

    def clear(self):
        self.plotter.clear_actors()

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v

    def update_param(self, key, value):
        """
        Update a nested visualization parameter used by the plotter.

        The ``key`` supports dotted paths to address structured fields that drive
        rendering of different primitives:

        - **point cloud** (`means`, `opacities`, `colors`):
          - ``means``: array-like of shape ``[num_frames, N, 3]`` giving 3D
            coordinates of points per frame (accessed as ``means[fid]``).
          - ``opacities``: scalar, 1D, or per-point array controlling point
            opacity.
          - ``colors``: RGB values in ``[0, 1]`` or ``[0, 255]`` used as scalars.
        - **graph / skeleton** (`graph.joints`, `graph.links`, `graph.joints_color`,
          `graph.links_color`):
          - ``graph.joints``: array-like of shape ``[num_frames, J, 3]`` with
            joint positions (indexed as ``graph['joints'][fid]``).
          - ``graph.links``: list of index pairs defining edges between joints.
          - ``graph.joints_color`` / ``graph.links_color``: optional per-joint /
            per-edge colors.
        - **meshes** (`meshes`):
          - ``meshes``: list of dictionaries (or lists over frames) where each
            dict contains at least:
              - ``mesh``: a `pyvista` mesh object.
              - ``kwargs``: keyword arguments forwarded to
                ``plotter.add_mesh`` (e.g. color, opacity, style, etc.).

        Examples
        --------
        >>> plotter.update_param("means", means_array)
        >>> plotter.update_param("graph.joints", joints_array)
        >>> plotter.update_param("meshes", mesh_list)
        """
        params = self.params
        keys = key.split(".")
        params = self.params
        keys = key.split(".")
        if keys[0] == "means":
            if isinstance(value, dict):
                # If value is a dict, we trust the user structure
                pass
            elif len(value.shape) == 2:
                value = value[None, ...]

        if keys[0] == "meshes":
            if not isinstance(value, list):
                value = [value]
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                params[k] = value
            else:
                if k not in params:
                    params[k] = {}
                params = params[k]

    @property
    def url(self):
        if (not self.backend.startswith("remote:")) and self._viewer is None:
            self.reset()
            if self.off_screen:
                return None

            def _menu():
                if self.frame_selection:
                    button(
                        click=button_play,
                        icon="mdi-play",
                        tooltip="Play",
                    )
                    text_field(
                        model=("frames", 0),
                        tooltip="Frames",
                        readonly=False,
                        type="number",
                        dense=True,
                        hide_details=True,
                        style="min-width: 40px; width: 100px",
                        classes="my-0 py-0 ml-1 mr-1",
                    )
                    slider(
                        model=("frames", 0),
                        tooltip="Frames",
                        min=0,
                        max=self.gui_state["app"]["num_frames"] - 1,
                        step=1,
                        dense=True,
                        hide_details=True,
                        style="width: 300px",
                        classes="my-0 py-0 ml-1 mr-1",
                    )
                if self.graph_enabled:
                    slider(
                        model=("node_size", 0.1),
                        tooltip="Node size",
                        min=0,
                        max=self.gui_state["ui"]["node_radius"],
                        step=0.002,
                        dense=True,
                        hide_details=True,
                        style="width: 100px",
                        classes="my-0 py-0 ml-1 mr-1",
                    )
                    # Individual toggles
                    checkbox(
                        model=("show_graph", True),
                        icons=("mdi-eye", "mdi-eye-off"),
                        tooltip="Toggle Graph",
                        label="Graph",
                        dense=True,
                        style="width: 80px",
                    )
                checkbox(
                    model=("show_meshes", True),
                    icons=("mdi-eye", "mdi-eye-off"),
                    tooltip="Toggle Mesh",
                    label="Mesh",
                    dense=True,
                    style="width: 80px",
                )

                # Check for point groups
                # Widgets handled in render() now

                # means = self.params.get("means")
                # if isinstance(means, dict):
                #     for name in means.keys():
                #         checkbox(
                #             model=(f"show_points_{name}", True),
                #             icons=("mdi-eye", "mdi-eye-off"),
                #             tooltip=f"Toggle {name}",
                #             label=name,
                #             dense=True,
                #             style="width: 80px",
                #         )
                if not isinstance(self.params.get("means"), dict):
                    # Single group fallback
                    checkbox(
                        model=("show_points", True),
                        icons=("mdi-eye", "mdi-eye-off"),
                        tooltip="Toggle Points",
                        label="Points",
                        dense=True,
                        style="width: 80px",
                    )

            def button_play():
                state.play = not state.play
                state.flush()

            self._viewer = self.plotter.show(
                return_viewer=True, jupyter_kwargs=dict(add_menu_items=_menu)
            )
            state, ctrl = (
                self._viewer.viewer.server.state,
                self._viewer.viewer.server.controller,
            )
            ctrl.view_update = self._viewer.viewer.update
            self.gui_state["pv"] = {"state": state, "ctrl": ctrl}

            if self.frame_selection:

                @state.change("play")
                async def _play(play, **kwargs):
                    while state.play:
                        if state.frames + self.gui_state["ui"]["play_step"] >= len(
                            self.params["means"]
                        ):
                            state.frames = len(self.params["means"]) - 1
                            state.play = False
                        else:
                            state.frames += self.gui_state["ui"]["play_step"]
                            state.flush()
                        await asyncio.sleep(self.gui_state["ui"]["play_sleep"])

                @state.change("frames")
                def update_frame_id(frames, **kwargs):
                    frame_id = frames
                    self.gui_state["ui"]["frame_id"] = int(frame_id)
                    self.render()
                    ctrl.view_update()

            @state.change("show_graph", "show_meshes", "show_points")
            def set_visibility(**kwargs):
                # Generic handler for simple flags
                for k, v in kwargs.items():
                    self.gui_state["ui"][k] = v
                self.render()
                ctrl.view_update()

            # Wrapper for dynamic toggles if needed.
            # But the PyVista widgets operate outside of Trame state (mostly)
            # So we rely on them modifying actors directly for now.

            @state.change("node_size")
            def set_node_size(node_size, **kwargs):
                self.gui_state["ui"]["node_radius"] = node_size
                self.render()
                ctrl.view_update()

        if not self.backend.startswith("remote:"):
            url = self._viewer.value.split('src="')[1].split('"')[0]
            if self.logging:
                logger.info(f"pyvista viewer url: {url}")
            return url
        else:
            return self.backend

    def __del__(self):
        try:
            self.plotter.close()
        except:
            pass


def render_pcds(means, colors=None, backend="/tmp/pvlib.state", logging=True):
    """
    Render point clouds using the Plotter class with a remote backend.

    Parameters
    ----------
    means : np.ndarray or torch.Tensor or dict
        Point cloud coordinates. Can be:
        - shape (N, 3) for a single static point cloud.
        - shape (T, N, 3) for a sequence of T point clouds.
        - dict of group_name -> array for multiple point groups.
    colors : np.ndarray or torch.Tensor or dict or None
        Point colors. Can be:
        - shape (N, 3) or (T, N, 3) matching `means`.
        - dict of group_name -> array/string matching keys in `means` dict.
    backend : str, optional
        Path to the shared memory state file for the remote backend.
        Defaults to "/tmp/pvlib.state".
    """
    plotter = Plotter(backend="remote:" + backend, logging=logging)
    plotter.update_param("means", means)  # points: (N, 3) or (T, N, 3)
    if colors is not None:
        plotter.update_param("colors", colors)  #
    plotter.render()


if __name__ == "__main__":
    import sys, time

    p = Plotter(backend="")
    logger.info(p.url)
    while True:
        txt = input("provide a pvstate full path>").strip()
        if txt == "":
            continue
        if txt == "q":
            break
        else:
            if osp.exists(txt):
                p.backend = f"from:{txt}"
                p.render()
            else:
                logger.info(f"file {txt} does not exist")
                continue
