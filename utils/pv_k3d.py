import k3d
import numpy as np
import ipywidgets as widgets
import ipywidgets.embed as embed
from loguru import logger
import matplotlib.colors as mcolors

try:
    import torch
    from torch import Tensor
except ImportError:
    Tensor = type(None)  # Fallback

DEFAULT_COLORS = (
    "darkred",
    "darkblue",
    "goldenrod",
    "darkmagenta",
    "darkgreen",
    "indigo",
    "sienna",
    "darkcyan",
    "mediumvioletred",
    "olivedrab",
)


def to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if "torch" in str(type(x)):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return np.array(x)
    return x


def to_rgb_int(colors):
    if colors is None:
        return None
    if isinstance(colors, (int, np.integer)):
        return int(colors)
    if isinstance(colors, str):
        colors = np.array(mcolors.to_rgb(colors))
    colors = to_np(colors)
    if colors is None:
        return None
    if colors.dtype != np.uint8:
        if colors.max() <= 1.05:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    if len(colors.shape) == 1 and colors.shape[0] == 3:
        return int(
            (colors[0].astype(np.uint32) << 16)
            | (colors[1].astype(np.uint32) << 8)
            | colors[2].astype(np.uint32)
        )
    if len(colors.shape) == 2 and colors.shape[1] == 3:
        return (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | (colors[:, 2].astype(np.uint32))
        )
    return None


class Plotter:
    def __init__(self, node_radius=0.01, **kwargs):
        self.plot = k3d.plot()
        self.params = kwargs
        self.objects = {}
        self.node_radius = node_radius
        self.num_frames = 1
        self._group_colors = {}
        self._group_has_custom_color = {}
        self._vbox = None
        self._current_frame = 0

    def update_param(self, key, value):
        keys = key.split(".")
        params = self.params
        if keys[0] == "means":
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, tuple) and len(v) == 2:
                        verts = to_np(v[0])
                        if hasattr(verts, "shape") and len(verts.shape) == 3:
                            self.num_frames = max(self.num_frames, verts.shape[0])
                    else:
                        v_np = to_np(v)
                        if hasattr(v_np, "shape") and len(v_np.shape) == 3:
                            self.num_frames = max(self.num_frames, v_np.shape[0])
            elif isinstance(value, tuple) and len(value) == 2:
                verts = to_np(value[0])
                if hasattr(verts, "shape") and len(verts.shape) == 3:
                    self.num_frames = max(self.num_frames, verts.shape[0])
            else:
                v_np = to_np(value)
                if v_np is not None and hasattr(v_np, "shape") and len(v_np.shape) == 3:
                    self.num_frames = max(self.num_frames, v_np.shape[0])
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                params[k] = value
            else:
                if k not in params:
                    params[k] = {}
                params = params[k]

    def clear(self):
        self.plot.objects = []
        self.objects, self.params, self._group_colors = {}, {}, {}
        self._group_has_custom_color = {}

    def _prepare_timeseries(self, data):
        if isinstance(data, list):
            return {str(i): to_np(v) for i, v in enumerate(data)}
        data_np = to_np(data)
        if hasattr(data_np, "shape") and len(data_np.shape) == 3:
            return {str(i): data_np[i] for i in range(data_np.shape[0])}
        return data_np

    def _render_points_group(
        self, name, means_data, colors_data=None, opacity=1.0, color=None
    ):
        ts_means = self._prepare_timeseries(means_data)
        ts_colors = (
            self._prepare_timeseries(colors_data) if colors_data is not None else None
        )
        if ts_colors is not None:
            if isinstance(ts_colors, dict):
                ts_colors = {k: to_rgb_int(v) for k, v in ts_colors.items()}
            else:
                ts_colors = to_rgb_int(ts_colors)
        elif color is not None:
            ts_colors = to_rgb_int(color)

        seed_means = ts_means["0"] if isinstance(ts_means, dict) else ts_means
        seed_colors = ts_colors["0"] if isinstance(ts_colors, dict) else ts_colors

        if seed_means is not None and seed_means.shape[-1] == 6:
            # Flow: use Python callback for temporal because vectors don't support TimeSeries
            subsample = self.params.get("flow_subsample")
            if subsample is not None:
                total_n = seed_means.shape[0]
                if isinstance(subsample, float) and subsample < 1.0:
                    subsample = int(total_n * subsample)
                if subsample < total_n:
                    indices = np.random.choice(total_n, subsample, replace=False)
                    self.objects[f"flow_indices_{name}"] = indices
                else:
                    indices = np.arange(total_n)
            else:
                indices = np.arange(seed_means.shape[0])

            # Apply subsampling
            sub_means = seed_means[indices]
            sub_colors = seed_colors
            if (
                isinstance(seed_colors, np.ndarray)
                and seed_colors.shape[0] == seed_means.shape[0]
            ):
                sub_colors = seed_colors[indices]

            flow_opacity = self.params.get("flow_opacity", opacity)
            point_shader = self.params.get("point_shader", "3d")

            obj_p_start = k3d.points(
                seed_means[:, :3],
                point_size=self.node_radius,
                opacity=opacity,
                shader=point_shader,
            )
            if isinstance(seed_colors, np.ndarray):
                obj_p_start.colors = seed_colors
            else:
                obj_p_start.color = (
                    int(seed_colors) if seed_colors is not None else 0xFF0000
                )
            self.objects[f"splats_start_{name}"] = obj_p_start
            self.plot += obj_p_start

            obj_p_end = k3d.points(
                sub_means[:, 3:],
                point_size=self.node_radius,
                opacity=opacity,
                shader=point_shader,
            )
            if isinstance(sub_colors, np.ndarray):
                obj_p_end.colors = sub_colors
            else:
                obj_p_end.color = (
                    int(sub_colors) if sub_colors is not None else 0x00FF00
                )
            self.objects[f"splats_end_{name}"] = obj_p_end
            self.plot += obj_p_end

            vectors = (sub_means[:, 3:] - sub_means[:, :3]).astype(np.float32)
            magnitudes = (
                np.linalg.norm(vectors, axis=1) if len(vectors) > 0 else np.array([])
            )

            obj_v = k3d.vectors(
                sub_means[:, :3],
                vectors,
                line_width=self.params.get("flow_line_width", self.node_radius * 0.5),
                use_head=True,
                opacity=flow_opacity,
                attribute=magnitudes,
                color_map=k3d.basic_color_maps.Jet,
                head_size=self.params.get("flow_head_size", 0.1),
            )
            self.objects[f"vectors_{name}"] = obj_v
            self.plot += obj_v

            # Save raw data for Python refresh
            self.objects[f"raw_means_{name}"] = ts_means
            self.objects[f"raw_colors_{name}"] = ts_colors
        else:
            # PCD: Use TimeSeries for HTML compatibility
            kwargs = {"point_size": self.node_radius, "opacity": opacity}
            if isinstance(seed_colors, np.ndarray):
                kwargs["colors"] = seed_colors
            else:
                kwargs["color"] = (
                    int(seed_colors) if seed_colors is not None else 0xFF0000
                )

            obj = k3d.points(
                seed_means, **kwargs, shader=self.params.get("point_shader", "3d")
            )
            if isinstance(ts_means, dict):
                obj.positions = ts_means
                if isinstance(ts_colors, dict):
                    obj.colors = ts_colors
            self.objects[f"splats_{name}"] = obj
            self.plot += obj

    def _render_mesh_group(self, name, mesh_data, color=None, opacity=1.0):
        verts_data, faces_data = mesh_data
        ts_verts = self._prepare_timeseries(verts_data)
        ts_faces = self._prepare_timeseries(faces_data)
        seed_verts = ts_verts["0"] if isinstance(ts_verts, dict) else ts_verts
        seed_faces = ts_faces["0"] if isinstance(ts_faces, dict) else ts_faces

        c_int = to_rgb_int(color) if color is not None else 0xCCCCCC
        obj = k3d.mesh(seed_verts, seed_faces, color=c_int, opacity=opacity)
        self.objects[f"mesh_{name}"] = obj
        self.plot += obj

        # Save raw for Python refresh
        self.objects[f"raw_mesh_verts_{name}"] = ts_verts
        self.objects[f"raw_mesh_faces_{name}"] = ts_faces

    def _refresh_python_objects(self, frame_idx):
        """Update objects that don't support native K3D TimeSeries."""
        fid = str(frame_idx)
        for name, obj in self.objects.items():
            # Flow points (start/end)
            if (
                name.startswith("splats_start_")
                and f"raw_means_{name[13:]}" in self.objects
            ):
                g_name = name[13:]
                ts = self.objects[f"raw_means_{g_name}"]
                indices = slice(None)  # Use all points for start splats
                if isinstance(ts, dict):
                    data = ts.get(fid, ts["0"])
                    obj.positions = data[indices, :3]
                tsc = self.objects.get(f"raw_colors_{g_name}")
                if isinstance(tsc, dict):
                    cdata = tsc.get(fid, tsc["0"])
                    if isinstance(cdata, np.ndarray) and len(cdata) > 0:
                        obj.colors = cdata[indices]

            if (
                name.startswith("splats_end_")
                and f"raw_means_{name[11:]}" in self.objects
            ):
                g_name = name[11:]
                ts = self.objects[f"raw_means_{g_name}"]
                indices = self.objects.get(f"flow_indices_{g_name}", slice(None))
                if isinstance(ts, dict):
                    data = ts.get(fid, ts["0"])
                    obj.positions = data[indices, 3:]
                tsc = self.objects.get(f"raw_colors_{g_name}")
                if isinstance(tsc, dict):
                    cdata = tsc.get(fid, tsc["0"])
                    if isinstance(cdata, np.ndarray) and len(cdata) > 0:
                        obj.colors = cdata[indices]

            # Flow vectors
            if name.startswith("vectors_") and f"raw_means_{name[8:]}" in self.objects:
                g_name = name[8:]
                ts = self.objects[f"raw_means_{g_name}"]
                indices = self.objects.get(f"flow_indices_{g_name}", slice(None))
                if isinstance(ts, dict):
                    data = ts.get(fid, ts["0"])[indices]
                    obj.origins = data[:, :3]
                    vecs = (data[:, 3:] - data[:, :3]).astype(np.float32)
                    obj.vectors = vecs
                    obj.attribute = (
                        np.linalg.norm(vecs, axis=1) if len(vecs) > 0 else np.array([])
                    )

            # Mesh
            if (
                name.startswith("mesh_")
                and f"raw_mesh_verts_{name[5:]}" in self.objects
            ):
                ts_v = self.objects[f"raw_mesh_verts_{name[5:]}"]
                ts_f = self.objects[f"raw_mesh_faces_{name[5:]}"]
                if isinstance(ts_v, dict):
                    obj.vertices = ts_v.get(fid, ts_v["0"])
                if isinstance(ts_f, dict):
                    obj.indices = ts_f.get(fid, ts_f["0"])

    def render(self):
        self.plot.objects = []
        self.objects = {}
        means = self.params.get("means")
        if means is not None:
            if isinstance(means, dict):
                colors = self.params.get("colors", {})
                for i, (name, data) in enumerate(means.items()):
                    user_color = colors.get(name)
                    if user_color is not None:
                        grp_color = user_color
                        self._group_has_custom_color[name] = True
                    else:
                        grp_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                        self._group_has_custom_color[name] = False

                    self._group_colors[name] = grp_color
                    if isinstance(data, tuple) and len(data) == 2:
                        self._render_mesh_group(name, data, grp_color)
                    else:
                        self._render_points_group(name, data, grp_color)
            else:
                self._render_points_group("", means, self.params.get("colors"))
        return self.show()

    def show(self):
        controls = []
        if self.num_frames > 1:
            slider = widgets.IntSlider(
                value=0, min=0, max=self.num_frames - 1, description="Frame"
            )
            # JS Link for native TimeSeries (PCDs)
            widgets.jslink((slider, "value"), (self.plot, "time"))

            # Python callback for others (Mesh, Vectors)
            def on_frame_change(change):
                self._refresh_python_objects(change["new"])

            slider.observe(on_frame_change, names="value")
            controls.append(slider)

        means = self.params.get("means")
        if isinstance(means, dict):
            for name in means.keys():
                target_objs = [
                    v
                    for k, v in self.objects.items()
                    if (
                        k.startswith("splats_")
                        or k.startswith("mesh_")
                        or k.startswith("vectors_")
                        or k.startswith("splats_start_")
                        or k.startswith("splats_end_")
                    )
                    and name in k
                ]
                if not target_objs:
                    continue

                cb = widgets.Checkbox(
                    value=True, description="", layout=widgets.Layout(width="auto")
                )
                for obj in target_objs:
                    widgets.jslink((cb, "value"), (obj, "visible"))

                if not self._group_has_custom_color.get(name, False):
                    color = self._group_colors.get(name, "white")
                    try:
                        color_css = (
                            mcolors.to_hex(color)
                            if isinstance(color, str)
                            else mcolors.to_hex(mcolors.to_rgba(color)[:3])
                        )
                        label_html = f"<span style='color: {color_css}; font-weight: bold;'>{name}</span>"
                    except:
                        label_html = f"<b>{name}</b>"
                else:
                    label_html = f"<b>{name}</b>"

                label = widgets.HTML(value=label_html)
                controls.append(widgets.HBox([cb, label]))

        # Group layer controls into a scrollable container if there are many
        if isinstance(means, dict) and len(controls) > (
            1 if self.num_frames > 1 else 0
        ):
            # Extract only the HBox controls (skip slider if present)
            layer_controls = controls[1:] if self.num_frames > 1 else controls

            layer_box = widgets.Box(
                layer_controls,
                layout=widgets.Layout(
                    flex_flow="row wrap",
                    max_height="200px",
                    overflow="auto",
                    border="1px solid #444",
                    padding="5px",
                    margin="5px 0",
                    width="100%",
                ),
            )
            # Replace layer controls with the compact box
            if self.num_frames > 1:
                controls = [controls[0], layer_box]
            else:
                controls = [layer_box]

        self._vbox = widgets.VBox([self.plot] + controls)
        return self._vbox

    def save_html(self, path):
        if self._vbox is None:
            self.render()
        embed.embed_minimal_html(path, views=[self._vbox])
        logger.info(
            f"Interactive plot saved to {path}. Note: Only PCDs support temporal scrubbing in static HTML."
        )


def render_pcds(means, colors=None, save_to=None, **kwargs):
    plotter = Plotter(**kwargs)
    plotter.update_param("means", means)
    if colors is not None:
        plotter.update_param("colors", colors)
    res = plotter.render()
    if save_to:
        plotter.save_html(save_to)
    return res
