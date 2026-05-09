import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils
from bokeh.plotting import figure, output_file, save, show
from bokeh.io import output_notebook
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Div,
    RangeSlider,
    CustomJS,
    BoxAnnotation,
)
from bokeh.layouts import column, gridplot


def sdf_to_colors(sdf_val, min_val=None, max_val=None, cmap_name="viridis"):
    """
    Converts SDF scalar values to an (N, 3) color array.

    Args:
        sdf_val (torch.Tensor): (N,) tensor of signed distance values.
        min_val (float): Minimum value for normalization. If None, uses min of sdf_val.
        max_val (float): Maximum value for normalization. If None, uses max of sdf_val.
        cmap_name (str): Name of the matplotlib colormap to use.

    Returns:
        np.ndarray: (N, 3) array of RGB colors in range [0, 1].
    """
    if torch.is_tensor(sdf_val):
        sdf_val = sdf_val.detach().cpu().numpy()

    if min_val is None:
        min_val = sdf_val.min()
    if max_val is None:
        max_val = sdf_val.max()

    # Normalize values to [0, 1]
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap(cmap_name)

    # Map to RGBA, then take only RGB
    colors = cmap(norm(sdf_val))[:, :3]
    return colors


def sdf_grad_to_colors(sdf_grad):
    """
    Converts SDF gradient vectors to an (N, 3) color array.
    Gradients are expected to be unit vectors in range [-1, 1].
    This function maps them to [0, 1] for visualization.

    Args:
        sdf_grad (torch.Tensor): (N, 3) tensor of gradient vectors.

    Returns:
        np.ndarray: (N, 3) array of RGB colors.
    """
    if torch.is_tensor(sdf_grad):
        sdf_grad = sdf_grad.detach().cpu().numpy()

    # Normalize from [-1, 1] to [0, 1]
    # (x + 1) / 2
    colors = (sdf_grad + 1.0) / 2.0

    # Clip just in case
    colors = np.clip(colors, 0.0, 1.0)
    return colors


def _to_torch_tensor(data):
    # 1. Unified conversion to Torch Tensor
    if isinstance(data, Image.Image):
        data = np.array(data)

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        tensor = data.detach().cpu()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # 2. Shape Normalization to (T, C, H, W)
    # Handle (H, W) -> (1, 1, H, W)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    elif tensor.ndim == 3:
        # Differentiate between (H, W, C) and (C, H, W) or (T, H, W)
        if tensor.shape[-1] in [1, 3]:  # Likely (H, W, C)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.shape[0] in [1, 3]:  # Likely (C, H, W)
            tensor = tensor.unsqueeze(0)
        else:  # Likely (T, H, W)
            tensor = tensor.unsqueeze(1)

    elif tensor.ndim == 4:
        # Handle (T, H, W, C) -> (T, C, H, W)
        if tensor.shape[-1] in [1, 3]:
            tensor = tensor.permute(0, 3, 1, 2)
        # If (T, C, H, W), it's already in the desired format

    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    return tensor


def to_pil(*data, captions=None, max_t=16, padding=2, normalize=True, font_size=20, nrow=None):
    """
    An omni-powerful utility to convert various data formats into a PIL Image.

    Args:
        *data: Variable number of input image data OR a single list/tuple of data.
            Supports shapes: (H, W), (1, H, W), (3, H, W), (H, W, 3),
            (T, H, W), (T, 3, H, W), (T, H, W, 3).
            If multiple inputs are provided, they are interleaved.
        captions (str | list[str], optional): Text to overlay on images.
            If list, length must match T (total frames after interleaving).
        max_t (int): Maximum number of frames to include in the grid for
            temporal data. Defaults to 16.
        padding (int): Pixel padding between grid items. Defaults to 2.
        normalize (bool): If True, rescales data to [0, 1] based on min/max.
            If False, assumes [0, 1] or [0, 255] logic.
        font_size (int): Size of the caption text.

    Returns:
        PIL.Image.Image: The processed image (or grid of images).
    """

    if len(data) == 0:
        raise ValueError("At least one data input is required.")

    # Handle single list/tuple input
    if len(data) == 1 and isinstance(data[0], (list, tuple)):
        data = data[0]

    # Process all inputs to (T, C, H, W)
    tensors = [_to_torch_tensor(d) for d in data]

    # Interleave logic
    if len(tensors) > 1:
        # Align lengths to absolute minimum to avoid shape mismatch
        min_T = min(t.shape[0] for t in tensors)
        tensors = [t[:min_T] for t in tensors]

        # Align channels
        max_C = max(t.shape[1] for t in tensors)
        if max_C == 3:
            tensors = [t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t for t in tensors]

        # Interleave: [T, C, H, W] -> stack dim 1 -> [T, N, C, H, W] -> flatten -> [T*N, C, H, W]
        tensor = torch.stack(tensors, dim=1).flatten(0, 1)
    else:
        tensor = tensors[0]

    # 3. Temporal Clipping
    T = tensor.shape[0]
    if T > max_t:
        tensor = tensor[:max_t]
        T = max_t
        if isinstance(captions, list):
            captions = captions[:max_t]

    # 4. Range Normalization (Auto-rescale to [0, 1])
    tensor = tensor.float()
    v_min, v_max = tensor.min(), tensor.max()

    if normalize:
        if v_max - v_min > 1e-5:
            tensor = (tensor - v_min) / (v_max - v_min)
        else:
            tensor = torch.zeros_like(tensor)
    else:
        # Auto-detect [0, 255] range if normalization is off
        if v_max > 1.01:
            tensor /= 255.0

    tensor = torch.clamp(tensor, 0, 1)

    # 5. Captioning Logic (Applied per-frame)
    if captions is not None:
        if isinstance(captions, str):
            captions = [captions] * T

        # Try to load a TrueType font for adjustable size
        try:
            # Common paths for Linux/macOS
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except Exception:
            try:
                # Common path for Windows
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                # Fallback to default (size won't change)
                font = ImageFont.load_default()

        annotated_frames = []
        for i in range(T):
            # Convert single frame to uint8 PIL for drawing
            frame_np = (tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if frame_np.shape[-1] == 1:
                frame_pil = Image.fromarray(frame_np.squeeze(-1), mode="L").convert(
                    "RGB"
                )
            else:
                frame_pil = Image.fromarray(frame_np, mode="RGB")

            draw = ImageDraw.Draw(frame_pil)
            # Basic text drawing (top-left)
            draw.text((5, 5), str(captions[i]), fill=(255, 255, 0), font=font)

            # Convert back to float tensor (C, H, W)
            annotated_frames.append(
                torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1).float() / 255.0
            )

        tensor = torch.stack(annotated_frames)

    # 6. Grid Generation
    if T > 1:
        if nrow is None:
            nrow = int(np.ceil(np.sqrt(T)))
        grid = vutils.make_grid(tensor, nrow=nrow, padding=padding)
    else:
        if tensor.ndim == 4:  # (1, C, H, W)
            grid = tensor[0]
        else:
            grid = tensor

    # 7. Final PIL Construction
    ndarr = (
        grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    )

    # Handle single-channel grayscale output
    if ndarr.shape[-1] == 1:
        return Image.fromarray(ndarr.squeeze(-1))
    return Image.fromarray(ndarr)


def annotate_images(
    images: torch.Tensor,
    texts: List[str],
    locations: Optional[List[Tuple[int, int]]] = None,
    colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (255, 255, 0),
    font_size: int = 20,
) -> torch.Tensor:
    """
    Annotate a batch of images with text.

    Args:
        images (torch.Tensor): (B, C, H, W) or (C, H, W) tensor in range [0, 1].
        texts (List[str]): List of B strings to draw.
        locations (List[Tuple[int, int]], optional): List of (x, y) coordinates.
            Defaults to (5, 5) for all.
        colors (tuple | list[tuple]): Color for the text.
        font_size (int): Size of the text.

    Returns:
        torch.Tensor: (B, C, H, W) or (C, H, W) annotated tensor.
    """
    is_batch = images.ndim == 4
    if not is_batch:
        images = images.unsqueeze(0)
    B, C, H, W = images.shape

    if locations is None:
        locations = [(5, 5)] * B
    if isinstance(colors, tuple):
        colors = [colors] * B

    # Try to load a TrueType font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    annotated_frames = []
    for i in range(B):
        # Convert to uint8 PIL
        frame_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if frame_np.shape[-1] == 1:
            frame_pil = Image.fromarray(frame_np.squeeze(-1), mode="L").convert("RGB")
        else:
            frame_pil = Image.fromarray(frame_np, mode="RGB")

        draw = ImageDraw.Draw(frame_pil)
        draw.text(locations[i], texts[i], fill=colors[i], font=font)

        # Convert back to (C, H, W)
        frame_tensor = (
            torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1).float() / 255.0
        )
        if C == 1:
            # Convert back to grayscale if input was grayscale
            frame_tensor = frame_tensor.mean(dim=0, keepdim=True)
        annotated_frames.append(frame_tensor)

    out = torch.stack(annotated_frames)
    if not is_batch:
        out = out.squeeze(0)
    return out.to(images.device)


def vis_distribution(
    tensor,
    dims=(),
    save_path=None,
    title="Distribution",
    show_plot=False,
    bins=50,
    outlier_quantile=0.0,
    density=False,
    subsample=100000,
    x_range=None,
    slider_range=None,
):
    """
    Visualizes the distribution of a tensor using Bokeh.

    Args:
        tensor (torch.Tensor): Input tensor of any shape.
        dims (tuple): Dimensions to aggregate over for hierarchical visualization.
            Defaults to (), which treats all elements as a single distribution.
        save_path (str): Path to save the interactive HTML file.
        title (str): Title for the plots.
        show_plot (bool): If True, will attempt to show the plot immediately.
            In a Jupyter notebook, this will render the plot in the cell output.
        bins (int): Number of bins for the histograms. Defaults to 50.
        outlier_quantile (float): If > 0, clips the histogram range to [q, 1-q]
            where q = outlier_quantile / 2. For example, 0.05 will clip to [0.025, 0.975].
        density (bool): If True, shows density instead of frequency.
        subsample (int): Max number of points to pass to JS for interactive selection.
        x_range (tuple): Manual override for the plot x-axis range (min, max).
        slider_range (tuple): Manual override for the slider range (min, max).
    """
    if torch.is_tensor(tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)

    # Flatten the data for global histogram
    flat_data = data.flatten()

    def create_hist_fig(vals, plot_title, color="navy"):
        h_range = x_range
        if h_range is None and outlier_quantile > 0:
            q = outlier_quantile / 2.0
            h_range = (float(np.quantile(vals, q)), float(np.quantile(vals, 1.0 - q)))

        # If still None, use full data range
        if h_range is None:
            h_range = (float(vals.min()), float(vals.max()))

        hist, edges = np.histogram(vals, bins=bins, range=h_range, density=density)
        y_label = "Density" if density else "Frequency"
        source = ColumnDataSource(
            data=dict(top=hist, left=edges[:-1], right=edges[1:], counts=hist)
        )
        p = figure(
            title=plot_title,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color="#fafafa",
            y_axis_label=y_label,
            x_range=h_range,
        )
        p.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            source=source,
            fill_color=color,
            line_color="white",
            alpha=0.5,
            hover_fill_alpha=1.0,
            hover_fill_color="firebrick",
        )

        hover = HoverTool(tooltips=[("Count", "@counts"), ("Range", "[@left, @right]")])
        p.add_tools(hover)
        return p

    p1 = create_hist_fig(flat_data, f"{title} (Global)")

    # Add a visual highlight for the selected range in the global plot
    range_highlight = BoxAnnotation(fill_alpha=0.1, fill_color="red")
    p1.add_layout(range_highlight)

    # Analyze Outliers
    g_min, g_max = flat_data.min(), flat_data.max()
    total_count = len(flat_data)

    # Stats Panel
    stats_text = f"<b>Total Elements:</b> {total_count}<br>"
    stats_text += f"<b>Global Min:</b> {g_min:.6f} | <b>Global Max:</b> {g_max:.6f}<br>"

    if outlier_quantile > 0:
        q = outlier_quantile / 2.0
        q_min, q_max = np.quantile(flat_data, q), np.quantile(flat_data, 1.0 - q)
        clipped_in = ((flat_data >= q_min) & (flat_data <= q_max)).sum()
        ratio_in = (clipped_in / total_count) * 100
        stats_text += f"<b>Main Body ({100 - outlier_quantile * 100:.1f}% Quantile):</b> [{q_min:.4f}, {q_max:.4f}]<br>"
        stats_text += f"<b>Main Body Ratio:</b> {ratio_in:.2f}% | <b>Outlier Ratio:</b> {100 - ratio_in:.2f}%<br>"

    stats_div = Div(text=stats_text, width=800)

    # Interactive Range Selector
    # Subsample for smooth JS interaction if necessary
    js_data = flat_data
    if total_count > subsample:
        js_data = np.random.choice(flat_data, subsample, replace=False)

    # Sort for potential fast search on JS side (though we'll use a loop for simplicity now)
    js_data = np.sort(js_data)

    source_js = ColumnDataSource(data=dict(vals=js_data))

    # Dual Slider Setup
    # 1. Precision Slider (Focus on main body)
    q = outlier_quantile / 2.0 if outlier_quantile > 0 else 0.0
    p_min, p_max = (
        float(np.quantile(flat_data, q)),
        float(np.quantile(flat_data, 1.0 - q)),
    )

    precision_slider = RangeSlider(
        start=p_min,
        end=p_max,
        value=(p_min, p_max),
        step=(p_max - p_min) / 2000 if p_max > p_min else 0.001,
        title="Precision Slider (Main Body)",
        width=800,
    )

    # 2. Global Slider (Full range including outliers)
    global_slider = RangeSlider(
        start=float(g_min),
        end=float(g_max),
        value=(p_min, p_max),
        step=(g_max - g_min) / 1000 if g_max > g_min else 0.1,
        title="Global Slider (Outlier Query)",
        width=800,
    )

    result_div = Div(
        text="<b>Proportion:</b> 100% (Move sliders to update)",
        width=800,
    )

    # Shared Callback logic: logic that updates the div and highlight based on ANY slider change
    shared_code = """
        const data = source.data['vals'];
        const [low, high] = cb_obj.value;
        
        let count = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] >= low && data[i] <= high) {
                count++;
            }
        }
        const percent = (count / data.length * 100).toFixed(4);
        div.text = `<b>Proportion in range [${low.toFixed(6)}, ${high.toFixed(6)}]:</b> ${percent}% (${count} / ${data.length} samples)`;
        
        // Update visual highlight
        highlight.left = low;
        highlight.right = high;
        
        // Sync the other slider if it's within its bounds
        if (other.start <= low && other.end >= high) {
            other.value = [low, high];
        } else {
            // If out of bounds of the other slider, we just let them stay unsynced 
            // to avoid jitter or snapping issues
        }
    """

    precision_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                source=source_js,
                div=result_div,
                highlight=range_highlight,
                other=global_slider,
            ),
            code=shared_code,
        ),
    )

    global_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                source=source_js,
                div=result_div,
                highlight=range_highlight,
                other=precision_slider,
            ),
            code=shared_code,
        ),
    )

    # Initialize highlight
    range_highlight.left = p_min
    range_highlight.right = p_max

    if len(dims) > 0:
        # Hierarchical visualization
        # We want to keep the dimensions in 'dims' as the "sample" dimensions
        # and aggregate over the other dimensions.
        # But the user said: "if the feature_dims is (0, 1), then it will compute
        # the histogram / distribution only on the last dim, and then visualize
        # the distribution parameter (like mean/std) of the last dim over (0, 1)"

        # This means 'dims' are the dimensions that define each individual sample
        # for which we compute mean/std.

        # Example: shape (10, 3, 4), dims=(0, 1)
        # We have 10*3 = 30 samples. Each sample has 4 values.
        # Compute mean/std of these 4 values.

        all_dims = list(range(data.ndim))
        # The dimensions NOT in dims are the ones we aggregate over for each sample
        agg_dims = [d for d in all_dims if d not in dims]

        # We first transpose to put 'dims' at the front, then reshape
        # But wait, it's easier to just compute mean/std over agg_dims
        # agg_dims = (2,) in the example

        means = np.mean(data, axis=tuple(agg_dims))
        stds = np.std(data, axis=tuple(agg_dims))

        # Additional Stats for Hierarchical Mode
        h_stats = "<br><b>--- Hierarchical Stats (over dims) ---</b><br>"
        m_flat, s_flat = means.flatten(), stds.flatten()
        m_q = np.quantile(m_flat, [0.025, 0.975])
        s_q = np.quantile(s_flat, [0.025, 0.975])
        h_stats += f"<b>Means:</b> min={m_flat.min():.6f}, max={m_flat.max():.6f}, mean={m_flat.mean():.6f}, std={m_flat.std():.6f}, 95%Q=[{m_q[0]:.6f}, {m_q[1]:.6f}]<br>"
        h_stats += f"<b>Stds:</b> min={s_flat.min():.6f}, max={s_flat.max():.6f}, mean={s_flat.mean():.6f}, std={s_flat.std():.6f}, 95%Q=[{s_q[0]:.6f}, {s_q[1]:.6f}]<br>"
        stats_div.text += h_stats

        # Plot Scatter: Mean vs Std
        source_scatter = ColumnDataSource(
            data=dict(mean=means.flatten(), std=stds.flatten())
        )
        p_scatter = figure(
            title=f"{title} (Mean vs Std)",
            x_axis_label="Mean",
            y_axis_label="Std",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p_scatter.scatter(
            "mean", "std", source=source_scatter, size=7, color="olive", alpha=0.5
        )
        p_scatter.add_tools(HoverTool(tooltips=[("Mean", "@mean"), ("Std", "@std")]))

        p_means = create_hist_fig(
            means.flatten(), f"{title} (Distribution of Means)", color="orange"
        )
        p_stds = create_hist_fig(
            stds.flatten(), f"{title} (Distribution of Stds)", color="green"
        )

        layout = gridplot(
            [[p1, p_scatter], [p_means, p_stds]], sizing_mode="scale_width"
        )
    else:
        layout = p1

    final_layout = column(
        stats_div,
        precision_slider,
        global_slider,
        result_div,
        layout,
        sizing_mode="scale_width",
    )

    if save_path:
        if not save_path.endswith(".html"):
            save_path += ".html"
        output_file(filename=save_path, title=title)
        save(final_layout)
        print(f"Distribution visualization saved to {save_path}")

    if show_plot:
        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                # Initialize notebook output if in IPython/Jupyter environment
                output_notebook(hide_banner=True)
            show(final_layout)
        except Exception:
            # Fallback to standard show (may open a browser tab)
            show(final_layout)

    return final_layout
