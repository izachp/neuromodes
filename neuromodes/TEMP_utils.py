from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import plotly.graph_objs as go
from matplotlib import colormaps

if TYPE_CHECKING:
    from lapy import TriaMesh
    from numpy.typing import ArrayLike
    from plotly.graph_objs import Figure
    from matplotlib.axes import Axes

def plot_mesh_data(
    geometry: TriaMesh,
    data: ArrayLike,
    cmap: str = 'seismic_r',
    cnorm: bool = False,
    size: tuple[int, int] = (700, 700),
    plot_edges: bool = False,
    plot_grid: bool = False,
    camera_eye: tuple[float, float, float] = (-2, 0, 0),
    flatshading: bool = False,
    lighting: dict[str, float] | None = None,
    lightposition: tuple[float, float, float] | None = None,
    nan_color: tuple[int, int, int] = (128, 128, 128),
    labels: str | list[str] | None = None,
    axs: Axes | list[Axes] | None = None,
) -> Figure | None:
    """
    Plot a colored mesh surface with overlaid edges.
    
    Parameters
    ----------
    geometry : lapy.TriaMesh
        The surface mesh geometry.
    data : ndarray
        Data values (n_vertices,).
    cmap : str, optional
        Matplotlib colormap name (default: 'seismic_r').
    cnorm : bool, optional
        If data is 2D, whether to normalize color scale across frames (default: True).
    size : tuple[int, int], optional
        Figure size as ``(width, height)`` in pixels (default: ``(700, 700)``).
    plot_edges : bool, optional
        Whether to overlay mesh edges (default: True).
    plot_grid : bool, optional
        Whether to show the 3D scene grid/background planes (default: False).
    camera_eye : tuple[float, float, float] or None, optional
        Initial camera eye position as ``(x, y, z)``. If None, Plotly default is used.
    flatshading : bool, optional
        Whether to use flat shading (faceted look) instead of smooth shading.
    lighting : dict[str, float] or None, optional
        Plotly ``Mesh3d.lighting`` parameters. Supported keys include
        ``ambient``, ``diffuse``, ``specular``, ``roughness``, and ``fresnel``.
        If None, Plotly defaults are used.
    lightposition : tuple[float, float, float] or None, optional
        Plotly ``Mesh3d.lightposition`` as ``(x, y, z)``. If None, Plotly default
        light position is used.
    nan_color : tuple[int, int, int], optional
        RGB color for NaN-valued vertices.
    labels : str, list[str], or None, optional
        Plot title labels. If ``axs is None``, provide a string to title the
        interactive figure (single or animated). If ``axs`` is provided,
        provide a list of strings to title each rendered subplot.
    axs : matplotlib.axes.Axes, list of Axes, or None, optional
        If None (default), returns an interactive Plotly figure with animation
        (if data is 2D). If provided, renders each frame as a static image to
        the corresponding matplotlib Axes and returns None. Axs can be a single
        Axes (renders frame 0) or a list/array of Axes (renders each frame).

    Returns
    -------
    plotly.graph_objs.Figure or None
        If axs is None, returns the Plotly Figure.
        If axs is provided, renders to axes and returns None.
    """
    # validate data shape
    n_verts = geometry.v.shape[0]
    data = np.asarray(data)
    if data.shape[0] != n_verts:
        raise ValueError(f"Data length {data.shape[0]} does not match number of vertices {n_verts}.")
    if len(nan_color) != 3:
        raise ValueError("nan_color must be a tuple of (r, g, b).")
    if any((c < 0 or c > 255) for c in nan_color):
        raise ValueError("nan_color values must be in the range [0, 255].")
    if labels is not None and not isinstance(labels, (str, list)):
        raise ValueError("labels must be None, a string, or a list of strings.")
    if isinstance(labels, list) and not all(isinstance(lbl, str) for lbl in labels):
        raise ValueError("labels list entries must all be strings.")
    if axs is None and isinstance(labels, list):
        raise ValueError("When axs is None, labels must be a string or None.")

    # Make colormap for plotly
    cmap_obj = colormaps.get_cmap(cmap)

    x, y, z = geometry.v.T
    i, j, k = geometry.t.T

    mesh_kwargs = {
        "flatshading": flatshading,
        "showscale": False,
    }
    if lighting is not None:
        mesh_kwargs["lighting"] = lighting
    if lightposition is not None:
        if len(lightposition) != 3:
            raise ValueError("lightposition must be a tuple of (x, y, z) or None.")
        mesh_kwargs["lightposition"] = dict(x=lightposition[0], y=lightposition[1], z=lightposition[2])

    # Helper function to create vertex colors, mapping NaN to grey
    def create_vertex_colors(data_frame, cmap_obj, normalize_range=None):
        """Convert data to RGB colors, with NaN mapped to grey."""
        colors = []
        for val in data_frame:
            if np.isnan(val):
                colors.append(f"rgb({nan_color[0]}, {nan_color[1]}, {nan_color[2]})")
            else:
                # Normalize value and map to colormap
                if normalize_range is None:
                    norm_val = val
                else:
                    norm_val = (val - normalize_range[0]) / (normalize_range[1] - normalize_range[0])
                    norm_val = np.clip(norm_val, 0, 1)
                r, g, b, _ = cmap_obj(norm_val)
                colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
        return colors

    # Edge coordinates (static, shared across frames)
    edges = []
    for idx_i, idx_j, idx_k in zip(i, j, k):
        edges.append((idx_i, idx_j))
        edges.append((idx_j, idx_k))
        edges.append((idx_k, idx_i))
    edges = list(set(edges))

    edge_x, edge_y, edge_z = [], [], []
    for idx_i, idx_j in edges:
        edge_x.extend([x[idx_i], x[idx_j], None])
        edge_y.extend([y[idx_i], y[idx_j], None])
        edge_z.extend([z[idx_i], z[idx_j], None])

    is_animated = data.ndim == 2
    if is_animated:
        n_frames = data.shape[1]
        # Get global min/max for normalization (if cnorm is True)
        if cnorm:
            cmin, cmax = np.nanmin(data), np.nanmax(data)
        else:
            cmin, cmax = None, None

        # Base frame (first timepoint)
        if cnorm:
            norm_range = (cmin, cmax)
        else:
            norm_range = (np.nanmin(data[:, 0]), np.nanmax(data[:, 0]))
        vertex_colors = create_vertex_colors(data[:, 0], cmap_obj, normalize_range=norm_range)
        
        base_mesh = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            vertexcolor=vertex_colors,
            **mesh_kwargs,
        )

        frames = [
            go.Frame(
                data=[
                    go.Mesh3d(
                        x=x, y=y, z=z, i=i, j=j, k=k,
                        vertexcolor=create_vertex_colors(
                            data[:, t],
                            cmap_obj,
                            normalize_range=(cmin, cmax) if cnorm else (np.nanmin(data[:, t]), np.nanmax(data[:, t]))
                        ),
                        **mesh_kwargs,
                    )
                ],
                name=str(t),
            )
            for t in range(n_frames)
        ]

        fig = go.Figure(data=[base_mesh], frames=frames)
    else:
        cmin, cmax = np.nanmin(data), np.nanmax(data)
        vertex_colors = create_vertex_colors(data, cmap_obj, normalize_range=(cmin, cmax))
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                vertexcolor=vertex_colors,
                **mesh_kwargs,
            )
        ])

    # Add static edge overlay
    if plot_edges:
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip',
        ))

    if len(size) != 2:
        raise ValueError("size must be a tuple of (width, height).")
    width, height = size
    if camera_eye is not None and len(camera_eye) != 3:
        raise ValueError("camera_eye must be a tuple of (x, y, z) or None.")

    # Layout + optional animation controls
    layout_kwargs = dict(
        width=width,
        height=height,
        scene=dict(
            aspectmode="data",
            xaxis=dict(
                showgrid=plot_grid,
                showbackground=plot_grid,
                showticklabels=plot_grid,
                title=("x" if plot_grid else ""),
            ),
            yaxis=dict(
                showgrid=plot_grid,
                showbackground=plot_grid,
                showticklabels=plot_grid,
                title=("y" if plot_grid else ""),
            ),
            zaxis=dict(
                showgrid=plot_grid,
                showbackground=plot_grid,
                showticklabels=plot_grid,
                title=("z" if plot_grid else ""),
            ),
        ),
    )

    if camera_eye is not None:
        layout_kwargs["scene_camera"] = dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))

    if is_animated:
        slider_steps = [
            dict(method="animate", args=[[str(t)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))], label=str(t))
            for t in range(n_frames)
        ]
        sliders = [dict(active=0, pad=dict(t=50), steps=slider_steps)]
        play_pause = [
            dict(label="▶️ Play", method="animate",
                 args=[None, dict(frame=dict(duration=50, redraw=True),
                                  transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸️ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate")]),
        ]
        layout_kwargs.update(
            sliders=sliders,
            updatemenus=[dict(type="buttons", showactive=False, buttons=play_pause, x=0, y=0)],
        )
    if isinstance(labels, str):
        layout_kwargs["title"] = labels

    fig.update_layout(**layout_kwargs)
    
    # Handle matplotlib axes rendering
    if axs is not None:
        from PIL import Image
        import io
        
        # Normalize axs to a list
        if not isinstance(axs, (list, np.ndarray)):
            axs = [axs]
        if isinstance(labels, str):
            labels = [labels]
        
        # Determine which frames to render
        if is_animated:
            frames_to_render = list(range(n_frames))
        else:
            frames_to_render = [0]
        
        # Render each frame
        for idx, frame_idx in enumerate(frames_to_render):
            if idx >= len(axs):
                break  # More frames than axes provided
            
            if is_animated:
                # Create a temporary figure with the frame data
                frame_mesh = go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k,
                    vertexcolor=create_vertex_colors(
                        data[:, frame_idx],
                        cmap_obj,
                        normalize_range=(cmin, cmax) if cnorm else (np.nanmin(data[:, frame_idx]), np.nanmax(data[:, frame_idx]))
                    ),
                    **mesh_kwargs,
                )
                temp_fig = go.Figure(data=[frame_mesh])
            else:
                temp_fig = fig
            
            # Add edge overlay if needed
            if plot_edges:
                temp_fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    hoverinfo='skip',
                ))
            
            # Apply layout settings
            temp_fig.update_layout(
                width=width,
                height=height,
                scene=dict(
                    aspectmode="data",
                    xaxis=dict(
                        showgrid=plot_grid,
                        showbackground=plot_grid,
                        showticklabels=plot_grid,
                        title=("x" if plot_grid else ""),
                    ),
                    yaxis=dict(
                        showgrid=plot_grid,
                        showbackground=plot_grid,
                        showticklabels=plot_grid,
                        title=("y" if plot_grid else ""),
                    ),
                    zaxis=dict(
                        showgrid=plot_grid,
                        showbackground=plot_grid,
                        showticklabels=plot_grid,
                        title=("z" if plot_grid else ""),
                    ),
                ),
            )
            if camera_eye is not None:
                temp_fig.update_layout(scene_camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])))
            
            # Render to image and display in axis
            img_bytes = temp_fig.to_image(format="png", width=width, height=height)
            img = Image.open(io.BytesIO(img_bytes))
            
            axs[idx].imshow(img)
            axs[idx].axis('off')
            if isinstance(labels, list) and idx < len(labels):
                axs[idx].set_title(labels[idx])
            elif is_animated:
                axs[idx].set_title(f"Frame {frame_idx}")
        
        return None
    
    return fig