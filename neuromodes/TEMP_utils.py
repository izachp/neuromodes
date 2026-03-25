from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import plotly.graph_objs as go
from matplotlib import colormaps

if TYPE_CHECKING:
    from lapy import TriaMesh
    from numpy.typing import ArrayLike

def plot_mesh_data(
    geometry: TriaMesh,
    data: ArrayLike,
    cmap: str = 'seismic_r',
    cnorm: bool = False,
    size: tuple[int, int] = (700, 700),
    plot_edges: bool = False,
    plot_grid: bool = False,
    camera_eye: tuple[float, float, float] = (-2, 0, 0),
):
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

    Returns
    -------
    plotly.graph_objs.Figure
        The generated Plotly figure.
    """
    # Make colormap for plotly
    cmap_obj = colormaps.get_cmap(cmap)
    vals = np.linspace(0, 1, 256)
    colorscale = [
        [i / (256 - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        for i, (r, g, b, _) in enumerate(cmap_obj(vals))
    ]

    x, y, z = geometry.v.T
    i, j, k = geometry.t.T

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
        n_vertices, n_frames = data.shape
        if n_vertices != len(x):
            raise ValueError("data shape[0] must match number of vertices.")
        cmin, cmax = (np.nanmin(data), np.nanmax(data)) if cnorm else (np.nanmin(data[:, 0]), np.nanmax(data[:, 0]))

        # Base frame (first timepoint)
        base_mesh = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=data[:, 0],
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            flatshading=False,
            showscale=False,
        )


        frames = [
            go.Frame(
                data=[
                    go.Mesh3d(
                        x=x, y=y, z=z, i=i, j=j, k=k,
                        intensity=data[:, t],
                        colorscale=colorscale,
                        cmin=(cmin if cnorm else np.nanmin(data[:, t])), cmax=(cmax if cnorm else np.nanmax(data[:, t])),
                        flatshading=False,
                        showscale=False,
                    )
                ],
                name=str(t),
            )
            for t in range(n_frames)
        ]

        fig = go.Figure(data=[base_mesh], frames=frames)
    else:
        cmin, cmax = np.nanmin(data), np.nanmax(data)
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                intensity=data,
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                flatshading=False,
                showscale=False,
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

    fig.update_layout(**layout_kwargs)
    return fig