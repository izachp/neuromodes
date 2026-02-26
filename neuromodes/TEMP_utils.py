"""Utility functions for mesh generation and visualization."""

from __future__ import annotations
import numpy as np
import plotly.graph_objs as go
from lapy import TriaMesh, TetMesh
from matplotlib import colormaps
from scipy import sparse

def make_thin_vol(surface_mesh, scaling=0.99, **kwargs):
    """
    Create a thin shell tetrahedral volume mesh from a surface mesh.
    
    Parameters
    ----------
    surface_mesh : trimesh.Trimesh
        The outer surface mesh.
    scaling : float, optional
        Scale factor for the inner surface (default: 0.99).
    **kwargs
        Additional keyword arguments for tetgen.TetGen.tetrahedralize().
    
    Returns
    -------
    tet_mesh : lapy.TetMesh
        Tetrahedral mesh of the shell volume.
    """
    import tetgen
    # Create inner mesh
    inner_mesh = surface_mesh.copy()
    inner_mesh.apply_scale(scaling)
    
    # Combine vertices and faces
    vertices = np.vstack([surface_mesh.vertices, inner_mesh.vertices])
    outer_faces = surface_mesh.faces
    inner_faces = inner_mesh.faces + len(surface_mesh.vertices)
    faces = np.vstack([outer_faces, inner_faces])
    
    # Tetrahedralize
    tet = tetgen.TetGen(vertices, faces)
    tet.tetrahedralize(**kwargs)
    
    # Extract tetrahedral mesh
    vol_vertices = tet.grid.points
    vol_tets = tet.grid.cells.reshape(-1, 5)[:, 1:5].astype(int)
    
    return TetMesh(v=vol_vertices, t=vol_tets)

def plot_mesh_data(geometry, data, cmap='seismic_r', cnorm=False, width=700, height=700, cmap_center=None, plot_edges=False):
    """
    Plot a colored mesh surface with overlaid edges.
    
    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        The surface mesh geometry.
    data : ndarray
        Data values (n_vertices,).
    cmap : str, optional
        Matplotlib colormap name (default: 'seismic_r').
    cnorm : bool, optional
        If data is 2D, whether to normalize color scale across frames (default: False).
    width : int, optional
        Figure width in pixels (default: 700).
    height : int, optional
        Figure height in pixels (default: 700).
    cmap_center : float or None, optional
        Center value for symmetric color scaling. If None, uses min/max. If float, color range is symmetric around this value.
    plot_edges : bool, optional
        If True, overlay mesh edges. Default: False.
    """
    # Make colormap for plotly
    cmap_obj = colormaps.get_cmap(cmap)
    vals = np.linspace(0, 1, 256)
    colorscale = [
        [i / (256 - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        for i, (r, g, b, _) in enumerate(cmap_obj(vals))
    ]

    if isinstance(geometry, TetMesh):
        geometry = geometry.boundary_tria()
        vkeep, _ = geometry.rm_free_vertices_()
        data = data[vkeep]
    elif not isinstance(geometry, TriaMesh):
        raise ValueError("plot_mesh_data currently only supports surface meshes and closed volumes.")

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
    epsilon = 1e-8
    if is_animated:
        n_vertices, n_frames = data.shape
        if n_vertices != len(x):
            raise ValueError("data shape[0] must match number of vertices.")
        if cnorm:
            # Global symmetric color range
            dmin, dmax = np.nanmin(data), np.nanmax(data)
            if cmap_center is not None:
                if dmin == dmax:
                    # All values are constant; symmetric range around center
                    dmax_abs = abs(dmin - cmap_center)
                    # If center is 0, range is -abs(constant) to +abs(constant)
                    cmin, cmax = cmap_center - dmax_abs, cmap_center + dmax_abs
                    # If constant is 0, fallback to small epsilon
                    if cmin == cmax:
                        cmin -= epsilon
                        cmax += epsilon
                else:
                    dmax_abs = max(abs(dmin - cmap_center), abs(dmax - cmap_center))
                    cmin, cmax = cmap_center - dmax_abs, cmap_center + dmax_abs
            else:
                cmin, cmax = dmin, dmax
                if cmin == cmax:
                    cmin -= epsilon
                    cmax += epsilon
            # Base frame
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
                            cmin=cmin, cmax=cmax,
                            flatshading=False,
                            showscale=False,
                        )
                    ],
                    name=str(t),
                )
                for t in range(n_frames)
            ]
        else:
            # Per-frame symmetric color range
            dmin0, dmax0 = np.nanmin(data[:, 0]), np.nanmax(data[:, 0])
            if cmap_center is not None:
                dmax_abs0 = max(abs(dmin0 - cmap_center), abs(dmax0 - cmap_center))
                cmin0, cmax0 = cmap_center - dmax_abs0, cmap_center + dmax_abs0
            else:
                cmin0, cmax0 = dmin0, dmax0
            if cmin0 == cmax0:
                cmin0 -= epsilon
                cmax0 += epsilon
            base_mesh = go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                intensity=data[:, 0],
                colorscale=colorscale,
                cmin=cmin0, cmax=cmax0,
                flatshading=False,
                showscale=False,
            )
            frames = []
            for t in range(n_frames):
                dmin_t, dmax_t = np.nanmin(data[:, t]), np.nanmax(data[:, t])
                if cmap_center is not None:
                    dmax_abs_t = max(abs(dmin_t - cmap_center), abs(dmax_t - cmap_center))
                    cmin_t, cmax_t = cmap_center - dmax_abs_t, cmap_center + dmax_abs_t
                else:
                    cmin_t, cmax_t = dmin_t, dmax_t
                if cmin_t == cmax_t:
                    cmin_t -= epsilon
                    cmax_t += epsilon
                frames.append(
                    go.Frame(
                        data=[
                            go.Mesh3d(
                                x=x, y=y, z=z, i=i, j=j, k=k,
                                intensity=data[:, t],
                                colorscale=colorscale,
                                cmin=cmin_t, cmax=cmax_t,
                                flatshading=False,
                                showscale=False,
                            )
                        ],
                        name=str(t),
                    )
                )
        fig = go.Figure(data=[base_mesh], frames=frames)
    else:
        dmin, dmax = np.nanmin(data), np.nanmax(data)
        if cmap_center is not None:
            dmax_abs = max(abs(dmin - cmap_center), abs(dmax - cmap_center))
            cmin, cmax = cmap_center - dmax_abs, cmap_center + dmax_abs
        else:
            cmin, cmax = dmin, dmax
        if cmin == cmax:
            cmin -= epsilon
            cmax += epsilon
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

    # Add static edge overlay if requested
    if plot_edges:
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Layout + optional animation controls
    layout_kwargs = dict(
        width=width,
        height=height,
        scene=dict(aspectmode="data"),
    )

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
    fig.show()

def _fem_tria_hetero(tria, lump=False, hetero=None):
    """
    Compute FEM matrices for a triangular mesh with heterogeneous elements.
    Adapted from lapy.fem.tria._fem_tria function to handle heterogeneous triangles.
    """
    import sys

    # Compute vertex coordinates and a difference vector for each triangle:
    t1 = tria.t[:, 0]
    t2 = tria.t[:, 1]
    t3 = tria.t[:, 2]
    v1 = tria.v[t1, :]
    v2 = tria.v[t2, :]
    v3 = tria.v[t3, :]
    v2mv1 = v2 - v1
    v3mv2 = v3 - v2
    v1mv3 = v1 - v3
    # Compute cross product and 4*vol for each triangle:
    cr = np.cross(v3mv2, v1mv3)
    vol = 2 * np.sqrt(np.sum(cr * cr, axis=1))
    # zero vol will cause division by zero below, so set to small value:
    vol_mean = 0.0001 * np.mean(vol)
    vol[vol < sys.float_info.epsilon] = vol_mean
    # compute cotangents for A
    # using that v2mv1 = - (v3mv2 + v1mv3) this can also be seen by
    # summing the local matrix entries in the old algorithm
    a12 = np.sum(v3mv2 * v1mv3, axis=1) / vol
    a23 = np.sum(v1mv3 * v2mv1, axis=1) / vol
    a31 = np.sum(v2mv1 * v3mv2, axis=1) / vol
    # compute diagonals (from row sum = 0)
    a11 = -a12 - a31
    a22 = -a12 - a23
    a33 = -a31 - a23
    # ----------------------------------- APPLY HETEROGENEITY ---------------------------------
    hetero_trias = np.sum(hetero[tria.t], axis=1) / 3.0
    a12 *= hetero_trias
    a23 *= hetero_trias
    a11 *= hetero_trias
    a22 *= hetero_trias
    a33 *= hetero_trias
    # -----------------------------------------------------------------------------------------
    # stack columns to assemble data
    local_a = np.column_stack(
        (a12, a12, a23, a23, a31, a31, a11, a22, a33)
    ).reshape(-1)
    i = np.column_stack((t1, t2, t2, t3, t3, t1, t1, t2, t3)).reshape(-1)
    j = np.column_stack((t2, t1, t3, t2, t1, t3, t1, t2, t3)).reshape(-1)
    # Construct sparse matrix:
    # a = sparse.csr_matrix((local_a, (i, j)))
    a = sparse.csc_matrix((local_a, (i, j)))
    # construct mass matrix (sparse or diagonal if lumped)
    if not lump:
        # create b matrix data (account for that vol is 4 times area)
        b_ii = vol / 24
        b_ij = vol / 48
        local_b = np.column_stack(
            (b_ij, b_ij, b_ij, b_ij, b_ij, b_ij, b_ii, b_ii, b_ii)
        ).reshape(-1)
        b = sparse.csc_matrix((local_b, (i, j)))
    else:
        # when lumping put all onto diagonal  (area/3 for each vertex)
        b_ii = vol / 12
        local_b = np.column_stack((b_ii, b_ii, b_ii)).reshape(-1)
        i = np.column_stack((t1, t2, t3)).reshape(-1)
        b = sparse.csc_matrix((local_b, (i, i)))
    return a, b