"""
Module for reading, validating, manipulating, and creating meshes of brain structures.
"""

from __future__ import annotations
from warnings import warn
from typing import TYPE_CHECKING
from lapy import TriaMesh
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

def mask_mesh_gemini(
    geometry: TriaMesh,
    mask: ArrayLike,
    return_mapping: bool = False,
) -> TriaMesh | tuple[TriaMesh, NDArray[np.int_]]:
    # Format / validate arguments
    mask_v = np.asarray_chkfinite(mask, dtype=bool)
    if mask_v.shape != (geometry.v.shape[0],):
        raise ValueError(f"mask must have shape ({geometry.v.shape[0]},), matching the number of "
                         "vertices in geometry.")

    # Remove vertices not in mask
    v_masked = geometry.v[mask_v].astype(np.float64)

    # Update vertex indices of elements (-1 represents removed vertices)
    map_v = np.full(len(mask_v), -1, dtype=int)
    map_v[mask_v] = np.arange(np.sum(mask_v))
    map_t = map_v[geometry.t]
    
    # Keep only elements where all vertices are in the mask
    mask_t = np.all(map_t != -1, axis=1)
    t_masked = map_t[mask_t]
    
    # Local-to-global mapping array contains the original indices of the kept vertices
    local_to_global = np.where(mask_v)[0]

    mesh = geometry.__class__(v=v_masked, t=t_masked)
    
    if return_mapping:
        return mesh, local_to_global
    return mesh

def mask_mesh(
    geometry: TriaMesh,
    mask: ArrayLike,
) -> TriaMesh | tuple[TriaMesh, NDArray[np.bool_]]:
    """
    Remove specified vertices and corresponding elements from a triangular surface mesh. Returns a
    ``lapy.TriaMesh`` object.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        The input surface or volume mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (``True``) or remove (``False``).
    allow_mask_dilation : bool, optional
        If ``True``, allows the mask to be dilated to include any vertices that are referenced by
        elements that would otherwise be removed. This ensures that the resulting mesh remains
        valid, but may include additional vertices beyond those specified in the original mask.
        Default is ``False``.
        
    Returns
    -------
    lapy.TriaMesh or lapy.TetMesh or tuple
        The masked mesh. If ``return_mask`` is ``True``, returns
        ``(masked_mesh, effective_mask)`` where ``effective_mask`` has the same length as the
        input ``mask`` and marks vertices that survived masking and face filtering.

    Raises
    ------
    ValueError
        If ``mask`` does not have a length matching the number of vertices in ``geometry``.
    """
    # Format / validate arguments
    mask_v = np.asarray_chkfinite(mask, dtype=bool)
    if mask_v.shape != (geometry.v.shape[0],):
        raise ValueError(f"mask must have shape ({geometry.v.shape[0]},), matching the number of "
                         "vertices in geometry.")

    # Remove vertices not in mask
    v_masked = geometry.v[mask_v].astype(np.float64)

    # Update vertex indices of elements (-1 represents removed vertices)
    map_v = np.full(len(mask_v), -1, dtype=int)
    map_v[mask_v] = np.arange(np.sum(mask_v))
    map_t = map_v[geometry.t]
    
    # Keep only elements where all vertices are in the mask
    mask_t = np.all(map_t != -1, axis=1)
    t_masked = map_t[mask_t]

    # # Deal with any unreferenced (faceless) vertices (TODO: support TetMesh)
    # v_isref = np.zeros(len(v_masked), dtype=bool)
    # v_isref[t_masked] = True
    # if not np.all(v_isref):
    #     if allow_mask_dilation:
    #         # Dilate the mask to include any unreferenced vertices that are part of removed elements
    #         mask_v_dilated = mask_v.copy()
    #         for i in np.where(~v_isref)[0]:
    #             # get index of original vertex in geometry.v
    #             i_orig = np.where(mask_v)[0][i]

    #             # Find elements that reference this vertex
    #             t_ref = geometry.t[np.any(geometry.t == i_orig, axis=1)]

    #             # Find elements that connect this vertex to the mask
    #             t_support = t_ref[np.sum(mask_v[t_ref], axis=1) >= 2]

    #             if len(t_support) >= 0:
    #                 # Get indices of vertices in t_support that are outside the mask
    #                 v_support = t_support[~mask_v[t_support]]

    #                 # Add closest vertex in v_support to the mask
    #                 dists = np.linalg.norm(geometry.v[v_support] - geometry.v[i_orig], axis=1)
    #                 v_closest = v_support[np.argmin(dists)]
    #                 mask_v_dilated[v_closest] = True
    #         return mask_mesh(geometry, mask_v_dilated)
         
    #     warn(f'{np.sum(~v_isref)} vertices (IDs: {np.where(~v_isref)[0]}) in the '
    #             'mask are not part of any kept element and will be removed.')
    #     # Create mapping from old to new vertex indices
    #     v_remap = np.full(len(v_masked), -1, dtype=int)
    #     v_remap[v_isref] = np.arange(np.sum(v_isref))
    #     v_masked = v_masked[v_isref]
    #     t_masked = v_remap[t_masked]

    # Create a new TriaMesh or TetMesh with the masked vertices and elements
    return geometry.__class__(v=v_masked, t=t_masked)

def fix_mask(
    mask: ArrayLike,
    geometry: TriaMesh
) -> NDArray[np.bool_]:
    # Format / validate arguments
    mask = np.asarray_chkfinite(mask, dtype=bool)
    mask_fixed = mask.copy()
    if mask.shape != (geometry.v.shape[0],):
        raise ValueError(f"mask must have shape ({geometry.v.shape[0]},), matching the number of "
                         "vertices in geometry.")
    
    # count in-mask neighbours of each in-mask vertex
    degree = geometry.adj_sym[:, mask].getnnz(axis=1)

    # find isolated vertices in mask
    v_iso_idx = np.where(mask & (degree == 0))[0]

    # remove isolated vertices from mask
    if len(v_iso_idx) > 0:
        warn(f'{len(v_iso_idx)} isolated vertices (IDs: {v_iso_idx}) will be removed.')
        mask_fixed[v_iso_idx] = False

    # find vertices with only one in-mask neighbour
    #n_nbr = [1] if isinstance(geometry, TriaMesh) else [1, 2]  # Tets have 4 vertices, so 2 in-mask
    #neighbours is also unreferenced
    n_nbr = 1  # TODO: revert to above line when TetMesh is supported
    v_unref_idx = np.where(mask & (degree == n_nbr))[0]

    # Add support for unreferenced vertices
    if len(v_unref_idx) > 0:

        # warn(f'{len(v_unref_idx)} unreferenced vertices (IDs: {v_unref_idx}) will be removed.')

        # for each unreferenced vertex, there will be 2 options for which element to add to the mask
        # to support the vertex. For TetMesh, we can first choose the element with the most in-mask
        # vertices. Otherwise, we can choose the element with the closest vertex (or vertices) to
        # the unreferenced vertex.

        # For now, we will assume a manifold TriaMesh and simply add the candidate vertex that is
        # closest to the unreferenced vertex to the mask. TODO: support TetMesh
        
        # Find each unreferenced vertex's in-mask neighbour
        v_nbrs_idx = geometry.adj_sym[:, v_unref_idx].nonzero()[0]
        v_nbr_idx = v_nbrs_idx[mask[v_nbrs_idx]]

        # Find each unreferenced vertex's two candidate neighbours that are outside the mask but
        # also neighbour the in-mask neighbour
        adj_v_unref = geometry.adj_sym[:, v_unref_idx]
        adj_v_nbr = geometry.adj_sym[:, v_nbr_idx]
        adj_overlap = adj_v_unref.copy().multiply(adj_v_nbr)

        # DEBUG NOTE
        print(f"v_unref_idx: {v_unref_idx}, v_nbr_idx: {v_nbr_idx}")
        print(f"adj_overlap nonzero: {adj_overlap.nonzero()[0]}")
        v_cands_idx = adj_overlap.nonzero()[0].reshape(len(v_unref_idx), 2)
        print(f"v_cands_idx: {v_cands_idx}")

        # For each unreferenced vertex, add the closer candidate to the mask
        coord_diffs = geometry.v[v_cands_idx] - geometry.v[v_unref_idx][:, None, :]  # shape (n_unref, 2, 3)
        dists = np.linalg.norm(coord_diffs, axis=2)  # shape (n_unref, 2)
        print(f"coord_diffs: {coord_diffs}, dists: {dists}")
        v_closest_idx = v_cands_idx[np.arange(len(v_unref_idx)), np.argmin(dists, axis=1)]
        print(f"v_closest_idx: {v_closest_idx}")
        mask_fixed[v_closest_idx] = True

    return mask_fixed

def unmask_data(
    data: ArrayLike,
    mask: ArrayLike,
    fill_val: float = np.nan
) -> NDArray:
    """
    Unmasks data by inserting it into a full array with the same length as the medial wall mask.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be unmasked, of shape ``(n_verts)`` or ``(n_verts, n_maps)``.
    mask : numpy.ndarray
        A boolean array-like of shape ``(n_verts + n_extra_verts)`` where ``True`` indicates the
        positions of the data in the full array. Must contain exactly ``n_verts`` ``True`` values.
    fill_val : float, optional
        The value to fill in the positions outside the mask. Default is NaN.

    Returns
    -------
    numpy.ndarray
        The unmasked data of shape ``(n_verts + n_extra_verts)`` or
        ``(n_verts + n_added_verts, n_maps)``.

    Raises
    ------
    ValueError
        If ``mask`` is not a 1D boolean array.
    ValueError
        If ``data`` does not have shape ``(n_verts,)`` or ``(n_verts, n_maps)``.
    """
    # Format / validate arguments
    data = np.asarray(data)
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("`mask` must be a 1D boolean array.")
    if data.ndim not in [1, 2] or data.shape[0] != np.sum(mask):
        raise ValueError("`data` must have shape (n_verts,) or (n_verts, n_maps), where n_verts "
                         f"matches the number of True values in `mask` ({np.sum(mask)}).")
    n_verts = len(mask)
    out_shape = (n_verts, data.shape[1]) if data.ndim == 2 else (n_verts,)

    # Initialise array of fill values
    data_unmasked = np.full(out_shape, fill_val)

    # Overwrite rows with data where mask is True
    data_unmasked[mask] = data

    return data_unmasked

def check_surf(
    surf: TriaMesh
) -> None:
    """
    Check if the surface mesh is contiguous with no unreferenced vertices.
    
    Parameters
    ----------
    surf : lapy.TriaMesh
        The surface mesh to check.

    Raises
    ------
    ValueError
        If the surface mesh contains unreferenced vertices.
    ValueError
        If the surface mesh is not contiguous.
    ValueError
        If the surface mesh is not manifold (i.e., contains edges belonging to more than two faces).
    """
    # Ensure surface has no unreferenced vertices
    referenced = np.zeros(len(surf.v), dtype=bool)
    referenced[surf.t] = True
    if np.any(~referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         f'vertices (i.e., not part of any face). IDs: {np.where(~referenced)[0]}')

    # Ensure surface is contiguous
    n_components = surf.connected_components()[0]
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

    # Ensure surface is manifold
    if not surf.is_manifold():
        raise ValueError('Surface mesh is not manifold: contains edges belonging to more than two '
                         'faces.')