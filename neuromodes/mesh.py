"""
Module for reading, validating, manipulating, and creating meshes of brain structures.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, TYPE_CHECKING
from lapy import TriaMesh, TetMesh
from nibabel.affines import apply_affine
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load
import numpy as np
from scipy.interpolate import griddata, Rbf
from neuromodes.io import fs_extensions

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image
    from numpy.typing import ArrayLike, NDArray

def is_vol(
    geometry: Union[TetMesh, TriaMesh, GiftiImage, str, Path, dict]
) -> bool:
    """
    Determine whether the given geometry represents a volume or surface mesh.

    Parameters
    ----------
    geometry : lapy.TetMesh, lapy.TriaMesh, nibabel.gifti.GiftiImage, str, Path, or dict
        The geometry to check. Can be an instance of `lapy.TetMesh`, `lapy.TriaMesh`,
        `nibabel.gifti.GiftiImage`, a path-like string, or a dictionary with mesh data.

    Returns
    -------
    bool
        True if the geometry is a volume mesh, False if it is a surface mesh.

    Raises
    ------
    ValueError
        If the geometry is a path-like string with an unrecognized file extension.
    ValueError
        If the geometry is a dictionary that does not have keys 'vertices' and 'faces', or if
        'faces' does not reference an array with shape (n_tetras, 4) for volumes or (n_trias, 3) for
        surfaces.
    """
    # Instances
    if isinstance(geometry, TetMesh):
        return True
    if isinstance(geometry, (TriaMesh, GiftiImage)):
        return False
    
    # Paths
    if isinstance(geometry, (str, Path)):
        if str(geometry).endswith('.tetra.vtk'):
            return True
        elif str(geometry).endswith(('.vtk', '.gii') + fs_extensions):
            return False
        raise ValueError(
            'Received path-like string for `geometry`, but file extension is not recognized. '
            'Please provide a path-like string to a mesh file for a surface (.vtk, .gii, '
            f'{", ".join(fs_extensions)}) or volume (.tetra.vtk).')
    
    # Dictionary
    if isinstance(geometry, dict):
        err_str = ('Received an invalid dictionary for `geometry`. `vertices` key should reference '
                   'an array of shape (n_verts, 3) and `faces` key should reference an array of '
                   'shape (n_tetras, 4) for volumes or (n_trias, 3) for surfaces.')
        if 'vertices' not in geometry:
            raise ValueError(err_str)
        try:
            verts_per_face = np.asarray(geometry['faces']).shape[1]
        except Exception:
            raise ValueError(err_str)
        if verts_per_face == 4:
            return True
        elif verts_per_face == 3:
            return False
        raise ValueError(err_str)

def mask_mesh(
    geometry: Union[TriaMesh, TetMesh],
    mask: ArrayLike
) -> Union[TriaMesh, TetMesh]:
    """
    Remove specified vertices and corresponding elements from a triangular surface or tetrahedral
    volume mesh. Returns a `lapy.TriaMesh` or `lapy.TetMesh` object.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        The input surface or volume mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (`True`) or remove (`False`).

    Returns
    -------
    lapy.TriaMesh or lapy.TetMesh
        The masked surface or volume mesh.

    Raises
    ------
    ValueError
        If `mask` does not have a length matching the number of vertices in `geometry`.
    """
    # Format / validate arguments
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.shape != (geometry.v.shape[0],):
        raise ValueError(f"`mask` must have shape (n_verts,) = ({geometry.v.shape[0]},).")

    # Remove vertices not in mask
    v_masked = geometry.v[mask]

    # Update vertex indices of elements (-1 represents removed vertices)
    v_map = np.full(len(mask), -1, dtype=int)
    v_map[mask] = np.arange(np.sum(mask))
    t_remapped = v_map[geometry.t]
    
    # Keep only elements where all vertices are in the mask
    elem_mask = np.all(t_remapped != -1, axis=1)
    t_masked = t_remapped[elem_mask]

    # Create a new TriaMesh or TetMesh with the masked vertices and elements
    return geometry.__class__(v=v_masked, t=t_masked)

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
        The data to be unmasked, which should have the same number of rows as the number of True
        values in `mask`. Can be 1D or 2D (n_masked_verts, n_maps).
    mask : numpy.ndarray
        A boolean array where True indicates the positions of the data in the full array.
    fill_val : float, optional
        The value to fill in the positions outside the mask. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The unmasked data, with the same shape as the medial mask.

    Raises
    ------
    ValueError
        If `mask` is not a 1D boolean array.
    ValueError
        If `data` does not have shape (n_masked_verts,) or (n_masked_verts, n_maps), where
        n_masked_verts is the number of True values in `mask`.
    """
    # Format / validate arguments
    data = np.asarray(data)
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("`mask` must be a 1D boolean array.")
    if data.ndim not in [1, 2] or data.shape[0] != np.sum(mask):
        raise ValueError(
            "`data` must have shape (n_masked_verts,) or (n_masked_verts, n_maps), where "
            f"n_masked_verts is the number of True values in `mask` ({np.sum(mask)})."
            )
    n_verts = len(mask)
    out_shape = (n_verts, data.shape[1]) if data.ndim == 2 else (n_verts,)

    # Initialise array of fill values
    data_unmasked = np.full(out_shape, fill_val)

    # Overwrite rows with data where mask is True
    data_unmasked[mask] = data

    return data_unmasked

def normalize_vol(
    geometry: TetMesh
) -> TetMesh:
    """
    Translate the mesh centroid to the origin and rescale to unit volume.

    Parameters
    ----------
    geometry : lapy.TetMesh
        The input volume mesh, to be modified in-place.
    """
    # Get edge vectors for each tetrahedron
    t0 = geometry.t[:, 0]
    t1 = geometry.t[:, 1]
    t2 = geometry.t[:, 2]
    t3 = geometry.t[:, 3]

    v0 = geometry.v[t0, :]
    v1 = geometry.v[t1, :]
    v2 = geometry.v[t2, :]
    v3 = geometry.v[t3, :]

    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Compute volume of each tetrahedron using triple product formula: V = |(e1 . (e2 x e3))| / 6
    tetra_vols = np.abs(np.einsum('ij,ij->i', e1, np.cross(e2, e3))) / 6

    # Compute centroid of each tetrahedron as simple average of its vertices
    tetra_centroids = (v0 + v1 + v2 + v3) / 4

    # Compute mesh centroid as volume-weighted average of tetrahedron centroids
    # Note: this is equivalent to LaPy's TriaMesh.centroid()
    centroid = np.sum(tetra_vols[:, np.newaxis] * tetra_centroids, axis=0) / np.sum(tetra_vols)

    # Translate centroid to origin
    geometry.v -= centroid

    # Rescale to unit volume
    geometry.v /= geometry.boundary_tria().volume() ** (1/3)

def check_vol(
    vol: TetMesh
) -> None:
    """
    Check if the volume mesh has no unreferenced vertices and a contiguous surface boundary.
    
    Parameters
    ----------
    vol : lapy.TetMesh
        The volume mesh to check.

    Raises
    ------
    ValueError
        If the volume mesh contains unreferenced vertices.
    ValueError
        If the volume mesh is not manifold (i.e., contains triangles shared by more than two
        tetrahedra).
    """
    if vol.has_free_vertices():
        raise ValueError('Volume mesh contains unreferenced vertices (i.e., not part of any '
                         'tetrahedron).')
    
    # Ensure volume is manifold (i.e., no faces shared by more than two tets)
    if not _is_vol_manifold(vol):
        raise ValueError('Volume mesh is not manifold: contains faces shared by more than two '
                         'tetrahedra.')

    # Validate surface boundary
    vol_boundary = vol.boundary_tria()
    vol_boundary.rm_free_vertices_()
    vol_boundary.orient_()
    check_surf(vol_boundary)

def _is_vol_manifold(vol) -> bool:
    """Check if the tetrahedral mesh is manifold.

    Returns
    -------
    bool
        True if every triangle face is shared by at most two tetrahedra.
    """
    # Extract all 4 triangles from each tetrahedron
    trias = np.concatenate([
        vol.t[:, [0, 1, 2]],
        vol.t[:, [0, 1, 3]],
        vol.t[:, [0, 2, 3]],
        vol.t[:, [1, 2, 3]],
    ])

    # Order vertices within each triangle for consistent representation
    trias.sort(axis=1)

    # Manifold if no triangle occurs more than twice
    tria_counts = np.unique(trias, axis=0, return_counts=True)[1]
    return np.all(tria_counts <= 2)

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
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         'vertices (i.e., not part of any face).')

    # Ensure surface is contiguous
    n_components = surf.connected_components()[0]
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

    # Ensure surface is manifold
    if not surf.is_manifold():
        raise ValueError('Surface mesh is not manifold: contains edges belonging to more than two '
                         'faces.')
    
def tetmesh_to_nifti(
    data: ArrayLike,
    tetmesh: TetMesh,
    nifti_mask: Union[str, Path, Nifti1Image]
) -> Nifti1Image:
    """
    Project data defined on a tetrahedral mesh to a volumetric NIFTI space.
    
    Parameters
    ----------
    nifti_mask : str, Path, or Nifti1Image
        Input NIFTI file path or loaded Nifti1Image object defining the target volume space.
    data : array-like
        Data values defined on the vertices of the tetmesh (shape should be (n_vertices,))
    tetmesh : lapy.TetMesh
        The tetrahedral mesh on which the data is defined. Must have vertex coordinates similar to
        the NIFTI space, after applying the appropriate affine transformation.
    """
    # Format / validate arguments
    data = np.asarray_chkfinite(data)
    if data.shape != (tetmesh.v.shape[0],):
        raise ValueError(f"`data` must have shape ({tetmesh.v.shape[0]},), got {data.shape}.")
    if not isinstance(tetmesh, TetMesh):
        raise ValueError("`tetmesh` must be an instance of `lapy.TetMesh`.")
    if isinstance(nifti_mask, (str, Path)):
        nifti_mask = load(nifti_mask)
    elif not isinstance(nifti_mask, Nifti1Image):
        raise ValueError("nifti_mask must be a Nifti1Image object or a path-like string to a valid "
                         "`.nii` or `.nii.gz` file.")
    
    # Get coordinates of nonzero voxels in physical space
    x, y, z = np.asarray(nifti_mask.get_fdata() > 0).nonzero()
    vox_coords = np.column_stack([x, y, z])
    apply_affine(nifti_mask.affine, vox_coords, inplace=True)

    # Initialise NIFTI array
    interp_data = np.zeros(nifti_mask.shape, dtype=np.result_type(data, np.float32))

    # Interpolate data and store at ROI coordinates
    interp_data[x, y, z] = griddata(tetmesh.v, data, vox_coords, method='linear')

    # Create a new NIFTI image with the interpolated values
    header = nifti_mask.header.copy().set_data_dtype(interp_data.dtype)
    return Nifti1Image(interp_data, nifti_mask.affine, header=header)

def nifti_to_tetmesh(
    nifti_data: Union[str, Path, Nifti1Image],
    tetmesh: TetMesh,
    nifti_mask: Union[str, Path, Nifti1Image, None] = None,
    **rbf_kwargs
) -> NDArray:
    """
    Project data from volumetric NIFTI space to a tetrahedral mesh using RBF interpolation.
    
    RBF (Radial Basis Function) interpolation is faster than griddata's linear method but
    more stable than nearest-neighbor on mesh boundaries. This avoids the zeros at boundary
    voxels that nearest-neighbor produces while being much faster than linear griddata.
    
    Parameters
    ----------
    nifti_data : str, Path, or Nifti1Image
        Input NIFTI file path or loaded Nifti1Image object.
    nifti_mask : str, Path, or Nifti1Image or None
        Optional mask to define the region of interest. If None, nonzero voxels are used.
    tetmesh : lapy.TetMesh
        The tetrahedral mesh to which data is projected.
    **rbf_kwargs
        Additional keyword arguments to pass to `scipy.interpolate.Rbf`, such as `function` and
        `smooth`.
    
    Returns
    -------
    interp_data : ndarray
        Data interpolated at mesh vertices, shape (n_vertices,).
    """
    # Load NIFTI if needed
    if isinstance(nifti_data, (str, Path)):
        nifti_data = load(nifti_data)
    elif not isinstance(nifti_data, Nifti1Image):
        raise ValueError("nifti_data must be a Nifti1Image object or path")
    
    # Get mask
    data = nifti_data.get_fdata()
    if nifti_mask is None:
        mask = data > 0
    else:
        if isinstance(nifti_mask, (str, Path)):
            nifti_mask = load(nifti_mask)
        mask = nifti_mask.get_fdata() > 0
    
    # Get voxel coordinates and values
    x, y, z = np.where(mask)
    vox_coords = np.column_stack([x, y, z])
    apply_affine(nifti_data.affine, vox_coords, inplace=True)
    
    # Build RBF interpolator from voxel space
    rbf = Rbf(vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2], data[mask], **rbf_kwargs)
    
    # Evaluate at mesh vertices  
    return rbf(tetmesh.v[:, 0], tetmesh.v[:, 1], tetmesh.v[:, 2])

def make_vol_mesh(
    vol: Union[str, Path, Nifti1Image]
) -> TetMesh:
    """
    Tetrahedral meshing using Gmsh's python API and marching cubes algorithm.
    Returns a lapy.TetMesh object.
    """
    import gmsh
    from skimage.measure import marching_cubes
    from nibabel.nifti1 import Nifti1Image

    # Format / validate arguments
    if isinstance(vol, (str, Path)):
        vol = load(vol)
    elif not isinstance(vol, Nifti1Image):
        raise ValueError("vol must be a Nifti1Image object or a path-like string to a valid "
                         "`.nii` or `.nii.gz` file.")
    
    # Get binary ROI from NIFTI
    roi = (vol.get_fdata() > 0).astype(np.uint8)

    # Marching cubes to extract smoothed surface (replacing mri_mc from FreeSurfer)
    surf_verts, trias, _, _ = marching_cubes(roi, level=0.5, allow_degenerate=False)
    apply_affine(vol.affine, surf_verts, inplace=True)
    surf = TriaMesh(v=surf_verts, t=trias)

    # match volume of mesh to ROI volume
    roi_vol = np.sum(roi) * np.abs(np.linalg.det(vol.affine[:3, :3]))
    surf_vol = surf.volume()
    centroid = surf.centroid()
    surf.v -= centroid
    surf.v *= (roi_vol / surf_vol) ** (1/3)
    surf.v += centroid

    check_surf(surf)

    # Gmsh tetrahedral meshing
    gmsh.initialize()
    gmsh.model.add("vol")

    vert_tags = [gmsh.model.geo.addPoint(float(x), float(y), float(z)) for x, y, z in surf.v]
    tria_tags = []
    for f in trias:
        l1 = gmsh.model.geo.addLine(vert_tags[f[0]], vert_tags[f[1]])
        l2 = gmsh.model.geo.addLine(vert_tags[f[1]], vert_tags[f[2]])
        l3 = gmsh.model.geo.addLine(vert_tags[f[2]], vert_tags[f[0]])

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3])
        s = gmsh.model.geo.addPlaneSurface([cl])
        tria_tags.append(s)

    sl = gmsh.model.geo.addSurfaceLoop(tria_tags)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # Set mesh options according to BrainEigenmodes
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    gmsh.model.mesh.generate(3)

    # Get mesh data
    verts = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    etypes, _, elems = gmsh.model.mesh.getElements()
    tetras = None
    for etype, nodes in zip(etypes, elems):
        if etype == 4:  # Gmsh tetrahedron element type
            tetras = nodes.reshape(-1, 4) - 1  # Convert to 0-based indexing
            break

    # Cleanup API
    gmsh.finalize()

    if tetras is None:
        raise RuntimeError("Gmsh did not generate any tetrahedra. Check if the input surface is"
                           "closed and valid.")

    # Convert to lapy
    mesh = TetMesh(v=verts.astype(np.float64), t=tetras.astype(np.int32))

    # Ensure the generated volume mesh is valid
    check_vol(mesh)

    return mesh