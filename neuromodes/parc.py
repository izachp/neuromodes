from typing import TYPE_CHECKING
import heapq
from scipy.sparse.linalg import eigsh
import numpy as np
from neuromodes.eigen import EigenSolver, _mask_fem_matrices
from lapy import Solver
from scipy.sparse.csgraph import connected_components
from neuromodes.mesh import mask_mesh, fix_mask, check_surf

if TYPE_CHECKING:
    from lapy import TriaMesh
    from numpy import integer
    from numpy.typing import NDArray
    from scipy.sparse import csc_matrix

def mask_mesh_gemini(
    geometry: TriaMesh,
    mask: NDArray[np.bool_],
    return_mapping: bool = False,
):
    # Copy to ensure we can modify it safely
    mask_v = np.array(mask, dtype=bool, copy=True)
    if mask_v.shape != (geometry.v.shape[0],):
        raise ValueError(f"mask must have shape ({geometry.v.shape[0]},)")

    v_masked = geometry.v[mask_v]#.astype(np.float64)

    map_v = np.full(len(mask_v), -1, dtype=int)
    map_v[mask_v] = np.arange(np.sum(mask_v))
    map_t = map_v[geometry.t]
    
    mask_t = np.all(map_t != -1, axis=1)
    t_masked = map_t[mask_t]

    # Clean up unreferenced vertices to prevent LaPy length mismatches
    referenced = np.zeros(len(v_masked), dtype=bool)
    referenced[t_masked] = True
    
    if not np.all(referenced):
        # Update the global mask so it stays perfectly in sync with the kept vertices
        valid_indices = np.where(mask_v)[0]
        mask_v[valid_indices[~referenced]] = False
        
        # Remap triangles to the new compressed vertex array
        v_remap = np.full(len(v_masked), -1, dtype=int)
        v_remap[referenced] = np.arange(np.sum(referenced))
        v_masked = v_masked[referenced]
        t_masked = v_remap[t_masked]

    mesh = geometry.__class__(v=v_masked, t=t_masked)
    
    if return_mapping:
        local_to_global = np.where(mask_v)[0]
        return mesh, local_to_global, mask_v
    return mesh

def make_parcellation_gemini_strict(
    geometry: TriaMesh,
    n_parcels: int,
    seed: int | None = None,
) -> NDArray[np.integer]:
    n_verts = geometry.v.shape[0]

    if n_parcels == 1:
        return np.ones(n_verts, dtype=int)
    if n_parcels == n_verts:
        # Estimate ordering by areas
        return np.argsort(geometry.vertex_areas())[::-1] + 1

    evals, emodes = Solver(geometry).eigs(2, rng=seed)

    empty_mask = np.zeros(n_verts, dtype=bool)
    pangcellation = np.ones(n_verts, dtype=int)

    heap = [(evals[1], np.ones(n_verts, dtype=bool), emodes[:, 1] > 0)]

    for i in range(1, n_parcels):
        print(len(heap))
        if not heap:
            print(f"Warning: Heap is empty. Stopping at {i} parcels.")
            break
        eval2, parc_mask, emode2_pos = heapq.heappop(heap)
        print(f'Parcel {i}: eval2={eval2}')

        _, local_to_global, _ = mask_mesh_gemini(geometry, parc_mask, return_mapping=True)

        mask_a = empty_mask.copy()
        mask_b = empty_mask.copy()
        mask_a[local_to_global[emode2_pos]] = True
        mask_b[local_to_global[~emode2_pos]] = True

        # Boundary Smoothing Filter
        for _ in range(10):
            deg_a = geometry.adj_sym[:, mask_a].getnnz(axis=1)
            deg_b = geometry.adj_sym[:, mask_b].getnnz(axis=1)

            to_b = mask_a & (deg_a < 2) & (deg_b > deg_a)
            to_a = mask_b & (deg_b < 2) & (deg_a > deg_b)

            if not np.any(to_b) and not np.any(to_a):
                break

            mask_a[to_b], mask_b[to_b] = False, True
            mask_a[to_a], mask_b[to_a] = True, False

        # Process both daughters
        try:
            geometry_a, _, mask_a = mask_mesh_gemini(geometry, mask_a, return_mapping=True)
            evals_a, emodes_a = Solver(geometry_a).eigs(2, rng=seed)
            heapq.heappush(heap, (evals_a[1], mask_a, emodes_a[:, 1] > 0))
        except ValueError as e:
            print(f"Parcel {i}, n_verts = {mask_a.sum()}, error in geometry_a: {e}")

        try:
            geometry_b, _, mask_b = mask_mesh_gemini(geometry, mask_b, return_mapping=True)
            evals_b, emodes_b = Solver(geometry_b).eigs(2, rng=seed)
            heapq.heappush(heap, (evals_b[1], mask_b, emodes_b[:, 1] > 0))
        except ValueError as e:
            print(f"Parcel {i}, n_verts = {mask_b.sum()}, error in geometry_b: {e}")

        # Assign labels
        pangcellation[mask_a] = i + 1
        
    return pangcellation

def make_parcellation_gemini(
    geometry: TriaMesh,
    n_parcels: int,
    seed: int | None = None,
) -> NDArray[np.int_]:
    n_verts = geometry.v.shape[0]

    if n_parcels == 1:
        return np.ones(n_verts, dtype=int)
    if n_parcels == n_verts:
        return np.arange(1, n_verts + 1, dtype=int)

    evals, emodes = Solver(geometry).eigs(2, rng=seed)

    empty_mask = np.zeros(n_verts, dtype=bool)
    pangcellation = np.ones(n_verts, dtype=int)

    counter = 0
    heap = [(evals[1], counter, np.ones(n_verts, dtype=bool), emodes[:, 1] > 0)]

    for i in range(1, n_parcels):
        if not heap:
            print(f"Warning: Heap is empty. Stopping at {i} parcels.")
            break

        eval2, _, parc_mask, emode_pos_local = heapq.heappop(heap)

        if eval2 == np.inf:
            v_indices = np.where(parc_mask)[0]
            mask_a, mask_b = empty_mask.copy(), empty_mask.copy()
            if len(v_indices) >= 2:
                mask_a[v_indices[0]] = True
                mask_b[v_indices[1:]] = True
            else:
                mask_a[parc_mask] = True

            eval2_a, eval2_b = np.inf, np.inf
            emode_pos_a, emode_pos_b = np.array([], dtype=bool), np.array([], dtype=bool)

        else:
            # Capture the updated parc_mask from mask_mesh
            _, local_to_global, parc_mask = mask_mesh_gemini(geometry, parc_mask, return_mapping=True)

            mask_a = empty_mask.copy()
            mask_a[local_to_global[emode_pos_local]] = True

            mask_b = empty_mask.copy()
            mask_b[local_to_global[~emode_pos_local]] = True

            # Boundary Smoothing Filter
            for _ in range(10):
                deg_a = geometry.adj_sym[:, mask_a].getnnz(axis=1)
                deg_b = geometry.adj_sym[:, mask_b].getnnz(axis=1)

                to_b = mask_a & (deg_a < 2) & (deg_b > deg_a)
                to_a = mask_b & (deg_b < 2) & (deg_a > deg_b)

                if not np.any(to_b) and not np.any(to_a):
                    if _ > 0:
                        print(f"Boundary smoothing converged after {_} iterations (parcel {i}).")
                    break

                mask_a[to_b], mask_b[to_b] = False, True
                mask_a[to_a], mask_b[to_a] = True, False

            # Helper function to solve or split by connectivity
            def process_submesh(mask_target):
                submesh, l2g, clean_mask = mask_mesh_gemini(geometry, mask_target, return_mapping=True)
                
                if submesh.t.shape[0] == 0 or submesh.v.shape[0] < 3:
                    print(f"Submesh is too small to compute eigenmodes. Returning infinite eigenvalue (parcel {i}).")
                    return submesh, clean_mask, np.inf, np.array([], dtype=bool)
                
                # Check for disconnected islands
                n_comp, labels = connected_components(submesh.adj_sym, directed=False)
                if n_comp > 1:
                    # Disconnected! Bypass LaPy and split the largest island from the rest
                    counts = np.bincount(labels)
                    largest = np.argmax(counts)
                    emode_pos = (labels == largest)
                    # A disconnected graph has a true Fiedler value of 0.0
                    print(f"Submesh is disconnected into {n_comp} components. Returning 0.0 eigenvalue for the largest component (parcel {i}).")
                    return submesh, clean_mask, 0.0, emode_pos
                
                # Normal connected mesh
                try:
                    e_vals, e_modes = Solver(submesh).eigs(2, rng=seed)
                    return submesh, clean_mask, e_vals[1], (e_modes[:, 1] > 0)
                except Exception:
                    print(f"Error computing eigenmodes for submesh (parcel {i}). Returning infinite eigenvalue.")
                    return submesh, clean_mask, np.inf, np.array([], dtype=bool)

            # Process both daughters
            geometry_a, mask_a, eval2_a, emode_pos_a = process_submesh(mask_a)
            geometry_b, mask_b, eval2_b, emode_pos_b = process_submesh(mask_b)

        # Assign labels
        pangcellation[mask_a] = i + 1

        heapq.heappush(heap, (eval2_a, counter, mask_a, emode_pos_a))
        heapq.heappush(heap, (eval2_b, counter, mask_b, emode_pos_b))
        counter += 2

    return pangcellation

def make_parcellation(
    geometry: TriaMesh,
    n_parcels: int,
    seed: int | None = None,
) -> NDArray[integer]:
    # compute eigenmode and eigenvalue of full mesh
    n_verts = geometry.v.shape[0]
    evals, emodes = Solver(geometry).eigs(2, rng=seed)

    # Initialise arrays
    empty_mask = np.zeros(n_verts, dtype=bool)
    parc_masks = [np.ones_like(empty_mask)]
    emode2s_pos = [emodes[:, 1] > 0]
    eval2s = [evals[1]]
    pangcellation = np.ones(n_verts, dtype=int)

    for i in range(1, n_parcels):
        # find index of smallest eigenvalue across all submeshes
        idx = np.argmin(eval2s)  # TODO: consider if it's better to instead insert evals into list where they belong, so that eval2s is always sorted

        # get corresponding data
        parc_mask = parc_masks[idx]
        emode_pos = emode2s_pos[idx]

        # create new masks for daughter parcels
        mask_a = empty_mask.copy()
        mask_a[parc_mask] = emode_pos

        # update pangcellation with new parcel labels
        pangcellation[mask_a] = i + 1
        if i == n_parcels - 1:
            return pangcellation

        mask_b = empty_mask.copy()
        mask_b[parc_mask] = ~emode_pos

        # Expand masks where necessary to ensure no unreferenced (faceless) vertices
        mask_a2 = fix_mask(mask_a, geometry)
        mask_b2 = fix_mask(mask_b, geometry)

        # Create daughter submeshes
        geometry_a = mask_mesh(geometry, mask_a2)
        geometry_b = mask_mesh(geometry, mask_b2)

        # NOTE: temporary
        try:
            check_surf(geometry_a)
        except ValueError as e:
            print(f"Error in geometry_a: {e}")
            return mask_a, mask_a2
        
        try:
            check_surf(geometry_b)
        except ValueError as e:
            print(f"Error in geometry_b: {e}")
            return mask_b, mask_b2

        # Compute eigenmodes and eigenvalues of each daughter submesh
        evals_a, emodes_a = Solver(geometry_a).eigs(2, rng=seed)
        evals_b, emodes_b = Solver(geometry_b).eigs(2, rng=seed)
        
        # Get eigenmode 2 and remove elements corresponding to vertices that were added to the mask
        # during expansion
        #print(f"emode2_a shape: {emodes_a[:, 1].shape}, mask_a2 sum: {mask_a2.sum()}, mask_a sum: {mask_a.sum()}")
        emode2_a = emodes_a[:, 1] # shape (mask_a2.sum(),)
        emode2_a = emode2_a[mask_a[mask_a2]] # shape (mask_a.sum(),)?
        emode2_b = emodes_b[:, 1][mask_b[mask_b2]]

        # Replace parent data with one daughter, and add other daughter to list
        parc_masks[idx] = mask_a
        emode2s_pos[idx] = emode2_a > 0  # I assume it's better to prioritise memory over speed, so calculate and store boolean mask rather than eigenmode values
        eval2s[idx] = evals_a[1]
        parc_masks.append(mask_b)
        emode2s_pos.append(emode2_b > 0)
        eval2s.append(evals_b[1])

        print(f"Created parcel {i+1}/{n_parcels} with {mask_a.sum()} vertices (parent had {parc_mask.sum()})")

def make_parcellation_hetero(
    geometry: TriaMesh,
    n_parcels: int,
    method: str = 'mode2',
    seed: int | None = None,
    **hetero_kwargs
) -> NDArray[integer]:
    # TODO: debug unreferenced vertices being excluded from both daughter parcels
    # TODO: support list of n_parcels, return parcellations along columns
    # TODO: support GMH's equal area method (sort tria areas by mode2, split at median, repeat)
    # TODO: support whatever Jace's method is (Voronoi?)
    # TODO: support custom method via Callable that takes geometry and mask and returns grad and
    # rank
    # TODO: profile speed when inserting evals/emodes/masks into list according to eval and removing
    # argmin call (or just delete parent -> append daughters -> use first element as evals should
    # come sorted as is? but maybe not if heterogeneity is involved?)
    # TODO: precompute initialisation vector and reuse masked versions?
    # TODO: test behaviour for n_parcels = n_verts

    # Format / validate arguments
    if method not in ['mode2']:
        raise ValueError("Method must be 'mode2'.")
    if n_parcels < 2 or n_parcels > geometry.v.shape[0] or int(n_parcels) != n_parcels:
        raise ValueError(f"n_parcels must be an integer in the range [2, {geometry.v.shape[0]}].")

    n_modes = 2

    # unpack heterogeneity kwargs for EigenSolver if applicable
    solver = EigenSolver(geometry, **hetero_kwargs).solve(n_modes, fix_mode1=False,
                                                          standardize=False, seed=seed)

    alpha = hetero_kwargs.get('alpha', None)
    scaling = hetero_kwargs.get('scaling', None)

    # Initialise arrays
    empty_mask = np.zeros(geometry.v.shape[0], dtype=bool)
    parc_masks = [np.ones_like(empty_mask)]
    parc_grads = [solver.emodes[:, -1]]
    parc_ranks = [solver.evals[-1]]
    pangcellation = np.full(geometry.v.shape[0], fill_value=np.nan, dtype=int)

    for i in range(1, n_parcels):
        # find index of smallest eigenvalue across all submeshes
        idx = np.argmin(parc_ranks)

        # get corresponding data
        mask_parent = parc_masks[idx]
        pos_grad = parc_grads[idx] > 0

        # create new masks for daughter parcels
        mask_a = empty_mask.copy()
        
        mask_a[mask_parent] = pos_grad

        submesh_a, mask_a = mask_mesh(geometry, mask_a, return_mask=True)

        mask_b = empty_mask.copy()
        #mask_b[mask_parent] = ~pos_grad
        mask_b[mask_parent & ~mask_a] = True
        submesh_b, mask_b = mask_mesh(geometry, mask_b, return_mask=True)

        # update pangcellation with new parcel labels
        pangcellation[mask_a] = i + 1

        if i == n_parcels - 1:
            return pangcellation

        # Get heterogeneity values for daughters if applicable
        hetero_a = hetero_kwargs['hetero'][mask_a] if solver.hetero is not None else None
        hetero_b = hetero_kwargs['hetero'][mask_b] if solver.hetero is not None else None

        # Compute eigenmode and eigenvalue of each daughter submesh
        solver_a = EigenSolver(
            submesh_a, hetero=hetero_a, alpha=alpha, scaling=scaling
            ).solve(n_modes, fix_mode1=False, standardize=False, atol=None, seed=seed)
        solver_b = EigenSolver(
            submesh_b, hetero=hetero_b, alpha=alpha, scaling=scaling
            ).solve(n_modes, fix_mode1=False, standardize=False, atol=None, seed=seed)

        # Replace parent data with one daughter, and add other daughter to list
        parc_masks[idx] = mask_a
        parc_grads[idx] = solver_a.emodes[:, -1]
        parc_ranks[idx] = solver_a.evals[-1]
        parc_masks.append(mask_b)
        parc_grads.append(solver_b.emodes[:, -1])
        parc_ranks.append(solver_b.evals[-1])

def make_pangcellation_maskfem(
    mass: csc_matrix,
    stiffness: csc_matrix,
    n_parcels: int,
    seed: int | None = 0
) -> NDArray[np.integer]:
    sigma = -0.01

    if n_parcels < 2 or n_parcels > mass.shape[0] or int(n_parcels) != n_parcels:
        raise ValueError(f"n_parcels must be an integer in the range [2, {mass.shape[0]}].")
    
    # Compute eigenmode and eigenvalue of full mesh
    evals, emodes = eigsh(stiffness, k=2, M=mass, sigma=sigma, rng=seed)

    # Initialise arrays
    n_verts = mass.shape[0]
    empty_mask = np.zeros(n_verts, dtype=bool)
    parc_masks = [np.ones_like(empty_mask)]
    parc_grads = [emodes[:, -1]]
    parc_ranks = [evals[-1]]
    pangcellation = np.ones(n_verts, dtype=int)

    for i in range(1, n_parcels):
        # find index of smallest eigenvalue across all submeshes
        idx = np.argmin(parc_ranks)

        # get corresponding data
        mask_parent = parc_masks[idx]
        pos_grad = parc_grads[idx] > 0

        # create new masks for daughter parcels
        mask_a = empty_mask.copy()
        mask_b = empty_mask.copy()

        mask_a[mask_parent] = pos_grad
        mask_b[mask_parent] = ~pos_grad

        mass_a, stiffness_a = _mask_fem_matrices(mask_a, mass, stiffness)
        mass_b, stiffness_b = _mask_fem_matrices(mask_b, mass, stiffness)

        # update pangcellation with new parcel labels
        pangcellation[mask_a] = i + 1

        if i == n_parcels - 1:
            return pangcellation

        # Compute eigenmode and eigenvalue of each daughter submesh
        evals_a, emodes_a = eigsh(stiffness_a, k=2, M=mass_a, sigma=sigma, rng=seed)
        evals_b, emodes_b = eigsh(stiffness_b, k=2, M=mass_b, sigma=sigma, rng=seed)

        # Replace parent data with one daughter, and add other daughter to list
        parc_masks[idx] = mask_a
        parc_grads[idx] = emodes_a[:, -1]
        parc_ranks[idx] = evals_a[-1]
        parc_masks.append(mask_b)
        parc_grads.append(emodes_b[:, -1])
        parc_ranks.append(evals_b[-1])

def make_pangcellation_maskfaces(
    geometry: TriaMesh,
    n_parcels: int,
    seed: int | None = 0
) -> NDArray[np.integer]:
    # Format / validate arguments
    if n_parcels < 2 or n_parcels > geometry.v.shape[0] or int(n_parcels) != n_parcels:
        raise ValueError(f"n_parcels must be an integer in the range [2, {geometry.v.shape[0]}].")

    evals, emodes = Solver(geometry).eigs(2, rng=seed)

    # Initialise arrays
    empty_mask = np.ones(geometry.t.shape[0], dtype=bool)
    parc_masks = [np.ones_like(empty_mask)]
    parc_grads = [emodes[geometry.t, -1].sum(axis=1)]
    parc_ranks = [evals[-1]]
    pangcellation = np.full(geometry.t.shape[0], fill_value=np.nan, dtype=int)

    for i in range(1, n_parcels):
        # find index of smallest eigenvalue across all submeshes
        idx = np.argmin(parc_ranks)

        # get mask of parent parcel, shape (geometry.t.shape[0],)
        mask_parent = parc_masks[idx]

        # create new face maska for daughter parcels, shapes (geometry.t.shape[0],)
        split = parc_grads[idx] > 0
        mask_daughter_a = empty_mask.copy()
        mask_daughter_a[mask_parent] = split
        mask_daughter_b = empty_mask.copy()
        mask_daughter_b[mask_parent] = ~split

        # update pangcellation with new parcel labels
        pangcellation[mask_daughter_a] = i + 1

        if i == n_parcels - 1:
            return pangcellation

        # create daughter submeshes
        mesh_daughter_a = geometry.__class__(geometry.v, geometry.t[mask_daughter_a])
        _ = mesh_daughter_a.rm_free_vertices_()
        mesh_daughter_b = geometry.__class__(geometry.v, geometry.t[mask_daughter_b])
        _ = mesh_daughter_b.rm_free_vertices_()

        # Compute eigenmode and eigenvalue of each daughter submesh
        evals_a, emodes_a = Solver(mesh_daughter_a).eigs(2, rng=seed)  # TODO: wrap in condition that n_verts > 2 to avoid error when submesh is too small to compute mode 2
        evals_b, emodes_b = Solver(mesh_daughter_b).eigs(2, rng=seed)

        # Replace parent data with one daughter, and add other daughter to list
        parc_masks[idx] = mask_daughter_a
        parc_grads[idx] = emodes_a[mesh_daughter_a.t, -1].sum(axis=1)  # TODO: store masks for memory efficiency (and to use full mesh indices?)
        parc_ranks[idx] = evals_a[-1]
        parc_masks.append(mask_daughter_b)
        parc_grads.append(emodes_b[mesh_daughter_b.t, -1].sum(axis=1))
        parc_ranks.append(evals_b[-1])    

# TODO: try method where we compute modes on disconnected mesh by removing connections in
# mass/stiffness