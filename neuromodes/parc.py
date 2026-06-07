from typing import TYPE_CHECKING
from scipy.sparse.linalg import eigsh
import numpy as np
from neuromodes.eigen import EigenSolver, _mask_fem_matrices
from neuromodes.mesh import mask_mesh

if TYPE_CHECKING:
    from lapy import TriaMesh
    from numpy import integer
    from numpy.typing import NDArray
    from scipy.sparse import csc_matrix

def make_parcellation(
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
    pangcellation = np.ones(geometry.v.shape[0], dtype=int)

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

def make_pangcellation_mask_method(
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

# TODO: try method where we compute modes on disconnected mesh by removing connections in
# mass/stiffness