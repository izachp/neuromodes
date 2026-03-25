from typing import TYPE_CHECKING
import numpy as np
from neuromodes import EigenSolver
from neuromodes.mesh import mask_mesh

if TYPE_CHECKING:
    from lapy import TriaMesh
    from numpy import integer
    from numpy.typing import NDArray

def make_parcellation(
    geometry: TriaMesh,
    n_parcels: int,
    method: str = 'mode2',
    seed: int | None = None
) -> NDArray[integer]:
    # TODO: see how James' and GMH's functions implement this and handle idiosyncracies
    # TODO: bugfix mode3 method
    # TODO: support list of n_parcels, return parcellations along columns
    # TODO: support GMH's equal area method
    # TODO: support whatever Jace's method is
    # TODO: support custom method via Callable that takes geometry and mask and returns grad and
    # rank
    # TODO: profile speed when inserting evals/emodes/masks into list according to eval and removing
    # argmin call

    # Format / validate arguments
    if method not in ['mode2', 'mode3']:
        raise ValueError("Method must be 'mode2' or 'mode3'.")
    if n_parcels < 2 or n_parcels > geometry.v.shape[0] or int(n_parcels) != n_parcels:
        raise ValueError(f"n_parcels must be an integer in the range [2, {geometry.v.shape[0]}].")
    
    n_modes = 2 if method == 'mode2' else 3
    solver = EigenSolver(geometry).solve(n_modes, fix_mode1=False, seed=seed)

    # Initialise arrays
    mask = np.zeros(geometry.v.shape[0], dtype=bool)
    parc_masks = [np.ones_like(mask)]
    parc_grads = [solver.emodes[:, -1]]
    parc_rank = [solver.evals[-1]]
    pangcellation = np.ones(geometry.v.shape[0], dtype=int)

    for i in range(1, n_parcels):
        # find index of smallest eigenvalue across all submeshes
        idx = np.argmin(parc_rank)

        # get corresponding data
        mask = parc_masks[idx]
        grad = parc_grads[idx]

        pos_grad = grad > 0

        # create new masks for daughter parcels
        mask_a = mask.copy()
        mask_b = mask.copy()
        mask_a[mask] = pos_grad
        mask_b[mask] = ~pos_grad

        # update pangcellation with new parcel labels
        pangcellation[mask_a] = i + 1

        if i == n_parcels - 1:
            return pangcellation, mask_a, mask_b

        # Compute eigenmode and eigenvalue of each daughter submesh
        solver_a = EigenSolver(geometry, mask=mask_a).solve(n_modes, fix_mode1=False,
                                                            standardize=False, atol=None, seed=seed)
        solver_b = EigenSolver(geometry, mask=mask_b).solve(n_modes, fix_mode1=False,
                                                            standardize=False, atol=None, seed=seed)

        # Replace parent data with one daughter, and add other daughter to list
        parc_masks[idx] = solver_a.mask
        parc_grads[idx] = solver_a.emodes[:, -1]
        parc_rank[idx] = solver_a.evals[-1]
        parc_masks.append(solver_b.mask)
        parc_grads.append(solver_b.emodes[:, -1])
        parc_rank.append(solver_b.evals[-1])