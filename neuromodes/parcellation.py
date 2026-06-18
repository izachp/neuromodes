
from __future__ import annotations
from typing import TYPE_CHECKING
from lapy import TriaMesh, Solver
from lapy.diffgeo import compute_geodesic_f
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import eye
import potpourri3d as pp3d
from scipy.spatial import KDTree
from neuromodes.stats import parcellate

if TYPE_CHECKING:
    from numpy.typing import NDArray

def make_cvt_parcellation(
    geometry: TriaMesh,
    n_parcels: int,
    emodes: NDArray[np.floating] | None = None,
    evals: NDArray[np.floating] | None = None,
    n_iterations: int = 10,
    n_modes: int | None = 200
) -> NDArray[np.int32]:
    """
    Create a cortical parcellation using a Centroidal Voronoi Tessellation (CVT) approach.

    Parameters
    ----------
    geometry : TriaMesh
        The cortical surface mesh to be parcellated.
    n_parcels : int
        The desired number of parcels.
    n_iterations : int, optional
        The maximum number of iterations for the CVT algorithm (default is 10).
    n_modes : int, optional
        The number of eigenmodes to use for the spectral embedding (default is 200).
    Returns
    -------
    NDArray[np.int32]
        An array of shape (n_vertices, n_iterations) containing the parcellation labels for each
        vertex at each iteration.
    """
    verts = geometry.v
    n_verts = len(verts)

    seeds = np.random.choice(n_verts, size=n_parcels, replace=False)
    # TODO: farthest point sampling to get faster convergence

    solver = Solver(geometry)
    if emodes is None or evals is None:
        evals, emodes = solver.eigs(n_modes)

    # 1. Affinity / Voronoi Time Step (Short time for sharp boundaries)
    t_affinity = geometry.avg_edge_length() ** 2
    lu = splu(solver.mass + t_affinity * solver.stiffness)

    # 2. Spectral Embedding Time Step (Longer time for smooth centroids)
    # Approximate expected area of a single parcel
    total_area = geometry.area()
    parcel_area = total_area / n_parcels
    t_spectral = parcel_area / np.pi # t ~ r^2
    
    # Pre-compute the k-dimensional spectral embedding for all vertices
    verts_heat = np.exp(-evals[None, :] * t_spectral) * emodes

    is_moved = np.ones(n_parcels, dtype=bool)
    all_parcellations = np.empty((n_verts, n_iterations), dtype=np.int32)
    
    for i in range(n_iterations):
        is_moved[:] = False 
        
        # Heat diffusion from current seeds
        b0 = np.zeros((n_verts, n_parcels), dtype=np.float64)
        for k, v in enumerate(seeds):
            b0[v, k] = 1.0
        dists = lu.solve(b0) 

        # Assign to nearest seed (highest heat)
        parcellation = np.argmax(dists, axis=1)

        for k in range(n_parcels):
            parcel_verts = np.where(parcellation == k)[0]
            
            # Handle edge case where a seed gets squeezed out and has no vertices
            if len(parcel_verts) == 0:
                continue 

            # Calculate centroid in k-dimensional spectral space
            centroid = np.mean(verts_heat[parcel_verts], axis=0)

            # Find vertex closest to centroid in spectral space
            dists_to_centroid = np.linalg.norm(verts_heat[parcel_verts] - centroid[None, :], axis=1)
            
            # THE FIX: Map local minimum index back to global vertex index
            new_seed_local = np.argmin(dists_to_centroid)
            new_seed = parcel_verts[new_seed_local] 
            
            if new_seed != seeds[k]:
                is_moved[k] = True
                seeds[k] = new_seed

        print(f"Iteration {i+1}: Moved {np.sum(is_moved)} seeds")
        all_parcellations[:, i] = parcellation
        
        # Early stopping if converged
        if not np.any(is_moved):
            print("Converged early!")
            break

    return all_parcellations

def make_cvt_parcellation_v2(
    geometry: TriaMesh,
    n_parcels: int,
    max_iterations: int,
    seed: int = 0
):
    m = 1.0
    n_verts = geometry.v.shape[0]

    # TODO: farthest point sampling to get faster convergence?
    seeds = np.random.default_rng(seed).choice(n_verts, size=n_parcels, replace=False)
    
    # Precompute LU for geodesic computation
    solver_lapy = Solver(geometry, lump=True)
    hmat = solver_lapy.mass + m * geometry.avg_edge_length()**2 * solver_lapy.stiffness
    lu = splu(hmat)

    # Precomputes for Karcher mean
    solver_pp3d = pp3d.MeshVectorHeatSolver(geometry.v, geometry.t)
    basisX, basisY, _ = solver_pp3d.get_tangent_frames()
    tree = KDTree(geometry.v)  # TODO: reconsider, as this is nearest in Euclidean space rather than on the surface

    # initialise arrays
    is_moved = np.ones(n_parcels, dtype=bool)
    sources = np.zeros((n_verts, n_parcels), dtype=np.float64)

    # Voronoi iteration
    for _ in range(max_iterations):
        is_moved[:] = False 
        
        # Heat diffusion from current seeds
        sources = np.zeros((n_verts, n_parcels), dtype=np.float64)
        for k, vert in enumerate(seeds):
            sources[vert, k] = 1.0
        heat = lu.solve(sources)  # shape (n_verts, n_parcels)

        # TODO: unwrap for speed, as this recomputes the same mass/stiffness
        dists = compute_geodesic_f(geometry, heat)  

        # Build parcellation as nearest seed
        parcellation = np.argmin(dists, axis=1)

        # for k in range(n_parcels):  # NOTE: old method, vectorised below
        #     seed_vert = seeds[k]
        #     parcel_verts = np.where(parcellation == k)[0]

        #     # Take Karcher mean of each parcel to get new seed
        #     log_map = solver_pp3d.compute_log_map(seeds[k])
        #     update_vector = 1/len(parcel_verts) * log_map[parcel_verts].sum(axis=0)
        #     update_vector_3d = update_vector[0] * basisX[seed_vert] + update_vector[1] * basisY[seed_vert]
        #     seed_vert_new = tree.query(geometry.v[seed_vert] + update_vector_3d)[1]
        #     if seed_vert_new != seed_vert:
        #         seeds[k] = seed_vert_new
        #         is_moved[k] = True

        # Calculate log map for each seed, keep values for each parcel, stitch into combined map
        log_map_combined = np.zeros((n_verts, 2), dtype=np.float64)
        for k in range(n_parcels):
            parcel_verts = np.where(parcellation == k)[0]
            log_map = solver_pp3d.compute_log_map(seeds[k])
            log_map_combined[parcel_verts, :] = log_map[parcel_verts, :]

        # parcellate the combined log map to get update vectors (TODO: check if non-eye mass is needed)
        update_vectors = parcellate(log_map_combined, parcellation, mass=eye(n_verts))  # shape (n_parcels, 2)
        update_vectors_3d = (update_vectors[:, [0]] * basisX[seeds]  # TODO: can make a single mult by stacking bases
                             + update_vectors[:, [1]] * basisY[seeds])  # shape (n_parcels, 3)
        seeds_updated = tree.query(geometry.v[seeds, :] + update_vectors_3d)[1]

        is_moved = seeds_updated != seeds
        
        # Early stopping if converged
        if not np.any(is_moved):
            break
        seeds = seeds_updated

    return parcellation